# Copyright (c) Alibaba, Inc. and its affiliates.

import ast
import secrets
from pathlib import Path
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.metric.semantics import MetricSelector
from evalscope.api.mixin import CodeExecutionSandboxMixin
from evalscope.api.registry import register_benchmark
from evalscope.api.sandbox import DockerImageSpec
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

PROMPT = """You are an expert Python programmer, and here is your task: {question} Your code should pass these tests:

{tests}"""  # noqa: E501

DESCRIPTION = """
## Overview

MBPP Plus is a fortified version of the MBPP benchmark, created to improve evaluation reliability for basic Python programming synthesis. It addresses quality issues in the original dataset and significantly increases test coverage for each problem.

## Task Description

- **Task Type**: Code Generation (Python)
- **Input**: Programming task description with test assertions
- **Output**: Python function implementation
- **Improvements**: Sanitized data with expanded test suites

## Key Features

- Corrected reference code and clarified problem statements
- Massively expanded test cases per problem (automated generation)
- LLM-based and mutation-based test generation strategies
- Exposes edge-case bugs missed by original MBPP
- More rigorous and accurate code correctness evaluation

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- **Security Warning**: By default, code is executed in the local environment. We strongly recommend using sandbox execution. See the [sandbox documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html) for details.
- Supports `pass@k` metric calculation
- Default timeout is 20 seconds per problem
- Stricter evaluation than original MBPP benchmark
"""  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='mbpp_plus',
        evaluation_version='v1.1',
        pretty_name='MBPP-Plus',
        tags=[Tags.CODING],
        description=DESCRIPTION,
        dataset_id='evalscope/mbppplus',
        subset_list=['default'],
        metric_list=['acc'],
        aggregation='mean_and_pass_at_k',
        primary_metric=MetricSelector(name='accuracy', aggregation='pass_at_k', dimensions={'k': 1}),
        eval_split='test',
        prompt_template=PROMPT,
        review_timeout=20,
        sandbox_config={
            'image': 'python3.11-numpy',
            'tools_config': {
                'shell_executor': {},
                'python_executor': {}
            }
        }
    )
)
class MBPPplusAdapter(CodeExecutionSandboxMixin, DefaultDataAdapter):
    """
    MBPPplus adapter using the new data processing framework.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_sandbox_image_spec(self) -> DockerImageSpec | None:
        """Reuse the NumPy image needed by the expanded EvalPlus test programs."""
        if self._task_config is not None and self._task_config.sandbox.default_config.get('image'):
            return None
        if self.sandbox_config.get('image') != 'python3.11-numpy':
            return None
        context = Path(__file__).parent.parent / 'humanevalplus' / 'docker'
        return DockerImageSpec(
            name_prefix='python3.11-numpy',
            context_dir=str(context),
            dockerfile=str(context / 'Dockerfile'),
            cache_key_parts=['sandbox', 'python3.11-numpy'],
        )

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a data record to a Sample object."""

        return Sample(
            input=record['prompt'],
            target=record['code'],
            metadata={
                'task_id': record['task_id'],
                'source_file': record['source_file'],
                'test_imports': record['test_imports'],
                'test_list': record['test_list'],
                'test': record['test'],
            }
        )

    def format_prompt_template(self, sample: Sample) -> str:
        tests = '\n'.join(sample.metadata['test_list'])
        return self.prompt_template.format(question=sample.input, tests=tests)

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Extract code from the prediction."""
        code = self.postprocess_completion(prediction)
        if 'if __name__ ==' in code:
            code = code[:code.index('if __name__ ==')]
        return code

    @classmethod
    def postprocess_completion(cls, completion, stop_words=['\nassert', '\n"""']):
        from evalscope.utils.code_utils import extract_code_from_freeform_completion

        if '[DONE]' in completion:
            completion = completion[:completion.index('[DONE]')]

        code, _ = extract_code_from_freeform_completion(completion, 'python', first_block_only=True)

        for st in stop_words:
            index = code.find(st)
            if index != -1:
                code = code[:index]
        return code

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """Execute the complete expanded tests, retaining test_list only in the prompt."""

        if not self.use_sandbox:
            raise RuntimeError(
                f'{self.pretty_name} benchmark requires sandboxed code '
                'execution for safety reasons. Please set use_sandbox in the task configuration.'
            )

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )
        problem = task_state.metadata
        test = problem['test']
        if not any(isinstance(node, (ast.Assert, ast.For)) for node in ast.parse(test).body):
            raise ValueError('MBPP+ expanded test data contains no executable checks.')
        imports = problem.get('test_imports', [])
        if isinstance(imports, str):
            imports = [imports]
        marker = 'EVALSCOPE_MBPP_PLUS_PASSED_' + secrets.token_hex(16)
        completion = '\n'.join([*imports, filtered_prediction, test, f'print("\\n" + {marker!r})'])

        res = self.execute_code_in_sandbox(code=completion, timeout=self.review_timeout, language='python')
        passed = res.get('status') == 'success' and marker in str(res.get('output', '')).splitlines()
        # Set score values
        score.value = {'acc': passed}
        score.metadata = {
            'task_id': problem['task_id'],
            'timeout': self.review_timeout,
            'execution_result': res,
            'scorer_protocol': 'full-expanded-tests-v1',
        }
        score.main_score_name = 'acc'

        return score
