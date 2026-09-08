"""Exercise automatic retries through the native loader, mock model, caches and publisher boundary."""

import json
from collections import Counter
from pathlib import Path
from unittest.mock import Mock

import pytest

from evalscope import TaskConfig, run_task
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import JudgeSummary, SampleScore
from evalscope.api.model import Model
from evalscope.benchmarks.general_mcq.general_mcq_adapter import GeneralMCQAdapter
from evalscope.constants import ScoreStatus
from evalscope.utils import scoreboard


def _config(tmp_path: Path, ignore_errors: bool) -> TaskConfig:
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    (dataset / 'example_val.jsonl').write_text(
        ''.join(json.dumps({'question': f'question {index}', 'A': 'yes', 'B': 'no', 'answer': 'A'}) + '\n'
                for index in range(3)), encoding='utf-8',
    )
    return TaskConfig(
        model='mock-recovery', eval_type='mock_llm', datasets=['general_mcq'],
        dataset_args={'general_mcq': {'local_path': str(dataset), 'subset_list': ['example']}},
        work_dir=str(tmp_path / 'run'), no_timestamp=True, eval_batch_size=2, eval_retries=1,
        ignore_errors=ignore_errors, collect_perf=False,
    )


def _publisher(monkeypatch: pytest.MonkeyPatch) -> Mock:
    publish = Mock(return_value={'task': 'completed-task'})
    monkeypatch.setattr(scoreboard, 'publish_benchmark_callback', publish)
    monkeypatch.setenv('SCOREBOARD_PUBLICATION_METADATA', 'publication.json')
    monkeypatch.setattr('evalscope.run.time.sleep', Mock())
    return publish


@pytest.mark.parametrize('stage,ignore_errors', [
    ('prediction', False), ('prediction', True), ('review', False), ('review', True), ('judge_transport', True),
])
def test_native_retry_reuses_completed_stages_and_publishes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, ignore_errors: bool,
) -> None:
    config = _config(tmp_path, ignore_errors)
    publish = _publisher(monkeypatch)
    predictions: Counter[int] = Counter()
    reviews: Counter[int] = Counter()
    infer = GeneralMCQAdapter.run_inference
    review = GeneralMCQAdapter.calculate_metrics

    def flaky_prediction(self: GeneralMCQAdapter, model: Model, sample: Sample, **kwargs: object) -> TaskState:
        predictions[sample.id] += 1
        if stage == 'prediction' and sample.id == 1 and predictions[1] == 1:
            raise ConnectionError('provider temporarily unavailable')
        return infer(self, model=model, sample=sample, **kwargs)

    def flaky_review(self: GeneralMCQAdapter, task_state: TaskState) -> SampleScore:
        index = task_state.sample_id
        reviews[index] += 1
        if stage == 'review' and index == 1 and reviews[1] == 1:
            raise ConnectionError('review temporarily unavailable')
        result = review(self, task_state)
        if stage == 'judge_transport' and index == 1 and reviews[1] == 1:
            result.score.value = {}
            result.score.status = ScoreStatus.EXCLUDED
            result.score.judge_summary = JudgeSummary(status=ScoreStatus.EXCLUDED, failures={'transport_error': 1})
        return result

    monkeypatch.setattr(GeneralMCQAdapter, 'run_inference', flaky_prediction)
    monkeypatch.setattr(GeneralMCQAdapter, 'calculate_metrics', flaky_review)
    report = run_task(config)['general_mcq']

    assert report.execution_summary.requested == report.execution_summary.succeeded == 3
    assert report.execution_summary.errored == 0
    assert predictions == {0: 1, 1: 2 if stage == 'prediction' else 1, 2: 1}
    assert reviews == {0: 1, 1: 1 if stage == 'prediction' else 2, 2: 1}
    prediction_path = tmp_path / 'run/predictions/mock-recovery/general_mcq_example.jsonl'
    assert sorted(json.loads(line)['index'] for line in prediction_path.read_text().splitlines()) == [0, 1, 2]
    publish.assert_called_once_with(config.work_dir, 'mock-recovery', 'general_mcq', 'publication.json')


def test_native_retry_exhaustion_keeps_progress_and_never_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, ignore_errors=True)
    publish = _publisher(monkeypatch)
    infer = GeneralMCQAdapter.run_inference
    predictions: Counter[int] = Counter()

    def unavailable(self: GeneralMCQAdapter, model: Model, sample: Sample, **kwargs: object) -> TaskState:
        predictions[sample.id] += 1
        if sample.id == 1:
            raise ConnectionError('provider still unavailable')
        return infer(self, model=model, sample=sample, **kwargs)

    monkeypatch.setattr(GeneralMCQAdapter, 'run_inference', unavailable)
    with pytest.raises(RuntimeError, match='execution failures remain'):
        run_task(config)
    assert predictions == {0: 1, 1: 2, 2: 1}
    review_path = tmp_path / 'run/reviews/mock-recovery/general_mcq_example.jsonl'
    assert sorted(json.loads(line)['index'] for line in review_path.read_text().splitlines()) == [0, 2]
    publish.assert_not_called()
