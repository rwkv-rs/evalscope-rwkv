from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig

pytest.importorskip('bfcl_eval')


def make_adapter(monkeypatch: pytest.MonkeyPatch, max_tokens: Optional[int], is_fc_model: bool = True) -> Any:
    from bfcl_eval.model_handler.api_inference import openai_completion

    client = MagicMock()
    client.chat.completions.create.return_value = 'response'
    monkeypatch.setattr(openai_completion, 'OpenAI', lambda **kwargs: client)
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://example.test/v1')
    config = TaskConfig(
        model='test-model',
        api_key='test-key',
        api_url='https://example.test/v1',
        datasets=['bfcl_v4'],
        generation_config={'max_tokens': max_tokens},
        dataset_args={'bfcl_v4': {'extra_params': {'is_fc_model': is_fc_model}}},
    )
    adapter = get_benchmark('bfcl_v4', config)
    # Exercise the configured handler during prerequisites as well as regular inference.
    monkeypatch.setattr(adapter, '_prereq_inference', lambda: adapter.handler._query_prompting({'message': []}))
    adapter._init_handler()
    return adapter


@pytest.mark.parametrize('is_fc_model', [True, False])
@pytest.mark.parametrize('max_tokens', [None, 2048])
def test_output_limit_reaches_prerequisite_and_all_turn_requests(
    monkeypatch: pytest.MonkeyPatch, max_tokens: Optional[int], is_fc_model: bool,
) -> None:
    adapter = make_adapter(monkeypatch, max_tokens, is_fc_model)
    handler = adapter.handler
    for _ in range(2):
        if is_fc_model:
            handler._query_FC({'message': [], 'tools': []})
        else:
            handler._query_prompting({'message': []})
    calls = handler.client.chat.completions.create.call_args_list
    assert len(calls) == 3
    for call in calls:
        if max_tokens is None:
            assert 'max_tokens' not in call.kwargs
        else:
            assert call.kwargs['max_tokens'] == max_tokens


def test_output_limit_is_per_handler_and_preserves_explicit_request_override(monkeypatch: pytest.MonkeyPatch) -> None:
    first = make_adapter(monkeypatch, 1024).handler
    second = make_adapter(monkeypatch, 4096).handler
    for handler, expected in [(first, 1024), (second, 4096), (first, 1024)]:
        response, elapsed = handler.generate_with_backoff(model='test', messages=[])
        assert response == 'response'
        assert elapsed >= 0
        assert handler.client.chat.completions.create.call_args.kwargs['max_tokens'] == expected
    first.generate_with_backoff(model='test', messages=[], max_tokens=17)
    assert first.client.chat.completions.create.call_args.kwargs['max_tokens'] == 17


IRRELEVANCE_ENTRY = {
    'id': 'irrelevance_0',
    'category': 'irrelevance',
    'question': [[{'role': 'user', 'content': 'What is the capital of France?'}]],
    'function': [{
        'name': 'calculate_area', 'description': 'Area of a triangle.',
        'parameters': {'type': 'dict', 'properties': {'base': {'type': 'integer'}}, 'required': ['base']},
    }],
}


def _score_after_failure(monkeypatch: pytest.MonkeyPatch, error: Exception) -> Any:
    from evalscope.api.dataset import Sample
    from evalscope.api.evaluator import TaskState

    adapter = make_adapter(monkeypatch, None)

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(adapter.handler, 'inference', fail)
    sample = Sample(input='q', target='{}', metadata=dict(IRRELEVANCE_ENTRY))
    from types import SimpleNamespace

    result = adapter._on_inference(SimpleNamespace(name='test-model'), sample)
    state = TaskState(model='test-model', sample=sample, output=result.output, completed=True)
    prediction = result.output.completion
    return result.output, adapter.match_score(prediction, prediction, '{}', state)


@pytest.mark.parametrize('error', [TimeoutError('request timed out'), ConnectionError('connection reset')])
def test_transport_failure_is_not_scored_as_irrelevance_success(
    monkeypatch: pytest.MonkeyPatch, error: Exception,
) -> None:
    """Finding 1: a timed-out request must not count as a correct 'no tool call' answer."""
    from evalscope.constants import ScoreStatus

    output, score = _score_after_failure(monkeypatch, error)
    assert output.error and type(error).__name__ in output.error
    assert score.status is ScoreStatus.TRANSPORT_ERROR
    assert not score.status.is_usable
    assert 'acc' not in score.value


def test_model_side_failure_keeps_official_grading(monkeypatch: pytest.MonkeyPatch) -> None:
    from evalscope.constants import ScoreStatus

    output, score = _score_after_failure(monkeypatch, ValueError('FC response is not valid JSON'))
    assert output.error is None
    assert score.status is ScoreStatus.SUCCESS
    assert score.value == {'acc': 1}


def test_full_generation_config_reaches_bfcl_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding 6: RWKV sampling and chat-template settings must reach every BFCL request."""
    from bfcl_eval.model_handler.api_inference import openai_completion

    client = MagicMock()
    client.chat.completions.create.return_value = 'response'
    monkeypatch.setattr(openai_completion, 'OpenAI', lambda **kwargs: client)
    config = TaskConfig(
        model='test-model', api_key='test-key', api_url='https://example.test/v1', datasets=['bfcl_v4'],
        generation_config={
            'max_tokens': 512, 'temperature': 0.3, 'top_p': 0.6, 'top_k': 20, 'presence_penalty': 0.5,
            'frequency_penalty': 0.1, 'seed': 7, 'stream': True,
            'extra_body': {'chat_template_kwargs': {'enable_thinking': False}, 'penalty_decay': 0.996},
        },
    )
    adapter = get_benchmark('bfcl_v4', config)
    monkeypatch.setattr(adapter, '_prereq_inference', lambda: None)
    adapter._init_handler()
    adapter.handler._query_FC({'message': [{'role': 'user', 'content': 'q'}], 'tools': []})
    sent = client.chat.completions.create.call_args.kwargs
    assert sent['max_tokens'] == 512 and sent['top_p'] == 0.6 and sent['seed'] == 7
    assert sent['presence_penalty'] == 0.5 and sent['frequency_penalty'] == 0.1
    assert sent['temperature'] == 0.3
    assert 'stream' not in sent and 'top_k' not in sent
    assert sent['extra_body'] == {
        'chat_template_kwargs': {'enable_thinking': False}, 'penalty_decay': 0.996, 'top_k': 20,
    }


def test_bfcl_shuffle_order_ignores_global_random_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding 2: consuming random numbers before loading must not change BFCL sample order."""
    import random

    records = [
        {'id': f'simple_{i}', 'question': [[{'role': 'user', 'content': str(i)}]], 'ground_truth': []}
        for i in range(12)
    ]
    adapter = make_adapter(monkeypatch, None)
    adapter.shuffle = True
    monkeypatch.setattr(
        'evalscope.benchmarks.bfcl.v4.bfcl_v4_adapter.process_test_entries',
        lambda **kwargs: (kwargs['test_entries'], []),
    )
    orders = []
    for consumed in (0, 5):
        for _ in range(consumed):
            random.random()
        dataset = adapter._create_dataset_for_category('simple_python', records, [])
        orders.append([sample.metadata['id'] for sample in dataset])
    assert orders[0] == orders[1]
    assert orders[0] != [record['id'] for record in records]


def test_actual_api_trace_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding 8: keep the real request/response, server tool-call ids and reasoning."""
    from types import SimpleNamespace

    from openai.types.chat import ChatCompletion

    from evalscope.api.dataset import Sample
    from evalscope.api.messages import ChatMessageAssistant, ContentReasoning

    adapter = make_adapter(monkeypatch, 256)
    adapter.handler.client.chat.completions.create.return_value = ChatCompletion.model_validate({
        'id': 'resp-1', 'object': 'chat.completion', 'created': 0, 'model': 'test-model',
        'prompt_text': 'User: area?\nAssistant:',
        'usage': {'prompt_tokens': 3, 'completion_tokens': 4, 'total_tokens': 7},
        'choices': [{
            'index': 0, 'finish_reason': 'tool_calls',
            'message': {
                'role': 'assistant', 'content': None, 'reasoning_content': 'Need the area tool.',
                'tool_calls': [{
                    'id': 'call_server_1', 'type': 'function',
                    'function': {'name': 'calculate_area', 'arguments': '{"base": 3}'},
                }],
            },
        }],
    })
    entry = dict(IRRELEVANCE_ENTRY, id='simple_python_0', category='simple_python', ground_truth=[])
    result = adapter._on_inference(SimpleNamespace(name='test-model'), Sample(input='q', metadata=entry))

    trace = result.output.metadata['bfcl_api_trace']
    assert len(trace) == 1
    assert trace[0]['request']['max_tokens'] == 256
    assert trace[0]['response']['prompt_text'] == 'User: area?\nAssistant:'
    assert result.output.metadata['bfcl_trace_aligned'] is True
    assistant = [m for m in result.messages if isinstance(m, ChatMessageAssistant)][0]
    assert [call.id for call in assistant.tool_calls] == ['call_server_1']
    assert any(isinstance(c, ContentReasoning) and c.reasoning == 'Need the area tool.' for c in assistant.content)
