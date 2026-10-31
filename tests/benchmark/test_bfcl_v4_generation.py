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
