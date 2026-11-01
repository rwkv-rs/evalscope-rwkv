import asyncio
from types import SimpleNamespace
from typing import Optional

import pytest
from openai.types.chat import ChatCompletionChunk

from evalscope.api.model import GenerateConfig, ModelOutput, ModelUsage
from evalscope.models.openai_compatible import OpenAICompatibleAPI


def _chunk(content: str, *, finish_reason: Optional[str] = None) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate({
        'id': 'completion-id',
        'created': 1,
        'model': 'test-model',
        'object': 'chat.completion.chunk',
        'choices': [{
            'index': 0,
            'finish_reason': finish_reason,
            'delta': {'content': content},
        }],
    })


def _prepare_api(monkeypatch: pytest.MonkeyPatch) -> OpenAICompatibleAPI:
    api = object.__new__(OpenAICompatibleAPI)
    api.base_url = 'https://example.test/v1'
    api.model_name = 'test-model'
    api.resolve_tools = lambda tools, tool_choice, config: (tools, tool_choice, config)
    api.completion_params = lambda config, tools: {'model': 'test-model', 'stream': True}
    api.validate_request_params = lambda request: None
    api.on_response = lambda response: None
    api.chat_choices_from_completion = lambda completion, tools: []

    monkeypatch.setattr('evalscope.models.openai_compatible.openai_chat_messages', lambda *args, **kwargs: [])

    def model_output(completion, choices):
        return SimpleNamespace(
            content=completion.choices[0].message.content,
            usage=None,
            message=SimpleNamespace(),
            time=None,
        )

    monkeypatch.setattr('evalscope.models.openai_compatible.model_output_from_openai', model_output)
    return api


def test_generate_retries_when_stream_consumption_fails(monkeypatch):
    api = _prepare_api(monkeypatch)
    attempts = 0

    def create(**request):
        nonlocal attempts
        attempts += 1
        attempt = attempts

        def stream():
            if attempt == 1:
                yield _chunk('discarded partial response')
                raise ConnectionError('stream interrupted by upstream gateway')
            yield _chunk('complete response', finish_reason='stop')

        return stream()

    api.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    result = api.generate([], [], None, GenerateConfig(retries=2, retry_interval=0, stream=True))

    assert attempts == 2
    assert result.content == 'complete response'


def test_generate_async_retries_when_stream_consumption_fails(monkeypatch):
    api = _prepare_api(monkeypatch)
    attempts = 0

    async def create(**request):
        nonlocal attempts
        attempts += 1
        attempt = attempts

        async def stream():
            if attempt == 1:
                yield _chunk('discarded partial response')
                raise ConnectionError('stream interrupted by upstream gateway')
            yield _chunk('complete response', finish_reason='stop')

        return stream()

    async_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(OpenAICompatibleAPI, 'async_client', property(lambda self: async_client))

    result = asyncio.run(
        api.generate_async([], [], None, GenerateConfig(retries=2, retry_interval=0, stream=True))
    )

    assert attempts == 2
    assert result.content == 'complete response'


def _completion_api(monkeypatch: pytest.MonkeyPatch, **model_args) -> OpenAICompatibleAPI:
    from openai.types.chat import ChatCompletion

    api = OpenAICompatibleAPI('test-model', base_url='https://example.test/v1', api_key='k', **model_args)
    completion = ChatCompletion.model_validate({
        'id': 'c', 'object': 'chat.completion', 'created': 1, 'model': 'test-model',
        'prompt_text': 'User: hi\nAssistant:',
        'usage': {'prompt_tokens': 3, 'completion_tokens': 4, 'total_tokens': 7},
        'choices': [{
            'index': 0, 'finish_reason': 'stop', 'stop_reason': '\nUser:',
            'message': {'role': 'assistant', 'content': 'done'},
        }],
    })
    monkeypatch.setattr(api.client.chat.completions, 'create', lambda **kwargs: completion)
    return api


def test_server_stop_reason_is_preserved_at_exact_output_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding 7: a normally finished response at the exact budget must stay ``stop``."""
    output = _completion_api(monkeypatch).generate([], [], None, GenerateConfig(max_tokens=4, stream=False))
    assert output.stop_reason == 'stop'
    assert 'stop_reason_corrected' not in (output.metadata or {})


def test_server_audit_fields_are_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding 8: the server-rendered prompt and matched stop are preserved."""
    output = _completion_api(monkeypatch).generate([], [], None, GenerateConfig(max_tokens=64, stream=False))
    assert output.metadata['prompt_text'] == 'User: hi\nAssistant:'
    assert output.metadata['server_stop_reasons'] == {'0': '\nUser:'}


def test_explicit_output_limit_correction_records_server_value(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _completion_api(monkeypatch, stop_reason_correction='output_limit')
    at_limit = api.generate([], [], None, GenerateConfig(max_tokens=4, stream=False))
    assert at_limit.stop_reason == 'max_tokens'
    assert at_limit.metadata['server_stop_reason'] == 'stop'
    assert at_limit.metadata['stop_reason_corrected'] == 'output_limit'
    below = api.generate([], [], None, GenerateConfig(max_tokens=5, stream=False))
    assert below.stop_reason == 'stop'


def test_unknown_stop_reason_correction_is_rejected() -> None:
    with pytest.raises(ValueError, match='stop_reason_correction'):
        OpenAICompatibleAPI('test-model', base_url='https://example.test/v1', api_key='k', stop_reason_correction='x')
