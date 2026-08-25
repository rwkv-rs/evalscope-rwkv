import argparse
import httpx
import json
import pytest
from openai.types.chat import ChatCompletion

from evalscope.api.messages import ChatMessageUser
from evalscope.api.model import GenerateConfig
from evalscope.api.registry import get_model_api
from evalscope.api.tool import ToolInfo
from evalscope.arguments import add_argument
from evalscope.config import TaskConfig
from evalscope.constants import EvalType
from evalscope.models.rwkv_openai import RWKVOpenAIAPI
from evalscope.models.utils.rwkv import parse_rwkv_text_tool_calls


TOOLS = [
    ToolInfo.model_validate({
        'name': 'bash',
        'description': 'Run a shell command.',
        'parameters': {
            'type': 'object',
            'properties': {'command': {'type': 'string'}},
            'required': ['command'],
            'additionalProperties': False,
        },
    }),
    ToolInfo.model_validate({
        'name': 'uber.ride',
        'description': 'Request a ride.',
        'parameters': {
            'type': 'object',
            'properties': {'ride_type': {'type': 'string'}},
            'required': ['ride_type'],
            'additionalProperties': False,
        },
    }),
]


def _completion(content: str, *, tool_calls=None, finish_reason: str = 'stop') -> ChatCompletion:
    return ChatCompletion.model_validate({
        'id': 'chatcmpl-rwkv-test',
        'created': 1,
        'model': 'rwkv-test',
        'object': 'chat.completion',
        'choices': [{
            'index': 0,
            'finish_reason': finish_reason,
            'message': {
                'role': 'assistant',
                'content': content,
                'tool_calls': tool_calls,
            },
        }],
        'usage': {
            'prompt_tokens': 10,
            'completion_tokens': 5,
            'total_tokens': 15,
        },
    })


@pytest.mark.parametrize(
    ('content', 'expected'),
    [
        (
            '<think>route</think>\n```json\n[{"bash":{"command":"pwd"}}]\n```',
            [('bash', '{"command":"pwd"}')],
        ),
        (
            '**Tool Call:**\n```json\n{"name":"uber_ride","arguments":{"ride_type":"comfort"}}\n```',
            [('uber.ride', '{"ride_type":"comfort"}')],
        ),
        (
            'Assistant: Tool Call: {"function":{"name":"bash","arguments":{"command":"pwd"}}}',
            [('bash', '{"command":"pwd"}')],
        ),
        (
            '### Tool Call\n[{"bash":{"command":"pwd"}},{"bash":{"command":"ls"}}]',
            [('bash', '{"command":"pwd"}'), ('bash', '{"command":"ls"}')],
        ),
        (
            '{"tool_calls":[{"function":{"name":"bash","arguments":"{\\"command\\":\\"ls\\"}"}}]}',
            [('bash', '{"command":"ls"}')],
        ),
        (
            '{"name":"bash","arguments":{"command":"pwd"}}'
            ', {"name":"bash","arguments":{"command":"ls"}}',
            [('bash', '{"command":"pwd"}'), ('bash', '{"command":"ls"}')],
        ),
    ],
)
def test_parse_rwkv_text_tool_call_shapes(content: str, expected: list[tuple[str, str]]) -> None:
    calls = parse_rwkv_text_tool_calls(content, TOOLS)

    assert [(call.name, call.arguments) for call in calls] == expected


def test_parse_rwkv_text_tool_calls_preserves_invalid_arguments() -> None:
    calls = parse_rwkv_text_tool_calls(
        '{"name":"bash","arguments":"{not-json"}',
        TOOLS,
    )

    assert calls[0].arguments == '{not-json'


@pytest.mark.parametrize(
    'content',
    [
        'not json',
        '{"name":"bash","arguments":{',
        '{"name":"bash","arguments":{}} trailing',
        '{"name":"bash","arguments":{}},',
        '{}[]',
        '[]',
        '[1]',
        '{"arguments":{}}',
        '{"tool_calls":[]}',
        '{"tool_calls":[1]}',
        '{"tool_calls":[{}]}',
    ],
)
def test_parse_rwkv_text_tool_calls_rejects_unrecognized_shapes(content: str) -> None:
    with pytest.raises(ValueError):
        parse_rwkv_text_tool_calls(content, TOOLS)


def test_rwkv_api_converts_text_call_and_keeps_raw_diagnostics() -> None:
    api = object.__new__(RWKVOpenAIAPI)
    api.text_tool_call_compat = True
    raw_content = '<think></think>\n```json\n[{"bash":{"command":"echo hi"}}]\n```'

    choices = api.chat_choices_from_completion(_completion(raw_content), TOOLS)

    choice = choices[0]
    assert choice.stop_reason == 'tool_calls'
    assert choice.message.text == ''
    assert choice.message.tool_calls[0].function.name == 'bash'
    assert choice.message.tool_calls[0].function.arguments == {'command': 'echo hi'}
    assert choice.message.tool_calls[0].internal['raw_arguments'] == '{"command":"echo hi"}'
    diagnostics = choice.message.metadata['rwkv_text_tool_call_compat']
    assert diagnostics == {
        'status': 'converted',
        'raw_content': raw_content,
        'tool_call_count': 1,
    }


def test_rwkv_api_transports_schema_invalid_and_malformed_arguments() -> None:
    api = object.__new__(RWKVOpenAIAPI)
    api.text_tool_call_compat = True

    schema_invalid = api.chat_choices_from_completion(
        _completion('[{"bash":{"unknown":1}}]'),
        TOOLS,
    )[0].message.tool_calls[0]
    malformed = api.chat_choices_from_completion(
        _completion('{"name":"bash","arguments":"{not-json"}'),
        TOOLS,
    )[0].message.tool_calls[0]

    assert schema_invalid.function.arguments == {'unknown': 1}
    assert schema_invalid.parse_error is None
    assert malformed.function.arguments == {}
    assert '{not-json' in malformed.parse_error
    assert malformed.internal['raw_arguments'] == '{not-json'


def test_rwkv_api_preserves_unknown_function_for_benchmark_scoring() -> None:
    api = object.__new__(RWKVOpenAIAPI)
    api.text_tool_call_compat = True

    tool_call = api.chat_choices_from_completion(
        _completion('[{"unknown_tool":{"value":1}}]'),
        TOOLS,
    )[0].message.tool_calls[0]

    assert tool_call.function.name == 'unknown_tool'
    assert tool_call.function.arguments == {'value': 1}


def test_rwkv_api_preserves_native_tool_calls() -> None:
    api = object.__new__(RWKVOpenAIAPI)
    api.text_tool_call_compat = True
    native_calls = [{
        'id': 'call-native',
        'type': 'function',
        'function': {'name': 'bash', 'arguments': '{"command":"pwd"}'},
    }]

    choices = api.chat_choices_from_completion(
        _completion('', tool_calls=native_calls, finish_reason='tool_calls'),
        TOOLS,
    )

    assert choices[0].message.tool_calls[0].id == 'call-native'
    assert choices[0].message.tool_calls[0].internal is None
    assert choices[0].message.metadata is None


def test_rwkv_api_leaves_malformed_text_and_records_parse_error() -> None:
    api = object.__new__(RWKVOpenAIAPI)
    api.text_tool_call_compat = True

    choices = api.chat_choices_from_completion(_completion('{"name":"bash"'), TOOLS)

    assert choices[0].stop_reason == 'stop'
    assert choices[0].message.tool_calls is None
    assert choices[0].message.text == '{"name":"bash"'
    diagnostics = choices[0].message.metadata['rwkv_text_tool_call_compat']
    assert diagnostics['status'] == 'unchanged'
    assert diagnostics['raw_content'] == '{"name":"bash"'
    assert 'complete JSON value' in diagnostics['error']


def test_rwkv_api_can_disable_text_compatibility() -> None:
    api = object.__new__(RWKVOpenAIAPI)
    api.text_tool_call_compat = False

    choices = api.chat_choices_from_completion(
        _completion('[{"bash":{"command":"pwd"}}]'),
        TOOLS,
    )

    assert choices[0].message.tool_calls is None
    assert choices[0].message.metadata is None


def test_rwkv_api_leaves_empty_content_unchanged() -> None:
    api = object.__new__(RWKVOpenAIAPI)
    api.text_tool_call_compat = True

    choices = api.chat_choices_from_completion(_completion(''), TOOLS)

    assert choices[0].message.text == ''
    assert choices[0].message.tool_calls is None
    assert choices[0].message.metadata is None


def test_rwkv_api_requires_boolean_compatibility_flag() -> None:
    with pytest.raises(TypeError, match='must be a boolean'):
        RWKVOpenAIAPI(
            model_name='rwkv-test',
            base_url='http://rwkv.test/v1',
            api_key='test-key',
            text_tool_call_compat='false',
        )


def test_rwkv_api_is_registered_and_exposed_by_cli() -> None:
    assert get_model_api(EvalType.RWKV_OPENAI_API) is RWKVOpenAIAPI
    parser = argparse.ArgumentParser()
    add_argument(parser)

    args = parser.parse_args([
        '--eval-type',
        'rwkv_openai_api',
        '--model-args',
        'text_tool_call_compat=false',
    ])

    assert args.eval_type == EvalType.RWKV_OPENAI_API
    assert args.model_args == {'text_tool_call_compat': False}


def test_rwkv_task_config_uses_remote_api_defaults() -> None:
    config = TaskConfig(
        model='rwkv-test',
        api_url='http://127.0.0.1:8000/v1',
        api_key='test-key',
        eval_type=EvalType.RWKV_OPENAI_API,
    )

    assert config.eval_batch_size == 8
    assert config.generation_config.temperature == 0.0


def test_rwkv_api_generate_end_to_end_with_mock_transport() -> None:
    requests = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json={
                'id': 'chatcmpl-rwkv-integration',
                'created': 1,
                'model': 'rwkv-test',
                'object': 'chat.completion',
                'choices': [{
                    'index': 0,
                    'finish_reason': 'stop',
                    'message': {
                        'role': 'assistant',
                        'content': '```json\n[{"bash":{"command":"pytest -q"}}]\n```',
                    },
                }],
                'usage': {
                    'prompt_tokens': 10,
                    'completion_tokens': 5,
                    'total_tokens': 15,
                },
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handle_request))
    api = RWKVOpenAIAPI(
        model_name='rwkv-test',
        base_url='http://rwkv.test/v1',
        api_key='test-key',
        http_client=http_client,
    )
    try:
        output = api.generate(
            input=[ChatMessageUser(content='Run the tests.')],
            tools=[TOOLS[0]],
            tool_choice='auto',
            config=GenerateConfig(temperature=0.0, max_tokens=128),
        )
    finally:
        api.client.close()

    assert requests[0]['tools'][0]['function']['name'] == 'bash'
    assert output.stop_reason == 'tool_calls'
    assert output.message.tool_calls[0].function.arguments == {'command': 'pytest -q'}
    assert output.message.metadata['rwkv_text_tool_call_compat']['status'] == 'converted'
