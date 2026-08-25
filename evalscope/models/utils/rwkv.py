import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

from evalscope.api.tool import ToolInfo


_TOOL_CALL_TEXT_PREFIXES = (
    '**Tool Call:**',
    '### Tool Call',
    'Tool Call:',
)


@dataclass(frozen=True)
class RWKVTextToolCall:
    """One tool call recovered from an RWKV text completion."""

    name: str
    arguments: str


def _candidate_source(text: str) -> str:
    source = str(text or '').strip()
    closing_think = source.rfind('</think>')
    if closing_think >= 0:
        suffix = source[closing_think + len('</think>'):].strip()
        if suffix:
            source = suffix
    if source.startswith('Assistant:'):
        source = source[len('Assistant:'):].lstrip()
    for prefix in _TOOL_CALL_TEXT_PREFIXES:
        if source.startswith(prefix):
            source = source[len(prefix):].lstrip()
            break
    if source.startswith('```'):
        lines = source.splitlines()
        if lines and lines[0].lstrip().startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        source = '\n'.join(lines).strip()
    return source


def _json_values(text: str) -> List[Any]:
    """Decode one JSON value or adjacent top-level JSON objects."""

    source = _candidate_source(text)
    if not source.startswith(('{', '[')):
        raise ValueError('completion must start with a JSON object or array')

    decoder = json.JSONDecoder()
    values: List[Any] = []
    while source:
        try:
            value, end = decoder.raw_decode(source)
        except json.JSONDecodeError as error:
            raise ValueError('completion did not contain a complete JSON value') from error
        values.append(value)
        source = source[end:].lstrip()
        if source.startswith(','):
            remainder = source[1:].lstrip()
            if not remainder:
                raise ValueError('completion contains a trailing comma after the JSON value')
            source = remainder
    if len(values) > 1 and not all(isinstance(value, dict) for value in values):
        raise ValueError('adjacent completion JSON values must all be objects')
    return values


def _native_calls(value: Mapping[str, Any]) -> List[Dict[str, Any]]:
    calls = value.get('tool_calls')
    if not isinstance(calls, list) or not calls:
        raise ValueError('native tool_calls must contain at least one call')
    output: List[Dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, Mapping):
            raise ValueError('native tool call must be an object')
        function = call.get('function')
        if not isinstance(function, Mapping):
            raise ValueError('native tool call must contain function')
        output.append({
            'name': function.get('name'),
            'arguments': function.get('arguments', {}),
        })
    return output


def _candidate_values(value: Any) -> List[Dict[str, Any]]:
    values = value if isinstance(value, list) else [value]
    output: List[Dict[str, Any]] = []
    candidate_fields = {'name', 'arguments', 'confidence', 'evidence', 'id', 'tool_call_id', 'tool_calls'}
    for item in values:
        if not isinstance(item, dict):
            raise ValueError('completion JSON tool call must be an object')
        if 'tool_calls' in item:
            output.extend(_native_calls(item))
        elif 'function' in item:
            output.extend(_native_calls({'tool_calls': [item]}))
        elif len(item) == 1 and next(iter(item)) not in candidate_fields:
            name, arguments = next(iter(item.items()))
            output.append({'name': name, 'arguments': arguments})
        else:
            output.append(item)
    return output


def _wire_tool_name(name: str, tools: List[ToolInfo]) -> str:
    names = [tool.name for tool in tools]
    if name in names:
        return name
    equivalents = [
        candidate for candidate in names
        if candidate.replace('.', '_') == name or candidate.replace('_', '.') == name
    ]
    return equivalents[0] if len(equivalents) == 1 else name


def _arguments_text(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False, separators=(',', ':'))


def parse_rwkv_text_tool_calls(text: str, tools: List[ToolInfo]) -> List[RWKVTextToolCall]:
    """Recover BFCL/RWKV JSON tool-call shapes without judging correctness.

    The parser only normalizes the wire representation. Unknown functions,
    schema-invalid arguments, and malformed argument strings remain available
    to EvalScope's normal tool-call parser and benchmark scorer.
    """

    decoded_values = _json_values(text)
    root: Any = decoded_values[0] if len(decoded_values) == 1 else decoded_values
    values = _candidate_values(root)
    if not values:
        raise ValueError('completion JSON must contain at least one tool call')

    calls: List[RWKVTextToolCall] = []
    for value in values:
        name = value.get('name')
        if not isinstance(name, str) or not name.strip():
            raise ValueError('tool call name must be a non-empty string')
        calls.append(
            RWKVTextToolCall(
                name=_wire_tool_name(name.strip(), tools),
                arguments=_arguments_text(value.get('arguments', {})),
            )
        )
    return calls


__all__ = ['RWKVTextToolCall', 'parse_rwkv_text_tool_calls']
