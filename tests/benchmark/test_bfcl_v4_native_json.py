import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any, Dict

import pytest

from evalscope.benchmarks.bfcl.v4.utils import compute_entry_result

pytest.importorskip('bfcl_eval')


def handler(native: bool = True) -> Any:
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler

    result = object.__new__(OpenAICompletionsHandler)
    result.is_fc_model = native
    return result


def entry(category: str, schema: Dict[str, Any], gold: Any) -> Dict[str, Any]:
    return {
        'id': category + '_0', 'category': category,
        'function': [{'name': 'fn', 'parameters': {
            'type': 'dict', 'required': ['v'], 'properties': {'v': schema},
        }}],
        'ground_truth': [{'fn': {'v': [gold]}}],
    }


@pytest.mark.parametrize('category,kind,value,gold,valid', [
    ('simple_java', 'boolean', True, True, True),
    ('simple_java', 'boolean', False, True, False),
    ('simple_java', 'boolean', 'true', True, False),
    ('simple_java', 'integer', 8, 8, True),
    ('simple_java', 'integer', '8', 8, False),
    ('simple_java', 'integer', True, 1, False),
    ('simple_java', 'String', 'quoted "value", Unicode 中文', 'quoted "value", Unicode 中文', True),
    ('simple_java', 'HashMap', {'x': 2}, {'x': [2]}, True),
    ('simple_java', 'double', 1.5, 1.5, True),
    ('simple_javascript', 'Boolean', True, True, True),
    ('simple_javascript', 'Boolean', 'true', True, False),
    ('simple_javascript', 'integer', 8, 8, True),
    ('simple_javascript', 'integer', '8', 8, False),
    ('simple_javascript', 'String', 'quoted "value", Unicode 中文', 'quoted "value", Unicode 中文', True),
    ('simple_javascript', 'dict', {'x': 2}, {'x': [2]}, True),
    ('simple_python', 'boolean', True, True, True),
    ('simple_python', 'integer', '8', 8, False),
])
def test_native_json_uses_official_value_checks(
    category: str, kind: str, value: Any, gold: Any, valid: bool,
) -> None:
    prompt = entry(category, {'type': kind}, gold)
    response = [{'fn': json.dumps({'v': value})}]
    original_prompt, original_response = deepcopy(prompt), deepcopy(response)
    result = compute_entry_result(handler(), response, prompt, True)
    assert result['valid'] is valid
    assert prompt == original_prompt
    assert response == original_response
    if not valid:
        assert result['prompt'] == original_prompt


@pytest.mark.parametrize('category,kind', [('simple_java', 'Array'), ('simple_javascript', 'array')])
@pytest.mark.parametrize('value,valid', [([1, 2], True), ([2, 1], False), (['1', '2'], False)])
def test_native_arrays_preserve_element_types_and_order(category: str, kind: str, value: Any, valid: bool) -> None:
    prompt = entry(category, {'type': kind, 'items': {'type': 'integer'}}, [1, 2])
    result = compute_entry_result(handler(), [{'fn': json.dumps({'v': value})}], prompt, True)
    assert result['valid'] is valid


@pytest.mark.parametrize('response', [
    [{'wrong': '{"v": true}'}],
    [{'fn': '{}'}],
    [{'fn': '{"v": true, "extra": 1}'}],
    [{'fn': '{"v": true}'}, {'fn': '{"v": true}'}],
    [{'fn': '{malformed json'}],
])
def test_native_json_keeps_official_name_arity_and_required_checks(response: Any) -> None:
    prompt = entry('simple_java', {'type': 'boolean'}, True)
    assert compute_entry_result(handler(), response, prompt, True)['valid'] is False


@pytest.mark.parametrize('category,kind', [('simple_java', 'boolean'), ('simple_javascript', 'Boolean')])
def test_prompting_source_decoder_is_unchanged(category: str, kind: str) -> None:
    prompt = entry(category, {'type': kind}, True)
    assert compute_entry_result(handler(False), 'fn(v=true);', prompt, True)['valid'] is True
    assert compute_entry_result(handler(False), 'fn(v=false);', prompt, True)['valid'] is False


def test_native_and_prompting_checks_can_run_concurrently() -> None:
    prompt = entry('simple_java', {'type': 'boolean'}, True)

    def check(index: int) -> bool:
        native = bool(index % 2)
        response = [{'fn': '{"v": true}'}] if native else 'fn(v=true);'
        return compute_entry_result(handler(native), response, prompt, True)['valid']

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(pool.map(check, range(16)))
    assert prompt['function'][0]['parameters']['properties']['v']['type'] == 'boolean'
