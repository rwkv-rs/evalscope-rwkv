"""Offline regressions for the native code benchmark scoring paths.

The sandbox double executes only the fixed, trusted snippets in this file.
It does not execute model responses or download benchmark datasets.
"""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from evalscope.api.benchmark import DefaultDataAdapter
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig

BCB_TEST = '''import unittest
class TestCases(unittest.TestCase):
    def test_answer(self):
        self.assertEqual(answer(), 42)
'''


@pytest.fixture
def make_adapter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Callable[..., DefaultDataAdapter]:
    def execute(code: str, timeout: int, language: str) -> dict[str, Any]:
        assert language == 'python'
        result = subprocess.run(
            [sys.executable, '-c', code], cwd=tmp_path, capture_output=True, text=True, timeout=timeout,
        )
        return {
            'status': 'success' if result.returncode == 0 else 'error',
            'output': result.stdout,
            'error': result.stderr,
        }

    def create(name: str, **dataset_args: Any) -> DefaultDataAdapter:
        config = TaskConfig(
            model='offline-test', eval_type='mock_llm', datasets=[name],
            dataset_args={name: dataset_args},
        )
        # This test supplies its own executor, so ms-enclave is not required.
        config.sandbox.enabled = True
        adapter = get_benchmark(name, config)
        monkeypatch.setattr(adapter, 'execute_code_in_sandbox', execute)
        return adapter

    return create


@pytest.mark.parametrize('name', ['bigcodebench', 'bigcodebench_hard'])
@pytest.mark.parametrize(('solution', 'passed'), [
    ('def answer(): return 42', True),
    ('def answer(): return 0', False),
    ('def answer(): raise RuntimeError("broken candidate")', False),
    ('def answer(:', False),
    ('raise SystemExit(0)', False),
    ('import os; os._exit(0)', False),
    ('print("candidate output", end="");\ndef answer(): return 42', True),
])
def test_bigcodebench_executes_tests(make_adapter: Callable[..., Any], name: str, solution: str, passed: bool) -> None:
    adapter = make_adapter(name, extra_params={'calibrate': False})
    state = SimpleNamespace(metadata={'task_id': 'synthetic', 'test': BCB_TEST})
    score = adapter.match_score('original response', solution, '', state)

    assert score.value == {'acc': passed}
    assert score.prediction == 'original response'
    assert score.extracted_prediction == solution
    assert score.metadata['scorer_protocol'] == 'explicit-unittest-v1'


def test_bigcodebench_runs_all_dataset_classes(make_adapter: Callable[..., Any]) -> None:
    adapter = make_adapter('bigcodebench', extra_params={'calibrate': False})
    tests = BCB_TEST + '''
class MoreTests(unittest.TestCase):
    def test_second_answer(self):
        self.assertEqual(second_answer(), 7)
'''
    state = SimpleNamespace(metadata={'task_id': 'synthetic', 'test': tests})
    score = adapter.match_score('', 'def answer(): return 42\ndef second_answer(): return 0', '', state)
    assert score.value == {'acc': False}


def test_bigcodebench_does_not_collect_candidate_test_classes(make_adapter: Callable[..., Any]) -> None:
    adapter = make_adapter('bigcodebench', extra_params={'calibrate': False})
    code = '''import unittest
class Unrelated(unittest.TestCase):
    def test_unrelated(self):
        self.fail('Not a benchmark test')
def answer(): return 42
'''
    state = SimpleNamespace(metadata={'task_id': 'synthetic', 'test': BCB_TEST})
    assert adapter.match_score('', code, '', state).value == {'acc': True}


@pytest.mark.parametrize('calibrate', [False, True])
def test_bigcodebench_calibration_is_preserved(make_adapter: Callable[..., Any], calibrate: bool) -> None:
    adapter = make_adapter('bigcodebench', extra_params={'calibrate': calibrate})
    state = SimpleNamespace(metadata={'task_id': 'synthetic', 'test': BCB_TEST, 'code_prompt': 'def answer():'})
    score = adapter.match_score('', '    return 42', '', state)
    assert score.value == {'acc': calibrate}


@pytest.mark.parametrize('test', ['', 'import unittest', 'import unittest\nclass TestCases(unittest.TestCase): pass'])
def test_bigcodebench_rejects_empty_test_data(make_adapter: Callable[..., Any], test: str) -> None:
    adapter = make_adapter('bigcodebench', extra_params={'calibrate': False})
    state = SimpleNamespace(metadata={'task_id': 'synthetic', 'test': test})
    with pytest.raises((ValueError, RuntimeError), match='test data'):
        adapter.match_score('', 'def answer(): return 42', '', state)


@pytest.mark.parametrize('name', ['bigcodebench', 'bigcodebench_hard'])
def test_success_status_without_completion_marker_is_not_a_pass(
    make_adapter: Callable[..., Any], monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    adapter = make_adapter(name, extra_params={'calibrate': False})
    monkeypatch.setattr(adapter, 'execute_code_in_sandbox', lambda **kwargs: {'status': 'success', 'output': ''})
    state = SimpleNamespace(metadata={'task_id': 'x', 'test': BCB_TEST})
    assert adapter.match_score('', 'pass', '', state).value == {'acc': False}


def test_coding_image_specs(make_adapter: Callable[..., Any], tmp_path: Path) -> None:
    bcb = make_adapter('bigcodebench').get_sandbox_image_spec()
    dockerfile = Path(bcb.context_dir, bcb.dockerfile).read_text()
    assert 'ENTRYPOINT []' in dockerfile
    assert 'CMD ["sleep", "infinity"]' in dockerfile
    custom = make_adapter('bigcodebench', extra_params={
        'docker_build_context': str(tmp_path), 'dockerfile': 'Customfile', 'force_rebuild': True,
    }).get_sandbox_image_spec()
    assert custom.context_dir == str(tmp_path)
    assert custom.dockerfile == 'Customfile'
    assert custom.force_rebuild


@pytest.mark.parametrize('name', ['bigcodebench', 'bigcodebench_hard'])
def test_explicit_sandbox_images_are_preserved(make_adapter: Callable[..., Any], name: str) -> None:
    adapter = make_adapter(name, sandbox_config={'image': 'custom-image:latest'})
    assert adapter.get_sandbox_image_spec() is None
    adapter = make_adapter(name)
    adapter._task_config.sandbox.default_config['image'] = 'custom-image:latest'
    assert adapter.get_sandbox_image_spec() is None


@pytest.mark.parametrize('name', ['bigcodebench', 'bigcodebench_hard'])
def test_changed_scoring_has_new_evaluation_version(make_adapter: Callable[..., Any], name: str) -> None:
    assert make_adapter(name).benchmark_meta.evaluation_version == 'v1.1'
