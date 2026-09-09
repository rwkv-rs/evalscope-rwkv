"""Bounded native retries reuse caches without changing scoring or cancellation."""

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from evalscope.api import registry
from evalscope.arguments import add_argument
from evalscope.config import TaskConfig, parse_task_config
from evalscope.run import _evaluate_with_retries, evaluate_model
from evalscope.utils.io_utils import OutputsStructure


def _config(**kwargs: object) -> TaskConfig:
    return TaskConfig(model='test-model', eval_type='mock_llm', seed=None, **kwargs)


def _evaluator(result: object, **kwargs: object) -> SimpleNamespace:
    evaluate = Mock(side_effect=result) if isinstance(result, BaseException) else Mock(return_value=result)
    return SimpleNamespace(benchmark_name='test-benchmark', model=object(), eval=evaluate, **kwargs)


@pytest.mark.parametrize('value', [-1, -10])
def test_retry_count_cannot_be_negative(value: int) -> None:
    with pytest.raises(ValidationError):
        _config(eval_retries=value)


@pytest.mark.parametrize('value', [0, -1])
def test_concurrency_must_be_positive(value: int) -> None:
    with pytest.raises(ValidationError):
        _config(eval_batch_size=value)


def test_cli_retry_count_and_default() -> None:
    parser = argparse.ArgumentParser()
    add_argument(parser)
    args = parser.parse_args(['--model', 'test', '--eval-type', 'mock_llm', '--eval-retries', '2'])
    assert parse_task_config(args).eval_retries == 2
    assert _config().eval_retries == 0


@pytest.mark.parametrize('option', ['shuffle', 'shuffle_choices', 'few_shot_random'])
def test_randomized_cache_retries_require_stable_seed(tmp_path: Path, option: str) -> None:
    config = _config(eval_retries=1, datasets=['general_mcq'], dataset_args={'general_mcq': {option: True}})
    with pytest.raises(ValueError, match='requires --seed'):
        evaluate_model(config, OutputsStructure(str(tmp_path)))


@pytest.mark.parametrize('rerun_review', [False, True])
def test_retry_rebuilds_evaluator_with_same_cache_and_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rerun_review: bool,
) -> None:
    config = _config(eval_retries=1, eval_batch_size=8, rerun_review=rerun_review)
    first = _evaluator(RuntimeError('judge temporarily unavailable'))
    result = SimpleNamespace(execution_summary=SimpleNamespace(errored=0))
    next_evaluator = _evaluator(result)
    benchmark = SimpleNamespace(validate_judge_strategy=Mock())
    get_benchmark = Mock(return_value=benchmark)
    create_evaluator = Mock(return_value=next_evaluator)
    sleep = Mock()
    monkeypatch.setattr(registry, 'get_benchmark', get_benchmark)
    monkeypatch.setattr(registry, 'create_evaluator', create_evaluator)
    monkeypatch.setattr('evalscope.run.time.sleep', sleep)

    assert _evaluate_with_retries(first, config, OutputsStructure(str(tmp_path))) is result

    retry = create_evaluator.call_args.kwargs
    assert retry['model'] is first.model
    assert retry['benchmark'] is benchmark
    assert retry['task_config'].use_cache == str(tmp_path)
    assert retry['task_config'].eval_batch_size == 8
    assert retry['task_config'].rerun_review is rerun_review
    assert config.use_cache is None
    assert retry['task_config'] is not config
    sleep.assert_called_once_with(5)
    first.eval.assert_called_once()
    next_evaluator.eval.assert_called_once()


@pytest.mark.parametrize('error', [KeyboardInterrupt(), SystemExit(143)])
def test_cancellation_never_restarts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    create = Mock()
    monkeypatch.setattr(registry, 'create_evaluator', create)
    with pytest.raises(type(error)):
        _evaluate_with_retries(_evaluator(error), _config(eval_retries=3), OutputsStructure(str(tmp_path)))
    create.assert_not_called()


def test_exhaustion_preserves_original_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    failure = RuntimeError('still offline')
    evaluator = _evaluator(failure)
    monkeypatch.setattr(registry, 'get_benchmark', Mock(return_value=SimpleNamespace(validate_judge_strategy=Mock())))
    monkeypatch.setattr(registry, 'create_evaluator', Mock(return_value=evaluator))
    sleep = Mock()
    monkeypatch.setattr('evalscope.run.time.sleep', sleep)
    with pytest.raises(RuntimeError) as raised:
        _evaluate_with_retries(evaluator, _config(eval_retries=2), OutputsStructure(str(tmp_path)))
    assert raised.value is failure
    assert evaluator.eval.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [5, 10]


@pytest.mark.parametrize('provider_failure', [False, True])
def test_failed_report_is_not_returned_as_completed_after_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider_failure: bool,
) -> None:
    report = SimpleNamespace(execution_summary=SimpleNamespace(errored=0 if provider_failure else 1))
    evaluator = _evaluator(report, has_retryable_errors=provider_failure)
    monkeypatch.setattr(registry, 'get_benchmark', Mock(return_value=SimpleNamespace(validate_judge_strategy=Mock())))
    monkeypatch.setattr(registry, 'create_evaluator', Mock(return_value=evaluator))
    monkeypatch.setattr('evalscope.run.time.sleep', Mock())
    with pytest.raises(RuntimeError, match='execution failures remain'):
        _evaluate_with_retries(evaluator, _config(eval_retries=1), OutputsStructure(str(tmp_path)))
    assert evaluator.eval.call_count == 2


def test_default_preserves_ignore_errors_report_behavior(tmp_path: Path) -> None:
    report = SimpleNamespace(execution_summary=SimpleNamespace(errored=1))
    evaluator = _evaluator(report)
    assert _evaluate_with_retries(evaluator, _config(ignore_errors=True), OutputsStructure(str(tmp_path))) is report
    evaluator.eval.assert_called_once()


def test_legitimate_zero_or_judge_exclusion_is_not_retried(tmp_path: Path) -> None:
    report = SimpleNamespace(metrics={'score': 0}, execution_summary=SimpleNamespace(errored=0, incomplete=True))
    evaluator = _evaluator(report)
    assert _evaluate_with_retries(evaluator, _config(eval_retries=3), OutputsStructure(str(tmp_path))) is report
    evaluator.eval.assert_called_once()
