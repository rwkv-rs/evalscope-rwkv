import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from evalscope.api.dataset import MemoryDataset, Sample
from evalscope.api.evaluator import CacheManager, TaskState
from evalscope.api.evaluator.cache import ModelResult, review_needs_retry
from evalscope.api.messages.perf_metrics import PerformanceMetrics
from evalscope.api.metric import JudgeSummary, SampleScore, Score
from evalscope.api.model import ModelOutput
from evalscope.constants import ScoreStatus
from evalscope.evaluator.evaluator import DefaultEvaluator, _WorkItem, logger
from evalscope.utils.io_utils import OutputsStructure


def state(index: int, *, error: str | None = None) -> TaskState:
    return TaskState(
        model='model', sample=Sample(id=index, input=f'input {index}'),
        output=ModelOutput.from_content(model='model', content=f'answer {index}', error=error), completed=True,
    )


def score(index: int, value: int = 0, status: ScoreStatus = ScoreStatus.SUCCESS) -> SampleScore:
    return SampleScore(sample_id=index, score=Score(value={'accuracy': value}, status=status))


def cache(tmp_path: Path) -> CacheManager:
    return CacheManager(OutputsStructure(str(tmp_path)), 'model', 'benchmark')


def evaluator(tmp_path: Path, *, use_cache: bool = False, rerun_review: bool = False) -> DefaultEvaluator:
    benchmark = SimpleNamespace(name='benchmark', save_metadata=True, use_batch_scoring=False, finalize=lambda: None)
    config = SimpleNamespace(
        model_id='model', use_cache=str(tmp_path) if use_cache else None, rerun_review=rerun_review,
        ignore_errors=False, collect_perf=False, eval_batch_size=2, limit=None,
    )
    return DefaultEvaluator(benchmark, object(), OutputsStructure(str(tmp_path)), config)


def test_concurrent_records_are_complete_and_synced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = cache(tmp_path)
    synced = []
    monkeypatch.setattr(os, 'fsync', synced.append)
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda index: manager.save_prediction_cache('test', state(index)), range(100)))
    manager.close()

    rows = [json.loads(line) for line in Path(manager.get_prediction_cache_path('test')).read_text().splitlines()]
    assert sorted(row['index'] for row in rows) == list(range(100))
    assert len(synced) == 100


@pytest.mark.parametrize('tail', [b'{"index":', b'{"text":"\xe4'])
def test_resume_repairs_only_partial_tail(tmp_path: Path, tail: bytes) -> None:
    manager = cache(tmp_path)
    manager.save_prediction_cache('test', state(0))
    manager.close()
    path = Path(manager.get_prediction_cache_path('test'))
    original = path.read_bytes()
    with path.open('ab') as stream:
        stream.write(tail)
    restored, remaining = manager.filter_prediction_cache('test', MemoryDataset([state(i)._sample for i in range(2)]))
    assert [item.sample_id for item in restored] == [0]
    assert [item.id for item in remaining] == [1]
    assert path.read_bytes() == original
    manager.save_prediction_cache('test', state(1))
    manager.close()
    assert len(manager._read_cache(str(path))) == 2


def test_complete_json_without_newline_is_retained(tmp_path: Path) -> None:
    manager = cache(tmp_path)
    path = Path(manager.get_prediction_cache_path('test'))
    path.write_text(ModelResult.from_task_state(state(0)).model_dump_json())
    assert len(manager._read_cache(str(path))) == 1
    manager.save_prediction_cache('test', state(1))
    manager.close()
    assert len(manager._read_cache(str(path))) == 2


@pytest.mark.parametrize('contents', [b'{broken}\n', b'{broken}\n{"index": 0}\n'])
def test_malformed_complete_records_are_not_silently_deleted(tmp_path: Path, contents: bytes) -> None:
    manager = cache(tmp_path)
    path = Path(manager.get_prediction_cache_path('test'))
    path.write_bytes(contents)
    with pytest.raises(json.JSONDecodeError):
        manager._read_cache(str(path))
    assert path.read_bytes() == contents


def test_cache_keeps_latest_id_and_retries_only_failed_stage(tmp_path: Path) -> None:
    manager = cache(tmp_path)
    for index in range(5):
        manager.save_prediction_cache('test', state(index))
        manager.save_review_cache('test', state(index), score(index))
    manager.save_prediction_cache('test', state(0, error='request timed out'))
    manager.save_prediction_cache('test', state(1))
    manager.save_review_cache('test', state(1), score(1, 1))
    manager.save_review_cache('test', state(2), score(2, status=ScoreStatus.TRANSPORT_ERROR))
    manager.save_review_cache('test', state(3), score(3, status=ScoreStatus.PARSE_ERROR))
    manager.close()

    predictions, remaining = manager.filter_prediction_cache('test', MemoryDataset([state(i)._sample for i in range(5)]))
    assert [item.id for item in remaining] == [0]
    scores, need_review = manager.filter_review_cache('test', predictions)
    assert [item.sample_id for item in need_review] == [2]
    assert {item.sample_id: item.score.value for item in scores} == {
        1: {'accuracy': 1}, 3: {'accuracy': 0}, 4: {'accuracy': 0},
    }


@pytest.mark.parametrize('status,retry', [(ScoreStatus.EXCLUDED, True), (ScoreStatus.DEGRADED, False)])
def test_native_judge_transport_summary_controls_retry(status: ScoreStatus, retry: bool) -> None:
    result = score(0, status=status)
    result.score.judge_summary = JudgeSummary(status=status, failures={'transport_error': 1})
    assert review_needs_retry(result) is retry
    result.score.judge_summary.failures = {'parse_error': 1}
    assert not review_needs_retry(result)


def test_judge_failure_resumes_without_regeneration_or_duplicate_prediction(tmp_path: Path) -> None:
    first = evaluator(tmp_path)
    first._predict_sample = lambda sample, directory: state(sample.id)

    def fail_review(task_state: TaskState) -> SampleScore:
        raise RuntimeError('judge unavailable')

    first._review_task_state = fail_review
    with pytest.raises(RuntimeError, match='judge unavailable'):
        first._process_work_item(_WorkItem('test', sample=state(0)._sample), str(tmp_path))
    first.finalize()

    resumed = evaluator(tmp_path, use_cache=True)
    resumed._predict_sample = lambda *args: pytest.fail('Completed prediction must not run again')
    resumed._review_task_state = lambda task_state: score(task_state.sample_id)
    context = resumed._collect_work_items({'test': MemoryDataset([state(0)._sample])})
    resumed._run_pool(context)
    resumed.finalize()
    assert len(resumed.cache_manager._read_cache(resumed.cache_manager.get_prediction_cache_path('test'))) == 1
    assert len(resumed.cache_manager._read_cache(resumed.cache_manager.get_review_cache_path('test'))) == 1


def test_resumed_performance_is_restored_once_for_review_only_items(tmp_path: Path) -> None:
    manager = cache(tmp_path)
    prediction = state(0)
    prediction.output.message.perf_metrics = PerformanceMetrics(latency=1.5)
    manager.save_prediction_cache('test', prediction)
    manager.save_prediction_cache('test', prediction)
    manager.close()
    resumed = evaluator(tmp_path, use_cache=True)
    resumed.task_config.collect_perf = True
    resumed._review_task_state = lambda task_state: score(task_state.sample_id)
    context = resumed._collect_work_items({'test': MemoryDataset([prediction._sample])})
    resumed._run_pool(context)
    resumed.finalize()
    assert resumed._perf_request_count == 1
    assert resumed._perf_metric_count == 1
    assert resumed.perf_collector.get_perf_dict()['summary']['n_samples'] == 1


def test_completed_worker_review_survives_other_worker_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    running_review = threading.Event()
    coordinator_failed = threading.Event()
    first = evaluator(tmp_path)
    first.benchmark.load_dataset = lambda: {'test': MemoryDataset([state(i)._sample for i in range(2)])}

    def predict(sample: Sample, directory: str) -> TaskState:
        if sample.id == 0:
            assert running_review.wait(5)
            raise RuntimeError('inference unavailable')
        return state(sample.id)

    def review(task_state: TaskState) -> SampleScore:
        running_review.set()
        assert coordinator_failed.wait(5)
        return score(task_state.sample_id)

    first._predict_sample = predict
    first._review_task_state = review
    monkeypatch.setattr(logger, 'error', lambda *args: coordinator_failed.set())
    with pytest.raises(RuntimeError, match='inference unavailable'):
        first.eval()

    restored = cache(tmp_path)
    predictions, _ = restored.filter_prediction_cache('test', first.benchmark.load_dataset()['test'])
    scores, pending = restored.filter_review_cache('test', predictions)
    assert [item.sample_id for item in scores] == [1]
    assert not pending
    assert not list(tmp_path.rglob('*.rerun-*'))


def test_interrupted_explicit_rerun_preserves_previous_reviews(tmp_path: Path) -> None:
    manager = cache(tmp_path)
    manager.save_prediction_cache('test', state(0))
    manager.save_review_cache('test', state(0), score(0, 1))
    manager.close()
    path = Path(manager.get_review_cache_path('test'))
    original = path.read_bytes()
    resumed = evaluator(tmp_path, use_cache=True, rerun_review=True)
    resumed.benchmark.load_dataset = lambda: {'test': MemoryDataset([state(0)._sample])}

    def interrupt(context: object) -> None:
        resumed.cache_manager.save_review_cache('test', state(0), score(0))
        raise KeyboardInterrupt()

    resumed._run_pool = interrupt
    with pytest.raises(KeyboardInterrupt):
        resumed.eval()
    assert path.read_bytes() == original
    assert not list(tmp_path.rglob('*.rerun-*'))


def test_completed_checkpoint_survives_hard_process_exit(tmp_path: Path) -> None:
    program = '''
import os, sys
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import CacheManager, TaskState
from evalscope.api.model import ModelOutput
from evalscope.utils.io_utils import OutputsStructure
manager = CacheManager(OutputsStructure(sys.argv[1]), 'model', 'benchmark')
state = TaskState(model='model', sample=Sample(id=0, input='input'),
                  output=ModelOutput.from_content(model='model', content='answer'))
manager.save_prediction_cache('test', state)
os._exit(37)
'''
    result = subprocess.run([sys.executable, '-c', program, str(tmp_path)], check=False, timeout=30)
    assert result.returncode == 37
    restored, remaining = cache(tmp_path).filter_prediction_cache('test', MemoryDataset([state(0)._sample]))
    assert [item.sample_id for item in restored] == [0]
    assert not remaining
