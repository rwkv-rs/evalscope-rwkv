import threading

from evalscope.utils.function_utils import run_in_threads_with_progress


def test_short_request_refills_slot_before_slow_request_finishes() -> None:
    third_started = threading.Event()

    def worker(index: int) -> int:
        if index == 0:
            # ALL_COMPLETED would wait its full one-second heartbeat before refilling.
            assert third_started.wait(0.8)
        if index == 2:
            third_started.set()
        return index

    assert run_in_threads_with_progress(range(3), worker, desc='refill', max_workers=2) == [0, 1, 2]


def test_live_concurrency_stays_bounded_and_results_remain_in_input_order() -> None:
    lock = threading.Lock()
    barrier = threading.Barrier(4)
    active = 0
    peak = 0

    def worker(index: int) -> int:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait(timeout=5)
        with lock:
            active -= 1
        return index

    assert run_in_threads_with_progress(range(40), worker, desc='bounded', max_workers=4) == list(range(40))
    assert peak == 4
