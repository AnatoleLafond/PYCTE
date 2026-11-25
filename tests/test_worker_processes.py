"""Worker processes of warningManager (speciationThreads / transportThreads > 1): each one joined by a pipe, its tasks
run in the order sent, errors and warnings sent back, a stopped process replaced at the next call."""
import operator
import os
import warnings
from concurrent.futures.process import BrokenProcessPool

import pytest

from pycte.Source import warningManager


@pytest.fixture(autouse=True)
def _clean():
    warningManager.shutdown()
    yield
    warningManager.shutdown()


def test_pinned_tasks_run_in_their_own_process():
    workers = warningManager.pinnedExecutors(2, group="test")
    pids = [[warningManager.submit(w, os.getpid) for w in workers] for _ in range(3)]
    pids = [[f.result() for f in row] for row in pids]
    assert pids[0] == pids[1] == pids[2] and pids[0][0] != pids[0][1] and os.getpid() not in pids[0]
    assert warningManager.pinnedExecutors(2, group="test") is workers          # kept between calls


def test_results_in_the_order_of_the_tasks():
    worker = warningManager.pinnedExecutors(1, group="test")[0]
    futures = [warningManager.submit(worker, operator.mul, k, 10) for k in range(20)]
    assert [f.result() for f in reversed(futures)] == [10 * k for k in reversed(range(20))]


def test_error_of_the_task_with_its_traceback():
    worker = warningManager.pinnedExecutors(1, group="test")[0]
    with pytest.raises(ZeroDivisionError) as error:
        warningManager.submit(worker, operator.truediv, 1, 0).result()
    assert "RemoteTraceback" in type(error.value.__cause__).__name__           # as with ProcessPoolExecutor
    assert warningManager.submit(worker, operator.add, 2, 3).result() == 5       # the worker goes on


def test_arguments_that_cannot_be_pickled():
    worker = warningManager.pinnedExecutors(1, group="test")[0]
    with pytest.raises(Exception):
        warningManager.submit(worker, operator.add, lambda: 0, 1).result()
    assert warningManager.submit(worker, operator.add, 2, 3).result() == 5       # nothing was sent: still in step


def test_stopped_process_replaced():
    workers = warningManager.pinnedExecutors(2, group="test")
    with pytest.raises(BrokenProcessPool):
        warningManager.submit(workers[0], os._exit, 3).result()
    assert workers[0]._broken
    renewed = warningManager.pinnedExecutors(2, group="test")
    assert renewed is not workers
    assert [warningManager.submit(w, operator.add, 1, 1).result() for w in renewed] == [2, 2]


def test_warm_then_tasks():
    warningManager.warm(2, pinned=True)
    workers = warningManager.pinnedExecutors(2)
    assert [warningManager.submit(w, operator.add, k, 1).result() for k, w in enumerate(workers)] == [1, 2]


def test_pool_runs_a_call_one_task_per_process():
    pool = warningManager.executor(3)
    pids = {f.result() for f in [warningManager.submit(pool, os.getpid) for _ in range(3)]}
    assert len(pids) == 3


def test_worker_warnings_relayed():
    worker = warningManager.pinnedExecutors(1, group="test")[0]
    warningManager.begin(console=False)
    try:
        warningManager.submit(worker, warnings.warn, "message from a worker").result()
        assert any("message from a worker" in text for text, _, _ in warningManager._pending)
    finally:
        warningManager.end()
