"""Warnings of a run -> warning.log (UTF-8): warn, info, block, abort; submit() relays worker warnings."""

import os
import threading
import traceback
import warnings
from concurrent.futures import Future
try:
    from . import display
except ImportError:
    import display

LOG_NAME = "warning.log"

_lock = threading.Lock()
_pending = []           # messages not yet written to warning.log
_started = False        # warning.log created for the current run
_console = True         # console display (off in the workers: the main process displays)
_saved = None           # filters and display of Python warnings before begin(), restored by end()


def _indented(message):
    return message.replace("\n", "\n    ") + "\n"


def warn(message):
    """Displays `message` and adds it to warning.log."""
    message = str(message).rstrip()
    with _lock:
        _pending.append((_indented(message), 1, message))
    if _console:
        display.message(f"WARNING {message}")


def info(message):
    """Adds `message` to warning.log, not displayed nor counted."""
    with _lock:
        _pending.append((_indented(str(message).rstrip()), 0, None))


def block(text, count=0):
    """Adds a solver report as it is, counted as `count` warnings."""
    text = str(text)
    with _lock:
        _pending.append((text if text.endswith("\n") else text + "\n", int(count), None))


def _fromPython(message, category, filename, lineno, file=None, line=None):
    warn(f"{category.__name__} ({os.path.basename(filename)}:{lineno}) : {message}")


def begin(console=True):
    """Run start: forgets previous warnings and captures Python warnings."""
    global _started, _console, _saved
    with _lock:
        _pending.clear()
        _started = False
    _console = console
    if _saved is None:
        _saved = warnings.catch_warnings()
        _saved.__enter__()
    warnings.showwarning = _fromPython
    warnings.simplefilter("always")


def _write(mode, header, centralDict):
    with _lock:
        messages = _pending[:]
        _pending.clear()
    if mode == "a" and not messages:
        return
    with open(LOG_NAME, mode, encoding="utf-8") as log:
        log.write(header + "".join(text for text, _, _ in messages))
    if centralDict is not None:
        centralDict['warningNbr'] = centralDict.get('warningNbr', 0) + sum(count for _, count, _ in messages)


def appendOutsideRun(text):
    """Appends `text` to warning.log outside a run."""
    with open(LOG_NAME, "a", encoding="utf-8") as log:
        log.write(text if text.endswith("\n") else text + "\n")


def start(header, centralDict=None):
    """Creates warning.log: `header`, then the pending warnings."""
    global _started
    _started = True
    _write("w", header, centralDict)


def flush(centralDict=None):
    """Writes the pending warnings (after start())."""
    if _started:
        _write("a", "", centralDict)


def abort(error):
    """Writes a fatal error and its traceback to warning.log."""
    text = "".join(traceback.format_exception(error)).rstrip()
    block(f"pycte : run aborted : {type(error).__name__} : {error}\n    " + text.replace("\n", "\n    ") + "\n\n", 1)
    if not _started:
        start("pycte : run aborted before the coupling started\n")


def end():
    """Run end: writes the pending warnings and restores the Python warning filters."""
    global _saved
    flush()
    if _saved is not None:
        _saved.__exit__(None, None, None)
        _saved = None


def _inWorker(fn, args, kwargs):
    """Runs fn in a worker -> (result, warnings emitted)."""
    begin(console=False)
    try:
        result = fn(*args, **kwargs)
    finally:
        end()
    return result, list(_pending)


def _serve(connection):
    """Worker process of a _Worker: runs the tasks (fn, args, kwargs) received on its pipe, one at a time, and sends
    back (True, result) or (False, error with its traceback, as ProcessPoolExecutor does), until None or the end of the
    pipe."""
    from concurrent.futures.process import _ExceptionWithTraceback
    while True:
        try:
            task = connection.recv()
        except (EOFError, OSError):
            return
        except BaseException as error:              # task that cannot be unpickled here: its error is the reply
            connection.send((False, _ExceptionWithTraceback(error, error.__traceback__)))
            continue
        if task is None:
            return
        fn, args, kwargs = task
        try:
            reply = (True, fn(*args, **kwargs))
        except BaseException as error:
            reply = (False, _ExceptionWithTraceback(error, error.__traceback__))
        try:
            connection.send(reply)
        except Exception as error:                  # reply that cannot be pickled
            problem = RuntimeError(f"pycte : the reply of a worker process could not be sent back : {error!r}")
            connection.send((False, _ExceptionWithTraceback(problem, error.__traceback__)))


class _Reply(Future):
    """Future of a task sent to a _Worker: result() and exception() read the replies of the worker (in the order of
    its tasks) until this one has come, in the thread that waits for it."""

    def __init__(self, worker):
        super().__init__()
        self._worker = worker

    def result(self, timeout=None):
        self._worker.wait(self)
        return super().result(timeout)

    def exception(self, timeout=None):
        self._worker.wait(self)
        return super().exception(timeout)


class _Worker:
    """One worker process kept between steps, joined by a pipe: its tasks run one at a time, in the order sent (as in
    a ProcessPoolExecutor(max_workers=1), which it replaces). The reply comes back on the pipe to the thread that waits
    for it, without the threads and queues of a pool: ~0.5 ms less per task here (0.07 instead of 0.6 ms for the
    round trip of two processes). A process that stops: BrokenProcessPool for its tasks, and _broken (new process at
    the next pinnedExecutors / executor)."""

    def __init__(self):
        import collections
        import multiprocessing
        parent, child = multiprocessing.Pipe()
        self._process = multiprocessing.Process(target=_serve, args=(child,), daemon=True)
        self._process.start()
        child.close()
        self._connection = parent
        self._waiting = collections.deque()         # replies to come, in the order of the tasks
        self._broken = False

    def _stopped(self):
        from concurrent.futures.process import BrokenProcessPool
        self._broken = True
        error = BrokenProcessPool("pycte : a worker process stopped (its task was not completed)")
        while self._waiting:
            self._waiting.popleft().set_exception(error)

    def submit(self, fn, *args, **kwargs):
        reply = _Reply(self)
        if self._broken:
            from concurrent.futures.process import BrokenProcessPool
            reply.set_exception(BrokenProcessPool("pycte : a worker process stopped"))
            return reply
        try:
            self._connection.send((fn, args, kwargs))
        except (EOFError, OSError):
            self._waiting.append(reply)
            self._stopped()
            return reply
        except Exception as error:                  # arguments that cannot be pickled: nothing was sent
            reply.set_exception(error)
            return reply
        self._waiting.append(reply)
        return reply

    def pending(self):
        return len(self._waiting)

    def wait(self, reply):
        while not reply.done() and self._waiting:
            first = self._waiting[0]
            try:
                ok, value = self._connection.recv()
            except (EOFError, OSError):
                self._stopped()
                break
            except Exception as error:              # reply that cannot be unpickled
                self._waiting.popleft()
                first.set_exception(error)
                continue
            self._waiting.popleft()
            if ok:
                first.set_result(value)
            else:
                first.set_exception(value)

    def shutdown(self, wait=True, cancel_futures=False):
        while self._waiting:
            self._waiting.popleft().cancel()
        try:
            self._connection.send(None)
        except (EOFError, OSError, ValueError):
            pass
        self._process.join(timeout=5 if wait else 0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        self._connection.close()


class _Pool:
    """`workers` _Worker kept between steps; a task goes to the worker with the fewest tasks waiting (the first one
    on a tie): the tasks of a call, sent at once, run one per process."""

    def __init__(self, workers):
        self._workers = [_Worker() for _ in range(workers)]

    @property
    def _broken(self):
        return any(w._broken for w in self._workers)

    def submit(self, fn, *args, **kwargs):
        worker = min(self._workers, key=lambda w: w.pending())
        return worker.submit(fn, *args, **kwargs)

    def shutdown(self, wait=True, cancel_futures=False):
        for worker in self._workers:
            worker.shutdown(wait=wait, cancel_futures=cancel_futures)


class _Driver:
    """wait() of the _Reply returned by submit: reads the reply of the task it relays (inner)."""

    def __init__(self, inner):
        self._inner = inner

    def wait(self, reply):
        if not reply.done():
            self._inner.exception()


_executors = {}


def executor(workers):
    """Pool of `workers` processes, kept between steps."""
    pool = _executors.get(workers)
    if pool is None or getattr(pool, '_broken', False):
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        pool = _executors[workers] = _Pool(workers)
    return pool


_pinned = {}


def pinnedExecutors(workers, group=None):
    """`workers` single-process executors kept between steps: task k always runs in process k (state kept per block).
    group: a solver with processes of its own (xGEMS), apart from those of PhreeqC (which keep their IPhreeqc
    instance)."""
    key = workers if group is None else (group, workers)
    pools = _pinned.get(key)
    if pools is None or any(getattr(pool, '_broken', False) for pool in pools):
        for pool in pools or []:
            pool.shutdown(wait=False, cancel_futures=True)
        pools = _pinned[key] = [_Worker() for _ in range(workers)]
    return pools


def _ready():
    return os.getpid()


def warm(workers, pinned=False, group=None):
    """Starts the processes of executor(workers) (or pinnedExecutors(workers, group)) without waiting for them."""
    pools = pinnedExecutors(workers, group) if pinned else [executor(workers)] * workers
    for pool in pools:
        pool.submit(_ready)


def shutdown():
    """Stops the pools of executor() and pinnedExecutors()."""
    while _executors:
        _executors.popitem()[1].shutdown(wait=True, cancel_futures=True)
    while _pinned:
        for pool in _pinned.popitem()[1]:
            pool.shutdown(wait=True, cancel_futures=True)


def submit(executor, fn, *args, **kwargs):
    """executor.submit, relaying the worker warnings to warning.log."""
    inner = executor.submit(_inWorker, fn, args, kwargs)
    # completed by relay below when the reply of the worker is read: outer.result() reads it (inner.exception())
    outer = _Reply(_Driver(inner)) if isinstance(inner, _Reply) else Future()

    def relay(done):
        try:
            result, messages = done.result()
        except BaseException as error:
            outer.set_exception(error)
            return
        with _lock:
            _pending.extend(messages)
        if _console:
            for _, _, shown in messages:
                if shown is not None:
                    display.message(f"WARNING {shown}")
        outer.set_result(result)

    inner.add_done_callback(relay)
    return outer
