"""Console display of the coupling loop: 'progressBar' (one line redrawn: steps done, elapsed and remaining time) or 'steps' (step by step messages of pycte and of the solvers)."""

import os
import shutil
import sys
import threading
import time

MODES = ('progressBar', 'steps')
REFRESH = 0.5
WIDTH = 30

_stepByStep = True
_bar = None
_lock = threading.RLock()     # bar redrawn by the main thread, messages also relayed by the pool callback threads


def canonical(mode):
    """Display mode name among MODES (case-insensitive), ValueError otherwise."""
    for name in MODES:
        if isinstance(mode, str) and mode.strip().lower() == name.lower():
            return name
    raise ValueError(f"consoleDisplay = {mode!r} : expected one of {', '.join(repr(m) for m in MODES)}")


def configure(mode):
    """Sets the display mode of this process."""
    global _stepByStep
    _stepByStep = canonical(mode) == 'steps'


def stepByStep():
    """True in 'steps' mode."""
    return _stepByStep


def stepPrint(*args, **kwargs):
    """print() of a per-step message, shown in 'steps' mode only."""
    if _stepByStep:
        print(*args, **kwargs)


def _writeTime(tps, arr=2):
    if tps >= 3600 * 24:
        return f"{tps / (3600 * 24):.{arr}f} d"
    elif tps >= 3600:
        return f"{tps / 3600:.{arr}f} h"
    elif tps >= 60:
        return f"{tps / 60:.{arr}f} min"
    elif tps < 1:
        return f"{tps * 1000:.{arr}f} msec"
    else:
        return f"{tps:.{arr}f} sec"


def _inKernel(stream):
    return type(stream).__module__.split('.')[0] in ('ipykernel', 'spyder_kernels')


def _isTerminal(stream):
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _encodes(stream, text):
    try:
        text.encode(getattr(stream, 'encoding', None) or 'ascii')
        return True
    except (LookupError, UnicodeEncodeError):
        return False


class _ProgressBar:
    """ When using the screen command in Unix, sometimes the progress bar is displayed twice. God knows why."""
    def __init__(self, nSteps, tEnd, start):
        self.nSteps, self.tEnd, self.start = nSteps, tEnd, start
        self.stream = sys.stdout
        inKernel = _inKernel(self.stream)
        self.live = inKernel or _isTerminal(self.stream)
        self.flush = self.live and not inKernel
        self.full, self.empty = ('█', '░') if _encodes(self.stream, '█░') else ('#', '-')
        self.nextDraw = 0.0
        self.tenth = -1
        self.width = 0
        self.open = False
        self.midLine = False      # console text written while the bar is shown, its line not ended yet
        self.last = (0, 0)
        self.shown = set()

    def line(self, done, t, now):
        filled = WIDTH * done // self.nSteps
        elapsed = now - self.start
        text = (f"|{self.full * filled}{self.empty * (WIDTH - filled)}| {100 * done // self.nSteps:3d} % "
                f"({done}/{self.nSteps} steps) | elapsed {_writeTime(elapsed)}")
        if done < self.nSteps:
            remaining = f"~ {_writeTime(elapsed * (self.tEnd - t) / t)}" if t > 0 else "..."
            text += f" | remaining {remaining}"
        return text

    def update(self, done, t):
        now = time.perf_counter()
        self.last = (done, t)
        last = done >= self.nSteps
        if self.live:
            if now < self.nextDraw and not last:
                return
            self.nextDraw = now + REFRESH
            text = self.line(done, t, now)
            if self.flush:
                # A line wider than the terminal wraps and "\r" then only goes back to its last row: one bar per redraw
                maxWidth = max(1, shutil.get_terminal_size().columns - 1)
                text = text[:maxWidth]
                self.width = min(self.width, maxWidth)
            if self.midLine:
                # the bar goes below an unfinished console line instead of overwriting it
                self.stream.write("\n")
                self.midLine = False
                self.width = 0
            self.stream.write("\r" + text + " " * max(0, self.width - len(text)))
            self.width = len(text)
            self.open = True
            if self.flush:
                self.stream.flush()
        else:
            tenth = 10 * done // self.nSteps
            if tenth == self.tenth and not last:
                return
            self.tenth = tenth
            self.stream.write(self.line(done, t, now) + "\n")

    def erase(self):
        """Clears the bar line, the cursor back at its start (live bar only)."""
        if self.open:
            self.stream.write("\r" + " " * self.width + "\r")
            self.width = 0
            self.open = False

    def clear(self):
        """Cursor at the start of an empty line: bar erased, unfinished console line ended."""
        self.erase()
        if self.midLine:
            self.stream.write("\n")
            self.midLine = False

    def redraw(self):
        if self.live:
            self.nextDraw = 0.0
            self.update(*self.last)

    def close(self):
        if self.open:
            self.stream.write("\n")
            self.stream.flush()
            self.open = False


class _Guard:
    """sys.stdout / sys.stderr while the live bar is shown: any other console output (print of a solver, Python
    traceback, ...) first erases the bar, which is redrawn below it once the line is ended, so the console keeps a
    single bar whatever prints. Pass-through in a forked worker process and once the bar is finished."""

    def __init__(self, stream, bar):
        self._stream, self._bar, self._pid = stream, bar, os.getpid()

    def write(self, text):
        bar = self._bar
        if not text or bar is not _bar or os.getpid() != self._pid:
            return self._stream.write(text)
        with _lock:
            bar.erase()
            if self._stream is not bar.stream:
                bar.stream.flush()
            n = self._stream.write(text)
            bar.midLine = not text.endswith("\n")
            if not bar.midLine:
                self._stream.flush()
                bar.redraw()
            return n

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _unguard():
    for name in ('stdout', 'stderr'):
        stream = getattr(sys, name)
        if isinstance(stream, _Guard):
            setattr(sys, name, stream._stream)


def start(nSteps, tEnd, startTime=None):
    global _bar
    finish()
    _bar = _ProgressBar(nSteps, tEnd, time.perf_counter() if startTime is None else startTime)
    if not _stepByStep:
        if _bar.live:
            sys.stdout, sys.stderr = _Guard(sys.stdout, _bar), _Guard(sys.stderr, _bar)
        _bar.update(0, 0)


def progress(done, t):
    if _bar is None:
        return
    if not _stepByStep:
        with _lock:
            _bar.update(done, t)
    elif done < _bar.nSteps:
        print(f"Remaining calculation time ~ {_writeTime((time.perf_counter() - _bar.start) * (_bar.tEnd - t) / t, 2)}")


def message(text, key=None):
    """print() of a message (a warning) during the loop: in 'progressBar' mode, written above the bar, which is
    redrawn below it, so the console keeps a single bar; a message already shown (same `key`, the text by default)
    is not shown again (warning.log keeps every occurrence)."""
    if _bar is None or _stepByStep:
        print(f"\n{text}", flush=True)
        return
    with _lock:
        key = text if key is None else key
        if key in _bar.shown:
            return
        _bar.shown.add(key)
        _bar.clear()
        print(text, file=_bar.stream, flush=True)
        _bar.redraw()


def finish():
    """Coupling loop end (or error): ends the bar line, console streams given back."""
    global _bar
    _unguard()
    if _bar is not None:
        _bar.close()
        _bar = None
