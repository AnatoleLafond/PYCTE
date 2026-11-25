"""PhreeqC interface shortcuts of 2026-11-20 that must give what the code they replace gives:
- phreeqc._nodeBlocks: the lines of a section of the node scripts, built once per call for every node, are the text
  the node loop appends line by line;
- pycte._phreeqcRow.after: the calls that follow the run of a node (warnings, row and column counts, last row) in one
  call, the same answers as the calls one by one; with the columns of phreeqc._readColumns (steps without speciation
  output), the values of these columns only, the same doubles;
- phreeqc._ScriptBuilder: the thread writing the next node scripts gives back each result (or the exception raised) in
  the order of the jobs, one per thread, stopped by resetPhreeqC;
- phreeqc.preloadIPhreeqc / takeIPhreeqc: an instance loaded in a thread during the set-up is the one newIPhreeqc() +
  loadDatabase() gives (same selected output for the same script), taken only for the same database settings, and a
  load that failed is left to the caller (its error raised there, as before)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pycte
from pycte.Source import phreeqc

PHREEQC_DAT = Path(pycte.__file__).parent / "Examples" / "CationExchange" / "phreeqc.dat"
try:
    import phreeqpy  # noqa: F401
    HAS_PHREEQPY = True
except ImportError:
    HAS_PHREEQPY = False


def _loopText(cells, key, names, pieces, cut, row, skipMissing):
    """What the node loop of speciationPhreeqC appends for one node (row of cells)."""
    text = ""
    for name, column, texts, head, end in phreeqc._nodeLines(cells, key, names, pieces):
        if column is None and skipMissing:
            continue
        amount = column[row]
        line = texts[row] if texts is not None else amount
        if amount > cut:
            text += f"{head}{line}{end}"
    return text


@pytest.mark.skipif(phreeqc._floatText is None, reason="pycte._floatText not built")
@pytest.mark.parametrize("seed", range(40))
def test_node_blocks_are_the_loop_text(seed):
    rng = np.random.default_rng(seed)
    n, m = int(rng.integers(1, 60)), int(rng.integers(0, 12))
    names = [f"S{k}" for k in range(m)]
    values = rng.lognormal(-20, 15, size=(n, m)) * rng.choice([1.0, -1.0, 0.0], size=(n, m), p=[0.7, 0.2, 0.1])
    values[rng.random((n, m)) < 0.05] = 1e-20                   # on the cut-off: not written
    frame = pd.DataFrame(values, columns=names, index=rng.permutation(n) + 100)
    pieces = lambda name: (f"\t{name} ", "\n") if seed % 2 else (f"\t{name} 0 ", f"  {name}\n")
    for skipMissing, asked in ((False, names), (True, names + ["Absent"])):
        cells = phreeqc._Cells(frame)
        blocks = phreeqc._nodeBlocks(cells, 'section', asked, pieces, 1e-20, skipMissing)
        assert blocks is not None and len(blocks) == n
        for row in range(n):
            assert blocks[row] == _loopText(cells, 'section', asked, pieces, 1e-20, row, skipMissing)
        assert phreeqc._nodeBlocks(cells, 'section', asked, pieces, 1e-20, skipMissing) is blocks   # once per call


def test_node_blocks_leave_the_other_cases_to_the_loop():
    frame = pd.DataFrame({'A': [1.0, 2.0], 'B': np.array([1, 2], dtype=np.int64)})
    pieces = lambda name: (f"\t{name} ", "\n")
    assert phreeqc._nodeBlocks(phreeqc._Cells(frame), 'k', ['A', 'B'], pieces, 0) is None        # int64: no texts
    assert phreeqc._nodeBlocks(phreeqc._Cells(frame), 'k', ['A', 'Absent'], pieces, 0) is None   # missing column
    assert phreeqc._nodeBlocks(phreeqc._Cells(frame), 'k', ['A'], lambda name: (None, None), 0) is None
    assert phreeqc._nodeBlocks(phreeqc._Cells(frame), 'k', [], pieces, 0) == ['', '']


SCRIPT = ("SOLUTION 1\n-units mol/kgw\n-temp 25\n\tpH 7\tcharge\n\tCa 1e-3\n\tCl 2e-3\n\tNa 1e-3\n\tK 1e-4\n"
          "\tBadElement 1e-3\n"           # unknown element: a warning
          "SELECTED_OUTPUT\n-reset false\n-molalities\tCa+2\tCl-\tNa+\tK+\tH+\tOH-\n-pH true\nEND\n")


@pytest.fixture
def solver():
    if not HAS_PHREEQPY or phreeqc._iphreeqcLibrary() is None:
        pytest.skip("phreeqpy / IPhreeqc library not found")
    s = phreeqc.newIPhreeqc()
    phreeqc.loadDatabase(s, {'chemPath': PHREEQC_DAT})
    yield s
    s.destroy_iphreeqc()


@pytest.mark.skipif(phreeqc._afterReader is None, reason="pycte._phreeqcRow without after()")
def test_after_gives_the_calls_one_by_one(solver):
    solver.run_string(SCRIPT)
    nrows, ncols = solver.row_count, solver.column_count
    header = phreeqc.selectedOutputRow(solver, 0, ncols)
    expected = phreeqc.selectedOutputRow(solver, nrows - 1, ncols)
    warning = phreeqc.warningText(solver)
    assert warning                                      # the unknown element
    block = np.full((3, ncols), -1.0)
    # header known, block row given: the last row read into it
    text, rows, cols, status = phreeqc._afterRun(solver, header, block, 1, block.ctypes.data)
    assert (text, rows, cols, status) == (warning, nrows, ncols, 0)
    assert block[1].tolist() == expected and block[0].tolist() == block[2].tolist() == [-1.0] * ncols
    # header unknown, other column count, no block: counts only
    for args in ((None, block, 0), (header[:-1], np.full((3, ncols - 1), -1.0), 0), (header, None, 0),
                 (header, block, 3)):
        text, rows, cols, status = phreeqc._afterRun(solver, *args, args[1].ctypes.data if args[1] is not None else 0)
        assert (text, rows, cols, status) == (warning, nrows, ncols, 1)
    # a value that is not a double (a text punched by USER_PUNCH): status 2, the block row left as it was, the row
    # then read value by value (selectedOutputValues) as selectedOutputRow reads it
    solver.run_string(SCRIPT.replace("END\n", "USER_PUNCH\n-headings\tword\n10 PUNCH \"abc\"\nEND\n"))
    nrows, ncols = solver.row_count, solver.column_count
    header = phreeqc.selectedOutputRow(solver, 0, ncols)
    block = np.full((1, ncols), -1.0)
    text, rows, cols, status = phreeqc._afterRun(solver, header, block, 0, block.ctypes.data)
    assert (rows, cols, status) == (nrows, ncols, 2) and block[0].tolist() == [-1.0] * ncols
    assert (phreeqc.selectedOutputValues(solver, nrows - 1, ncols)
            == phreeqc.selectedOutputRow(solver, nrows - 1, ncols, np.empty(ncols)))
    assert phreeqc.selectedOutputValues(solver, nrows - 1, ncols)[-1] == "abc"
    # no selected output: no row, no column count asked
    solver.run_string("SELECTED_OUTPUT\n-reset false\n-active false\nEND\nSOLUTION 1\nEND\n")
    text, rows, cols, status = phreeqc._afterRun(solver, header, block, 0, block.ctypes.data)
    assert status == 1 and rows == solver.row_count and cols == (solver.column_count if rows else 0)


def test_after_without_warning(solver):
    if phreeqc._afterReader is None:
        pytest.skip("pycte._phreeqcRow without after()")
    solver.run_string(SCRIPT.replace("\tBadElement 1e-3\n", ""))
    assert phreeqc.warningText(solver) == ''
    ncols = solver.column_count
    header = phreeqc.selectedOutputRow(solver, 0, ncols)
    block = np.empty((1, ncols))
    text, rows, cols, status = phreeqc._afterRun(solver, header, block, 0, block.ctypes.data)
    assert (text, rows, cols, status) == ('', solver.row_count, ncols, 0)
    assert block[0].tolist() == phreeqc.selectedOutputRow(solver, solver.row_count - 1, ncols)


def test_read_columns_choice():
    if phreeqc._afterColumns is None:
        pytest.skip("pycte._phreeqcRow without the columns of after() (VERSION 2)")
    header = ['mass_H2O', 'm_Ca+2(mol/kgw)', 'Calcite', 'd_Calcite', 'pH']
    positions, address, count, names = phreeqc._readColumns(header, {'mass_H2O', 'Calcite', 'pH', 'k_Calcite'})
    assert positions.dtype == np.intc and positions.tolist() == [0, 2, 4] and address == positions.ctypes.data
    assert count == 3 and names == ['mass_H2O', 'Calcite', 'pH']
    assert phreeqc._readColumns(header, None) is None                            # speciation output: every column
    assert phreeqc._readColumns(header, set(header)) is None                     # every column needed
    assert phreeqc._readColumns(header, {'absent'}) is None                      # none
    assert phreeqc._readColumns(header + ['pH'], {'pH'}) is None                  # names not distinct
    assert phreeqc._readColumns(header + [3], {'pH'}) is None                     # not a string


@pytest.mark.skipif(phreeqc._afterColumns is None, reason="pycte._phreeqcRow without the columns of after()")
def test_after_reads_the_given_columns(solver):
    script = SCRIPT.replace("\tBadElement 1e-3\n", "").replace(
        "END\n", "USER_PUNCH\n-headings\tword\tvalue\n10 PUNCH \"abc\", 1.5\nEND\n")
    solver.run_string(script)
    nrows, ncols = solver.row_count, solver.column_count
    header = phreeqc.selectedOutputRow(solver, 0, ncols)
    full = phreeqc.selectedOutputValues(solver, nrows - 1, ncols)
    assert header[-2:] == ['word', 'value'] and full[-2] == 'abc'
    # the text column not read: status 0, the values of the columns in the block row, bit for bit
    columns = phreeqc._readColumns(header, set(header) - {'word', 'pH'})
    assert columns[3] == [h for h in header if h not in ('word', 'pH')]
    block = np.full((2, columns[2]), -1.0)
    text, rows, cols, status = phreeqc._afterRun(solver, header, block, 1, block.ctypes.data, columns)
    assert (text, rows, cols, status) == ('', nrows, ncols, 0)
    expected = np.array([full[j] for j in columns[0].tolist()])
    assert block[1].tobytes() == expected.tobytes() and block[0].tolist() == [-1.0] * columns[2]
    # the text column read: status 2, the row then read whole (selectedOutputValues)
    columns = phreeqc._readColumns(header, {'word', 'value'})
    block = np.full((1, 2), -1.0)
    assert phreeqc._afterRun(solver, header, block, 0, block.ctypes.data, columns)[3] == 2
    assert block[0].tolist() == [-1.0, -1.0]
    # block row of another width, header unknown: not read
    assert phreeqc._afterRun(solver, header, np.empty((1, 3)), 0, 0, columns)[3] == 1
    assert phreeqc._afterRun(solver, None, block, 0, block.ctypes.data, columns)[3] == 1
    # a column outside the row: not read
    vars_, size = phreeqc._rowBuffers(solver, ncols)[6:8]
    warning, rowsFn, colsFn, value, typeOffset, valueOffset = phreeqc._afterAddresses(solver)
    outside = np.array([0, ncols], dtype=np.intc)
    out = np.full(2, -1.0)
    result = phreeqc._afterColumns(warning, rowsFn, colsFn, value, solver.id_, ncols, vars_, size, typeOffset,
                                   valueOffset, out.ctypes.data, outside.ctypes.data, 2)
    assert result[1:] == (nrows, ncols, 1) and out.tolist() == [-1.0, -1.0]


def test_script_builder_gives_back_results_in_order():
    builder = phreeqc._ScriptBuilder()
    try:
        error = KeyError('Absent')

        def failing():
            raise error
        builder.submit(lambda: "script 1")
        builder.submit(failing)
        builder.submit(lambda: {2: "script 2", 3: "script 3"})
        assert builder.result() == (True, "script 1")
        assert builder.result() == (False, error)                     # the exception itself, given back
        assert builder.result() == (True, {2: "script 2", 3: "script 3"})
    finally:
        builder.stop()
    builder.thread.join(5)
    assert not builder.thread.is_alive()
    builder.submit(lambda: "never written")                             # thread stopped: no wait without end
    assert builder.result() == (False, None)


def test_script_builder_per_thread_and_stopped_by_reset():
    import threading
    phreeqc._stopScriptBuilders()
    first = phreeqc._scriptBuilder()
    assert phreeqc._scriptBuilder() is first                            # kept from one call to the next
    other = []
    worker = threading.Thread(target=lambda: other.append(phreeqc._scriptBuilder()))
    worker.start()
    worker.join()
    assert other[0] is not first                                        # its own in another thread
    phreeqc._stopScriptBuilders()
    for builder in (first, other[0]):
        builder.thread.join(5)
        assert not builder.thread.is_alive()
    assert phreeqc._builders == {}


def _selectedOutput(s, script):
    s.run_string(script)
    return [phreeqc.selectedOutputRow(s, r, s.column_count) for r in range(s.row_count)]


def test_small_database_is_not_preloaded():
    if not HAS_PHREEQPY or phreeqc._iphreeqcLibrary() is None:
        pytest.skip("phreeqpy / IPhreeqc library not found")
    assert PHREEQC_DAT.stat().st_size < phreeqc._PRELOAD_MIN_TEXT
    try:
        phreeqc.preloadIPhreeqc(PHREEQC_DAT, None, None, 2)
        assert phreeqc._preloaded == []
        assert phreeqc.takeIPhreeqc({'chemPath': PHREEQC_DAT}) is None
    finally:
        phreeqc._releasePreloaded()


def test_preloaded_instance_is_a_loaded_one(monkeypatch):
    if not HAS_PHREEQPY or phreeqc._iphreeqcLibrary() is None:
        pytest.skip("phreeqpy / IPhreeqc library not found")
    monkeypatch.setattr(phreeqc, '_PRELOAD_MIN_TEXT', 0)          # phreeqc.dat preloaded too
    settings = {'chemPath': PHREEQC_DAT, 'knobs': {'convergence_tolerance': 1e-12}, 'kineticRates': None}
    try:
        phreeqc.preloadIPhreeqc(PHREEQC_DAT, settings['knobs'], None, 2)
        assert len(phreeqc._preloaded) == 2
        phreeqc.preloadIPhreeqc(PHREEQC_DAT, settings['knobs'], None, 2)      # already 2: none more
        assert len(phreeqc._preloaded) == 2
        assert phreeqc.takeIPhreeqc(dict(settings, knobs=None)) is None      # other settings: not taken
        taken = phreeqc.takeIPhreeqc(settings)
        assert taken is not None and len(phreeqc._preloaded) == 1
        fresh = phreeqc.newIPhreeqc()
        phreeqc.loadDatabase(fresh, settings)
        try:
            script = SCRIPT.replace("\tBadElement 1e-3\n", "")
            assert _selectedOutput(taken, script) == _selectedOutput(fresh, script)
        finally:
            fresh.destroy_iphreeqc()
            taken.destroy_iphreeqc()
    finally:
        phreeqc._releasePreloaded()
    assert phreeqc._preloaded == [] and phreeqc._preloaders == []


def test_failed_preload_is_left_to_the_caller(tmp_path, monkeypatch):
    if not HAS_PHREEQPY or phreeqc._iphreeqcLibrary() is None:
        pytest.skip("phreeqpy / IPhreeqc library not found")
    monkeypatch.setattr(phreeqc, '_PRELOAD_MIN_TEXT', 0)
    bad = tmp_path / "bad.dat"
    bad.write_text("UNKNOWN_KEYWORD\n")
    settings = {'chemPath': bad, 'knobs': None, 'kineticRates': None}
    try:
        phreeqc.preloadIPhreeqc(bad, None, None, 1)
        assert phreeqc.takeIPhreeqc(settings) is None                       # load failed: made by the caller
        s = phreeqc.newIPhreeqc()
        try:
            with pytest.raises(ValueError, match="could not read the database"):
                phreeqc.loadDatabase(s, settings)
        finally:
            s.destroy_iphreeqc()
    finally:
        phreeqc._releasePreloaded()
