"""nativeTransport._writeColumns (pass 12): the frame of the Nernst-Planck transports, built at once, is the frame the
loop `df[name] = arrays[name]` gave (same values bit for bit, dtypes, labels), in one block; the other cases go through
the loop (same frame, same errors).
Pass 13: what is kept from one call to the next for the same columns Index gives what was computed at every call:
nativeTransport._presentIn, orchestra._literbulkLayout / _literbulkValues, orchestra._columnsSplit."""
import numpy as np
import pandas as pd
import pytest

from pycte.Source import nativeTransport


def _orchestra():
    try:
        from pycte.Source import orchestra
    except ImportError as error:                   # PyORCHESTRA not built
        pytest.skip(f"pycte.Source.orchestra cannot be imported : {error}")
    return orchestra


def _literbulkBefore(frame, colAq, waterMass, cross):
    """orchestra._literbulkValues before pass 13 (everything computed at every call)."""
    from pycte.Source import frames
    columns = frame.columns
    if not (columns.is_unique and frames.allFloat64(frame)):
        return None
    if len(set(colAq)) != len(colAq) or any(c not in columns for c in cross):
        return None
    aq = columns.get_indexer(colAq)
    if np.any(aq < 0):
        return None
    water = np.asarray(waterMass)
    if water.dtype.kind not in 'fiu' or not (water.ndim == 0 or (water.ndim == 1 and len(water) == len(frame))):
        return None
    block = frame.to_numpy().T.copy()
    block[aq] = block[aq] * water
    keptColumns = columns
    if cross:
        crossSet = set(cross)
        keep = [k for k, c in enumerate(columns.tolist()) if c not in crossSet]
        block = block[keep]
        keptColumns = columns[keep]
    values = block.T
    return values, keptColumns, {name: k for k, name in enumerate(keptColumns.tolist())}


@pytest.mark.parametrize("seed", range(12))
def test_literbulk_values_kept_layout_is_the_computed_one(seed):
    orchestra = _orchestra()
    rng = np.random.default_rng(seed)
    n, k = int(rng.integers(1, 40)), int(rng.integers(2, 30))
    names = [f"c{j}" for j in range(k)]
    columns = pd.Index(names)
    cross = [names[j] for j in sorted(rng.choice(k, size=int(rng.integers(0, k)), replace=False))]
    aqPool = [c for c in names if c not in cross] or names
    colAq = list(rng.choice(aqPool, size=int(rng.integers(1, len(aqPool) + 1)), replace=False))
    for call in range(3):                          # the same Index at every call, as from one step to the next
        frame = pd.DataFrame(rng.lognormal(-8, 4, size=(n, k)), columns=columns)
        water = rng.uniform(0.1, 2.0, size=n) if call != 1 else 0.7
        got, expected = orchestra._literbulkValues(frame, colAq, water, cross), _literbulkBefore(frame, colAq, water, cross)
        assert (got is None) == (expected is None)
        if got is not None:
            assert got[0].tobytes() == expected[0].tobytes() and got[0].strides == expected[0].strides
            assert got[1].equals(expected[1]) and got[2] == expected[2]
    other = pd.DataFrame(np.ones((n, k)), columns=pd.Index(names[::-1]))   # another Index: looked up again
    assert orchestra._literbulkValues(other, colAq, 1.0, cross)[2] == _literbulkBefore(other, colAq, 1.0, cross)[2]
    assert orchestra._literbulkValues(frame, colAq, 1.0, cross + ["absent"]) is None


def _rowsOf(primary, inputs, n):
    """The rows given to ORCHESTRA, as speciationOrchestra builds them from a dict or from a frame (per column)."""
    rows = np.empty((n, len(inputs)), dtype=np.float64, order="C")
    for j, col in enumerate(inputs):
        rows[:, j] = (np.asarray(primary[col], dtype=np.float64) if isinstance(primary, dict)
                      else primary[col].to_numpy(dtype=np.float64))
    return rows


@pytest.mark.parametrize("seed", range(8))
def test_cross_arrays_give_the_rows_of_the_concat(seed):
    orchestra = _orchestra()
    rng = np.random.default_rng(seed)
    n, k = int(rng.integers(1, 50)), int(rng.integers(1, 12))
    species = [f"s{j}" for j in range(k)]
    comm = pd.DataFrame({"x": rng.random(n), "time": rng.integers(0, 10**6, size=n), "pH": rng.uniform(2, 12, n),
                         "Ca": rng.lognormal(-8, 3, n), "porosity": np.full(n, 0.3)})
    comm.index = pd.RangeIndex(n) if seed % 2 else pd.Index(rng.permutation(n) + 5)
    cross = ["time", "pH", "porosity"][: 1 + seed % 3]
    product = rng.lognormal(-9, 4, size=(n, k))
    product[rng.random((n, k)) < 0.05] = np.nan
    concat = pd.concat([comm[cross].reset_index(drop=True), pd.DataFrame(product, columns=species)], axis=1)
    found = orchestra._crossArrays(comm, cross)
    assert found is not None and list(found) == cross
    found.update((name, product[:, j]) for j, name in enumerate(species))
    inputs = list(rng.permutation(cross + species))
    assert _rowsOf(found, inputs, n).tobytes() == _rowsOf(concat, inputs, n).tobytes()
    assert orchestra._crossArrays(comm, cross + ["absent"]) is None                         # concat path
    assert orchestra._crossArrays(comm.assign(pH=comm["pH"].astype(object)), ["pH"]) is None   # object column
    assert orchestra._crossArrays(comm.assign(pH=comm["pH"] > 7), ["pH"]) is None             # bool column


def test_columns_split_kept_is_the_computed_one():
    orchestra = _orchestra()
    comm = pd.Index(["x", "a", "b", "c"])
    for spct, col in ((pd.Index(["a", "b", "c"]), ["x"]), (pd.Index(["a", "b"]), ["x"]),
                      (pd.Index(["a", "b", "c", "x"]), ["x"]), (pd.Index(["a", "a", "b", "c"]), ["x"]),
                      (pd.Index(["a", "b", "c"]), ["x", "x"])):
        computed = spct.tolist()
        expected = (not set(col) & set(computed) and set(comm.tolist()) == set(col) | set(computed)
                    and len(set(col)) == len(col) and len(set(computed)) == len(computed))
        for _ in range(2):
            assert orchestra._columnsSplit(comm, spct, col, computed) == expected


def test_mixed_numeric_columns_choice():
    n = 4
    df = pd.DataFrame({"x": np.linspace(0, 1, n), "time": np.full(n, 1800), "Na+": np.ones(n), "pH": np.full(n, 7.0)})
    assert nativeTransport._mixedNumericColumns(df, {"Na+"}) == ["time"]              # int column not transported
    assert nativeTransport._mixedNumericColumns(df.assign(time=1800.0), {"Na+"}) is None    # every column float64
    assert nativeTransport._mixedNumericColumns(df, {"Na+", "time"}) is None          # an int column transported
    assert nativeTransport._mixedNumericColumns(df.assign(x=np.arange(n)), {"Na+"}) is None  # x not float
    assert nativeTransport._mixedNumericColumns(df.assign(pH="7"), {"Na+"}) is None    # a text column
    assert nativeTransport._mixedNumericColumns(df.assign(pH=True), {"Na+"}) is None   # a bool column
    assert nativeTransport._mixedNumericColumns(df.drop(columns="x"), {"Na+"}) is None
    dup = pd.concat([df, df[["pH"]]], axis=1)
    assert nativeTransport._mixedNumericColumns(dup, {"Na+"}) is None                 # repeated column names


@pytest.mark.parametrize("seed", range(6))
def test_renewal_due_times_are_those_of_the_full_scan(seed, tmp_path):
    from pycte.Source import renewal
    rng = np.random.default_rng(seed)
    scale = [1.0, 720.0, 3.6e6, 1e-3, 1e9, 1.0][seed]
    gaps = rng.choice([1e-10, 5e-10, 2e-9, 1e-3, 1.0, 37.0], size=400) * scale
    times = np.cumsum(gaps).tolist()
    if seed == 5:
        times = [t - times[len(times) // 2] for t in times]          # negative times as well
    tol = lambda tr: 1e-9 * max(abs(tr), 1.0)
    steps = sorted(set([tr + d for tr in times[::3] for d in (-2 * tol(tr), -tol(tr), 0.0, tol(tr), 3 * tol(tr))]
                       + rng.uniform(times[0], times[-1], 200).tolist()))
    comm = pd.DataFrame({"x": [0.0, 1.0], "A": [1.0, 2.0]})
    group = {'cells': [0], 'times': list(times), 'next': 0, 'concentration': None, 'water': 1.0}
    cd = {'commMtrx': comm, 'renewalInitial': comm.copy(), 'renewal': [group], 'renewalSpeciesCols': ['A'],
          'renewalZeroCols': ['A'], 'renewalWaterMass': None, 'renewalFile': tmp_path / "renewal.txt",
          'timeUnit': 's'}
    refNext, applied = 0, 0
    for t in steps:
        due = [tr for tr in times[refNext:] if t >= tr - 1e-9 * max(abs(tr), 1.0)]   # before: every pending time
        refNext += len(due)
        cd['tStep'] = t
        renewal.apply(cd)
        assert group['next'] == refNext
        if due:
            applied += 1
            last = pd.read_csv(tmp_path / "renewal.txt", sep="\t")["tRenewal"].iloc[-1]
            assert last == due[-1]
    assert applied > 0


def _conditionBefore(entries, t):
    tol = 1e-9 * max(1.0, abs(t))
    return [cond for ts, cond in entries if ts <= t + tol][-1]


def _segmentsBefore(cd):
    schedule = cd.get('boundarySchedule')
    areaTimes = [ts for ts, _ in cd.get('areaSchedule') or ()]
    dtU, t0 = cd['dtStep'], cd.get('tTransportStart')
    if not (schedule or areaTimes) or t0 is None:
        return [(cd['boundaryConditions'], dtU, t0)]
    t1 = t0 + dtU
    tol = 1e-9 * max(1.0, abs(t1))
    starts = [ts for entries in schedule or () for ts, _ in entries] + areaTimes
    cuts = sorted({ts for ts in starts if t0 + tol < ts < t1 - tol})
    bounds = [t0] + cuts + [t1]
    conditions = lambda a: [_conditionBefore(e, a) for e in schedule] if schedule else cd['boundaryConditions']
    return [(conditions(a), b - a, a) for a, b in zip(bounds[:-1], bounds[1:])]


def _areaBefore(cd, t, before=False):
    area = cd['areaBase']
    tol = 1e-9 * max(1.0, abs(t))
    for ts, a in cd['areaSchedule']:
        if not ((ts < t - tol) if before else (ts <= t + tol)):
            break
        area = a
    return area


@pytest.mark.parametrize("seed", range(6))
def test_schedules_by_bisection_are_the_full_scans(seed):
    rng = np.random.default_rng(seed)
    scale = [1.0, 720.0, 3.6e6, 1e-3, 1e9, 86400.0][seed]
    def times(n):
        return np.cumsum(rng.choice([1e-10, 5e-10, 2e-9, 1e-3, 1.0, 37.0], size=n) * scale).tolist()
    # as boundarySchedule gives them: sorted by start time, distinct, the first at time 0
    left = [[(ts, f"c{k}") for k, ts in enumerate([0.0] + times(60))], [(0.0, "closed")]]
    area = [(ts, np.full(3, float(k + 1))) for k, ts in enumerate(times(40))]
    base = np.ones(3)
    probes = sorted({ts + d for entries in (left[0], area) for ts, _ in entries[::2]
                     for d in (-2e-9 * max(abs(ts), 1.0), -1e-9 * max(abs(ts), 1.0), 0.0, 1e-9 * max(abs(ts), 1.0))}
                    | set(rng.uniform(0, left[0][-1][0] * 1.1, 80).tolist()) | {0.0})
    def outcome(f, *args):
        try:
            return f(*args)
        except IndexError as error:
            return ("IndexError", str(error))
    for t in probes:
        for entries in left:
            assert outcome(nativeTransport.conditionAt, entries, t) == outcome(_conditionBefore, entries, t)
        cd = {'areaBase': base, 'areaSchedule': area}
        for before in (False, True):
            assert nativeTransport.areaAt(cd, t, before) is _areaBefore(cd, t, before)
    for t0, dt in [(probes[k], probes[k + 3] - probes[k]) for k in range(0, len(probes) - 3, 5)] + [(5.0, 0.0)]:
        for sched, areas in ((left, area), (left, None), (None, area)):
            cd = {'boundarySchedule': sched, 'areaSchedule': areas, 'dtStep': dt, 'tTransportStart': t0,
                  'boundaryConditions': ['closed', 'closed']}
            assert outcome(nativeTransport.transportSegments, cd) == outcome(_segmentsBefore, cd)
    with pytest.raises(IndexError):
        nativeTransport.conditionAt([(1.0, "late")], 0.0)                 # no condition yet: the same IndexError


def test_capture_output_reads_what_is_written():
    import os
    from pycte.Source import warningManager
    orchestra = _orchestra()
    start = len(warningManager._pending)
    try:
        with orchestra.captureOutput("t1"):
            pass                                                      # nothing written
        with orchestra.captureOutput("t2"):
            os.write(1, b"hello\r\nworld\n")
        with orchestra.captureOutput("t3"):
            pass                                                      # nothing, after a call that wrote
        with orchestra.captureOutput("t4"):
            os.write(2, b"err")
        with orchestra.captureOutput("t5"):
            os.write(1, b"x")                                          # short text after a longer one
        texts = [text for text, _, _ in warningManager._pending[start:]]
    finally:
        del warningManager._pending[start:]
    lines = [[line.strip() for line in text.strip().splitlines()] for text in texts]
    assert lines == [["ORCHESTRA, t2 :", "hello", "world"], ["ORCHESTRA, t4 :", "err"], ["ORCHESTRA, t5 :", "x"]]


def test_parse_condition_kept_is_the_parsed_one():
    for bc in ["constant", "closed", "flux", "electrode", "electrode+constant", "electrode + flux", "electrode+closed"]:
        for _ in range(2):
            assert nativeTransport.parseCondition(bc) == nativeTransport.parseCondition(bc + "")
    assert nativeTransport.parseCondition("electrode") == (True, "closed")
    assert nativeTransport.parseCondition("electrode + flux") == (True, "flux")
    for _ in range(2):
        with pytest.raises(ValueError, match="unknown boundary condition 'wrong'"):
            nativeTransport.parseCondition("wrong")                  # an error raised each time, never kept
    with pytest.raises(ValueError):
        nativeTransport.parseCondition(None)


def test_transported_values_frame_is_the_returned_frame():
    rng = np.random.default_rng(3)
    names = ["Na+", "Cl-", "K+"]
    index = pd.RangeIndex(7)
    values = rng.random((7, 3))
    kept = nativeTransport._TransportedValues(values, index, names)
    expected = pd.DataFrame(values.T.copy().T, index=index, columns=nativeTransport.frames.columnIndex(names))
    got = kept.frame()
    assert got.equals(expected) and got.to_numpy().tobytes() == expected.to_numpy().tobytes()
    assert got.to_numpy().strides == expected.to_numpy().strides and got.columns is expected.columns
    assert kept.to_numpy() is values and kept.columns is expected.columns


@pytest.mark.parametrize("order", ["C", "F"])
def test_tot_input_arrays_are_those_of_the_columns(order):
    orchestra = _orchestra()
    from pycte.Source import frames
    rng = np.random.default_rng(1 if order == "C" else 2)
    n, m = 37, 24
    names = [f"s{j}" for j in range(m)]
    values = np.asarray(rng.lognormal(-8, 3, size=(n, m)), order=order)
    comm = pd.DataFrame(values, columns=pd.Index(names))
    col = names[2:]                                          # the frame given before: commMtrx[col]
    part = orchestra._takeColumns(comm, col)
    species, cross = names[3:20], names[20:23]
    stoich = rng.integers(-2, 3, size=(len(species), 6)).astype(float)
    cd = {'couplingFormalism': 'tot', 'output': {}, 'paths': {}, 'lStep': 0, 'systemSpecies': species,
          'crossDependencies': {'speciation': {'total': cross}}}
    assert orchestra._totInput(cd, comm, col) is comm
    a, b = orchestra._takeValues(comm, species), orchestra._takeValues(part, species)
    assert a.tobytes() == b.tobytes() and a.strides == b.strides and a.flags.c_contiguous == b.flags.c_contiguous
    assert (a @ stoich).tobytes() == (b @ stoich).tobytes()
    pa, pb = orchestra._positions(comm, cross), orchestra._positions(part, cross)
    for p, q in zip(pa.tolist(), pb.tolist()):
        assert comm.to_numpy()[:, p].tobytes() == part.to_numpy()[:, q].tobytes()
    # cases left to commMtrx[col]
    assert orchestra._totInput(dict(cd, couplingFormalism='literbulk'), comm, col) is None
    assert orchestra._totInput(cd, comm, col + ["absent"]) is None
    assert orchestra._totInput(dict(cd, systemSpecies=species + ["s0"]), comm, col) is None   # not in col
    assert orchestra._totInput(cd, comm.assign(s5=1), col) is None                           # int column
    one = pd.DataFrame(values[:1].copy(), columns=pd.Index(names))
    assert orchestra._totInput(cd, one, col) is None                     # one row: its block is C- and F-ordered
    sliced = comm.iloc[:5]                                               # a view: neither, the same rule both ways
    if orchestra._totInput(cd, sliced, col) is sliced:
        a = orchestra._takeValues(sliced, species)
        b = orchestra._takeValues(orchestra._takeColumns(sliced, col), species)
        assert a.tobytes() == b.tobytes() and a.strides == b.strides and (a @ stoich).tobytes() == (b @ stoich).tobytes()
    wanted = dict(cd, output={'speciation': {1}}, paths={'speciation': 'x'})
    assert orchestra._totInput(wanted, comm, col) is None                                   # output written


def test_orchestra_output_frame_is_the_returned_frame():
    orchestra = _orchestra()
    from pycte.Source import frames
    out = np.random.default_rng(4).random((9, 5))
    names = ["a", "b", "c"]
    kept = orchestra._OrchestraOutput(out[:, :3], names)
    expected = pd.DataFrame(out[:, :3], columns=frames.columnIndex(names))
    assert kept.frame().equals(expected) and kept.columns is expected.columns and len(kept) == 9
    assert kept.to_numpy().tobytes() == expected.to_numpy().tobytes()


def test_present_in_is_the_index_membership():
    columns = pd.Index(["x", "Ca+2", "Cl-", "H2O", "Na+"])
    names = ["Na+", "K+", "Cl-", "H2O", "SO4-2"]
    expected = [s for s in names if s in columns]
    for _ in range(3):
        found = nativeTransport._presentIn(columns, names)
        assert found == expected
        found.append("changed")                      # a new list at every call
    assert nativeTransport._presentIn(pd.Index(["Na+", "x"]), names) == ["Na+"]          # another Index
    assert nativeTransport._presentIn(columns, iter(names)) == expected                  # any iterable of names
    mixed = pd.Index(["x", 1, "Na+"], dtype=object)                                      # not only str: Index lookups
    assert nativeTransport._presentIn(mixed, ["Na+", 1, 1.0]) == [s for s in ["Na+", 1, 1.0] if s in mixed]


def _loop(df, names, arrays):
    for name in names:
        df[name] = arrays[name]
    return df


def _special(rng, shape):
    values = rng.lognormal(-10, 8, size=shape) * rng.choice([1.0, -1.0], size=shape)
    flat = values.reshape(-1)
    for k, v in enumerate([0.0, -0.0, np.nan, np.inf, -np.inf, 5e-324, 1e308]):
        if k < flat.size:
            flat[rng.integers(flat.size)] = v
    return values


def _same(a, b):
    assert a.columns.equals(b.columns) and a.index.equals(b.index)
    assert [str(t) for t in a.dtypes] == [str(t) for t in b.dtypes]
    assert a.to_numpy().tobytes() == b.to_numpy().tobytes()       # bit for bit: NaN, -0.0 ...


@pytest.mark.parametrize("seed", range(20))
def test_write_columns_is_the_loop(seed):
    rng = np.random.default_rng(seed)
    n, k = int(rng.integers(1, 120)), int(rng.integers(1, 40))
    names = [f"S{j}" for j in range(k)]
    index = pd.RangeIndex(n) if seed % 2 else pd.Index(rng.permutation(n) + 7)
    df = pd.DataFrame(_special(rng, (n, k)), index=index, columns=pd.Index(names))
    c = _special(rng, (k, n))
    arrays = {name: c[j] for j, name in enumerate(names)}           # rows of the state array, as the solvers give
    expected = _loop(df.copy(), names, arrays)
    out = nativeTransport._writeColumns(df.copy(), names, arrays)
    _same(out, expected)
    assert out._mgr.nblocks == 1 and expected._mgr.nblocks == k
    assert not any(np.shares_memory(out.to_numpy(), a) for a in arrays.values())   # copies, as after the loop
    before = arrays['S0'][0]
    out.iat[0, 0] = 123.0 if before != 123.0 else 124.0             # a write into the frame (boundaries) ...
    assert arrays['S0'][0] == before or np.isnan(before)             # ... leaves the arrays of the solver as they were


def test_write_columns_other_cases_go_through_the_loop():
    df = pd.DataFrame(np.arange(12, dtype=float).reshape(4, 3), columns=["a", "b", "c"])
    arrays = {"a": np.ones(4), "b": np.full(4, 2.0), "c": np.full(4, 3.0)}
    for names in (["b", "a", "c"], ["a", "b"]):                     # other order, some columns only
        _same(nativeTransport._writeColumns(df.copy(), names, arrays), _loop(df.copy(), names, arrays))
    ints = dict(arrays, b=np.arange(4))                               # an int64 array: int64 column, as the loop
    _same(nativeTransport._writeColumns(df.copy(), list("abc"), ints), _loop(df.copy(), list("abc"), ints))
    listed = dict(arrays, c=[7.0, 8.0, 9.0, 10.0])                    # a list
    _same(nativeTransport._writeColumns(df.copy(), list("abc"), listed), _loop(df.copy(), list("abc"), listed))
    mixed = df.assign(b=np.arange(4))                                 # an int64 column of the frame
    _same(nativeTransport._writeColumns(mixed.copy(), list("abc"), arrays), _loop(mixed.copy(), list("abc"), arrays))
    short = dict(arrays, a=np.ones(3))                                # wrong length: the loop's error
    with pytest.raises(ValueError) as fromLoop:
        _loop(df.copy(), list("abc"), short)
    with pytest.raises(ValueError) as fromHelper:
        nativeTransport._writeColumns(df.copy(), list("abc"), short)
    assert str(fromHelper.value) == str(fromLoop.value)
    with pytest.raises(KeyError):                                     # a missing array: the loop's KeyError
        nativeTransport._writeColumns(df.copy(), list("abc"), {"a": np.ones(4)})
