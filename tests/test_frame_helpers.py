"""Shortcuts of the interface that must give exactly what the pandas / loop code they replace gives: the frames of
frames.columnsOf (values, layout, blocks, labels), the Index of frames.columnIndex, the positions kept by
frames.indexer for two Index objects with the same names, phreeqc.primaryColumns (same sums, bit for bit) and the
ORCHESTRA console capture."""
import os

import numpy as np
import pandas as pd
import pytest

from pycte.Source import frames, orchestra, phreeqc


def _layout(df, source=None):
    """What the frame holds and how: labels, index, values (bytes), dtype and layout of each block; with pandas 3
    (frames._columnsOf3 reproduces it exactly) also its strides and, for a frame taken from `source`, whether the
    block is a view of the block of source, where, and its copy-on-write references."""
    blocks = []
    for b in df._mgr.blocks:
        v = b.values
        if not frames._PANDAS3:
            blocks.append((b.mgr_locs.as_array.tolist(), v.shape, v.flags['C_CONTIGUOUS'], str(b.dtype)))
            continue
        view = None
        if source is not None and np.shares_memory(v, source._mgr.blocks[0].values):
            view = (v.__array_interface__['data'][0]
                    - source._mgr.blocks[0].values.__array_interface__['data'][0], b.refs.has_reference())
        blocks.append((b.mgr_locs.as_array.tolist(), v.shape, v.strides, v.flags['C_CONTIGUOUS'],
                       v.flags['F_CONTIGUOUS'], v.flags['WRITEABLE'], str(b.dtype), view))
    return (list(df.columns), str(df.columns.dtype), df.columns.name, df.index.tolist(),
            df.to_numpy().tobytes(), str(df.to_numpy().dtype), blocks)


def _frame(n=7, columns=('x', 'Ca', 'Cl', 'Na', 'K', 'time'), order='C'):
    """Frame of one float64 block, C-ordered (columns x nodes, as pycte builds commMtrx) or F-ordered (a frame built
    on a C-ordered nodes x columns array). copy=False: pandas 3 would copy the array in Fortran order (C block)."""
    rng = np.random.default_rng(3)
    values = rng.standard_normal((n, len(columns)))
    df = pd.DataFrame(values if order == 'F' else np.ascontiguousarray(values.T).T, columns=list(columns),
                      copy=False)
    block = df._mgr.blocks[0].values
    assert block.flags['C_CONTIGUOUS' if order == 'C' else 'F_CONTIGUOUS']
    return df


# positions 1, 3 (step 2) / 5, 0 / one name / all: with pandas 3 the first three are views of the block
SELECTIONS = [['Ca', 'Na'], ['time', 'x'], ['K'], ['x', 'Ca', 'Cl', 'Na', 'K', 'time'], ['Na', 'x', 'K']]


@pytest.mark.parametrize("order", ['C', 'F'])
@pytest.mark.parametrize("names", SELECTIONS)
def test_selections_are_the_pandas_ones(names, order):
    df = _frame(order=order)
    expected = df[names]
    for got in (frames.columnsOf(df, names), phreeqc._selectColumns(df, names), orchestra._takeColumns(df, names)):
        assert _layout(got, df) == _layout(expected, df)
    values, reference = orchestra._takeValues(df, names), expected.to_numpy()
    assert values.tobytes() == reference.tobytes()
    # takeArray copies: the layout of pandas, except a selection pandas 3 gives as a strided view of the block
    # (positions of constant step other than 1), for which it gives a contiguous array (unchanged since 2026-11-16)
    if reference.flags['C_CONTIGUOUS'] or reference.flags['F_CONTIGUOUS']:
        assert values.flags['C_CONTIGUOUS'] == reference.flags['C_CONTIGUOUS']
        assert values.flags['F_CONTIGUOUS'] == reference.flags['F_CONTIGUOUS']


@pytest.mark.skipif(not frames._PANDAS3, reason="pandas 3 selections (frames._columnsOf3)")
def test_columns_of_is_df_names_on_random_selections():
    """columnsOf against df[names] on random frames (C / F blocks, 1 to 40 nodes) and selections (slices of any
    step, permutations, subsets): the same frame, and the same copy-on-write behaviour when either is written."""
    rng = np.random.default_rng(11)
    count = 0
    for nodes in (1, 2, 40):
        for ncols in (1, 3, 21):
            names = [f"S{k}{'+' * (k % 3)}" for k in range(ncols)]
            for order in 'CF':
                values = rng.standard_normal((nodes, ncols))
                selections = [names[:1], names[::-1], names[1::2], names[::-2], names[-1:]]
                selections += [list(rng.permutation(names)[:k]) for k in range(1, ncols + 1)]
                for sel in selections:
                    sel = [str(s) for s in sel]
                    if not sel:
                        continue
                    df = _frameOf(values, names, order)
                    expected, got = df[sel], frames.columnsOf(df, sel)
                    assert _layout(got, df) == _layout(expected, df), (nodes, ncols, order, sel)
                    before = df.to_numpy().copy()
                    got.iloc[0, 0] = expected.iloc[0, 0] = 1e300
                    assert df.to_numpy().tobytes() == before.tobytes()
                    assert got.to_numpy().tobytes() == expected.to_numpy().tobytes()
                    count += 1
    assert count > 100


def _frameOf(values, names, order):
    if order == 'F':
        return pd.DataFrame(np.ascontiguousarray(values), columns=names, copy=False)
    return pd.DataFrame(np.asfortranarray(values), columns=names, copy=False)


def test_columns_of_falls_back_to_pandas():
    df = _frame()
    multi = pd.concat([df[['x', 'Ca']], df[['Cl', 'Na']]], axis=1)          # two blocks
    assert len(multi._mgr.blocks) == 2
    mixed = df.assign(tag='a')                                              # object column
    named = df.copy()
    named.columns.name = 'species'
    for frame, names in ((multi, ['Ca', 'Cl']), (mixed, ['Ca', 'Na']), (named, ['Ca', 'Na']), (df, [])):
        assert _layout(frames.columnsOf(frame, names)) == _layout(frame[names])
    # several blocks or not only float64: the array of df[names] is taken by pandas
    assert frames.takeArray(multi, ['Ca', 'Cl']) is None and frames.takeArray(mixed, ['Ca', 'Na']) is None
    taken = frames.takeArray(named, ['Ca', 'Na'])            # an array: the name of the columns plays no part
    assert taken.tobytes() == named[['Ca', 'Na']].to_numpy().tobytes()
    with pytest.raises(KeyError):
        frames.columnsOf(df, ['Ca', 'missing'])


def test_column_index_is_the_index_pandas_builds():
    names = ['Ca+2', 'Cl-', 'm_Na+(mol/kgw)']
    built = pd.DataFrame(np.zeros((2, 3)), columns=names).columns
    index = frames.columnIndex(names)
    assert index.equals(built) and index.dtype == built.dtype and index.name == built.name
    assert frames.columnIndex(list(names)) is index                           # kept for the same names
    assert pd.DataFrame(np.zeros((2, 3)), columns=index).columns is index    # the frame keeps the object


def test_indexer_keeps_two_index_objects_with_the_same_names():
    names = ['Cl', 'Ca']
    a = pd.Index(['x', 'Ca', 'Cl'])
    b = pd.Index(['Ca', 'Cl', 'y', 'z'])
    for _ in range(3):
        assert frames.indexer(a, names).tolist() == a.get_indexer(names).tolist()
        assert frames.indexer(b, names).tolist() == b.get_indexer(names).tolist()
    assert frames.indexer(a, names) is frames.indexer(a, names)


def _primaryColumnsLoop(tab, stoich):
    """phreeqc.primaryColumns as it was written: out[k] += coeffs[j, k] * conc, species by species."""
    values = tab.to_numpy()
    coeffs = stoich.to_numpy()
    out = np.zeros((coeffs.shape[1], len(tab)))
    for j, comp in enumerate(stoich.index):
        conc = values[:, tab.columns.get_loc(comp)]
        for k in np.flatnonzero(coeffs[j]):
            out[k] += coeffs[j, k] * conc
    touched = coeffs.any(axis=0)
    return {prim: out[k] if touched[k] else np.zeros(len(tab), dtype=np.int64)
            for k, prim in enumerate(stoich.columns)}


def test_primary_columns_bit_for_bit():
    rng = np.random.default_rng(11)
    for _ in range(300):
        names = [f"s{i}" for i in range(rng.integers(1, 25))]
        rng.shuffle(names)
        n = int(rng.integers(0, 30))
        values = rng.standard_normal((n, len(names))) * 10.0 ** rng.integers(-18, 4, size=(n, len(names)))
        values[rng.random(values.shape) < 0.05] = -0.0
        values[rng.random(values.shape) < 0.01] = np.nan
        tab = pd.DataFrame(values, columns=names)
        species = list(rng.choice(names, size=int(rng.integers(0, len(names) + 1)), replace=False))
        coeffs = rng.integers(-3, 4, size=(len(species), int(rng.integers(1, 9)))) * rng.choice([1.0, 0.5, 1 / 3])
        coeffs[rng.random(coeffs.shape) < 0.5] = 0.0
        stoich = pd.DataFrame(coeffs, index=species, columns=[f"p{k}" for k in range(coeffs.shape[1])])
        expected = _primaryColumnsLoop(tab, stoich)
        for got in (phreeqc.primaryColumns(tab, stoich), phreeqc.primaryColumnsOf(tab, species, stoich)):
            assert list(got) == list(expected)
            for key in expected:
                assert got[key].dtype == expected[key].dtype
                assert got[key].tobytes() == expected[key].tobytes()


@pytest.mark.skipif(phreeqc._primarySums is None, reason="pycte._phreeqcRow without primary()")
def test_primary_sums_in_c_are_the_numpy_steps(monkeypatch):
    # _phreeqcRow.primary against the numpy steps of _primaryColumnsFast, bit for bit: special values, both memory
    # orders, zero, signed zero and NaN coefficients
    rng = np.random.default_rng(22)
    special = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 5e-324, 1e308, -1e308, 2.0 ** -1022])
    for case in range(200):
        n, ncols = int(rng.integers(1, 40)), int(rng.integers(1, 30))
        values = rng.standard_normal((n, ncols)) * 10.0 ** rng.integers(-30, 30, (n, ncols))
        mask = rng.random(values.shape) < 0.05
        values[mask] = rng.choice(special, int(mask.sum()))
        names = [f"c{j}" for j in range(ncols)]
        species = list(rng.choice(names, size=int(rng.integers(1, ncols + 1)), replace=False))
        coeffs = rng.choice([0.0, -0.0, 1.0, -1.0, 2.0, 0.5, 1 / 3, np.nan], (len(species), int(rng.integers(1, 9))))
        tab = pd.DataFrame(np.asfortranarray(values) if case % 2 else values, columns=names)
        stoich = pd.DataFrame(coeffs, index=species, columns=[f"p{k}" for k in range(coeffs.shape[1])])
        with np.errstate(all="ignore"):
            inC = phreeqc._primaryColumnsFast(tab, stoich, tab.to_numpy(), stoich.to_numpy())
            monkeypatch.setattr(phreeqc, "_primarySums", None)
            steps = phreeqc._primaryColumnsFast(tab, stoich, tab.to_numpy(), stoich.to_numpy())
            monkeypatch.undo()
        assert list(inC) == list(steps)
        for key in steps:
            assert inC[key].dtype == steps[key].dtype
            assert inC[key].tobytes() == steps[key].tobytes()


def test_orchestra_capture_reads_each_call_once():
    got = []
    saved = orchestra.warningManager.info
    orchestra.warningManager.info = got.append
    try:
        for k, payload in enumerate([b"", b"first\n", b"", b"second\r\nline\n", b""]):
            with orchestra.captureOutput(f"call {k}"):
                if payload:
                    os.write(1, payload)
    finally:
        orchestra.warningManager.info = saved
    assert got == ["ORCHESTRA, call 1 :\nfirst", "ORCHESTRA, call 3 :\nsecond\nline"]


def test_constant_is_built_once_for_the_same_sources():
    """frames.constant: built once while the sources are the same objects, again for another object (even an equal
    one), an error raised at every call and never kept."""
    names, other = ['Na', 'Cl'], ['Na', 'Cl']
    built = []

    def build(source):
        built.append(1)
        return [s + '+' for s in source]

    first = frames.constant('test', (names,), lambda: build(names))
    assert frames.constant('test', (names,), lambda: build(names)) is first and len(built) == 1
    assert frames.constant('test', (other,), lambda: build(other)) == first and len(built) == 2
    assert frames.constant('test2', (names,), lambda: build(names)) is not first and len(built) == 3

    def fails():
        built.append(1)
        raise ValueError("bad setting")

    for _ in range(2):
        with pytest.raises(ValueError, match="bad setting"):
            frames.constant('testError', (names,), fails)
    assert len(built) == 5
