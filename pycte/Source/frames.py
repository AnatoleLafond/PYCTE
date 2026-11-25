"""Small helpers on the DataFrames exchanged at each step (commMtrx and the frames built from it)."""
import numpy as np
import pandas as pd

try:
    from pandas._libs.lib import maybe_indices_to_slice as _indicesToSlice
except ImportError:                                 # another pandas layout: df[names] (_PANDAS3 False below)
    _indicesToSlice = None

# pandas 3: column labels of str dtype, frames built from arrays copied in Fortran order by default, selections of
# columns that are views when their positions form a slice (_columnsOf3)
_PANDAS3 = (int(pd.__version__.split('.')[0]) >= 3 and _indicesToSlice is not None
            and callable(getattr(pd.DataFrame, '_slice', None)))

_indexers = {}
_columnIndexes = {}


def columnIndex(names):
    """The Index pd.DataFrame(..., columns=list(names)) builds (pd.Index(list(names), tupleize_cols=False)), kept for
    the same names (strings): the frames built at every step on the same names share one Index object (immutable),
    whose positions indexer() then finds again instead of looking them up at every call."""
    names = labels(names)
    if not all(type(name) is str for name in names):
        return pd.Index(names, tupleize_cols=False)
    key = tuple(names)
    found = _columnIndexes.get(key)
    if found is None:
        if len(_columnIndexes) >= 256:
            _columnIndexes.clear()
        found = _columnIndexes[key] = pd.Index(names, tupleize_cols=False)
    return found


def takeArray(df, names):
    """df[names].to_numpy() for a frame made of one float64 block, with unique columns holding every name once; None
    otherwise. pandas takes the rows of the names from the block into a new block of the same order (take_nd: an
    F-ordered block gives an F-ordered block, any other a C-ordered one) and to_numpy gives its transpose: the same
    array here (values and memory layout), taken from df.to_numpy() (the block transposed) without building a frame.
    pd.DataFrame(takeArray(df, names)) has the block of df[names] (its transpose)."""
    blocks = getattr(getattr(df, '_mgr', None), 'blocks', None)
    if (blocks is None or len(blocks) != 1 or blocks[0].dtype != np.float64 or blocks[0].values.ndim != 2
            or not df.columns.is_unique or len(set(names)) != len(names)):
        return None
    positions = indexer(df.columns, names)
    if not np.all(positions >= 0):
        return None
    values = df.to_numpy()                          # nodes x columns: the block transposed
    if blocks[0].values.flags.f_contiguous:
        return np.take(values, positions, axis=1)   # C-ordered: transpose of the F-ordered block pandas takes
    return values.T[positions].T                    # F-ordered: transpose of the C-ordered block pandas takes


def columnsOf(df, names):
    """df[names] for a list of names. When df is one block of float64 columns, unique, named by strings (an object
    Index without name) and holding every name once: the same frame (values, block and its memory layout, see
    takeArray) built from the array of df, with the same index object and an Index of the names (the labels and dtype
    of df.columns.take, no name), ~15 us instead of ~0.12 ms. Otherwise df[names].

    pandas 3 (columns of str dtype, which the object test above left to df[names] at every call, ~0.2 ms, with a new
    Index of the names defeating the caches of positions of the next steps): _columnsOf3."""
    names = list(names)
    columns = df.columns
    if _PANDAS3:
        if names and columns.name is None and all(type(name) is str for name in names):
            found = _columnsOf3(df, names)
            if found is not None:
                return found
        return df[names]
    if names and columns.dtype == object and columns.name is None and all(type(name) is str for name in names):
        taken = takeArray(df, names)
        if taken is not None:
            return pd.DataFrame(taken, index=df.index, columns=columnIndex(names))
    return df[names]


def _columnsOf3(df, names):
    """df[names] with pandas 3, for a frame of one float64 block in column order (block row k = column k), unique
    columns named by strings, an Index of the dtype pandas gives to these names, no attrs: the frame pandas builds, by
    the block operation pandas chooses, with columnIndex(names) as columns (same labels, dtype, no name); None
    otherwise (df[names]).

    pandas takes the rows of the names from the block (BlockManager._slice_take_blocks_ax0): when their positions form
    a slice (lib.maybe_indices_to_slice: one position, or a constant step), a view of the block (getitem_block_columns,
    with its copy-on-write references), otherwise a new block (take_nd). The view is the frame df.iloc[:, slice] gives
    (the same getitem_block_columns of the same slice, by DataFrame._slice which iloc calls, ~25 us instead of
    ~0.2 ms); the new block is allocated and filled as take_nd does (_takeRows: same shape, order and strides) and
    given to the frame without copy (pandas 3 would copy it in Fortran order)."""
    columns = df.columns
    blocks = getattr(getattr(df, '_mgr', None), 'blocks', None)
    if (type(columns) is not pd.Index or df.attrs or not df.flags.allows_duplicate_labels or blocks is None
            or len(blocks) != 1 or blocks[0].dtype != np.float64 or blocks[0].values.ndim != 2
            or not columns.is_unique or len(set(names)) != len(names)):
        return None
    locs = blocks[0].mgr_locs
    if not (locs.is_slice_like and locs.indexer == slice(0, len(columns), 1)):
        return None
    index = columnIndex(names)
    if index.dtype != columns.dtype:
        return None
    positions = indexer(columns, names)
    if not np.all(positions >= 0):
        return None                                 # df[names] raises its KeyError
    taker = positions.astype(np.intp, copy=False)
    found = _indicesToSlice(taker, max(len(taker), int(taker.max()) + 1))
    if isinstance(found, slice):
        # df.iloc[:, found] (_iLocIndexer._get_slice_axis: the bounds of a slice from positions of the frame are valid)
        result = df._slice(found, axis=1)
        result.columns = index
        return result
    block = _takeRows(blocks[0].values, taker)
    return pd.DataFrame(block.T, index=df.index, columns=index, copy=False)


def _takeRows(values, taker):
    """Rows `taker` of the 2D block `values` in a new array allocated as pandas' _take_nd_ndarray allocates it (an
    F-contiguous block taken transposed, the result F-ordered when the source then is F-contiguous along the taken
    axis): the same shape, order and strides, also for the axes of length 1 whose strides numpy reads (BLAS). The
    values are copied by np.take (mode 'clip': every position is valid, no buffered copy)."""
    arr, axis = values, 0
    flip = arr.flags.f_contiguous
    if flip:
        arr, axis = arr.T, 1
    shape = list(arr.shape)
    shape[axis] = len(taker)
    out = np.empty(shape, dtype=arr.dtype, order='F' if arr.flags.f_contiguous and axis == 1 else 'C')
    np.take(arr, taker, axis=axis, out=out, mode='clip')
    return out.T if flip else out


def labels(values):
    """list(values): Index.tolist for an Index (the same labels; iterating a str Index of pandas 3 reads them one by
    one, ~0.2 us each, ~15 times slower), list(values) for anything else."""
    return values.tolist() if isinstance(values, pd.Index) else list(values)


def allFloat64(df):
    """all(dt == np.float64 for dt in df.dtypes), read from the blocks of the frame: each column is in one block, so one
    comparison per block (usually one) instead of one per column (~20-40 us for 150 columns, several times per step);
    a block holding no column is left out. Without blocks (another frame implementation), from df.dtypes."""
    blocks = getattr(getattr(df, '_mgr', None), 'blocks', None)
    if blocks is None:
        return all(dt == np.float64 for dt in df.dtypes)
    return all(block.dtype == np.float64 for block in blocks if len(block.mgr_locs))


def indexer(columns, names):
    """columns.get_indexer(list(names)), as a read-only array, kept for the same Index object and the same names.

    commMtrx is rebuilt at each step on the same columns Index, and get_indexer builds an Index of the names at each call
    (~0.1 ms, several times per step). The cache holds the Index itself, compared by identity: another Index, even an
    equal one, is looked up again (an Index is immutable: same object, same positions). The key holds the id of the
    Index as well: the same names looked up in two Index objects at every step (commMtrx and a frame of its columns)
    keep one entry each instead of replacing each other (the entry keeps its Index alive, so the id is not reused)."""
    names = labels(names)
    key = (id(columns), tuple(names))
    found = _indexers.get(key)
    if found is not None and found[0] is columns:
        return found[1]
    positions = columns.get_indexer(names)
    positions.flags.writeable = False
    if len(_indexers) >= 256:
        _indexers.clear()
    _indexers[key] = (columns, positions)
    return positions


_constants = {}


def constant(name, sources, build, cache=None, size=256):
    """build(), computed once and kept while `sources` are the same objects: for the values that a solver call derives
    from settings only (lists of names, flags, script texts), the same at every call of a run.

    The settings of a run are fixed once the solvers are set up: centralDict gets new objects when they change (set-up
    of an instance, MultiCompoundTransport), never lists or dicts changed in place during the coupling. The entry keeps
    its sources alive: no other object can have their ids while it is kept, so the ids of the key are those objects
    themselves (another object, even an equal one, builds the value again). An error of build() is raised as before and
    nothing is kept. The value is shared by every call that finds it: the callers only read it. name: any hashable
    (values that are not objects of the settings, such as a number of nodes, go there). cache, size: a store of the
    caller (large values, emptied at the end of a run) and its number of entries, instead of the common store."""
    store = _constants if cache is None else cache
    key = (name, *map(id, sources))
    found = store.get(key)
    if found is not None:
        return found[1]
    value = build()
    if len(store) >= size:
        store.clear()
    store[key] = (tuple(sources), value)
    return value
