"""Intermediate outputs: output = {'stepEnd' | 'speciation' | 'transport': [steps]}, txt and / or vtu (outputFormat)."""
import difflib
import numbers
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

try:
    # compiled with pycte (setup.py, pycte/text_src): the values written as repr / numpy astype(str) write them (same
    # characters), ~6 times faster, without holding the GIL; without it (pycte installed without a C compiler) numpy
    from .. import _floatText
except ImportError:
    _floatText = None
try:
    from .frames import allFloat64 as _allFloat64
except ImportError:
    from frames import allFloat64 as _allFloat64

OUTPUT_KEYS = ('stepEnd', 'speciation', 'transport')
OUTPUT_FORMATS = ('txt', 'vtu')
LEGACY_FOLDERS = ('CouplingHistory', 'coupling', 'coupling_vtu', 'Speciation', 'PrimarySpecies', 'Transport', 'renouvellement', 'renewal')
LEGACY_PREFIXES = ('outputTransport', 'outputVTU')
RENAMED_KEYS = {'coupling': 'stepEnd'}


def _stepValue(value):
    """Whole step number, or None."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    try:
        return int(value) if int(value) == value else None
    except (OverflowError, ValueError):
        return None


def normalize(output, nSteps, dropped=None):
    """output -> {key: set of step numbers}; ValueError if malformed. With a dict `dropped` (output steps of a bundled
    example, the run made shorter), steps beyond the run are left out and listed there instead of being an error; a
    key left without step keeps the last step."""
    if output is None:
        return {'stepEnd': {nSteps}}
    if output is False:
        return {}
    if not isinstance(output, dict):
        raise ValueError(f"output must be a dict {{'stepEnd' | 'speciation' | 'transport': [steps]}} (got {output!r})")
    result = {}
    for key, steps in output.items():
        if key not in OUTPUT_KEYS:
            hint = [RENAMED_KEYS[key]] if key in RENAMED_KEYS else difflib.get_close_matches(str(key), OUTPUT_KEYS, n=1)
            raise ValueError(f"output : unknown key {key!r}" + (f" (did you mean '{hint[0]}' ?)" if hint else "")
                             + f" : use {', '.join(OUTPUT_KEYS)}")
        if isinstance(steps, numbers.Real) and not isinstance(steps, bool):
            steps = [steps]
        if isinstance(steps, (str, bytes)) or not hasattr(steps, '__iter__'):
            raise ValueError(f"output['{key}'] must be a list of time step numbers (got {steps!r})")
        steps = list(steps)
        # a value that is not a whole number ('5', 2.5, True) used to be dropped without notice
        wrong = [s for s in steps if _stepValue(s) is None]
        if wrong:
            raise ValueError(f"output['{key}'] : {wrong} are not time step numbers (whole numbers, 0 = initial state)")
        steps = [_stepValue(s) for s in steps]
        bad = [s for s in steps if not 0 <= s <= nSteps]
        if bad and dropped is not None:
            dropped[key] = sorted(bad)
            steps = [s for s in steps if 0 <= s <= nSteps] or [nSteps]
        elif bad:
            raise ValueError(f"output['{key}'] : step(s) {bad} outside [0, {nSteps}] ({nSteps} coupling steps, "
                             "0 = initial state)")
        result[key] = set(steps)
    return result


def normalizeFormat(outputFormat):
    """outputFormat -> tuple of formats; ValueError if malformed."""
    if outputFormat is None:
        return ('txt',)
    if isinstance(outputFormat, str):
        outputFormat = [outputFormat]
    if not hasattr(outputFormat, '__iter__'):
        raise ValueError(f"outputFormat must be a list among {list(OUTPUT_FORMATS)} (got {outputFormat!r})")
    formats = []
    for value in outputFormat:
        name = value.strip().lower().lstrip('.') if isinstance(value, str) else value
        if name not in OUTPUT_FORMATS:
            hint = difflib.get_close_matches(str(value), OUTPUT_FORMATS, n=1)
            raise ValueError(f"outputFormat : unknown format {value!r}" + (f" (did you mean '{hint[0]}' ?)" if hint else "")
                             + f" : use {', '.join(OUTPUT_FORMATS)}")
        if name not in formats:
            formats.append(name)
    if not formats:
        raise ValueError(f"outputFormat must hold at least one format among {list(OUTPUT_FORMATS)}")
    return tuple(formats)


def stepNumber(centralDict):
    """Output number of the current step (0: initial speciation)."""
    return 0 if centralDict.get('firstStepEquilibrium') else centralDict['lStep'] + 1


def wanted(centralDict, key, step=None):
    """Should output `key` be written at the current step (or at step `step`)?"""
    steps = (centralDict.get('output') or {}).get(key)
    step = stepNumber(centralDict) if step is None else step
    paths = centralDict.get('paths') or {}
    return bool(steps) and step in steps and (key in paths or f"{key}_vtu" in paths)


def filePath(centralDict, pathKey, prefix, step=None):
    """Path <paths[pathKey]>/<prefix>_<n>.txt."""
    step = stepNumber(centralDict) if step is None else step
    return os.path.join(centralDict['paths'][pathKey], f"{prefix}_{step}.txt")


_writer = None
_writes = []
_MAX_PENDING_WRITES = 8


def _submit(task, *args):
    """Runs task(*args) on the output thread; raises the error of a finished write."""
    global _writer
    if _writer is None:
        _writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix='pycte-output')
    _writes.append(_writer.submit(task, *args))
    while _writes and (_writes[0].done() or len(_writes) > _MAX_PENDING_WRITES):
        _writes.pop(0).result()


def flush():
    """Waits for the files being written; raises the first write error."""
    error = None
    while _writes:
        try:
            _writes.pop(0).result()
        except BaseException as err:
            error = error or err
    if error is not None:
        raise error


def close():
    """End of run: waits for the files being written and stops the output thread."""
    global _writer
    try:
        flush()
    finally:
        if _writer is not None:
            _writer.shutdown(wait=True)
            _writer = None


def _textFast(frame, header=True):
    """Rows of `frame` as to_csv(index=False, sep='\\t', header=header) writes them, or None outside the float64
    case."""
    names = frame.columns.tolist()
    if header and not all(isinstance(c, str) and c and not any(ch in c for ch in '\t"\r\n') for c in names):
        return None
    if not _allFloat64(frame):
        return None
    values = frame.to_numpy()
    if np.isnan(values).any():
        return None
    if _floatText is not None and values.dtype == np.float64 and values.ndim == 2:
        # the same text: each value as astype(str) writes it (repr of the float), rows joined by tabs, each ended by
        # os.linesep, after the header line
        body = _floatText.rows(values, '\t', os.linesep)
        if header:
            return '\t'.join(names) + os.linesep + body
        return body if len(values) else os.linesep
    lines = ['\t'.join(names)] if header else []
    lines += ['\t'.join(row) for row in values.astype(str).tolist()]
    return os.linesep.join(lines) + os.linesep


def writeText(path, frame, header=True):
    """`frame` -> tab-separated text file, same bytes as frame.to_csv(path, index=False, header=header, sep='\\t')."""
    # without header, an empty frame gives an empty file (to_csv), not an empty line
    text = _textFast(frame, header) if len(frame.columns) and (header or len(frame)) else None
    if text is None:
        frame.to_csv(path, index=False, header=header, sep='\t')
        return
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        handle.write(text)


def _writeFiles(textPath, vtuPath, frame, coord, reference):
    if textPath is not None:
        writeText(textPath, frame)
    if vtuPath is not None:
        writeVTU(vtuPath, frame, coord, reference)


def write(centralDict, frame, pathKey, prefix, step=None):
    """Writes `frame` as <prefix>_<n>.txt and / or .vtu, following outputFormat (on the output thread). Solver instance
    of a role with several instances (solverInstances): in the sub-folder of the instance, e.g. transport/<name>/."""
    step = stepNumber(centralDict) if step is None else step
    paths = centralDict['paths']
    instance = centralDict.get('solverInstance') if pathKey != 'stepEnd' else None
    folder = (lambda key: os.path.join(paths[key], instance)) if instance else (lambda key: paths[key])
    textPath = os.path.join(folder(pathKey), f"{prefix}_{step}.txt") if pathKey in paths else None
    vtuPath = (os.path.join(folder(f"{pathKey}_vtu"), f"{prefix}_{step}.vtu") if f"{pathKey}_vtu" in paths else None)
    if textPath is None and vtuPath is None:
        return
    if instance:
        for path in (textPath, vtuPath):
            if path is not None:
                os.makedirs(os.path.dirname(path), exist_ok=True)
    coord = centralDict.get('coord')
    reference = centralDict.get('commMtrx') if vtuPath is not None else None
    _submit(_writeFiles, textPath, vtuPath, frame.copy(), None if coord is None else list(coord),
            None if reference is None else reference.copy())


def _points(frame, coord, reference):
    """(n, 3) node coordinates and coordinate columns of `frame` (or of `reference`)."""
    for source in (frame, reference):
        if source is None or len(source) != len(frame):
            continue
        present = [(axis, name) for axis, name in enumerate(coord or ()) if name in source.columns]
        if present:
            points = np.zeros((len(frame), 3))
            for axis, name in present:
                points[:, axis] = np.asarray(source[name], dtype=float).reshape(len(frame), -1)[:, 0]
            return points, [name for _, name in present] if source is frame else []
    return None, []


_meshCache = {}


def _cells(points):
    """VTK cells of the nodes: segments (1D), quads / hexahedra (rectilinear grid) or vertices."""
    n = len(points)
    key = (n, hash(points.tobytes()))
    if key in _meshCache:
        return _meshCache[key]
    axes = [a for a in range(3) if n and np.ptp(points[:, a]) > 0]
    mesh = (np.arange(n, dtype=np.int64), 1, 1)
    if len(axes) == 1:
        order = np.argsort(points[:, axes[0]], kind='stable').astype(np.int64)
        mesh = (np.column_stack([order[:-1], order[1:]]).ravel(), 2, 3)
    elif len(axes) > 1:
        inverse = [np.unique(points[:, a], return_inverse=True) for a in axes]
        shape = tuple(len(u) for u, _ in inverse)
        if int(np.prod(shape)) == n:
            flat = np.ravel_multi_index(tuple(i.ravel() for _, i in inverse), shape)
            if np.bincount(flat, minlength=n).max() == 1:
                grid = np.empty(n, dtype=np.int64)
                grid[flat] = np.arange(n, dtype=np.int64)
                g = grid.reshape(shape)
                if len(axes) == 2:
                    corners = [g[:-1, :-1], g[1:, :-1], g[1:, 1:], g[:-1, 1:]]
                    mesh = (np.stack([c.ravel() for c in corners], axis=1).ravel(), 4, 9)
                else:
                    corners = [g[:-1, :-1, :-1], g[1:, :-1, :-1], g[1:, 1:, :-1], g[:-1, 1:, :-1],
                               g[:-1, :-1, 1:], g[1:, :-1, 1:], g[1:, 1:, 1:], g[:-1, 1:, 1:]]
                    mesh = (np.stack([c.ravel() for c in corners], axis=1).ravel(), 8, 12)
    if len(_meshCache) > 8:
        _meshCache.clear()
    _meshCache[key] = mesh
    return mesh


def writeVTU(path, frame, coord, reference=None):
    """`frame` -> VTK unstructured grid file `path`."""
    points, skip = _points(frame, coord, reference)
    if points is None:
        from . import warningManager
        warningManager.warn(f"pycte : {os.path.basename(path)} not written : no coordinate column {list(coord or ())}")
        return
    connectivity, size, cellType = _cells(points)
    nCells = len(connectivity) // size
    try:
        values = frame.to_numpy(dtype=np.float64)
        columns = [(name, values[:, k]) for k, name in enumerate(frame.columns)]
    except (TypeError, ValueError):
        columns = []
        for k, name in enumerate(frame.columns):
            try:
                columns.append((name, frame.iloc[:, k].to_numpy(dtype=np.float64)))
            except (TypeError, ValueError):
                pass

    blocks, xml = [], []

    def array(data, dtype, attributes):
        data = np.ascontiguousarray(data, dtype=dtype)
        offset = sum(8 + b.nbytes for b in blocks)
        xml.append(f'<DataArray {attributes} format="appended" offset="{offset}"/>')
        blocks.append(data)

    xml.append('<PointData>')
    for name, data in columns:
        if name not in skip:
            array(data, '<f8', f'type="Float64" Name={quoteattr(str(name))}')
    xml.append('</PointData>\n<Points>')
    array(points, '<f8', 'type="Float64" NumberOfComponents="3"')
    xml.append('</Points>\n<Cells>')
    array(connectivity, '<i8', 'type="Int64" Name="connectivity"')
    array(np.arange(size, size * nCells + 1, size), '<i8', 'type="Int64" Name="offsets"')
    array(np.full(nCells, cellType), 'u1', 'type="UInt8" Name="types"')
    xml.append('</Cells>')

    head = ('<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" '
            f'header_type="UInt64">\n<UnstructuredGrid>\n<Piece NumberOfPoints="{len(frame)}" NumberOfCells="{nCells}">\n'
            + '\n'.join(xml) + '\n</Piece>\n</UnstructuredGrid>\n<AppendedData encoding="raw">\n_')
    with open(path, 'wb') as handle:
        handle.write(head.encode('utf-8'))
        for data in blocks:
            handle.write(np.uint64(data.nbytes).tobytes())
            handle.write(data.tobytes())
        handle.write(b'\n</AppendedData>\n</VTKFile>\n')


def folders(runFolder, output, comsolTags=(), formats=('txt',)):
    """Requested output folders {paths key: path}, with their _vtu twins."""
    run = Path(runFolder)
    tree = {}
    if 'stepEnd' in output:
        tree['stepEnd'] = ('stepEnd',)
    if 'speciation' in output:
        tree['speciation'] = ('speciation',)
        tree['primarySpecies'] = ('speciation', 'primarySpecies')
    if 'transport' in output:
        tree['transport'] = ('transport',)
        for tag in comsolTags:
            tree[f'Transport{tag}'] = ('transport', str(tag))
    paths = {}
    if 'txt' in formats:
        paths.update({key: str(run.joinpath(*parts)) for key, parts in tree.items()})
    if 'vtu' in formats:
        paths.update({f"{key}_vtu": str(run.joinpath(*(f"{p}_vtu" for p in parts))) for key, parts in tree.items()})
    return paths


MARKER = '.pycte_output'        # written in each output folder pycte creates
# StepEnd_3.txt, nativeTransport_5.vtu, COMSOL_0.txt, renewal/renewal.txt
OUTPUT_FILE = re.compile(r'^([A-Za-z][\w-]*_\d+\.(txt|vtu)|renewal\.txt)$')


def _pycteFolder(folder):
    """True if the folder holds pycte outputs only: its MARKER, or (outputs of a version without the marker) only output
    files and folders of output files."""
    if (folder / MARKER).is_file():
        return True
    for entry in folder.rglob('*'):
        if entry.is_file() and not OUTPUT_FILE.match(entry.name):
            return False
    return True


def prepareFolders(runFolder, paths):
    """Removes the output folders of a previous run and creates the requested ones, with their MARKER. A folder named
    like an output folder that holds other files (a folder of the user) is never removed: renamed <name>_<date-time>
    when the run needs its name, left alone otherwise. Returns the warnings (renamed folders)."""
    run = Path(runFolder)
    known = set(OUTPUT_KEYS) | {f"{key}_vtu" for key in OUTPUT_KEYS} | set(LEGACY_FOLDERS)
    needed = {Path(path).relative_to(run).parts[0].lower() for path in paths.values()}
    warnings = []
    for entry in run.iterdir():
        if not (entry.is_dir() and (entry.name in known or entry.name.startswith(LEGACY_PREFIXES))):
            continue
        try:
            if _pycteFolder(entry):
                shutil.rmtree(entry)
            elif entry.name.lower() in needed:          # lower: Transport and transport are one folder on Windows
                kept = entry.with_name(f"{entry.name}_{time.strftime('%Y%m%d-%H%M%S')}")
                entry.rename(kept)
                warnings.append(f"pycte : the folder {entry.name} holds files that are not pycte outputs : it is kept, "
                                f"renamed {kept.name}, and {entry.name} is created for the outputs of this run")
        except PermissionError as error:
            # Windows: a file open in Excel, a viewer or another Python session cannot be removed nor its folder renamed
            raise RuntimeError(f"pycte : the outputs of the previous run in {entry} cannot be replaced : "
                               f"{error.filename or entry} is open in another program (Excel, a viewer, another "
                               "Python session) : close it, or run in another folder") from None
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    for name in {Path(path).relative_to(run).parts[0] for path in paths.values()}:
        (run / name / MARKER).touch()
    return warnings
