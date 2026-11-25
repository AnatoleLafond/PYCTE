import ctypes
import difflib
import numpy as np
import time
import pandas as pd
import importlib.util
import os
import platform
import re
import sys
import threading
import queue
from pathlib import Path
import concurrent.futures
try:
    from . import outputManager
    from . import timeUnits
    from . import warningManager
    from . import display
    from . import crossDependenciesManager
    from . import frames
except ImportError:
    import outputManager
    import timeUnits
    import warningManager
    import display
    import crossDependenciesManager
    import frames
try:
    # compiled with pycte (setup.py, pycte/phreeqc_src): a row of the selected output read in one call (same calls of
    # GetSelectedOutputValue, same values); without it (pycte installed without a C compiler) the Python loop reads it
    from .. import _phreeqcRow
except ImportError:
    _phreeqcRow = None
try:
    # compiled with pycte (pycte/text_src): float64 values written as repr writes them (the text of f"{x}"), faster
    from .. import _floatText
except ImportError:
    _floatText = None

phreeqc = None

_EMPTY_FRAME = pd.DataFrame()


def _dtypes(df):
    """Column dtypes of df, without building a Series."""
    try:
        return df._mgr.get_dtypes()
    except AttributeError:
        return df.dtypes.to_numpy()


def _selectColumns(df, names):
    """df[names] (new frame, same values, same block and memory layout): frames.columnsOf."""
    return frames.columnsOf(df, names)


def resetPhreeqC(verbose=False):

    global phreeqc, _threads
    _releasePreloaded()
    _stopScriptBuilders()
    # blocks of speciationThreads > 1 (threads): instances released, the next run starts from new ones
    for block in _blocks:
        block.release()
    _blocks.clear()
    if _threads is not None:
        _threads.shutdown(wait=True)
        _threads = None
    if phreeqc is None:
        return False
    for meth in ('destroy_iphreeqc', 'destroy', 'DestroyIPhreeqc'):
        f = getattr(phreeqc, meth, None)
        if callable(f):
            try:
                f()
                if verbose: print(f"IPhreeqc released via {meth}() (PID={os.getpid()})")
            except Exception as r:
                warningManager.warn(f"PhreeqC : IPhreeqc not released by {meth}() : {r}")
            break
    phreeqc = None
    return True

def writeTime(tps, arr=2):
    if tps >= 3600 * 24:
        return f"{tps / (3600 * 24):.{arr}f} d"
    elif tps >= 3600:
        return f"{tps / 3600:.{arr}f} h"
    elif tps >= 60:
        return f"{tps / 60:.{arr}f} min"
    elif tps < 1:
        return f"{tps*1000 :.{arr}f} msec"
    else:
        return f"{tps:.{arr}f} sec"

DATABASE_KEYWORDS = {
    'masterSpecies': 'SOLUTION_MASTER_SPECIES',
    'solutionSpecies': 'SOLUTION_SPECIES',
    'phases': 'PHASES',
    'masterExchange': 'EXCHANGE_MASTER_SPECIES',
    'exchangeSpecies': 'EXCHANGE_SPECIES',
    'masterSurface': 'SURFACE_MASTER_SPECIES',
    'surfaceSpecies': 'SURFACE_SPECIES',
}

_DATA_BLOCKS = {
    "SOLUTION_MASTER_SPECIES", "SOLUTION_SPECIES", "PHASES", "EXCHANGE_MASTER_SPECIES", "EXCHANGE_SPECIES",
    "SURFACE_MASTER_SPECIES", "SURFACE_SPECIES", "RATES", "END", "SIT", "PITZER", "LLNL_AQUEOUS_MODEL_PARAMETERS",
    "GAS_BINARY_PARAMETERS", "MEAN_GAMMAS", "NAMED_EXPRESSIONS", "CALCULATE_VALUES", "ISOTOPES", "ISOTOPE_RATIOS",
    "ISOTOPE_ALPHAS", "SOLID_SOLUTIONS", "EQUILIBRIUM_PHASES", "KINETICS", "REACTION", "REACTION_TEMPERATURE",
    "REACTION_PRESSURE", "SOLUTION", "SOLUTION_SPREAD", "EXCHANGE", "SURFACE", "GAS_PHASE", "KNOBS", "USE", "SAVE",
    "SELECTED_OUTPUT", "USER_PUNCH", "USER_PRINT", "USER_GRAPH", "PRINT", "TITLE", "DATABASE", "INCLUDE$", "COPY",
    "DELETE", "DUMP", "MIX", "RUN_CELLS", "TRANSPORT", "ADVECTION", "INVERSE_MODELING", "INCREMENTAL_REACTIONS",
}


def _blockKeyword(line):
    """PhreeqC keyword opening a data block on this line, or None."""
    words = line.split('#', 1)[0].split()
    if not words:
        return None
    word = words[0].upper()
    return word if word in _DATA_BLOCKS or word.endswith(('_MODIFY', '_RAW')) else None


# reaction of the pseudo-phases of fixedPhases whose name is known ; any other name needs its reaction as 4th item
FIXED_PHASE_REACTIONS = {'Fix_ph': 'H+ = H+', 'Fix_H+': 'H+ = H+', 'Fix_pe': 'e- = e-'}


def fixedPhaseSpecs(fixedPhases):
    """fixedPhases {name: (SI, reactant, amount[, reaction])} -> [(name, SI, reactant, amount, reaction)]."""
    specs = []
    for name, value in (fixedPhases or {}).items():
        if not isinstance(value, (tuple, list)) or len(value) not in (3, 4):
            raise ValueError(f"fixedPhases['{name}'] must be (SI, reactant, amount) or (SI, reactant, amount, "
                             f"reaction), got {value!r}")
        reaction = value[3] if len(value) == 4 else FIXED_PHASE_REACTIONS.get(name)
        if reaction is None:
            raise ValueError(f"fixedPhases['{name}'] : unknown pseudo-phase, give its reaction as 4th item "
                             f"(known : {', '.join(FIXED_PHASE_REACTIONS)})")
        specs.append((name, float(value[0]), value[1], float(value[2]), reaction))
    return specs


def userDatabaseBlocks(values, fixpH=None, fixedPhases=None):
    """Database keywords (and fixpH / fixedPhases phases) -> [(PhreeqC keyword, lines)]."""
    allowed = set(DATABASE_KEYWORDS.values())
    blocks = []
    for key, header in DATABASE_KEYWORDS.items():
        text = values.get(key)
        if not isinstance(text, str):
            if not text or (key == 'phases' and isinstance(text, (list, tuple))):
                continue
            raise ValueError(f"{key} must be PhreeqC input text (content of a {header} block), got a "
                             f"{type(text).__name__}")
        current = None
        for line in text.splitlines():
            keyword = _blockKeyword(line)
            if keyword is not None:
                if keyword not in allowed:
                    raise ValueError(f"{key} : the keyword {keyword} is not a database block that pycte can add "
                                     f"(allowed : {', '.join(sorted(allowed))})")
                current = [keyword, []]
                blocks.append(current)
                continue
            if current is None:
                if not line.split('#', 1)[0].strip():
                    continue
                current = [header, []]
                blocks.append(current)
            current[1].append(line)
    specs = fixedPhaseSpecs(fixedPhases)
    if fixpH and 'Fix_ph' not in [s[0] for s in specs]:
        blocks.append(['PHASES', ['Fix_ph', '    H+ = H+', '    log_k 0']])
    for name, _, _, _, reaction in specs:
        blocks.append(['PHASES', [name, f'    {reaction}', '    log_k 0']])
    return [(keyword, lines) for keyword, lines in blocks]


def mergeDatabase(text, blocks):
    """Database text with the user blocks added to the blocks of the same keyword."""
    lines = text.splitlines(keepends=True)
    newline = '\r\n' if lines and lines[0].endswith('\r\n') else '\n'
    keywords = [_blockKeyword(line) for line in lines]
    continued, created = {}, {}
    for keyword, content in blocks:
        added = [line + newline for line in content]
        start = next((i for i, k in enumerate(keywords) if k == keyword), None)
        if start is None:
            at = next((i for i, k in enumerate(keywords) if k == 'END'), len(lines))
            created.setdefault(at, {}).setdefault(keyword, []).extend(added)
        else:
            at = next((i for i in range(start + 1, len(lines)) if keywords[i] is not None), len(lines))
            continued.setdefault(at, []).extend(added)
    out = []
    for i in range(len(lines) + 1):
        if i in continued or i in created:
            if out and not out[-1].endswith(('\n', '\r')):
                out[-1] += newline
            out += continued.get(i, [])
            for keyword, added in created.get(i, {}).items():
                out += [keyword + newline] + added
        if i < len(lines):
            out.append(lines[i])
    return ''.join(out)


def prepareDatabase(chemPath, runFolder, values, fixpH=None, fixedPhases=None):
    """chemPath, or its copy completed with the database keywords (<name>_pycte<ext> in the run folder)."""
    from pathlib import Path
    blocks = userDatabaseBlocks(values, fixpH, fixedPhases)
    if not blocks:
        return Path(chemPath)
    from charset_normalizer import from_path
    chemPath = Path(chemPath)
    best = from_path(chemPath).best()
    encoding = best.encoding if best else 'utf-8'
    with open(chemPath, 'r', encoding=encoding, newline='') as f:
        text = f.read()
    merged = Path(runFolder) / f"{chemPath.stem}_pycte{chemPath.suffix}"
    with open(merged, 'w', encoding=encoding, newline='') as f:
        f.write(mergeDatabase(text, blocks))
    return merged


def knobsText(knobs):
    """KNOBS block of the knobs keyword (text or dict), '' without it."""
    if not knobs:
        return ''
    if isinstance(knobs, dict):
        lines = [f"-{key.lstrip('-')} {value}" for key, value in knobs.items()]
    else:
        lines = [line for line in str(knobs).strip().splitlines() if line.strip()]
        if lines and lines[0].split()[0].upper() == 'KNOBS':
            lines = lines[1:]
    return "KNOBS\n" + "".join(f"\t{line.strip()}\n" for line in lines) + "END\n"


def selectedOutputNames(centralDict):
    """Names read back from PhreeqC at each step: (totals, molalities, equilibrium phases) of SELECTED_OUTPUT."""
    phasesPrim = centralDict['primarySpecies']['phases']
    mctTotals = centralDict.get('mctTotals') if centralDict.get('MultiCompoundTransport') else None
    molalities = [spc for spc in centralDict['systemSpecies'] if spc not in phasesPrim]
    totals = []
    if mctTotals:
        totals = [t for t in mctTotals if t not in ('H', 'O')]
        molalities = [spc for spc in molalities if spc not in mctTotals]
    return totals, molalities, list(phasesPrim)


_libraryPath = []


def _iphreeqcLibrary():
    """IPhreeqc library of phreeqpy (the file phreeqpy.iphreeqc.phreeqc_dll.IPhreeqc chooses), for the functions phreeqpy
    does not wrap (GetWarningString); None if it is not there (other phreeqpy version). Found once per process."""
    if not _libraryPath:
        _libraryPath.append(_findIphreeqcLibrary())
    return _libraryPath[0]


def _findIphreeqcLibrary():
    import phreeqpy.iphreeqc.phreeqc_dll as phreeqc_mod
    if sys.platform == 'win32':
        # one name whatever the processor: platform.machine() (a WMI query on Windows, ~0.1 s) is not needed
        name = 'IPhreeqc-3.7.3.dll'
        path = os.path.join(os.path.dirname(phreeqc_mod.__file__), 'phreeqc3', name)
        return path if os.path.isfile(path) else None
    machine = platform.machine().lower()
    if sys.platform.startswith('linux'):
        name = {'x86_64': 'libiphreeqc-3.7.3.so', 'aarch64': 'linux_arm_libiphreeqc-3.7.3.so',
                'arm64': 'linux_arm_libiphreeqc-3.7.3.so'}.get(machine)
    elif sys.platform == 'darwin':
        name = {'x86_64': 'libiphreeqc-3.7.3.dylib', 'arm64': 'libiphreeqc-3.7.3-m1.dylib'}.get(machine)
    else:
        name = None
    path = os.path.join(os.path.dirname(phreeqc_mod.__file__), 'phreeqc3', name) if name else None
    return path if path and os.path.isfile(path) else None


WATER_MOLES_PER_KG = 55.50868      # mol of H2O in 1 kg of water

_warningReader = {}
_warningsSeen = {'run': None, 'kinds': set()}


def _warningFunction():
    """GetWarningString of the IPhreeqc library of phreeqpy (ctypes), None if the library is not found."""
    if 'function' not in _warningReader:
        library = _iphreeqcLibrary()
        function = None
        if library:
            function = ctypes.cdll.LoadLibrary(library).GetWarningString
            function.argtypes, function.restype = [ctypes.c_int], ctypes.c_char_p
        _warningReader['function'] = function
    return _warningReader['function']


def warningText(solver):
    """Warnings of the last run of an IPhreeqc instance ('' if the IPhreeqc library of phreeqpy is not found)."""
    function = _warningFunction()
    return (function(solver.id_) or b'').decode('utf-8', 'replace').strip() if function else ''


# _phreeqcRow.after (pycte/phreeqc_src): the calls that follow the run of a node in one call; None with an older
# build of _phreeqcRow, or without it
_afterReader = getattr(_phreeqcRow, 'after', None)
# the same, reading the given columns of the row only (VERSION 2 of _phreeqcRow); None with an older build
_afterColumns = _afterReader if getattr(_phreeqcRow, 'VERSION', 1) >= 2 else None
# _phreeqcRow.primary: the sums of primaryColumns in C (_primaryColumnsFast); None with an older build, or without it
_primarySums = getattr(_phreeqcRow, 'primary', None)


def _readColumns(header, needed):
    """Columns of the selected output read back at a step without speciation output: (their positions in the header, as
    an array of C ints, its address, their count, their names) for the names of `needed` (the names read from the
    frame of the selected output after the node loop). None, every column being read: needed None (speciation output
    written: every column), without _afterColumns, a header not made of distinct strings, no column or every column
    needed."""
    if needed is None or _afterColumns is None:
        return None
    if not all(type(name) is str for name in header) or len(set(header)) != len(header):
        return None
    positions = [j for j, name in enumerate(header) if name in needed]
    if not positions or len(positions) == len(header):
        return None
    array = np.array(positions, dtype=np.intc)
    return array, array.ctypes.data, len(positions), [header[j] for j in positions]


def _afterAddresses(solver):
    """(GetWarningString or 0 without it, GetSelectedOutputRowCount, GetSelectedOutputColumnCount,
    GetSelectedOutputValue) addresses and the offsets of the type and value of a VAR, for _phreeqcRow.after."""
    found = getattr(solver, '_pycteAfter', None)
    if found is None:
        warning = _warningFunction()
        address = lambda f: ctypes.cast(f, ctypes.c_void_p).value
        varType = type(solver.var)
        found = (address(warning) if warning is not None else 0, address(solver._get_selected_output_row_count),
                 address(solver._get_selected_output_column_count), _valueReader(solver)[3],
                 varType.type.offset, varType.value.offset)
        solver._pycteAfter = found
    return found


def _afterRun(solver, header, block, filled, blockAddress, columns=None):
    """(warningText, row_count, column_count of the selected output, status) after the run of a node, in one call of
    _phreeqcRow.after (the same calls of IPhreeqc, in the order of speciationPhreeqC: GetWarningString, the row count,
    the column count when there is a row). With the header known and the block of the rows: the last row is read into
    block[filled] (status 0) as selectedOutputRow reads it, when it has len(header) columns, all doubles; status 1:
    not read; 2 (not all doubles), 3 (error code): to be read value by value, as selectedOutputRow does then.
    columns (_readColumns of the header): only these columns of the row are read, into block[filled] (as many values),
    status 2 / 3 when one of them is not a double / gives an error code."""
    warning, rows, cols, value, typeOffset, valueOffset = _afterAddresses(solver)
    expected, buffer, size, dest = -1, 0, 0, 0
    if (header is not None and block is not None and filled < len(block)
            and block.shape[1] == (len(header) if columns is None else columns[2])):
        expected = len(header)
        buffers = _rowBuffers(solver, expected)
        buffer, size = buffers[6], buffers[7]
        dest = blockAddress + filled * block.strides[0]
    if columns is not None and _afterColumns is not None:
        text, nrows, ncols, status = _afterColumns(warning, rows, cols, value, solver.id_, expected, buffer, size,
                                                   typeOffset, valueOffset, dest, columns[1], columns[2])
    else:
        text, nrows, ncols, status = _afterReader(warning, rows, cols, value, solver.id_, expected, buffer, size,
                                                  typeOffset, valueOffset, dest)
    return (text.decode('utf-8', 'replace').strip() if text else ''), nrows, ncols, status


def newWarnings(text, run, seen=None):
    """Lines of `text` of a kind (numbers left out) not reported yet in this run by this process (seen: by this block of
    nodes, see _Block): a warning repeated at every node and step is written once in warning.log."""
    state = _warningsSeen if seen is None else seen
    if state['run'] != run:
        state.update(run=run, kinds=set())
    new = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        kind = re.sub(r'[-+]?\d+(\.\d*)?([eE][-+]?\d+)?', '#', line)
        if kind not in state['kinds']:
            state['kinds'].add(kind)
            new.append(line)
    return new


def newIPhreeqc():
    """IPhreeqc instance on the library of phreeqpy that warningText reads; a missing phreeqpy, or an IPhreeqc library
    that cannot be loaded, stops the run with the fix (the first instance is made when the run is set up)."""
    try:
        import phreeqpy.iphreeqc.phreeqc_dll as phreeqc_mod
    except ImportError as err:
        raise ImportError("chemistryModule = 'PhreeqC' : the phreeqpy package is not installed (it comes with pycte) : "
                          "pip install phreeqpy") from err
    try:
        return phreeqc_mod.IPhreeqc(_iphreeqcLibrary())
    except (OSError, NotImplementedError) as err:
        raise RuntimeError(f"PhreeqC : the IPhreeqc library of phreeqpy cannot be loaded on this machine ({err}) : "
                           "reinstall phreeqpy (pip install --force-reinstall phreeqpy), or check that this platform "
                           "is one phreeqpy supports (Windows, Linux x86_64 / aarch64, macOS)") from err


# IPhreeqc instances loaded while the run is set up (preloadIPhreeqc): PhreeqC reads the database in a thread (~75 ms
# for ThermoChimie, the GIL released by ctypes) while Python extracts it and sets up the transport. Each one is what
# newIPhreeqc() + loadDatabase() gives where it is taken (takeIPhreeqc): the same database text, KNOBS and RATES,
# nothing else run on it. Taken by the kineticRates check, checkOutputNames and the first speciation (instance of the
# process, or of each block of nodes); those not taken are released by resetPhreeqC. A load that failed is done again
# where the instance is taken, as before, so that its error is raised there. The instance is made, and the texts
# prepared, by the thread of the set-up: the thread of the load only makes the calls of IPhreeqc (each return from
# one needs the GIL, which the set-up holds most of the time: a few returns only, otherwise the load would end later
# than the set-up needs it).
_preloaded = []                 # [(database settings, future of the loaded instance)]
_preloaders = []                # threads of the loads (one per instance), stopped by _releasePreloaded
_preloadLock = threading.Lock()
# smallest database text preloaded: each return of a thread from IPhreeqc waits for the GIL (up to the 5 ms switch
# interval) while the set-up runs Python, so that two loads of phreeqc.dat (34 kB, 2 x 6 ms) end ~55 ms later and the
# set-up waits for them (+12 to +20 ms on CationExchange and CalciteDolomite), while ThermoChimie (636 kB, 2 x 76 ms)
# loads in 165 ms during the set-up (CementClayInterface: set-up -30 to -40 ms, initialisation -75 ms)
_PRELOAD_MIN_TEXT = 200_000


def _databaseSettings(centralDict):
    """What loadDatabase reads: the database file (absolute path, modification time and size), knobs, kineticRates."""
    path = os.path.abspath(str(centralDict['chemPath']))
    stat = os.stat(path)
    return (path, stat.st_mtime_ns, stat.st_size, centralDict.get('knobs'), centralDict.get('kineticRates'))


def _sameSettings(a, b):
    try:
        return bool(a == b)
    except Exception:
        return False


def _loadInto(solver, text, knobs, rates):
    """loadDatabase(solver, ...) of a database without INCLUDE$, read as `text`, then its KNOBS block (knobsText, ''
    without it) and RATES script (_ratesScript, None without it): the calls of IPhreeqc of loadDatabase, in its order
    (thread of preloadIPhreeqc). An error is raised, the instance released (then made and loaded again where it is
    taken, whose error is raised there)."""
    try:
        solver.load_database_string(text)
        if solver.phc_database_error_count:
            raise ValueError("PhreeqC could not read the database")
        if knobs:
            solver.run_string(knobs)
        if rates is not None:
            solver.run_string(rates)
    except BaseException:
        try:
            solver.destroy_iphreeqc()
        except Exception:
            pass
        raise
    return solver


def preloadIPhreeqc(chemPath, knobs, kineticRates, count):
    """Starts loading IPhreeqc instances with this database (see _preloaded), so that `count` of them are being loaded
    or waiting; nothing without phreeqpy or the IPhreeqc library that warningText reads, for a small database
    (_PRELOAD_MIN_TEXT) or one read with its INCLUDE$ files (loaded from its path)."""
    settings = {'chemPath': chemPath, 'knobs': knobs, 'kineticRates': kineticRates}
    try:
        if count <= 0 or _iphreeqcLibrary() is None:
            return
        key = _databaseSettings(settings)
        # read here (encoding detected, cached for the extraction and every load), not in the threads
        text = _databaseText(key[0])
        if len(text) < _PRELOAD_MIN_TEXT or 'INCLUDE$' in text.upper():
            return
        knobsScript = knobsText(knobs)
        ratesScript = _ratesScript(kineticRates) if kineticRates else None
    except Exception:
        return                          # the error comes again where the instance is made, as before
    with _preloadLock:
        missing = count - sum(1 for known, _ in _preloaded if _sameSettings(known, key))
        if missing <= 0:
            return
        # every instance made before the first load starts: in the IPhreeqc library, making an instance while another
        # one loads its database took 16 ms instead of 0.2 ms, and slowed that load down (CationExchange); loads of
        # instances made beforehand run side by side (3 loads of ThermoChimie in 72 ms, one in 68 ms), one per thread
        solvers = []
        for _ in range(missing):
            try:
                solvers.append(newIPhreeqc())
            except Exception:
                break                   # made where it is taken, its error raised there
        if not solvers:
            return
        loader = concurrent.futures.ThreadPoolExecutor(max_workers=len(solvers), thread_name_prefix='pycte-iphreeqc')
        _preloaders.append(loader)
        for solver in solvers:
            _preloaded.append((key, loader.submit(_loadInto, solver, text, knobsScript, ratesScript)))


def takeIPhreeqc(centralDict):
    """An instance preloaded with the database of centralDict (newIPhreeqc() + loadDatabase(..., centralDict)), waited
    for if it is still loading; None when there is none, or when its load failed: the caller then makes and loads its
    instance as before."""
    if not _preloaded:
        return None
    try:
        key = _databaseSettings(centralDict)
    except Exception:
        return None
    with _preloadLock:
        found = next((k for k, (known, _) in enumerate(_preloaded) if _sameSettings(known, key)), None)
        if found is None:
            return None
        _, future = _preloaded.pop(found)
    if future.exception() is not None:
        return None
    return future.result()


def _releasePreloaded():
    """End of run: the preloaded instances not taken are released, the threads of the loads stopped."""
    with _preloadLock:
        pending = [future for _, future in _preloaded]
        _preloaded.clear()
        loaders = _preloaders[:]
        _preloaders.clear()
    for future in pending:
        try:
            solver = future.result()
        except Exception:
            continue
        try:
            solver.destroy_iphreeqc()
        except Exception as r:
            warningManager.warn(f"PhreeqC : IPhreeqc not released by destroy_iphreeqc() : {r}")
    for loader in loaders:
        loader.shutdown(wait=True)


def checkOutputNames(centralDict):
    """Stops the run when PhreeqC does not know a name it is asked for at each step (species, totals, phases): PhreeqC
    only warns ('Did not find species, Ca2+.') and returns 0, so that a misspelled species (Ca2+ or Ca++ for Ca+2, the
    element Cl for Cl-) would be set to 0 from the first step. PhreeqC itself checks the names, on the database of the
    run (keywords of pycte merged), with the SELECTED_OUTPUT of the run."""
    totals, molalities, phases = selectedOutputNames(centralDict)
    # loaded while the database was extracted (preloadIPhreeqc: only with the IPhreeqc library found)
    solver = takeIPhreeqc(centralDict)
    loaded = solver is not None
    if not loaded:
        solver = newIPhreeqc()          # first instance of the run: a missing phreeqpy stops it here, with the fix
        if _iphreeqcLibrary() is None:
            solver.destroy_iphreeqc()
            warningManager.warn("PhreeqC : IPhreeqc library of phreeqpy not found, the names of the coupled variables "
                                "are not checked against the database (an unknown name would be set to 0)")
            return
    try:
        if not loaded:
            loadDatabase(solver, centralDict)
        text = "SELECTED_OUTPUT\n-reset false\n"
        if totals: text += "-totals" + "".join(f"\t{t}" for t in totals) + "\n"
        if molalities: text += "-molalities" + "".join(f"\t{s}" for s in molalities) + "\n"
        if phases: text += "-equilibrium_phases" + "".join(f"\t{p}" for p in phases) + "\n"
        solver.run_string(text + "END\n")
        warnings = warningText(solver)
    finally:
        solver.destroy_iphreeqc()

    asked = {'species': molalities, 'master species': totals, 'phase': phases}
    unknown = {}
    for kind, name in re.findall(r"Did not find (master species|species|phase), (.+?)\.?\s*$", warnings, re.M):
        if name in asked[kind] and name not in unknown.get(kind, []):
            unknown.setdefault(kind, []).append(name)
    if not unknown:
        return

    try:
        from . import extractDB
    except ImportError:
        import extractDB
    try:
        db = extractDB.extract_master_species(centralDict['chemPath'])
        known = {'species': sorted(extractDB.extract_solution_species(centralDict['chemPath']))
                 + db['EXCHANGE_SPECIES'] + db['SURFACE_SPECIES'] + db['SURFACE_MASTER_SPECIES'],
                 'master species': db['SOLUTION_MASTER_ELEMENT'], 'phase': db['PHASES']}
    except Exception:
        known = {}

    def hint(kind, name):
        # same species with the charge written the PhreeqC way first: Ca++ -> Ca+2, Ca2+ -> Ca+2, CO32- -> CO3-2
        swapped = re.match(r'^(.*\D)(\d)([+-])$', name)
        same = [c for c in dict.fromkeys([extractDB._canonical_charge(name)]
                                         + ([f"{swapped[1]}{swapped[3]}{swapped[2]}"] if swapped else []))
                if c != name and c in known.get(kind, [])]
        close = same + [c for c in difflib.get_close_matches(name, known.get(kind, []), n=3, cutoff=0.6) if c not in same]
        return f"{name} (did you mean {' or '.join(close[:3])} ?)" if close else name

    what = {'species': "species (SOLUTION_SPECIES, EXCHANGE_SPECIES or SURFACE_SPECIES, read with -molalities)",
            'master species': "totals (SOLUTION_MASTER_SPECIES, read with -totals)",
            'phase': "phases (PHASES, read with -equilibrium_phases)"}
    raise ValueError(f"PhreeqC does not know these coupled variables in {Path(centralDict['chemPath']).name} : it "
                     "would set them to 0 at every step\n"
                     + "\n".join(f"  {what[kind]} : {', '.join(hint(kind, n) for n in names)}"
                                 for kind, names in unknown.items())
                     + "\nGive the names of the database : the charge after the formula (Ca+2, not Ca2+ or Ca++), "
                     "a species and not an element (Cl-, not Cl), the case of the database.")


_databaseTexts = {}


def _databaseText(path):
    """Text of the database file at the absolute `path` (encoding detected), read once per file version."""
    try:
        from . import extractDB
    except ImportError:
        import extractDB
    key = (path, os.path.getmtime(path))
    if key not in _databaseTexts:
        with open(path, 'r', encoding=extractDB.databaseEncoding(path), errors='replace') as f:
            _databaseTexts[key] = f.read()
    return _databaseTexts[key]


def _loadDatabaseFile(solver, chemPath):
    """chemPath into an IPhreeqc instance. Its text is read here (encoding detected) and given to PhreeqC, which then
    opens no file: a path with accents (Données) is no problem, nor a latin-1 database. A database with INCLUDE$ is
    loaded from its path (the included files are found next to it). A load error (syntax) stops the run with the
    messages of PhreeqC, instead of 'No database is loaded' at the first step."""
    try:
        from . import extractDB
    except ImportError:
        import extractDB
    path = os.path.abspath(str(chemPath))
    text = _databaseText(path)
    if 'INCLUDE$' in text.upper():
        solver.load_database(extractDB.asciiPath(path))
    else:
        solver.load_database_string(text)
    if solver.phc_database_error_count:
        raise ValueError(f"PhreeqC could not read the database {path} ({solver.phc_database_error_count} error(s)) :\n"
                         + solver.get_error_string().strip())


def loadDatabase(solver, centralDict):
    """Loads chemPath, the KNOBS and the kineticRates into an IPhreeqc instance."""
    _loadDatabaseFile(solver, centralDict['chemPath'])
    knobs = knobsText(centralDict.get('knobs'))
    if knobs:
        try:
            solver.run_string(knobs)
        except Exception as e:
            raise ValueError(f"knobs : PhreeqC could not read the KNOBS block :\n{knobs}\n{e}") from None
    rates = centralDict.get('kineticRates')
    if not rates:
        return
    try:
        solver.run_string(_ratesScript(rates))
    except Exception as e:
        raise ValueError(f"kineticRates : PhreeqC could not read the RATES block :\n{e}") from None


def _ratesScript(rates):
    """Script of the kineticRates (content of a RATES block, with or without its RATES line) run by loadDatabase."""
    text = str(rates).strip('\n')
    words = text.split()
    if not words or words[0].upper() != 'RATES':
        text = "RATES\n" + text
    return text + "\nEND\n"


_KINETICS_OPTIONS = {'formula', 'm', 'm0', 'parms', 'tol', 'steps', 'step_divide', 'runge_kutta', 'bad_step_max',
                     'cvode', 'cvode_steps', 'cvode_order'}


def kineticsBlocks(text):
    """KINETICS text -> [[rate name or None, lines], ...]."""
    blocks = [[None, []]]
    for physical in str(text).split('\n'):
        for line in physical.split('#', 1)[0].split(';'):
            words = line.split()
            if words and not words[0].startswith('-') and words[0].lower() not in _KINETICS_OPTIONS:
                blocks.append([words[0], [line]])
            else:
                blocks[-1][1].append(line)
    return blocks


def kineticsRates(blocks, species):
    """kineticsSpecies -> {species: rate name}; ValueError if a species has no rate block."""
    names = [name for name, _ in blocks if name is not None]
    rates, missing = {}, []
    for spc in species:
        if spc in names:
            rates[spc] = spc
            continue
        same = [name for name in names if name.lower() == spc.lower()]
        if len(same) == 1:
            rates[spc] = same[0]
        else:
            missing.append(spc)
    if missing:
        raise ValueError(f"kineticsSpecies {missing} : no rate of that name in the kinetics text (rates : {names}) ; "
                         "each kinetics species must be the name of a rate block, which receives its -m0")
    return rates


def kineticsText(blocks, amounts, initial=None):
    """
    KINETICS text with the amount of each rate of amounts {rate name: moles} written at the end of its block. With its
    initial amount in initial {rate name: moles} (> 0): '-m0 initial' and '-m amount', so that the rate laws keep
    M / M0 (surface area) from one step to the next ; otherwise '-m0 amount' (M = M0 at the start of the step).
    """
    lines = []
    for name, body in blocks:
        lines += body
        if name in amounts:
            m0 = (initial or {}).get(name)
            if m0 is not None and m0 > 0:
                lines += [f"-m0 {m0}", f"-m {amounts[name]}"]
            else:
                lines.append(f"-m0 {amounts[name]}")
    return "\n".join(lines)


def speciesToNode(v, ranges, labels):
    """Counter-charge species of a node, from speciesChargeGeometry."""
    ranges = np.asarray(ranges, dtype=float)  
    
    results = []
    
    mask = (v >= ranges[:, 0]) & (v <= ranges[:, 1])
    
    if np.any(mask):
        idx = np.where(mask)[0][0]
        matched_label =  labels[idx]
    else:
        matched_label = None
    
    results.append(matched_label)

    return results


def chargeCandidates(label):
    """Counter-charge candidates of a node, as a list."""
    candidates = label[0] if label else None
    if candidates is None:
        return []
    return [candidates] if isinstance(candidates, str) else list(candidates)


def chargeSpecies(*candidates):
    """Species carrying the charge in SOLUTION: first candidate of the first non-empty argument."""
    for candidate in candidates:
        if isinstance(candidate, str):
            return candidate
        if candidate:
            return candidate[0]
    return None


def chargeTarget(centralDict, spc):
    """
    Counter-charge species (speciesCharge, speciesChargeGeometry) -> what receives 'charge' in SOLUTION : 'pH' for pH,
    H or H+ ; a solution master species (Cl, C(+4)) as it is ; a species by its master species ('Cl-' -> 'Cl',
    'CO3-2' -> 'C(+4)') ; None when it is none of these.
    """
    if spc is None:
        return None
    if spc in ('ph', 'pH', 'H', 'H+'):
        return 'pH'
    solution = centralDict['primarySpecies']['solution']
    if spc in solution:
        return spc
    masters = [p for p in centralDict['primToSecSpecies'].get(spc, {}) if p not in ('H', 'O', 'H2O')]
    return masters[0] if len(masters) == 1 and masters[0] in solution else None


def stoichMatrix(centralDict, species):
    """Stoichiometric matrix species -> primary species, cached in centralDict['stoichPhreeqC']."""
    prims = centralDict['primarySpecies']['total']
    cached = centralDict.get('stoichPhreeqC')
    if (cached is not None and cached.index.tolist() == frames.labels(species)
            and cached.columns.tolist() == frames.labels(prims)):
        return cached
    phantom = centralDict['primarySpecies']['primarySpeciesPhantom']
    stoich = pd.DataFrame(0.0, index=list(species), columns=list(prims))
    for comp in species:
        for prim, coeff in centralDict['primToSecSpecies'][comp].items():
            if prim in phantom and prim not in stoich.columns:
                continue
            if prim not in stoich.columns:
                raise ValueError(f"PhreeqC : {comp} decomposes into {prim}, which is not a primary species "
                                 f"({centralDict['primToSecSpecies'][comp]})")
            stoich.loc[comp, prim] = coeff
    centralDict['stoichPhreeqC'] = stoich
    return stoich


def toPrimarySpecies(tab, stoich):
    """tab (nodes x species) -> primary species, summed species by species."""
    return pd.DataFrame(primaryColumns(tab, stoich), index=tab.index)


def primaryColumns(tab, stoich):
    """Columns of toPrimarySpecies as {primary species: array}."""
    values = tab.to_numpy()
    coeffs = stoich.to_numpy()
    fast = _primaryColumnsFast(tab, stoich, values, coeffs)
    if fast is not None:
        return fast
    out = np.zeros((coeffs.shape[1], len(tab)))
    for j, comp in enumerate(stoich.index):
        conc = values[:, tab.columns.get_loc(comp)]
        for k in np.flatnonzero(coeffs[j]):
            out[k] += coeffs[j, k] * conc
    touched = coeffs.any(axis=0)
    return {prim: out[k] if touched[k] else np.zeros(len(tab), dtype=np.int64)
            for k, prim in enumerate(stoich.columns.tolist())}


def primaryColumnsOf(frame, names, stoich):
    """primaryColumns(frame[names], stoich), stoich being stoichMatrix(centralDict, names): for unique float64 columns
    of frame holding every name, read from frame itself (the columns of the names, the same values; frame[names] would
    build a frame of them, ~0.25 ms), otherwise from frame[names]."""
    if (frame.columns.is_unique and frames.allFloat64(frame) and stoich.index.tolist() == frames.labels(names)
            and np.all(frames.indexer(frame.columns, names) >= 0)):
        found = _primaryColumnsFast(frame, stoich, frame.to_numpy(), stoich.to_numpy())
        if found is not None:
            return found
    return primaryColumns(frame[names], stoich)


def _primaryColumnsFast(tab, stoich, values, coeffs):
    """primaryColumns for unique float64 columns of tab holding every species of stoich (None otherwise).

    The loop adds to out[k], from 0.0, the products coeffs[j, k] * column j of every species j with a nonzero
    coefficient, j ascending. Here every product is computed at once (the same multiplications), ordered by primary
    species then j (np.nonzero of coeffs.T), and the additions made step by step: at step s, the s-th product of every
    primary species having more than s terms is added to its sum (primary species sorted by number of terms, so these
    are the first rows). Each sum receives the same additions in the same order, from the same 0.0: the same values,
    in a few numpy calls per step instead of two per coefficient."""
    if not (values.dtype == np.float64 and coeffs.dtype == np.float64 and values.ndim == 2 and coeffs.ndim == 2
            and tab.columns.is_unique and frames.allFloat64(tab)):
        return None
    positions = frames.indexer(tab.columns, stoich.index)
    if not np.all(positions >= 0):
        return None                                 # a missing species: the loop raises its KeyError
    n = len(tab)
    if _primarySums is not None and n and coeffs.shape[0] and coeffs.shape[1]:
        # the same sums in C (_phreeqcRow.primary: the products and additions of the loop, in its order, from 0.0),
        # without the numpy calls of each step below
        sums = np.zeros((coeffs.shape[1], n))
        try:
            touched = _primarySums(values, np.asarray(positions, dtype=np.int64), coeffs, sums)
        except (TypeError, ValueError, BufferError):
            touched = None                  # an array the C code does not take: the numpy steps below
        if touched is not None:
            return {prim: sums[k] if touched[k] else np.zeros(n, dtype=np.int64)
                    for k, prim in enumerate(stoich.columns.tolist())}
    prims, species = np.nonzero(coeffs.T)           # pairs (k, j): k ascending, then j ascending
    products = coeffs[species, prims][:, None] * values[:, positions[species]].T
    lengths = np.bincount(prims, minlength=coeffs.shape[1])    # terms of each primary species (> 0: coeffs.any)
    starts = np.zeros(len(lengths), dtype=np.intp)
    np.cumsum(lengths[:-1], out=starts[1:])
    order = np.argsort(-lengths, kind='stable')
    sortedLengths = lengths[order]
    count = int(np.count_nonzero(sortedLengths))
    sums = np.zeros((count, n))
    first = starts[order[:count]]
    for step in range(int(sortedLengths[0]) if count else 0):
        rows = int(np.count_nonzero(sortedLengths > step))
        sums[:rows] += products[first[:rows] + step]
    row = {k: i for i, k in enumerate(order[:count].tolist())}
    out = {}
    for k, prim in enumerate(stoich.columns.tolist()):
        i = row.get(k)
        out[prim] = sums[i] if i is not None else np.zeros(n, dtype=np.int64)
    return out


def selectedOutputFrame(results, block=None):
    """(header, row) of every node -> one DataFrame. A row is a list of values, or the position of the row in `block`
    (rows of doubles read into it by selectedOutputRow): when every row is there, the frame is built from the block
    (the array np.array(rows) built, same values, same layout); otherwise from the rows as lists (block rows as
    their tolist(), the lists selectedOutputRow returned before)."""
    header = results[0][0]
    if block is not None and all(h is header and type(row) is int for h, row in results):
        values = block[:len(results)]
        if [row for _, row in results] == list(range(len(results))) and not np.isnan(values).any():
            return pd.DataFrame(values, columns=frames.columnIndex(header))
    if block is not None:
        results = [(h, block[row].tolist() if type(row) is int else row) for h, row in results]
    if all(h is header for h, _ in results):
        rows = [row for _, row in results]
        if all(type(value) is float for row in rows for value in row):
            values = np.array(rows, dtype=float)
            if not np.isnan(values).any():
                return pd.DataFrame(values, columns=frames.columnIndex(header))
        table = pd.DataFrame(rows, columns=header)
        if not table.isna().to_numpy().any():
            return table
    return pd.concat([pd.DataFrame([row], columns=h) for h, row in results], ignore_index=True)


def _floatFrame(df):
    """Whether df has unique column names and float64 columns only."""
    return df.columns.is_unique and frames.allFloat64(df)


def _cellValues(array):
    """float64 / int64 column -> list of Python numbers (same text in the scripts, faster to read and format)."""
    if isinstance(array, np.ndarray) and array.ndim == 1 and array.dtype in (np.float64, np.int64):
        return array.tolist()
    return array


class _LazyColumns(dict):
    """{column: _cellValues(array)}, each column converted at its first use (most columns of a call are never read by
    the scripts): the same lists, the same KeyError for a missing column."""

    def __init__(self, arrays):
        super().__init__()
        self._arrays = arrays

    def __missing__(self, column):
        values = self[column] = _cellValues(self._arrays[column])
        return values

    def get(self, column, default=None):
        return self[column] if column in self._arrays else default

    def __contains__(self, column):
        return column in self._arrays


class _Cells:
    """cells[index, column]: the DataFrame.loc value (Python number for float64 / int64 columns), read from the columns.
    cache: lines of the node scripts built once per call from these columns (speciationPhreeqC)."""
    def __init__(self, df, columns=None):
        self.cache = {}
        self.rows = {label: i for i, label in enumerate(df.index)}
        self.length = len(df.index)
        if columns is not None:
            arrays = columns
        else:
            dtypes = set(_dtypes(df))
            if len(dtypes) == 1 and isinstance(next(iter(dtypes)), np.dtype) and df.columns.is_unique:
                values = df.to_numpy()
                arrays = {c: values[:, j] for j, c in enumerate(df.columns.tolist())}
            else:
                arrays = {c: df[c].to_numpy() for c in df.columns.tolist()}
        self.values = _LazyColumns(arrays)
        # float64 columns (their values are Python floats in self.values): their texts can be written at once
        self.arrays = {c: v for c, v in arrays.items()
                       if isinstance(v, np.ndarray) and v.ndim == 1 and v.dtype == np.float64}

    def texts(self, column):
        """[f"{v}" for v in the values of the column] for a float64 column (_floatText: the same text, faster), None
        for another column or without _floatText."""
        array = self.arrays.get(column) if _floatText is not None else None
        return _floatText.texts(array) if array is not None else None

    def __contains__(self, column):
        return column in self.values

    def __getitem__(self, key):
        index, column = key
        return self.values[column][self.rows[index]]


def _nodeLines(cells, key, names, pieces):
    """Lines of `names` in the node scripts of a call: [(name, column, texts, head, end)], built once per cells and key.
    column: the values of the name in cells (the list cells[index, name] reads, None without column), texts: their
    texts f"{value}" (cells.texts, None if not written at once), head and end: the constant parts of the line,
    pieces(name), around the amount of the node."""
    found = cells.cache.get(key)
    if found is None:
        found = cells.cache[key] = [(name, cells.values.get(name), cells.texts(name)) + tuple(pieces(name))
                                    for name in names]
    return found


def _nodeBlocks(cells, key, names, pieces, cut, skipMissing=False):
    """The lines of `names` (_nodeLines) of every row of cells at once, built once per call: [text of row 0, ...], the
    text of a row being, name after name, head + f"{amount}" + end where its amount > cut ('' elsewhere), what the
    node loop appends line by line. None when a name has no float64 column with its texts or no constant pieces: the
    lines are then written node by node. skipMissing: a name without column is left out (EXCHANGE), as in the loop."""
    found = cells.cache.get(('blocks', key), False)
    if found is not False:
        return found
    columns = []
    for name, column, texts, head, end in _nodeLines(cells, key, names, pieces):
        if column is None and skipMissing:
            continue
        if column is None or texts is None or head is None or end is None or type(column) is not list:
            columns = None
            break
        columns.append([head + text + end if amount > cut else '' for amount, text in zip(column, texts)])
    if columns is None:
        blocks = None
    elif columns:
        blocks = [''.join(parts) for parts in zip(*columns)]
    else:
        blocks = [''] * cells.length
    cells.cache[('blocks', key)] = blocks
    return blocks


def _sectionBlocks(direct, cells, key, names, pieces, cut, skipMissing=False):
    """_nodeBlocks, None when the script is written direct (amount by amount, see _sectionLines)."""
    return None if direct else _nodeBlocks(cells, key, names, pieces, cut, skipMissing)


def _sectionLines(direct, cells, key, names, pieces):
    """_nodeLines; direct: the same lines without the texts of the columns (None: each amount written by the f-string of
    the line, the text _floatText gives) and without building them for the whole call (cells.cache left as it is)."""
    if not direct:
        return _nodeLines(cells, key, names, pieces)
    return [(name, cells.values.get(name), None) + tuple(pieces(name)) for name in names]


def _valueReader(phreeqc):
    """(GetSelectedOutputValue without ctypes argument conversion, its VAR, pointer to the VAR, address of the function)
    of the IPhreeqc instance."""
    reader = getattr(phreeqc, '_pycteValueReader', None)
    if reader is None:
        getValue = phreeqc._get_value
        address = ctypes.cast(getValue, ctypes.c_void_p).value
        reader = (type(getValue)(address), phreeqc.var, ctypes.byref(phreeqc.var), address)
        phreeqc._pycteValueReader = reader
    return reader


def runNode(phreeqc, script, tail, body=False):
    """run_string of a node script ending with `tail` (USER_PUNCH / SELECTED_OUTPUT ... END): the IPhreeqc instance keeps
    these definitions, so they are sent again only when they change or after a failed run. body: `script` is the node
    script without its tail (the same text is run: body + tail, or body + END when the instance has the tail)."""
    if body:
        text = script + "\nEND\n" if getattr(phreeqc, '_pycteTail', None) == tail else script + tail
    elif getattr(phreeqc, '_pycteTail', None) == tail and script.endswith(tail):
        text = script[:len(script) - len(tail)] + "\nEND\n"
    else:
        text = script
    phreeqc._pycteTail = None
    phreeqc.run_string(text)
    phreeqc._pycteTail = tail


def _rowBuffers(phreeqc, ncols):
    """ncols VAR structures of the instance (one per column), a pointer to each and a numpy view of their type and
    double value: a row is read into them, then converted at once instead of value by value. Last: address of the
    first VAR and size of a VAR (for _phreeqcRow)."""
    buffers = getattr(phreeqc, '_pycteRowBuffers', None)
    if buffers is None or buffers[0] != ncols:
        varType = type(phreeqc.var)
        size = ctypes.sizeof(varType)
        array = (varType * ncols)()
        pointers = [ctypes.byref(array, k * size) for k in range(ncols)]
        layout = np.dtype({'names': ['type', 'value'], 'formats': [np.intc, np.float64],
                           'offsets': [varType.type.offset, varType.value.offset], 'itemsize': size})
        view = np.frombuffer(memoryview(array).cast('B'), dtype=layout)
        allDoubles = np.full(ncols, 3, dtype=np.intc).tobytes()        # type of every VAR holding a double (TT_DOUBLE)
        buffers = (ncols, array, list(enumerate(pointers)), view['type'], view['value'], allDoubles,
                   ctypes.addressof(array), size)
        phreeqc._pycteRowBuffers = buffers
    return buffers


def selectedOutputRow(phreeqc, row, ncols, out=None):
    """Selected output row `row` of the IPhreeqc instance. Each value is read into its own VAR, then the row is
    converted at once when every value is a double (the usual case); otherwise, or on an error code, the row is read
    again value by value (selectedOutputValues: same values, same errors). The values are read by _phreeqcRow (the
    loop below, in C: ~0.07 ms less per node on CementClayInterface), or by the loop below without it.
    out: float64 array of ncols values receiving a row of doubles (None returned), instead of its list (the doubles
    copied as they are, no Python float made per value)."""
    reader = _valueReader(phreeqc)
    _, _, pointers, types, values, allDoubles, address, size = _rowBuffers(phreeqc, ncols)
    id_ = phreeqc.id_
    if _phreeqcRow is not None:
        if _phreeqcRow.read(reader[3], id_, row, ncols, address, size):
            return selectedOutputValues(phreeqc, row, ncols)
    else:
        getValue = reader[0]
        for col, pointer in pointers:
            if getValue(id_, row, col, pointer):
                return selectedOutputValues(phreeqc, row, ncols)
    if types.tobytes() == allDoubles:
        if out is not None:
            out[...] = values
            return None
        return values.tolist()
    return selectedOutputValues(phreeqc, row, ncols)


def selectedOutputValues(phreeqc, row, ncols):
    """Selected output row `row` of the IPhreeqc instance, value by value."""
    getValue, var, pointer, _ = _valueReader(phreeqc)
    id_, value = phreeqc.id_, var.value
    out = []
    for col in range(ncols):
        error_code = getValue(id_, row, col, pointer)
        if error_code != 0:
            phreeqc.raise_ipq_error(error_code)
        type_ = var.type
        if type_ == 3:
            val = value.double_value
        elif type_ == 2:
            val = value.long_value
        elif type_ == 4:
            val = value.string_value.decode('utf-8')
        elif type_ == 0:
            val = None
        if type_ == 1:
            phreeqc.raise_error(value.error_code)
        out.append(val)
    return out


class _ScriptBuilder:
    """A thread writing the script of the next node of speciationPhreeqC while the IPhreeqc instance runs the current
    one (run_string, a ctypes call, releases the GIL): a job is a function returning the script; its result, or the
    exception it raised, is given back in the order of the jobs. It runs Python code only (no call of IPhreeqc), while
    the thread that gave the job waits in run_string."""

    def __init__(self):
        self._jobs, self._done = queue.SimpleQueue(), queue.SimpleQueue()
        self.thread = threading.Thread(target=self._loop, name='pycte-phreeqc-scripts', daemon=True)
        self.thread.start()

    def _loop(self):
        while True:
            job = self._jobs.get()
            if job is None:
                return
            try:
                self._done.put((True, job()))
            except BaseException as error:          # given back: the script is then written again by the loop
                self._done.put((False, error))

    def submit(self, job):
        self._jobs.put(job)

    def result(self):
        """(True, script) or (False, the exception raised) of the oldest job not given back yet (waits for it); (False,
        None) if the thread has stopped without giving it back (the caller then writes the scripts itself)."""
        while True:
            try:
                return self._done.get(timeout=1.0)
            except queue.Empty:
                if not self.thread.is_alive():
                    return False, None

    def stop(self):
        self._jobs.put(None)


_builders = {}                      # thread ident -> its _ScriptBuilder (the main thread, the threads of the blocks)
_buildersLock = threading.Lock()
_AHEAD_NODES = 8                    # scripts written ahead per job of the _ScriptBuilder (0: every node script written
                                    # in the node loop, as before)


def _scriptBuilder():
    """_ScriptBuilder of the calling thread (made at its first use, kept from one call to the next)."""
    key = threading.get_ident()
    with _buildersLock:
        builder = _builders.get(key)
        if builder is None or not builder.thread.is_alive():
            builder = _builders[key] = _ScriptBuilder()
    return builder


def _stopScriptBuilders():
    """The threads of the _ScriptBuilders stopped (resetPhreeqC)."""
    with _buildersLock:
        for builder in _builders.values():
            builder.stop()
        _builders.clear()


def _kineticsSettings(centralDict):
    """(kinetic phases, KINETICS blocks, {species: rate name}) of speciationPhreeqC: from the settings only."""
    kineticPhases = ([s for s in (centralDict['kineticsSpecies'] or []) if s in centralDict['primarySpecies']['phases']]
                     if centralDict['kinetics'] else [])
    kinBlocks = kineticsBlocks(centralDict['kinetics']) if centralDict['kinetics'] and centralDict['kineticsSpecies'] else None
    kinRates = kineticsRates(kinBlocks, centralDict['kineticsSpecies']) if kinBlocks else {}
    return kineticPhases, kinBlocks, kinRates


def _solutionSettings(primary):
    """(solution primary species written in SOLUTION, the same as a set) of speciationPhreeqC."""
    excludedSolution = set(primary['primarySpeciesPhantom'] + ['H', 'O', 'H2O'])
    solutionPrim = [s for s in primary['solution'] if s not in excludedSolution]
    return solutionPrim, set(solutionPrim)


def _equilibriumSettings(centralDict, phasesPrim, kineticPhases):
    """(EQUILIBRIUM_PHASES header or None, equilibrium phases) of speciationPhreeqC: from the settings only."""
    fixed = fixedPhaseSpecs(centralDict.get('fixedPhases'))
    equilibriumHeader = None
    if centralDict['fixpH'] or fixed or phasesPrim:
        equilibriumHeader = "\nEQUILIBRIUM_PHASES 1\n"
        if centralDict['fixpH'] and 'Fix_ph' not in [f[0] for f in fixed]:
            equilibriumHeader += f"Fix_ph {centralDict['fixpH']}\n"
        # pseudo-phases of fixedPhases, rewritten at every call (never saved) : the reservoir of reactant is
        # the same at every step and the target SI is held, as Fix_H+ / Fix_pe in a PhreeqC TRANSPORT
        for name, si, reactant, amount, _ in fixed:
            equilibriumHeader += f"{name} {si} {reactant + ' ' if reactant else ''}{amount}\n"
    equilibriumPhases = [p for p in phasesPrim if p not in kineticPhases]
    return equilibriumHeader, equilibriumPhases


def _tailText(centralDict, crossSpc, mctTotals, phasesPrim, kineticPhases, kineticsRan):
    """USER_PUNCH and SELECTED_OUTPUT ending the node scripts of speciationPhreeqC: from the settings only (and whether
    the kinetics run at this call)."""
    tail = ""
    if crossSpc:
        punchNames = list(crossSpc['output'])
        punchCmds = [crossDependenciesManager.PUNCH.get(crossSpc['quantity'][n], crossSpc['quantity'][n]) for n in punchNames]
        if mctTotals:
            punchNames += ['H(mol/kgw)', 'O(mol/kgw)']
            punchCmds += ['TOT("H")', 'TOT("O")']
        tail += "\nUSER_PUNCH\n\t-headings"
        for val in punchNames:
            tail += f"\t{val}"
        tail += '\n'
        for it, val in enumerate(punchCmds, start = 1):
            tail += f'\t{it} PUNCH {val} \n'

    tail += "\nSELECTED_OUTPUT\n-reset false\n"
    totals, molalities, _ = selectedOutputNames(centralDict)
    if centralDict.get('molesStorage') and not centralDict['userVarBool'].get('water'):
        tail += "-water true\n"
    if mctTotals:
        tail += "-totals" + "".join(f"\t{t}" for t in totals)
        if molalities: tail += "\n-molalities" + "".join(f"\t{spc}" for spc in molalities)
    else:
        tail += "-molalities" + "".join(f"\t{spc}" for spc in molalities)

    if phasesPrim:
        tail += "\n-equilibrium_phases"
        for phase in phasesPrim: tail +=f"\t{phase}"
    tail += "\n"
    kineticOutput = [s for s in kineticPhases if s not in (centralDict['userVarList'].get('kinetic_reactants') or [])]
    if kineticsRan and kineticOutput:
        tail += "-kinetic_reactants" + "".join(f"\t{s}" for s in kineticOutput) + "\n"
    for val in centralDict['userVarBool'].keys():
        if centralDict['userVarBool'][val]:
            tail += f"-{val} True\n"
    for val in centralDict['userVarList'].keys():
        if centralDict['userVarList'][val]:
            tail += f"\n-{val}"
            for lst in centralDict['userVarList'][val]:
                tail += f"\t{lst}"
    tail += "\nEND\n"
    return tail


def speciationPhreeqC(centralDict, commMtrxPart, beforeTrsptMtrx, block=None):
    """Speciation of the nodes of commMtrxPart, on the IPhreeqc instance of the process (global phreeqc), or on that
    of `block` (a block of nodes computed in a thread, _Block), created and loaded here at its first call."""
    global phreeqc
    ref = time.perf_counter()
    
    # instance loaded during the set-up (takeIPhreeqc), otherwise made and loaded here
    if block is None:
        if phreeqc is None:
            phreeqc = takeIPhreeqc(centralDict)
            if phreeqc is None:
                phreeqc = newIPhreeqc()
                loadDatabase(phreeqc, centralDict)
        solver, seen = phreeqc, None
    else:
        if block.solver is None:
            block.solver = takeIPhreeqc(centralDict)
            if block.solver is None:
                block.solver = newIPhreeqc()
                loadDatabase(block.solver, centralDict)
        solver, seen = block.solver, block.seen
    
    initWorker = time.perf_counter() - ref

    mctTotals = centralDict.get('mctTotals') if centralDict.get('MultiCompoundTransport') else None
    # values derived from the settings only, the same at every call: built once per run (frames.constant), only read
    kineticPhases, kinBlocks, kinRates = frames.constant(
        'phreeqcKinetics', (centralDict['kinetics'], centralDict['kineticsSpecies'],
                            centralDict['primarySpecies']['phases']), lambda: _kineticsSettings(centralDict))
    kineticsRan = bool(centralDict['kinetics']) and centralDict['dtStep'] > 0

    # initial amounts of the kinetic reactants (engine.py), the M0 of the rate laws ; nodes it does not know (another
    # grid) keep M0 = current amount
    kinInitial = centralDict.get('kineticsM0')

    def kineticsScript(amountCells):
        if not kinBlocks:
            return f"{centralDict['kinetics']}\n"
        initial = ({kinRates[k]: kinInitial.at[index, k] for k in centralDict['kineticsSpecies'] if k in kinInitial}
                   if kinInitial is not None and index in kinInitial.index else None)
        return kineticsText(kinBlocks, {kinRates[k]: amountCells[index, k] for k in centralDict['kineticsSpecies']},
                            initial) + "\n"

    crossSpc = centralDict['crossDependencies']['speciation']
    if not crossSpc['total']:
        crossSpc = None
    speciationBlock = centralDict['crossDependencies']['speciation']
    cmdToNameInput = frames.constant('phreeqcInputNames', (speciationBlock,),
                                     lambda: crossDependenciesManager.byQuantity(speciationBlock, 'input'))
    hasDebye = "DebyeLength" in cmdToNameInput
    peName = cmdToNameInput.get('-la("e-")')
    ehName = cmdToNameInput.get('Eh')
    hasPe = bool(peName or ehName)
    phName = cmdToNameInput.get('-la("H+")')
    hasPH = bool(phName)
    cbName = cmdToNameInput.get('CHARGE_BALANCE')
    hasCB = bool(cbName)
    tkName = cmdToNameInput.get('tk')
    pressureName = cmdToNameInput.get('pressure')

    def temperature(cells):
        '''Temperature of the node in °C, the unit of PhreeqC: tk column or tempDefault, both in K.'''
        if tkName:
            return cells[index, tkName] - 273.15
        return centralDict['tempDefault'] - 273.15

    def pe(cells):
        '''pe of the node: pe column, or Eh converted.'''
        if peName:
            return cells[index, peName]
        return cells[index, ehName] / (crossDependenciesManager.EH_PER_PE_K * (temperature(cells) + 273.15))

    def cellText(cells, name):
        '''cells[index, name] in a script: its text f"{value}" when the float64 column is written at once (cells.texts),
        else the value itself (the same text in the f-string)'''
        texts = cells.cache.get(('text', name), False)
        if texts is False:
            texts = cells.cache[('text', name)] = cells.texts(name)
        row = cells.rows.get(index) if texts is not None else None
        return texts[row] if row is not None else cells[index, name]

    def peText(cells):
        '''pe(cells) in a script (its text for a pe column, see cellText)'''
        return cellText(cells, peName) if peName else pe(cells)

    def pressure(cells):
        '''-pressure line (atm) of the node, '' without pressure column.'''
        return f"-pressure {cells[index, pressureName] / crossDependenciesManager.ATM}\n" if pressureName else ""

    primary = centralDict['primarySpecies']
    phasesPrim, surfacesPrim, exchangesPrim = primary['phases'], primary['surface'], primary['exchange']
    solutionPrim, solutionSet = frames.constant(
        'phreeqcSolution', (primary['primarySpeciesPhantom'], primary['solution']), lambda: _solutionSettings(primary))
    cutoffs = centralDict['cutoffs']
    cutSolution, cutPhases, cutSurface, cutExchange = (cutoffs['solution'], cutoffs['phases'], cutoffs['surface'],
                                                       cutoffs['exchange'])
    preliminary = centralDict['preliminaryEquilibrium']
    # solMod: SURFACE is rebuilt from its numbers of sites (master form) ; the charge of the surface species, minus that
    # of their sites in master form, goes back to the solution with the coupled charge balance (as in rmphreeqc.py)
    surfaceColumns = {}
    if centralDict['solMod'] and hasCB and surfacesPrim:
        try:
            from . import rmphreeqc
        except ImportError:
            import rmphreeqc
        surfaceColumns = rmphreeqc._surfaceColumns(centralDict, centralDict['systemSpecies'])[1]

    def surfaceCharge(cells, primCells):
        '''charge of the surface species of the node minus that of their sites, where SURFACE defines the sites'''
        charge = 0.0
        for s, (sites, z, siteZ) in surfaceColumns.items():
            amount = cells[index, s]
            charge += amount * z - (amount * siteZ if all(primCells[index, m] > cutSurface for m in sites) else 0.0)
        return charge
    molesWater = centralDict.get('molesStorage') and centralDict.get('cellWaterMass') is not None
    # water coupled: an H2O column, or the H and O totals (MultiCompoundTransport), which already hold the water
    waterCoupled = 'H2O' in commMtrxPart.columns or {'H', 'O'} <= set(commMtrxPart.columns)

    # preliminaryEquilibrium: the kinetics run in the second simulation only (with the solution after transport)
    kineticsBefore = (bool(centralDict['kinetics']) and not centralDict['solMod'] and not preliminary
                      and centralDict['dtStep'] > 0)
    kineticsAfter = bool(centralDict['kinetics']) and centralDict['dtStep'] > 0
    stepText = ""
    if kineticsAfter:
        # in seconds: PhreeqC does not know every timeUnit ('-step 2 a' would be 2 s)
        stepText = f'\n-step {timeUnits.toSeconds(centralDict["dtStep"], centralDict["timeUnit"])}\n'
        if centralDict['step_divide']:
            stepText += f"-step_divide {centralDict['step_divide']}\n"

    equilibriumHeader, equilibriumPhases = frames.constant(
        'phreeqcEquilibrium', (centralDict['fixpH'], centralDict.get('fixedPhases'), phasesPrim, kineticPhases),
        lambda: _equilibriumSettings(centralDict, phasesPrim, kineticPhases))
    SI, reversibility = centralDict['SI'], centralDict['mineralReversibility']

    # constant parts (head, end) of the line of a name in the node scripts, the amount between them: the same text as
    # the f-strings of each line (each field formatted on its own); (None, None): line written as before
    def phasePieces(phase):
        '''f"\t{phase} {SI[phase]} {amount}  {reversibility[phase]}\n", for SI and mineralReversibility given as dicts
        holding the phase (no lookup that could fail or change them before the amount is known)'''
        try:
            if type(SI) is dict and type(reversibility) is dict and phase in SI and phase in reversibility:
                return f"\t{phase} {SI[phase]} ", f"  {reversibility[phase]}\n"
        except TypeError:
            pass
        return None, None

    def surfacePieces(surface):
        '''f"\t{surface} {amount} 100 1 \n"'''
        return f"\t{surface} ", " 100 1 \n"

    def linePieces(name):
        '''f"\t{name} {amount}\n"'''
        return f"\t{name} ", "\n"

    # preliminaryEquilibrium: the second simulation (solution after transport) uses the reactants as the first one defines
    # them (equilibrium phases with their amounts, exchanger and surface equilibrated with the solution before
    # transport). The batch reaction of the first simulation is not kept: its mass transfer would reach the reactants
    # but not the solution after transport (mass created when the state before transport is not at equilibrium)
    useText = ""
    if equilibriumHeader is not None: useText += "use equilibrium_phases 1\n"
    if surfacesPrim: useText += "use surface 1\n"
    if exchangesPrim: useText += "use exchange 1\n"

    tail = frames.constant(
        'phreeqcTail', (crossSpc, mctTotals, centralDict.get('MultiCompoundTransport'), centralDict['systemSpecies'],
                        phasesPrim, centralDict.get('molesStorage'), centralDict['userVarBool'],
                        centralDict['userVarList'], kineticPhases, kineticsRan),
        lambda: _tailText(centralDict, crossSpc, mctTotals, phasesPrim, kineticPhases, kineticsRan))

    def nodeSpeciation(commMtrxPart,commMtrx_primSpecies,spcChargeDefaut,spcChargeGeom,speciationCharge,solMod,currentPrimSpecies,
                       withTail=True, direct=False):
        # direct: the lines of the sections written amount by amount (_sectionLines, the same text), the texts of the
        # whole columns and the blocks of every node not built here (the first node of a call: the _ScriptBuilder
        # builds them while IPhreeqc runs it)
        # species carrying the charge balance: first candidate of the node (speciesChargeGeometry), else of speciesCharge,
        # as written in SOLUTION ('pH' or a master species, see chargeTarget)
        chargeSpc = chargeTarget(centralDict, chargeSpecies(spcChargeGeom, spcChargeDefaut))
        chargeOnPH = speciationCharge and chargeSpc == 'pH'

        def pHText(cells):
            '''pH line of SOLUTION, 'charge' on that line (PhreeqC syntax: pH 7 charge), from pHdefault when the pH is
            not coupled'''
            if hasPH:
                return f"\tpH {cellText(cells, phName)}" + ("\tcharge\n" if chargeOnPH else "\n")
            return f"\tpH {centralDict['pHdefault']}\tcharge\n" if chargeOnPH else ""
        scriptMain =f"# node n°{index}\n"     # a comment: bare text before the first keyword is 'Unknown input' to PhreeqC
        # molesStorage: initial guess set to the water mass computed at the previous step (kg); SOLUTION_MODIFY recomputes it
        # from the O balance, the guess is only the starting point of the iterations
        if molesWater:
            water = centralDict['cellWaterMass'][index]
        elif centralDict['water']:
            water = centralDict['water'][index]
        else: water = 1

        if kineticsBefore:
            scriptMain += "\nKINETICS 1\n"
            scriptMain += kineticsScript(commMtrx_primSpecies)
            scriptMain += stepText

        # amounts of the node read from the columns of the cells (cells[index, name] as before when the column or the
        # node is not there: same value, same error), with their text f"{amount}" written at once for the column
        # (_nodeLines), lines with their constant parts built once per call; the lines of a whole section, for every
        # node, built once per call when each name has its float64 column (_nodeBlocks: the same text)
        primRow = commMtrx_primSpecies.rows.get(index)
        if equilibriumHeader is not None:
            scriptMain += equilibriumHeader
            blocks = (_sectionBlocks(direct, commMtrx_primSpecies, 'phases', equilibriumPhases, phasePieces, cutPhases)
                      if primRow is not None else None)
            if blocks is not None:
                scriptMain += blocks[primRow]
            else:
                for phase, column, texts, head, end in _sectionLines(direct, commMtrx_primSpecies, 'phases', equilibriumPhases,
                                                                  phasePieces):
                    if column is not None and primRow is not None:
                        amount = column[primRow]
                        text = texts[primRow] if texts is not None else amount
                    else:
                        amount = text = commMtrx_primSpecies[index,phase]
                    if amount > cutPhases:
                        scriptMain += (f"{head}{text}{end}" if head is not None else
                                       f"\t{phase} {SI[phase]} {amount}  {reversibility[phase]}\n")

        if surfacesPrim:
            scriptMain +="\nSURFACE 1\n"
            if preliminary: scriptMain += "equilibrate with solution 1\n"
            blocks = (_sectionBlocks(direct, commMtrx_primSpecies, 'surface', surfacesPrim, surfacePieces, cutSurface)
                      if primRow is not None else None)
            if blocks is not None:
                scriptMain += blocks[primRow]
                edl = blocks[primRow] != ''             # a surface line written (its head is not empty)
            else:
                edl = False
                for surface, column, texts, head, end in _sectionLines(direct, commMtrx_primSpecies, 'surface', surfacesPrim,
                                                                    surfacePieces):
                    if column is not None and primRow is not None:
                        amount = column[primRow]
                        text = texts[primRow] if texts is not None else amount
                    else:
                        amount = text = commMtrx_primSpecies[index,surface]
                    if amount > cutSurface :
                        scriptMain += f"{head}{text}{end}"
                        edl = True
            if edl and hasDebye and commMtrxPart[index,cmdToNameInput["DebyeLength"]] > 0:
                scriptMain +=f"-diffuse_layer {commMtrxPart[index,cmdToNameInput['DebyeLength']]}\n"
            else: scriptMain += "-no_edl\n"
            if centralDict['surfaceCounterIons']: scriptMain += "-only_counter_ions true\n"

        if exchangesPrim:
            scriptMain +="\nEXCHANGE 1\n"
            if preliminary: scriptMain += "equilibrate with solution 1\n"

            # a name without column (exchange not in commMtrx_primSpecies) is left out, as before
            blocks = (_sectionBlocks(direct, commMtrx_primSpecies, 'exchange', exchangesPrim, linePieces, cutExchange, True)
                      if primRow is not None else None)
            if blocks is not None:
                scriptMain += blocks[primRow]
            else:
                for exchange, column, texts, head, end in _sectionLines(direct, commMtrx_primSpecies, 'exchange', exchangesPrim,
                                                                     linePieces):
                    if column is None:
                        continue
                    if primRow is not None:
                        amount = column[primRow]
                        text = texts[primRow] if texts is not None else amount
                    else:
                        amount = text = commMtrx_primSpecies[index,exchange]
                    if amount > cutExchange:
                            scriptMain += f"{head}{text}{end}"

        if solMod:
            if preliminary:

                scriptMain +=f"\nSOLUTION 1 #node n°{index}\n-units mol/kgw\ntemp {temperature(beforeCells)}\n{pressure(beforeCells)}"
                if 'H2O' in commMtrx_primSpecies:
                    scriptMain += f"-water {beforeCells[index,'H2O']*18.015/1000}\n"
                else:

                    scriptMain += "-water 1\n"

                if crossSpc:
                    if hasPe:
                        scriptMain += f"\tpe {peText(beforeCells)}\n"
                    if hasPH:
                        scriptMain += f"\tpH {cellText(beforeCells, phName)}\n"

                else:
                    scriptMain += f"\tpH {centralDict['pHdefault']}"

                scriptMain += "\n"
                perKgw = (beforeCells[index,'H2O']*18.015/1000 if centralDict.get('molesStorage') and 'H2O' in commMtrx_primSpecies else 1)
                for species in solutionPrim:
                    if commMtrx_primSpecies[index,species] > cutSolution:
                        scriptMain += f"\n\t{species} {commMtrx_primSpecies[index,species] / perKgw}"

                scriptMain += "\n"

                if centralDict['supplementarySolution'] : scriptMain += centralDict['supplementarySolution'] + '\n'
                # no RUN_CELLS here: cell 1 keeps the reactants as defined above (see useText), RUN_CELLS below
                # reacts them with the solution after transport. SOLUTION 1 is defined again, empty, as without
                # preliminaryEquilibrium: SOLUTION_MODIFY 1 replaces totals of a new solution only (on the solution
                # above, '-totals C(+4)' would be added to its C(4): carbon doubled)
                scriptMain += f"\nend\nSOLUTION 1\n-units mmol/kgw\n-water {water}\nEND\n"
            else:
                currentPrimSpecies = commMtrx_primSpecies
                firstLine, rest = scriptMain.split("\n", 1)
                scriptMain = firstLine + f"""
    SOLUTION 1
    -units mmol/kgw
    # Na 1
    # Cl 1
    -water {water}
    END
        """ + rest

            if kineticsAfter:
                scriptMain += "\nKINETICS 1\n"
                scriptMain += kineticsScript(commMtrx_primSpecies)
                scriptMain += stepText

            scriptMain += "\nSOLUTION_MODIFY 1\n"

            # pH, pe, charge balance: the columns coupled by crossDependencies (any alias), each one on its own
            if hasPH:
                scriptMain += f"\npH {cellText(commMtrxPart, phName)}"
            if hasPe:
                scriptMain += f"\npe {peText(commMtrxPart)}"
            if hasPH or hasPe:
                scriptMain += "\n"

            # temperature of the node (the solution above is created at 25 °C)
            scriptMain += f"-temp {temperature(commMtrxPart)}\n{pressure(commMtrxPart)}"
            # H and O of the water: coupled (H2O column, or H and O totals, then H2O is 0 here), otherwise the water of
            # the node (kg, as in SOLUTION 1 above). Without it the totals held only the H and O of the solutes (2e-7
            # In fact, please phreeqcRM, that is simpler for pe and pH transport. 
            waterMoles = currentPrimSpecies[index,'H2O'] if waterCoupled else water * WATER_MOLES_PER_KG
            scriptMain += f"-total_h {waterMoles*2 + currentPrimSpecies[index,'H'] }\n"
            scriptMain += f"-total_o {waterMoles + currentPrimSpecies[index,'O'] }\n"

            if hasCB:
                if surfaceColumns:
                    cb = commMtrxPart[index,cbName]
                    cb += surfaceCharge(commMtrxPart, commMtrx_primSpecies)
                else:
                    cb = cellText(commMtrxPart, cbName)
                scriptMain += f"-cb {cb}\n-totals\n"
            else:
                scriptMain += "-cb 0\n-totals\n"

            currentRow = currentPrimSpecies.rows.get(index)
            blocks = (_sectionBlocks(direct, currentPrimSpecies, 'solution', solutionPrim, linePieces, cutSolution)
                      if currentRow is not None else None)
            if blocks is not None:
                scriptMain += blocks[currentRow]
            else:
                for species, column, texts, head, end in _sectionLines(direct, currentPrimSpecies, 'solution', solutionPrim,
                                                                    linePieces):
                    if column is not None and currentRow is not None:
                        amount = column[currentRow]
                        text = texts[currentRow] if texts is not None else amount
                    else:
                        amount = text = currentPrimSpecies[index,species]
                    if amount > cutSolution:
                        scriptMain += f"{head}{text}{end}"

            # RUN_CELLS reacts the reactants of cell 1 (USE is ignored), with preliminaryEquilibrium those defined above
            scriptMain +="\nRUN_CELLS\n-cells 1\n"

        else:
            scriptMain +=f"""\nSOLUTION 1 #node n°{index}
-units mol/kgw
-water {water}
-temp {temperature(beforeCells) if preliminary else temperature(commMtrxPart)}
{pressure(beforeCells) if preliminary else pressure(commMtrxPart)}"""
            if centralDict['densityPhreeqC']: scriptMain += f"-density {centralDict['densityPhreeqC']}\n"

            cells = beforeCells if preliminary else commMtrxPart
            if hasPe:
                scriptMain += f"\tpe {peText(cells)}\n"
            scriptMain += pHText(cells)
            scriptMain += "\n"
            # lines ended by "\n" (linePieces), but that of the charge species (speciationCharge), ended by "\tcharge\n"
            chargeLine = bool(speciationCharge) and chargeSpc in solutionSet
            blocks = (_sectionBlocks(direct, commMtrx_primSpecies, 'solution', solutionPrim, linePieces, cutSolution)
                      if primRow is not None and not chargeLine else None)
            if blocks is not None:
                scriptMain += blocks[primRow]
            else:
                for species, column, texts, head, _ in _sectionLines(direct, commMtrx_primSpecies, 'solution', solutionPrim,
                                                                  linePieces):
                    if column is not None and primRow is not None:
                        amount = column[primRow]
                        text = texts[primRow] if texts is not None else amount
                    else:
                        amount = text = commMtrx_primSpecies[index,species]
                    if amount > cutSolution:
                        scriptMain += f"{head}{text}"
                        scriptMain += "\tcharge\n" if speciationCharge and chargeSpc == species else "\n"

            if centralDict['supplementarySolution'] : scriptMain += centralDict['supplementarySolution'] + '\n'
            if preliminary:
                scriptMain += f"end\nSOLUTION 2\n-units mol/kgw\n-temp {temperature(commMtrxPart)}\n{pressure(commMtrxPart)}"
                scriptMain += pHText(commMtrxPart)
                if hasPe:
                    scriptMain += f"\tpe {peText(commMtrxPart)}\n"
                scriptMain += '\n'
                currentRow = currentPrimSpecies.rows.get(index)
                blocks = (_sectionBlocks(direct, currentPrimSpecies, 'solution', solutionPrim, linePieces, cutSolution)
                          if currentRow is not None and not chargeLine else None)
                if blocks is not None:
                    scriptMain += blocks[currentRow]
                else:
                    for species, column, texts, head, _ in _sectionLines(direct, currentPrimSpecies, 'solution', solutionPrim,
                                                                      linePieces):
                        if column is not None and currentRow is not None:
                            amount = column[currentRow]
                            text = texts[currentRow] if texts is not None else amount
                        else:
                            amount = text = currentPrimSpecies[index,species]
                        if amount > cutSolution:
                            scriptMain += f"{head}{text}"
                            scriptMain += "\tcharge\n" if speciationCharge and chargeSpc == species else "\n"

                scriptMain += useText
                if kineticsAfter:
                    scriptMain += "\nKINETICS 1\n"
                    scriptMain += kineticsScript(commMtrx_primSpecies)
                    scriptMain += stepText

        # withTail False: the script without its tail, run by runNode(..., body=True) (the same text is run)
        return scriptMain + tail if withTail else scriptMain

    warnings = 0
    scriptWarnings =""


    excluded = centralDict['crossDependencies']['speciation']['total']


    stoich = stoichMatrix(centralDict, [c for c in commMtrxPart.columns.tolist() if c not in excluded])

    if not beforeTrsptMtrx.empty :
        primSource, primColumns = beforeTrsptMtrx, primaryColumns(beforeTrsptMtrx, stoich)
        currentCells = _Cells(commMtrxPart, primaryColumns(commMtrxPart, stoich))
    else:
        currentCells = _Cells(_EMPTY_FRAME)
        primSource, primColumns = commMtrxPart, primaryColumns(commMtrxPart, stoich)
    primCells = _Cells(primSource, primColumns)
    speciationOut = outputManager.wanted(centralDict, 'speciation')
    commMtrx_primSpecies = pd.DataFrame(primColumns, index=primSource.index) if speciationOut else _EMPTY_FRAME

    pd.set_option('display.max_columns', None)
    partCells = _Cells(commMtrxPart)
    beforeCells = _Cells(beforeTrsptMtrx)
    calcTime = 0
    abort = False
    phreeqcOutput = _EMPTY_FRAME
    results = []
    header = None
    block, filled, blockAddress = None, 0, 0

    # columns of the selected output read after the node loop: speciation columns, crossDependencies, kinetic
    # reactants, water mass. At a step without speciation output, only the columns of these names are read from
    # IPhreeqc (_readColumns, not the d_<phase> of -equilibrium_phases ...: the frame of the selected output is not
    # written then), the others at every step. rowNames: the names of the columns of the rows read
    speciationColumns  = []
    phasesSet, excludedSet = set(centralDict['primarySpecies']['phases']), set(excluded)
    if mctTotals:
        totalsSet = set(mctTotals)
        speciationColumns += [spc if spc in phasesSet else f"{spc}(mol/kgw)" if spc in totalsSet else f"m_{spc}(mol/kgw)" for spc in centralDict['systemSpecies'] if spc not in excludedSet]
    elif centralDict['primarySpecies']['phases']: speciationColumns += [spc if spc in phasesSet else f"m_{spc}(mol/kgw)" for spc in centralDict['systemSpecies'] if spc not in excludedSet]
    else: speciationColumns += [f"m_{spc}(mol/kgw)" for spc in centralDict['systemSpecies'] if spc not in excludedSet]
    crossOut = centralDict['crossDependencies']['speciation']['output']
    needed = None
    if not speciationOut:
        needed = set(crossOut) | set(speciationColumns) | {f"k_{c}" for c in kineticPhases}
        if centralDict.get('molesStorage'):
            needed.add('mass_H2O')
    columns, rowNames, fullRow = None, None, None
    # speciesChargeGeometry = [(node range, counter-charge species), ...]: species of each range, in the order of the ranges
    geometrySpecies = [spc for _, spc in centralDict['speciesChargeGeometry'] or []]

    # the scripts of the next nodes (up to _AHEAD_NODES) are written by a thread (_ScriptBuilder) while IPhreeqc runs
    # the current node: the same function (nodeSpeciation, which reads the cells and the settings only) on the same
    # values, so the same text; every call of IPhreeqc stays in this thread, in the same order. The job sets `index`
    # (read by the helpers of nodeSpeciation) to the node it writes, this thread sets it back once the job is done; both
    # never run together (this thread is in run_string meanwhile, and waits for the job before going on). A script whose
    # writing raised is written again in the loop (the same error, at the same place), the next ones too
    nodes = list(commMtrxPart.index)

    def nextScripts(first, last):
        nonlocal index
        written = {}
        for position in range(first, last):
            index = nodes[position]
            try:
                if centralDict['chargeGeometryCells']: candidates = chargeCandidates(speciesToNode(index, centralDict['chargeGeometryCells'], geometrySpecies))
                else: candidates = []
                written[position] = nodeSpeciation(partCells, primCells, centralDict['speciesCharge'], candidates,
                                                   centralDict['speciationCharge'], centralDict['solMod'], currentCells,
                                                   withTail=False)
            except Exception:
                break
        return written

    builder = _scriptBuilder() if _AHEAD_NODES > 0 and len(nodes) > 1 else None
    ahead = {}                      # position -> script of the node, written by the builder
    for position, index in enumerate(nodes):
        if abort : break
        # counter-charge candidates of this node (speciesChargeGeometry); none: speciesCharge applies
        if centralDict['chargeGeometryCells']: cellChargeCandidates = chargeCandidates(speciesToNode(index, centralDict['chargeGeometryCells'], geometrySpecies))
        else: cellChargeCandidates = []

        try:
            script = ahead.pop(position, None)
            if script is None:
                # the first node written direct: the lines of every node (_nodeBlocks) are built by the builder, with
                # the next scripts, while IPhreeqc runs it
                script = nodeSpeciation(partCells, primCells, centralDict['speciesCharge'],cellChargeCandidates, centralDict['speciationCharge'],centralDict['solMod'],currentCells,
                                        withTail=False, direct=builder is not None and position == 0)
            building = builder is not None and position + 1 < len(nodes) and position + 1 not in ahead
            if building:
                builder.submit(lambda first=position + 1, last=min(position + 1 + _AHEAD_NODES, len(nodes)):
                               nextScripts(first, last))

            ref = time.perf_counter()
            try:
                runNode(solver, script, tail, body=True)
            finally:
                elapsed = time.perf_counter() - ref
                if building:
                    done, written = builder.result()
                    index = nodes[position]
                    if done:
                        ahead.update(written)
            calcTime += elapsed
        except Exception as e:
            where = f"PhreeqC : node n°{index}, time step n°{centralDict['lStep']+1}, t={centralDict['tStep']}{centralDict['timeUnit']}, PID={os.getpid()} :"
            # counter-charge candidates of the node (speciesChargeGeometry), else speciesCharge; the first one has failed
            candidates = ((cellChargeCandidates or list(centralDict['speciesCharge'] or []))
                          if centralDict['speciationCharge'] else [])
            kind = "geometrical dependent counter-balance" if cellChargeCandidates else "counter-balance"
            if candidates:
                scriptWarnings += f"{where} assuming {candidates[0]} as counter-charge species :\n {e}\n"
            else:
                scriptWarnings += f"{where} assuming no counter-charge species :\n {e}\n"

            solved, tried = False, None
            for spc in candidates[1:]:
                warnings += 1
                tried = spc
                scriptWarnings += f"{where} testing {spc} as {kind} species ...\n"
                retry = nodeSpeciation(partCells, primCells, None, spc, True, centralDict['solMod'], currentCells)
                if "\tcharge" not in retry:
                    scriptWarnings += f"{where} {spc} is absent from the solution of this node : speciation without charge balance\n"
                try:
                    ref = time.perf_counter()
                    solver.run_string(retry)
                    calcTime += time.perf_counter() - ref
                    scriptWarnings += f"{where} success !\n"
                    solved = True
                    break
                except Exception as r:
                    scriptWarnings += f"{where} {r}\n"

            # no candidate left (or none at all): the node cannot be computed
            if not solved:
                if tried is None:
                    warnings += 1
                scriptWarnings += f"{where} Fatal PhreeqC error. Aborted PhreeqC batch :\n"
                scriptWarnings += (nodeSpeciation(partCells, primCells, None, tried, True, centralDict['solMod'], currentCells)
                                   if tried is not None else
                                   nodeSpeciation(partCells, primCells, centralDict['speciesCharge'], cellChargeCandidates,
                                                  centralDict['speciationCharge'], centralDict['solMod'], currentCells))
                print('\nSpeciation batch aborted. See warning.log file.', end = '')
                abort = True

        # after the run of the node: its warnings, the row and column counts of the selected output and its last row
        # (into the block, header known) asked in one call (_phreeqcRow.after: the same calls of IPhreeqc in the same
        # order), otherwise call by call; status None: counts not asked yet, 0: row read into block[filled]
        status = None
        if not abort and _afterReader is not None:
            text, nrows, ncols, status = _afterRun(solver, header, block, filled, blockAddress, columns)
        # non-fatal PhreeqC warnings (an element unknown to the database set to 0, ...), each kind once per run (an
        # empty text gives none)
        if not abort and (status is None or text):
            for line in newWarnings(text if status is not None else warningText(solver), centralDict.get('phreeqcRun'),
                                    seen):
                warnings += 1
                scriptWarnings += (f"PhreeqC : node n°{index}, time step n°{centralDict['lStep']+1}, t={centralDict['tStep']}"
                                   f"{centralDict['timeUnit']}, PID={os.getpid()} : {line} (warning of this kind written "
                                   "once per run)\n")

        if status is None:
            nrows = solver.row_count
        if nrows and not abort and status == 0:
            results.append((rowNames, filled))
            filled += 1
        elif nrows and not abort:
            if status is None:
                ncols = solver.column_count
            if header is None or ncols != len(header):
                # the headings follow from the SELECTED_OUTPUT / USER_PUNCH definitions (tail) only: kept on the
                # instance while its last run used these definitions (runNode sets _pycteTail after a run without
                # error) and gave as many columns, read again otherwise
                known = getattr(solver, '_pycteHeader', None)
                if (known is not None and known[0] == tail and known[1] == ncols
                        and getattr(solver, '_pycteTail', None) == tail):
                    header = known[2]
                else:
                    header = selectedOutputRow(solver, 0, ncols)
                    if getattr(solver, '_pycteTail', None) == tail:
                        solver._pycteHeader = (tail, ncols, header)
                columns = _readColumns(header, needed)
                rowNames = header if columns is None else columns[3]
            # rows of doubles read into the block (one row per node), the others as lists
            width = len(rowNames)
            if block is None:
                block = np.empty((len(commMtrxPart.index), width))
                blockAddress = block.ctypes.data
            target = block[filled] if block.shape[1] == width and filled < len(block) else None
            if status in (2, 3):
                # read by _phreeqcRow.after: a value that is not a double, or an error code; read value by value, as
                # selectedOutputRow reads it then
                row = selectedOutputValues(solver, nrows - 1, ncols)
            elif columns is None:
                row = selectedOutputRow(solver, nrows - 1, ncols, target)
            else:
                # row not read by _phreeqcRow.after (header just known): read whole, then its columns taken
                if fullRow is None or len(fullRow) != ncols:
                    fullRow = np.empty(ncols)
                row = selectedOutputRow(solver, nrows - 1, ncols, fullRow if target is not None else None)
                if row is None:
                    target[...] = fullRow[columns[0]]
            if row is not None and columns is not None:
                row = [row[j] for j in columns[0].tolist()]
            if row is None:
                results.append((rowNames, filled))
                filled += 1
            else:
                results.append((rowNames, row))
        elif not nrows:
            warnings += 1
            scriptWarnings +=f"PhreeqC : node n°{index}, time step n°{centralDict['lStep']+1}, t={centralDict['tStep']}{centralDict['timeUnit']}, PID={os.getpid()} : no PhreeqC ouput ...\n"
            scriptWarnings +=f"PhreeqC : node n°{index}, time step n°{centralDict['lStep']+1}, t={centralDict['tStep']}{centralDict['timeUnit']}, PID={os.getpid()} : aborted PhreeqC batch :\n"
            scriptWarnings += nodeSpeciation(partCells, primCells, None, None, centralDict['speciationCharge'], centralDict['solMod'],currentCells)
            abort = True

    if abort:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), warnings, scriptWarnings, abort, calcTime, initWorker
    else:
        phreeqcOutput = selectedOutputFrame(results, block)
        phreeqcOutput.index = commMtrxPart.index

        # speciationColumns, crossOut: before the node loop
        crossIn = centralDict['crossDependencies']['speciation']['input']
        crossTrspt = centralDict['crossDependencies']['transport']['total']
        comm = centralDict['commMtrx']
        coupledVariables = list(centralDict['coupledVariables'])
        # nodes of commMtrx read for the crossDependencies: all of them, or (block of nodes of a worker process, whose
        # commMtrx holds every node) the rows of this block, found by label as the frame of the columns below aligns
        # them (both indexes unique, every label found: the same values)
        sameNodes = comm.index.equals(phreeqcOutput.index)
        rows = None
        if not sameNodes and comm.index.is_unique and phreeqcOutput.index.is_unique:
            rows = comm.index.get_indexer(phreeqcOutput.index)
            if not np.all(rows >= 0):
                rows = None
        if (_floatFrame(phreeqcOutput) and _floatFrame(comm) and _floatFrame(commMtrxPart)
                and (sameNodes or rows is not None) and len(set(coupledVariables)) == len(coupledVariables)):
            outputValues = phreeqcOutput.to_numpy()
            outputPos = {name: j for j, name in enumerate(phreeqcOutput.columns.tolist())}
            water = outputValues[:, outputPos['mass_H2O']] if centralDict.get('molesStorage') else None
            speciesCols = iter(speciationColumns)
            values = np.empty((len(coupledVariables), len(phreeqcOutput)))
            # rows of values taken from columns of the selected output: (row, column) pairs found in the order of the
            # coupled variables (the same lookups, the same errors), then copied at once; the species in mol/kgw times
            # the water mass (the same products, element by element)
            plainRows, plainCols, waterRows, waterCols = [], [], [], []
            for j, c in enumerate(coupledVariables):
                if c in crossOut:
                    plainRows.append(j); plainCols.append(outputPos[c])
                elif c in crossIn or c in crossTrspt:
                    values[j] = comm[c].to_numpy() if rows is None else comm[c].to_numpy()[rows]
                elif c in kineticPhases:
                    next(speciesCols)
                    if kineticsRan:
                        plainRows.append(j); plainCols.append(outputPos[f"k_{c}"])
                    else:
                        values[j] = commMtrxPart[c].to_numpy()
                else:
                    col = next(speciesCols)
                    if water is not None and col.endswith('(mol/kgw)'):
                        waterRows.append(j); waterCols.append(outputPos[col])
                    else:
                        plainRows.append(j); plainCols.append(outputPos[col])
            if next(speciesCols, None) is not None:
                raise ValueError("PhreeqC output : more species columns than coupledVariables entries (check systemSpecies)")
            if plainRows:
                values[plainRows] = outputValues[:, plainCols].T
            if waterRows:
                values[waterRows] = (outputValues[:, waterCols] * water[:, None]).T
            commMtrxSpct = pd.DataFrame(values.T, index=phreeqcOutput.index, columns=frames.columnIndex(coupledVariables))
        else:
            waterKg = phreeqcOutput['mass_H2O'] if centralDict.get('molesStorage') else None
            speciesCols = iter(speciationColumns)
            columns = {}
            for c in centralDict['coupledVariables']:
                if c in crossOut:
                    columns[c] = phreeqcOutput[c]
                elif c in crossIn or c in crossTrspt:
                    columns[c] = centralDict['commMtrx'][c]
                elif c in kineticPhases:
                    next(speciesCols)
                    columns[c] = phreeqcOutput[f"k_{c}"] if kineticsRan else commMtrxPart[c]
                else:
                    col = next(speciesCols)
                    columns[c] = phreeqcOutput[col] * waterKg if waterKg is not None and col.endswith('(mol/kgw)') else phreeqcOutput[col]
            if next(speciesCols, None) is not None:
                raise ValueError("PhreeqC output : more species columns than coupledVariables entries (check systemSpecies)")
            commMtrxSpct = pd.DataFrame(columns, index=phreeqcOutput.index)

        return commMtrxSpct, commMtrx_primSpecies, phreeqcOutput, warnings, scriptWarnings, abort, calcTime, initWorker


_workerDatabase = None


def workerSpeciationPhreeqC(centralDict, commMtrxPart, beforeTrsptMtrx):
    """speciationPhreeqC in the process of a block of nodes, whose IPhreeqc instance is kept while the database settings
    are the same (the block always runs in the same process: same results from one run to the next)."""
    global _workerDatabase
    database = (str(centralDict['chemPath']), centralDict.get('knobs'), centralDict.get('kineticRates'))
    if database != _workerDatabase:
        resetPhreeqC()
        _workerDatabase = database
    return speciationPhreeqC(centralDict, commMtrxPart, beforeTrsptMtrx)


class _Block:
    """A block of nodes of speciationThreads > 1, computed in a thread: its own IPhreeqc instance (created at its first
    call, kept while the database settings are the same, released at the end of the run) and its own record of the
    warnings already reported, as the worker process of the block had."""

    def __init__(self):
        self.solver = None
        self.database = None
        self.seen = {'run': None, 'kinds': set()}

    def release(self):
        if self.solver is not None:
            try:
                self.solver.destroy_iphreeqc()
            except Exception as r:
                warningManager.warn(f"PhreeqC : IPhreeqc not released by destroy_iphreeqc() : {r}")
            self.solver = None


_blocks = []            # block k of the nodes: always _blocks[k]
_threads = None         # ThreadPoolExecutor of the blocks


def blockSpeciationPhreeqC(block, centralDict, commMtrxPart, beforeTrsptMtrx):
    """speciationPhreeqC of a block of nodes on the instance of the block (workerSpeciationPhreeqC in a thread)."""
    database = (str(centralDict['chemPath']), centralDict.get('knobs'), centralDict.get('kineticRates'))
    if database != block.database:
        block.release()
        block.database = database
    return speciationPhreeqC(centralDict, commMtrxPart, beforeTrsptMtrx, block)


def _blockThreads(n):
    """The n blocks and the threads that compute them (kept between steps, released by resetPhreeqC)."""
    global _threads
    while len(_blocks) < n:
        _blocks.append(_Block())
    if _threads is None or _threads._max_workers < n:
        if _threads is not None:
            _threads.shutdown(wait=True)
        from concurrent.futures import ThreadPoolExecutor
        _threads = ThreadPoolExecutor(max_workers=n, thread_name_prefix='pycte-phreeqc')
    return _blocks[:n], _threads


# keys of centralDict read by workerSpeciationPhreeqC and the functions it calls (tests/test_worker_dicts.py checks
# that the list holds every key they read): with speciationThreads > 1, the workers receive these keys only, pickled at
# every call, instead of the whole dictionary (dtpycte, ~370 kB for 100 000 steps, the transport settings and arrays,
# the state of the other solvers ...)
WORKER_KEYS = frozenset({
    'MultiCompoundTransport', 'SI', 'cellWaterMass', 'chargeGeometryCells', 'chemPath', 'commMtrx', 'coupledVariables',
    'crossDependencies', 'cutoffs', 'densityPhreeqC', 'dtStep', 'firstStepEquilibrium', 'fixedPhases', 'fixpH',
    'kineticRates', 'kinetics', 'kineticsM0', 'kineticsSpecies', 'knobs', 'lStep', 'mctTotals',
    'mineralReversibility', 'molesStorage', 'output', 'pHdefault', 'paths', 'phreeqcRun', 'preliminaryEquilibrium',
    'primToSecSpecies', 'primarySpecies', 'solMod', 'speciationCharge', 'speciesCharge', 'speciesChargeGeometry',
    'step_divide', 'stoichPhreeqC', 'supplementarySolution', 'surfaceCounterIons', 'systemSpecies', 'tStep',
    'tempDefault', 'timeUnit', 'userVarBool', 'userVarList', 'water',
})


def _workerDict(centralDict):
    """centralDict as the workers read it: the WORKER_KEYS (in the order of centralDict); commMtrx reduced to the
    columns they read (crossDependencies of the speciation and of the transport, every node, same index: the block
    of each worker its own nodes are given apart), output to its 'speciation' entry (outputManager.wanted)."""
    light = {k: v for k, v in centralDict.items() if k in WORKER_KEYS}
    comm = light.get('commMtrx')
    try:
        crossDep = centralDict['crossDependencies']
        read = set(crossDep['speciation']['input']) | set(crossDep['transport']['total'])
    except (KeyError, TypeError):
        read = None                                 # unexpected layout: commMtrx sent whole
    if read is not None and isinstance(comm, pd.DataFrame) and comm.columns.is_unique:
        light['commMtrx'] = comm[[c for c in comm.columns.tolist() if c in read]]
    output = light.get('output')
    if isinstance(output, dict):
        light['output'] = {k: v for k, v in output.items() if k == 'speciation'}
    return light


def multiCompoundSetup(centralDict):
    """MultiCompoundTransport: commMtrx, boundaries and renewal solutions converted to totals, H, O and chargebalance."""
    from . import extractDB
    cd = centralDict

    def fail(message):
        raise ValueError(f"MultiCompoundTransport : {message}")

    def warn(message):
        warningManager.warn(f"MultiCompoundTransport : {message}")

    crossSpc = cd['crossDependencies']['speciation']
    cbName = crossDependenciesManager.byQuantity(crossSpc, 'input').get('CHARGE_BALANCE')
    if not cbName:
        fail("the charge balance must be coupled : add 'chargebalance' to coupledVariables, to the initial conditions "
             "and to crossDependencies = {'speciation' : ['chargebalance', ..]}")
    for key in ('preliminaryEquilibrium', 'NernstPlanck', 'donnan', 'activityGradient'):
        if cd.get(key):
            fail(f"{key} is not available (it needs the species)")
    if not cd['solMod']:
        cd['solMod'] = True
        warn("solMod forced to True (SOLUTION_MODIFY with the totals, -total_h, -total_o and -cb)")

    db = extractDB.extract_master_species(cd['chemPath'])
    masters = db['SOLUTION_MASTER_SPECIES']
    masterOf = dict(zip(masters, db['SOLUTION_MASTER_ELEMENT']))
    redox = {el: states for el, states in extractDB.group_redox_states(masters).items() if el not in ('H', 'O')}
    primaryState = {el: next((s for s in states if masterOf.get(s) == masterOf.get(el)), None) for el, states in redox.items()}
    element = lambda s: s.split('(')[0]
    present = {element(s) for s in cd['primarySpecies']['solution'] if s not in ('H', 'O', 'H2O')}
    totals = list(dict.fromkeys(s for s in masters if element(s) in present and element(s) not in ('H', 'O')
                                and (s in redox[element(s)] if element(s) in redox else s == element(s))))
    totals += ['H', 'O']

    crossNames = list(crossSpc['total']) + list(cd['crossDependencies']['transport']['total'])
    surfaceSpecies = [s for s in db['SURFACE_SPECIES'] if s not in db['SOLUTION_MASTER_ELEMENT']]
    immobile = set(db['SURFACE_MASTER_SPECIES'] + surfaceSpecies + db['EXCHANGE_SPECIES'] + db['PHASES']
                   + list(cd['kineticsSpecies'] or []))
    cmdToName = crossDependenciesManager.byQuantity(crossSpc, 'total')
    pHName, peName, ehName = cmdToName.get('-la("H+")'), cmdToName.get('-la("e-")'), cmdToName.get('Eh')
    tkName = cmdToName.get('tk')

    def composition(name):
        if name == 'H2O':
            return {'H': 2, 'O': 1}
        if name in totals:
            return {name: 1}
        if name in redox and primaryState[name]:
            return {primaryState[name]: 1}
        return cd['primToSecSpecies'].get(name)

    def linear(tab):
        position = {t: k for k, t in enumerate(totals)}
        out = np.zeros((len(tab), len(totals)))
        touched = set()
        for col in tab.columns:
            comp = composition(col)
            if comp is None:
                fail(f"'{col}' is neither a species of {cd['chemPath'].name} nor a total")
            values = None
            for prim, coeff in comp.items():
                if prim not in totals:
                    fail(f"'{col}' decomposes into '{prim}', which is not a transported total")
                if values is None:
                    values = tab[col].astype(float).to_numpy()
                out[:, position[prim]] += coeff * values
                touched.add(prim)
        kept = [t for t in totals if t in touched]
        return pd.DataFrame(out[:, [position[t] for t in kept]], index=tab.index, columns=kept)

    def initialSolution(tab, pH, pe, temp):
        solver = newIPhreeqc()
        loadDatabase(solver, cd)
        punch = ("SELECTED_OUTPUT\n-reset false\n-totals" + "".join(f"\t{t}" for t in totals if t not in ('H', 'O'))
                 + '\nUSER_PUNCH\n-headings\tH\tO\tchargebalance\n10 PUNCH TOT("H"), TOT("O"), CHARGE_BALANCE\nEND\n')
        rows = []
        for k, index in enumerate(tab.index):
            given = {}
            for col in tab.columns:
                if col in masters:
                    comp = {col: 1}
                else:
                    comp = composition(col)
                    if comp is None:
                        fail(f"'{col}' is neither a species of {cd['chemPath'].name} nor a total")
                for prim, coeff in comp.items():
                    if prim not in ('H', 'O'):
                        given[prim] = given.get(prim, 0.0) + coeff * float(tab.at[index, col])
            script = (f"SOLUTION 1\n-units mol/kgw\n-water 1\n-temp {temp[k]}\n"
                      f"pH {pH[k]}\npe {pe[k]}\n")
            script += "".join(f"\t{name} {value}\n" for name, value in given.items() if value > cd['cutoffs']['solution'])
            try:
                solver.run_string(script + punch)
            except Exception as e:
                fail(f"PhreeqC initial solution failed (node {index}) :\n{e}\n{script}")
            out = solver.get_selected_output_array()
            res = dict(zip(out[0], out[-1]))
            rows.append({**{t: res[f"{t}(mol/kgw)"] for t in totals if t not in ('H', 'O')},
                         'H': res['H'], 'O': res['O'], cbName: res['chargebalance']})
        return pd.DataFrame(rows, index=tab.index)

    def convert(tab, what):
        cols = [c for c in tab.columns if c not in crossNames and c not in immobile and c not in cd['coord']]
        if 'H2O' in cols or {'H', 'O'} <= set(cols):
            return linear(tab[cols]), "species / totals decomposition"
        if 'H' in cols or 'O' in cols:
            fail(f"{what} : give both H and O totals (water included), or H2O")
        pH = tab[pHName].to_numpy() if pHName in tab else [cd['pHdefault']] * len(tab)
        # °C for PhreeqC, from the tk column or tempDefault (K)
        temp = (tab[tkName].to_numpy() if tkName in tab else np.full(len(tab), float(cd['tempDefault']))) - 273.15
        pe = (tab[peName].to_numpy() if peName in tab
              else tab[ehName].to_numpy() / (crossDependenciesManager.EH_PER_PE_K * (np.asarray(temp, dtype=float) + 273.15))
              if ehName in tab else [4] * len(tab))
        return initialSolution(tab[cols], pH, pe, temp), "PhreeqC initial solution (pH, pe, totals)"

    ci = cd['commMtrx']
    tot, how = convert(ci, 'initial conditions')
    missing = [t for t in totals if t not in tot.columns]
    for t in missing:
        tot[t] = 0.0
    crossCols = [s for s in cd['coupledVariables'] if s in crossNames]
    kept = [s for s in cd['coupledVariables'] if s not in crossNames and s in immobile]
    comm = pd.concat([ci[cd['coord'] + crossCols], tot[totals], ci[kept]], axis=1)
    if cbName in tot:
        comm[cbName] = tot[cbName]

    speciation = crossCols + totals + kept
    ps = cd['primarySpecies']
    ps['solution'] = totals + ['H2O']
    ps['total'] = ps['solution'] + ps['exchange'] + ps['surface'] + ps['phases']
    cd['primToSecSpecies'].update({t: {t: 1} for t in totals})
    cd.update({
        'commMtrx': comm[cd['coord'] + speciation].copy(),
        'coupledVariables': speciation,
        'systemSpecies': [s for s in speciation if s not in crossNames],
        'mctTotals': totals,
        'transportedSpecies': totals + [cbName],
        })
    cd['anythingButSpecies'] = [c for c in cd['commMtrx'].columns if c not in speciation]

    for key in ('firstBoundary', 'secondBoundary'):
        values = cd.get(key)
        if not values:
            continue
        unknown = [k for k in values if k not in crossNames and k not in immobile and composition(k) is None and k not in masters]
        if unknown:
            fail(f"{key} : {unknown} are neither totals nor species of the initial conditions (give the boundary "
                 f"with the same names as the initial conditions, or as totals)")
        tab = pd.DataFrame([{k: float(v) for k, v in values.items()}])
        if not [c for c in tab.columns if c not in crossNames and c not in immobile and c not in cd['coord']]:
            continue
        btot, _ = convert(tab, key)
        converted = {t: float(btot[t].iloc[0]) for t in btot.columns if t in totals}
        if cbName in btot:
            converted[cbName] = float(btot[cbName].iloc[0])
        elif cbName in values:
            converted[cbName] = float(values[cbName])
        else:
            warn(f"{key} has no chargebalance : Neumann condition for the charge balance on this side")
        cd[key] = converted

    for k, group in enumerate(cd.get('renewal') or []):
        values = group['concentration']
        if not values:
            continue
        where = f"renewal, group {k}"
        unknown = [c for c in values if c not in crossNames and c not in immobile and composition(c) is None and c not in masters]
        if unknown:
            fail(f"{where} : {unknown} are neither totals nor species of the initial conditions")
        kept = {c: float(v) for c, v in values.items() if c in immobile or (c in crossNames and c != cbName)}
        tab = pd.DataFrame([{c: float(v) for c, v in values.items()}])
        converted = {}
        if [c for c in tab.columns if c not in crossNames and c not in immobile and c not in cd['coord']]:
            btot, _ = convert(tab, where)
            converted = {t: float(btot[t].iloc[0]) for t in btot.columns if t in totals}
            if cbName in btot:
                converted[cbName] = float(btot[cbName].iloc[0])
        if cbName not in converted:
            if cbName not in values:
                warn(f"{where} has no chargebalance : 0 is used for the renewed cells")
            converted[cbName] = float(values.get(cbName, 0.0))
        group['concentration'] = {**converted, **kept}

    for key in ('speciesDiffCoeff', 'speciesPorosity'):
        stale = [s for s in (cd.get(key) or {}) if s not in cd['transportedSpecies']]
        if stale:
            warn(f"{key} : {stale} are not transported totals (ignored)")

    print(f"MultiCompoundTransport : initial conditions -> totals ({how}), "
          f"transported : {', '.join(totals)}, {cbName}")
    if missing:
        warn(f"totals not given in the initial conditions, set to 0 : {', '.join(missing)}")
    return centralDict


def debyeWarning(centralDict):
    """Warns once when the DebyeLength column is not used."""
    if centralDict.get('debyeWarned'):
        return
    name = crossDependenciesManager.byQuantity(centralDict['crossDependencies']['speciation'], 'input').get('DebyeLength')
    if not name or centralDict.get('debyeWarned'):
        return
    centralDict['debyeWarned'] = True
    if not centralDict['primarySpecies']['surface']:
        warningManager.warn(f"crossDependencies : {name} (DebyeLength) not used, the system has no SURFACE")
        return
    values = centralDict['commMtrx'][name].to_numpy(dtype=float)
    notPositive = np.flatnonzero(~(values > 0))
    if notPositive.size:
        warningManager.warn(f"crossDependencies : {name} (DebyeLength) <= 0 at {notPositive.size} node(s) "
                            f"(first : {centralDict['commMtrx'].index[notPositive[0]]}) : no diffuse layer there (-no_edl)")


def _inputColumns(centralDict):
    """(columns given to PhreeqC, those of them that are species) of spct."""
    excluded = centralDict['crossDependencies']['speciation']['total']
    transportOnly = [c for c in centralDict['crossDependencies']['transport']['total'] if c not in excluded]
    phreeqcInput = [c for c in centralDict['coupledVariables'] if c not in transportOnly]
    return phreeqcInput, [s for s in phreeqcInput if s not in excluded]


def spct(centralDict):
    startPhreeqC = time.perf_counter()
    display.stepPrint("PhreeqC", end=" ", flush=True)
    
    excluded = centralDict['crossDependencies']['speciation']['total']
    # columns given to PhreeqC: the coupled variables but the crossDependencies of the transport alone (a temperature
    # coupled with both the speciation and the transport is read by PhreeqC); from the settings only: built once
    phreeqcInput, phreeqcSpecies = frames.constant(
        'phreeqcInput', (excluded, centralDict['crossDependencies']['transport']['total'], centralDict['coupledVariables']),
        lambda: _inputColumns(centralDict))
    stoichMatrix(centralDict, phreeqcSpecies)
    debyeWarning(centralDict)

    if centralDict['speciationThreads'] > 1:
        
        chunk_size = int(np.ceil(len(centralDict['commMtrx']) / centralDict['speciationThreads']))
        commMtrxInput = _selectColumns(centralDict['commMtrx'], phreeqcInput)
        commMtrxSplit = [commMtrxInput.iloc[i:i + chunk_size] for i in range(0, len(commMtrxInput), chunk_size)]


        if centralDict['preliminaryEquilibrium']:
            
            assert len(centralDict['commMtrx'][phreeqcInput]) == len(centralDict["beforeTrsptMtrx"][phreeqcInput])

            beforeTrsptMtrx = centralDict["beforeTrsptMtrx"][phreeqcInput].copy()
            commMtrxSplitbeforeTrspt = [beforeTrsptMtrx[phreeqcInput].iloc[i:i + chunk_size] for i in range(0, len(centralDict['commMtrx'][phreeqcInput]), chunk_size)]
        else: 
            commMtrxSplitbeforeTrspt = [_EMPTY_FRAME] * len(commMtrxSplit)     # the same empty frame
        
        # block k computed in a thread on its own IPhreeqc instance (_Block, kept from one step to the next): PhreeqC
        # runs in parallel, ctypes releasing the GIL during each call. Each block is given what its worker process was
        # given before: the dictionary of the keys it reads (its own copy, _workerDict) and a copy of its nodes. No
        # process to start (~2 s per run here: Python and pycte imports), nothing to pickle nor send at each call
        blocks, threads = _blockThreads(len(commMtrxSplit))
        light = _workerDict(centralDict)            # what the blocks read, not the whole dictionary
        futures = {}
        for i,chunk in enumerate(commMtrxSplit):
            fut = threads.submit(blockSpeciationPhreeqC, blocks[i], dict(light), chunk.copy(),
                                 commMtrxSplitbeforeTrspt[i].copy())
            futures[fut] = i

        results = [None] * len(commMtrxSplit)
        for future, i in futures.items():
            try:
                results[i] = future.result()
            except Exception:
                import traceback
                traceback.print_exc()
                raise
    
        
        df1, df3, df4, intgr, strg, Bool, calc, initWorker = zip(*results)

        commMtrxSpct = pd.concat(df1, ignore_index=True)

        commMtrx_primSpecies = pd.concat(df3, ignore_index=True)
        phreeqcOutput = pd.concat(df4, ignore_index=True)
        totalWarnings = sum(intgr)
        calcPrcsTime = sum(calc)
        calcWallClock = max(calc)
        abort = any(Bool)
        init = max(initWorker)
        totalScriptWarnings = "".join(strg)

    else:
        if centralDict['preliminaryEquilibrium']:
            beforeTrspt = centralDict["beforeTrsptMtrx"][phreeqcInput].copy()
        else:
            beforeTrspt = _EMPTY_FRAME

        commMtrxSpct, commMtrx_primSpecies, phreeqcOutput, totalWarnings, totalScriptWarnings, abort, calcWallClock, init = speciationPhreeqC(centralDict,
                                                        _selectColumns(centralDict['commMtrx'], phreeqcInput),beforeTrspt )
        calcPrcsTime = calcWallClock

    if totalWarnings or abort:
        warningManager.block(f"PhreeqC, time = {centralDict['tStep']}{centralDict['timeUnit']}, time step n°{centralDict['lStep']+1} : the {totalWarnings} following warnings occurred ...\n"
                             f"{totalScriptWarnings}\n", totalWarnings)
    if abort:
        raise RuntimeError(f"PhreeqC : fatal error at time step n°{centralDict['lStep']+1}, the speciation batch was "
                           "aborted (PhreeqC messages and script of the node above in warning.log)")

    
    comm = centralDict['commMtrx']
    coord = list(centralDict['coord'])
    computed = commMtrxSpct.columns.tolist()
    if (_floatFrame(comm) and _floatFrame(commMtrxSpct) and commMtrxSpct.index.equals(comm.index)
            and len(set(coord)) == len(coord) and not set(coord) & set(computed)
            and set(comm.columns.tolist()) <= set(coord) | set(computed)):
        spctValues = commMtrxSpct.to_numpy()
        spctPos = {c: j for j, c in enumerate(computed)}
        full = np.empty((len(comm.columns), len(comm)))
        # computed columns copied at once (rows of full, positions in spctValues), the coordinates one by one
        fromSpct, spctCols = [], []
        for j, c in enumerate(comm.columns.tolist()):
            if c in coord:
                full[j] = comm[c].to_numpy()
            else:
                fromSpct.append(j); spctCols.append(spctPos[c])
        if fromSpct:
            full[fromSpct] = spctValues[:, spctCols].T
        commMtrxSpct = pd.DataFrame(full.T, index=comm.index, columns=comm.columns)
    else:
        commMtrxSpct = pd.concat([comm[coord],commMtrxSpct], axis=1)
        commMtrxSpct = commMtrxSpct[comm.columns]
    if centralDict.get('molesStorage'):
        waterMass = (phreeqcOutput['mass_H2O'] if phreeqcOutput.index.equals(comm.index)
                     else pd.concat([comm[coord],phreeqcOutput], axis=1)['mass_H2O'])
        centralDict['cellWaterMass'] = waterMass.to_numpy(dtype=float)

    centralDict.update({
        "commMtrx": commMtrxSpct,
        "PhreeqCCalcTime_WallClock": centralDict['PhreeqCCalcTime_WallClock']  + calcWallClock,
        "PhreeqCCalcTime_ProcessorTime": centralDict["PhreeqCCalcTime_ProcessorTime"] + calcPrcsTime,
        "PhreeqCInterfTime_WallClock": centralDict['PhreeqCInterfTime_WallClock'] + time.perf_counter() - startPhreeqC - calcWallClock - init,
        "PhreeqCInitTime" : centralDict["PhreeqCInitTime"] + init,
        "PhreeqCTotalTime" : centralDict['PhreeqCTotalTime'] + time.perf_counter() - startPhreeqC,
        })


    if outputManager.wanted(centralDict, 'speciation'):
        coords = frames.columnsOf(comm, coord)
        commMtrx_primSpecies = pd.concat([coords,commMtrx_primSpecies], axis=1)
        phreeqcOutput = pd.concat([coords,phreeqcOutput], axis=1)
        outputManager.write(centralDict, commMtrx_primSpecies, 'primarySpecies', 'PrimarySpecies')
        outputManager.write(centralDict, phreeqcOutput, 'speciation', 'PhreeqC')

    display.stepPrint(f"({writeTime((time.perf_counter() - startPhreeqC))})")
    return centralDict