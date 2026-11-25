import time
import numpy as np
import os
import concurrent.futures
import subprocess
import re
import shutil
from pathlib import Path
import pandas as pd
import h5py
import sys
try:
    from . import outputManager
    from . import timeUnits
    from . import warningManager
    from . import display
except ImportError:
    import outputManager
    import timeUnits
    import warningManager
    import display



def writeTime(tps, arr=2):
    if tps >= 3600 * 24:
        return f"{tps / (3600 * 24):.{arr}f} d"
    elif tps >= 3600:
        return f"{tps / 3600:.{arr}f} h"
    elif tps >= 60:
        return f"{tps / 60:.{arr}f} min"
    elif tps < 1:
        return f"{tps *1000 :.{arr}f} msec"
    else:
        return f"{tps:.{arr}f} sec"


_writtenTime = {}

secondsPerUnit = timeUnits.secondsPerUnit


WORK_FOLDER = "pflotranWork"


def _readText(path):
    """Text of a PFLOTRAN input and its encoding."""
    raw = Path(path).read_bytes()
    for encoding in ("utf-8", "cp1252"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1"), "latin-1"


def workingCopy(input_path, runFolder):
    """Copies the PFLOTRAN input and the files it names to <run folder>/pflotranWork; returns the copy."""
    source = Path(input_path).resolve()
    work = (Path(runFolder) / WORK_FOLDER).resolve()
    if work == source.parent or work in source.parents:
        raise ValueError(f"trsptPath = {input_path} : the PFLOTRAN input must not lie in {work}, the working folder "
                         "that pycte empties at every run")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    text, _ = _readText(source)
    outside = []
    for word in sorted(set(re.findall(r"[^\s'\"]+", text))):
        if Path(word).is_absolute() or not (source.parent / word).is_file() or (source.parent / word).resolve() == source:
            continue
        target = (work / word).resolve()
        if work not in target.parents:
            outside.append(word)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source.parent / word, target)
    if outside:
        warningManager.warn(f"PFLOTRAN : {outside} named by {source.name} lie outside its folder : not copied to "
                            f"{work}, give them as absolute paths in the input")
    shutil.copy2(source, work / source.name)
    return work / source.name


def _speciesBlock(input_path, block):
    """Names listed in a CHEMISTRY block of the PFLOTRAN input (PRIMARY_SPECIES, IMMOBILE_SPECIES, ...)."""
    names, inside = [], False
    with open(input_path, encoding=_readText(input_path)[1]) as f:
        for line in f:
            word = line.split('#', 1)[0].strip()
            if not word:
                continue
            if word.upper() == block:
                inside = True
                continue
            if inside:
                if word == '/' or word.upper() == 'END':
                    break
                names.append(word.split()[0])
    return names


def immobileSpecies(input_path):
    return [n[:-len('pycte')] if n.endswith('pycte') else n for n in _speciesBlock(input_path, 'IMMOBILE_SPECIES')]


NEGATIVE_SUFFIX = '_neg'


def signedSpecies(input_path):
    """Species transported as a positive and a negative part: <name>pycte and <name>_negpycte in PRIMARY_SPECIES."""
    tail = NEGATIVE_SUFFIX + 'pycte'
    return [n[:-len(tail)] for n in _speciesBlock(input_path, 'PRIMARY_SPECIES') if n.endswith(tail)]


_flooredSpecies = set()


def reactionArray(centralDict, key):
    species = list(centralDict['transportedSpecies'])
    out = np.zeros((len(centralDict['commMtrx']), len(species)))
    values = centralDict.get(key)
    if isinstance(values, pd.DataFrame):
        for j, sp in enumerate(species):
            if sp in values.columns:
                out[:, j] = values[sp].to_numpy(dtype=float)
    return out


def hasReaction(centralDict):
    return centralDict.get('transportSource') is not None or centralDict.get('transportSink') is not None


def update_pflotran_time(input_path, new_time, unit):
    text, encoding = _readText(input_path)

    pattern_final_time = re.compile(
        r'^(\s*FINAL_TIME\s+)([^\s#]+)(\s+)([^\s#]+)',
        re.MULTILINE,
    )
    text, n1 = pattern_final_time.subn(
        lambda m: f"{m.group(1)}{new_time}{m.group(3)}{unit}", text, count=1
    )
    if n1 == 0:
        raise ValueError(f"PFLOTRAN : line FINAL_TIME not found in {Path(input_path).name}")

    pattern_times = re.compile(
        r'^(\s*TIMES\s+)(\S+)(\s+).*$',
        re.MULTILINE,
    )
    text, n2 = pattern_times.subn(
        lambda m: f"{m.group(1)}{unit}{m.group(3)}{new_time}", text, count=1
    )
    if n2 == 0:
        raise ValueError("Line TIMES not found in OUTPUT block.")

    with open(input_path, "w", encoding=encoding, newline="") as f:
        f.write(text)


def normalizeVarName(name):

    s = name.lower()
    s = re.sub(r"^total[ _]", "", s)
    s = s.replace("minus", "-")
    s = s.replace("plus", "+")
    s = re.sub(r"dc\b", "", s)
    # unit of a PFLOTRAN output ('Total Ca+2pycte [M]' -> 'ca+2pycte'); a name ending with ']' (Calcite[s]) is kept
    s = re.sub(r"\s+\[[^\]]*\]?\s*$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_approximateMatches = set()     # (target, source) associations by substring already reported


def matchVariables(source_names, target_names):

    norm_source = {name: normalizeVarName(name) for name in source_names}
    norm_target = {name: normalizeVarName(name) for name in target_names}

    def priority(name):
        return 1 if "rate" in name.lower() else 0

    mapping = {}
    used_source = set()
    remaining = list(target_names)

    def try_pass(candidate_fn, unique=False):
        nonlocal remaining
        still_remaining = []
        for tgt in remaining:
            tgt_norm = norm_target[tgt]
            candidates = [s for s, sn in norm_source.items()
                          if s not in used_source and candidate_fn(tgt_norm, sn)]
            if len(candidates) >= 1 and not (unique and len(candidates) > 1):
                best = sorted(candidates, key=lambda s: priority(s))[0]
                mapping[tgt] = best
                used_source.add(best)
                if unique and (tgt, best) not in _approximateMatches:
                    _approximateMatches.add((tgt, best))
                    warningManager.warn(f"PFLOTRAN : variable {best!r} read as {tgt!r} (name containing it, no "
                                        "exact match)")
            else:
                still_remaining.append(tgt)
        remaining = still_remaining

    try_pass(lambda tn, sn: sn == tn)
    try_pass(lambda tn, sn: sn.removesuffix("pycte") == tn)
    try_pass(lambda tn, sn: bool(tn) and tn in sn.split())
    # last resort, a name containing the target: only when a single output variable does ('o2' is also in 'no2-')
    try_pass(lambda tn, sn: bool(tn) and tn in sn, unique=True)

    if remaining:
        raise ValueError(f"PFLOTRAN : no output variable (or several ones) for {remaining} among {source_names} : "
                         "check the OUTPUT block of the PFLOTRAN input and the species names")

    return mapping

_UNIT_SUFFIX_RE = re.compile(r"^(.*)\(([A-Za-z]+)\)$")
_KNOWN_UNITS = {"K", "C"}


def extractTargetUnit(name):
    m = _UNIT_SUFFIX_RE.match(name)
    if m and m.group(2) in _KNOWN_UNITS:
        return m.group(1).strip(), m.group(2)
    return name, None


def convertUnit(value, from_unit, to_unit):
    if from_unit is None or to_unit is None or from_unit.upper() == to_unit.upper():
        return value
    fu, tu = from_unit.upper(), to_unit.upper()
    if fu == "C" and tu == "K":
        return value + 273.15
    if fu == "K" and tu == "C":
        return value - 273.15
    raise ValueError(f"Cannot convert units: '{from_unit}' -> '{to_unit}'")


def lastOutputFile(input_path, ext, runStart):
    # PFLOTRAN output written by the current run: <stem>.h5, <stem>-NNN.h5 or <stem>-NNN.tec
    stem = re.escape(Path(input_path).stem)
    pattern = re.compile(rf"{stem}(?:-(\d+))?\.{ext}$")
    files = [f for f in Path(input_path).parent.iterdir()
             if pattern.match(f.name) and f.stat().st_mtime >= runStart - 1]
    if not files:
        return None
    return max(files, key=lambda f: int(pattern.match(f.name).group(1) or 0))


def splitNameUnit(raw):
    m = re.match(r"(.*?)\s+\[([^\[\]]*)\]\s*$", raw)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return raw.strip(), None


def readTecplotOutput(last_file):
    # TECPLOT format: values written with 7 significant digits only
    variables_line = None
    with last_file.open() as f:
        for line in f:
            if line.startswith("VARIABLES="):
                variables_line = line
                break

    if variables_line is None:
        raise ValueError(f"No line VARIABLES= found in {last_file}")

    variables = []
    variableUnits = {}
    for raw in re.findall(r'"([^"]+)"', variables_line):
        name, unit = splitNameUnit(raw)
        variables.append(name)
        variableUnits[name] = unit

    df = pd.read_csv(
        last_file,
        sep=r"\s+",
        skiprows=3,
        names=variables,
        engine="python"
    )
    return df, variableUnits


def readHdf5Output(last_file, centralDict):
    # HDF5 format: double precision values, last output time of the file
    with h5py.File(last_file, 'r') as f:
        timeGroups = {}
        for key in f.keys():
            m = re.match(r"Time:\s*(\S+)", key)
            if m:
                timeGroups[float(m.group(1))] = key
        if not timeGroups:
            raise ValueError(f"No 'Time:' group found in {last_file}")
        group = f[timeGroups[max(timeGroups)]]

        data = {}
        variableUnits = {}
        for raw, dset in group.items():
            if not isinstance(dset, h5py.Dataset):
                continue
            name, unit = splitNameUnit(raw)
            # same names as in the TECPLOT header ("Total_X" -> "Total X")
            name = re.sub(r"^(Total|Free|Material)_", r"\1 ", name)
            values = dset[()]
            # structured grid: (nx, ny, nz) array, natural cell order is x fastest
            data[name] = np.ravel(values, order='F') if values.ndim > 1 else values
            variableUnits[name] = unit

    df = pd.DataFrame(data)
    if len(df) != len(centralDict['commMtrx']):
        raise ValueError(f"{last_file.name}: {len(df)} cells read, {len(centralDict['commMtrx'])} expected")
    # cell centers are not stored for structured grids, the grid is the one of commMtrx
    for c in centralDict['coord']:
        df[c] = centralDict['commMtrx'][c].to_numpy()
        variableUnits[c] = None
    return df, variableUnits


def executable():
    """PFLOTRAN executable, $PFLOTRAN_DIR/src/pflotran/pflotran, run through mpirun: checked when the run is set up
    (not after the first speciation step). On Windows, without mpirun (MS-MPI provides mpiexec, which pycte does not
    call), a clear error: PFLOTRAN is run under Linux or WSL."""
    folder = os.environ.get("PFLOTRAN_DIR")
    if sys.platform == 'win32' and shutil.which("mpirun") is None:
        raise RuntimeError("PFLOTRAN : pycte runs PFLOTRAN as 'mpirun -n N $PFLOTRAN_DIR/src/pflotran/pflotran', and "
                           "this Windows machine has no mpirun (MS-MPI provides mpiexec only) : run pycte with "
                           "PFLOTRAN under Linux or WSL (Windows Subsystem for Linux)")
    if not folder:
        raise RuntimeError("PFLOTRAN : the environment variable PFLOTRAN_DIR (folder of the PFLOTRAN installation, "
                           "with src/pflotran/pflotran) is not set")
    path = os.path.join(folder, "src", "pflotran", "pflotran")
    if not (os.path.isfile(path) or os.path.isfile(path + ".exe")):
        raise RuntimeError(f"PFLOTRAN : no executable {path} (PFLOTRAN_DIR = {folder}) : build PFLOTRAN, or set "
                           "PFLOTRAN_DIR to the folder holding src/pflotran/pflotran")
    if shutil.which("mpirun") is None:
        raise RuntimeError("PFLOTRAN : mpirun is not in the PATH (pycte runs 'mpirun -n N pflotran') : install an MPI "
                           "(OpenMPI, MPICH) or add its bin folder to the PATH")
    return path


def transport(centralDict):
    sandbox = centralDict.get('pflotranReactionTerms') == 'sandbox'

    # PFLOTRAN receives and returns the transported species (or totals, MultiCompoundTransport), the immobile species of
    # its input (IMMOBILE_SPECIES) and the crossDependencies of the transport (temperature, porosity, ...); the other
    # columns (crossDependencies of the speciation, Zz, solids unknown to PFLOTRAN) are copied back by trspt. Floor of
    # 1e-40 on the species only (pe, a temperature in °C can be negative)
    crossTrspt = centralDict['crossDependencies']['transport']
    crossNames = crossTrspt['total']
    quantity = crossTrspt['quantity']       # column -> quantity (tk -> temperature(K))
    given = (set(centralDict['transportedSpecies']) | set(immobileSpecies(centralDict['trsptPath'])) | set(crossNames)) - {'Zz'}
    signed = [s for s in signedSpecies(centralDict['trsptPath']) if s in given and s not in crossNames]
    spcArray = {}
    for col in centralDict['commMtrx'].columns:
        if col in centralDict['coord'] or col not in given:
            continue
        arr = centralDict['commMtrx'][col].to_numpy()
        if col in crossNames:
            spcArray[col] = arr
            continue
        if col in signed:
            spcArray[col + NEGATIVE_SUFFIX] = np.where(-arr < 1e-40, 1e-40, -arr)
        elif (arr < 0).any() and col not in _flooredSpecies:
            _flooredSpecies.add(col)
            warningManager.warn(f"PFLOTRAN : {col} has negative values, set to 1e-40 : add {col}{NEGATIVE_SUFFIX}pycte "
                                f"to PRIMARY_SPECIES of {Path(centralDict['trsptPath']).name} to transport its "
                                "negative part")
        spcArray[col] = np.where(arr < 1e-40, 1e-40, arr)

    
    # temperature sent in the unit of PFLOTRAN (°C), whatever the unit of its column (tk: K), as it is read back
    for col in crossNames:
        base, unit = extractTargetUnit(quantity.get(col, col))
        if col in spcArray and base.lower() == 'temperature' and unit:
            spcArray[col] = convertUnit(spcArray[col], unit, 'C')
    scale = centralDict.get('pflotranScale') or {}
    with h5py.File(centralDict['trsptPath'].parent / 'testpycte.h5', 'w') as f:
        f.create_dataset('Cell Ids', data = np.array(list(range(1,len(centralDict['commMtrx'])+1)), dtype = 'i8'))
        for s in spcArray:
            f.create_dataset(s+'i', data = spcArray[s] * scale.get(s, 1.0))
        if sandbox:
            perSecond = secondsPerUnit(centralDict['timeUnit'])
            for prefix, key in (('S_', 'transportSource'), ('K_', 'transportSink')):
                values = reactionArray(centralDict, key) / perSecond
                for j, sp in enumerate(centralDict['transportedSpecies']):
                    f.create_dataset(f'{prefix}{sp}i', data=values[:, j])

    timeKey = (centralDict['dtStep'], centralDict['timeUnit'])
    if centralDict['lStep'] == 0 or _writtenTime.get(str(centralDict['trsptPath'])) != timeKey:
        # PFLOTRAN notation: s, min, h, d, y ('m' would be read as metres)
        update_pflotran_time(centralDict['trsptPath'],centralDict['dtStep'], timeUnits.shortName(centralDict['timeUnit']))
        _writtenTime[str(centralDict['trsptPath'])] = timeKey

    pflotran = executable()
    logPath = centralDict['trsptPath'].parent / 'pflotran.log'
    runStart = time.time()
    with open(logPath, "w") as f:
        ref = time.perf_counter()
        try:
            subprocess.run(
                [
                    "mpirun",
                    "-n",
                    str(centralDict.get("transportThreads", 1)),
                    pflotran,
                    "-pflotranin",
                    centralDict["trsptPath"].name,
                ],
                stdout=f,
                cwd=centralDict["trsptPath"].parent,
                stderr=f,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as error:
            failed = error
        else:
            failed = None
    if failed is not None:
        tail = logPath.read_text(errors='replace').strip().splitlines()[-15:]
        raise RuntimeError(f"PFLOTRAN stopped with exit status {failed.returncode} at time step n°"
                           f"{centralDict['lStep'] + 1} : its log is {logPath} ; last lines :\n" + "\n".join(tail))

    calcTime = time.perf_counter() - ref

    outputFile = lastOutputFile(centralDict['trsptPath'], 'h5', runStart)
    if outputFile is not None:
        df, variableUnits = readHdf5Output(outputFile, centralDict)
    else:
        outputFile = lastOutputFile(centralDict['trsptPath'], 'tec', runStart)
        if outputFile is None:
            raise FileNotFoundError(f"No '{Path(centralDict['trsptPath']).stem}*.h5' or "
                                    f"'{Path(centralDict['trsptPath']).stem}-*.tec' output file found.")
        df, variableUnits = readTecplotOutput(outputFile)

    df = df.drop(columns="Material ID", errors="ignore")
    integralColumns = {m.group(1): name for name in df.columns if (m := re.match(r'^I_(.+)pycte$', name))}
    integrals = df[list(integralColumns.values())].copy()
    df = df.drop(columns=list(integralColumns.values()))
    if sandbox and hasReaction(centralDict):
        missing = [sp for sp in centralDict['transportedSpecies'] if sp not in integralColumns]
        if missing:
            raise ValueError(f"pflotranReactionTerms = 'sandbox' : {outputFile.name} has no column I_<species>pycte for "
                             f"{missing} (the reaction sandbox must integrate c over time into the immobile species "
                             "I_<species>pycte, output in CHEMISTRY / OUTPUT)")
        I = np.column_stack([integrals[integralColumns[sp]].to_numpy(dtype=float)
                             for sp in centralDict['transportedSpecies']]) / secondsPerUnit(centralDict['timeUnit'])
        centralDict['transportReactionIntegral'] = pd.DataFrame(I, index=centralDict['commMtrx'].index,
                                                                columns=list(centralDict['transportedSpecies']))

    targetNames = centralDict['coord'] + list(spcArray.keys())

    # a crossDependency is matched by its quantity (column tk -> temperature, in K)
    targetBaseAndUnit = [extractTargetUnit(quantity.get(t, t)) for t in targetNames]
    matchTargets = [base for base, _ in targetBaseAndUnit]

    varMapping = matchVariables(list(df.columns), matchTargets)   

    df = df[[varMapping[base] for base in matchTargets]]          
    df.columns = targetNames                                      

    for target, (base, desiredUnit) in zip(targetNames, targetBaseAndUnit):
        if desiredUnit is None:
            continue
        sourceName = varMapping[base]
        sourceUnit = variableUnits.get(sourceName)
        df[target] = convertUnit(df[target], sourceUnit, desiredUnit)

    for s in signed:
        df[s] = df[s] - df.pop(s + NEGATIVE_SUFFIX)

    for s, factor in (centralDict.get('pflotranScale') or {}).items():
        if s in df:
            df[s] = df[s] / factor

    return df, calcTime


def trspt(centralDict):
    display.stepPrint("PFLOTRAN", end=" ", flush=True)
    startPflotran = time.perf_counter()

    comm, calcWallClock = transport(centralDict)

    # columns not given to PFLOTRAN (crossDependencies of the speciation, Zz, ...): copied back after the transport
    toAdd = [c for c in centralDict['commMtrx'].columns if c not in comm.columns]
    if toAdd:
        comm = pd.concat([comm,centralDict['commMtrx'][toAdd]], axis=1)
    comm = comm[[c for c in centralDict['commMtrx'].columns if c in comm.columns]]      # columns in the order of commMtrx

    if outputManager.wanted(centralDict, 'transport'):
       outputManager.write(centralDict, comm, 'transport', 'pflotran')

    centralDict.update({
        "commMtrx": comm,
        "PFLOTRANCalcTime_WallClock": centralDict["PFLOTRANCalcTime_WallClock"] + calcWallClock,
        'PFLOTRANInterfTime_WallClock' : centralDict["PFLOTRANInterfTime_WallClock"] + time.perf_counter() - startPflotran - calcWallClock,
        "PFLOTRANCalcTime_ProcessorTime": centralDict["PFLOTRANCalcTime_ProcessorTime"] + calcWallClock,
        "PFLOTRANTotalTime" : centralDict["PFLOTRANTotalTime"] + time.perf_counter() - startPflotran,})


    display.stepPrint(f"({writeTime((time.perf_counter() - startPflotran))})")


    return centralDict