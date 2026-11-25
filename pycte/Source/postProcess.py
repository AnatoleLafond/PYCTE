"""Post-processing of pycte outputs into DataFrames: postProcess, availableSpecies and getTimes."""

import os
import re
import warnings

import pandas as pd

__all__ = ["postProcess", "getTimes", "availableSpecies"]

_COORD_COLS = ("x", "y", "z")
_LOG_NAME = "warning.log"
_SETTINGS_HEADER = "pycte : Settings of the run"


def _stepNumber(fileName):
    """Time step number contained in the file name (PhreeqC_12.txt -> 12)."""
    match = re.search(r"(\d+)", os.path.splitext(os.path.basename(fileName))[0])
    return int(match.group(1)) if match else float("inf")


_OUTPUT_TXT = re.compile(r"^[A-Za-z][\w-]*_\d+\.txt$")      # StepEnd_3.txt, PhreeqC_12.txt (not ic.txt)


def _outputFiles(folder):
    return [f for f in os.listdir(folder) if _OUTPUT_TXT.match(f)]


def _resolveFolders(folder=None):
    """folder -> (output folder, run folder): stepEnd/ first (the outputs at the end of the steps), then speciation/
    and the folders of older versions; a folder of outputs can also be given itself."""
    if folder in (None, "", "."):
        folder = os.getcwd()
    folder = os.path.abspath(os.path.expanduser(str(folder)))

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"pycte.postProcess : folder not found: {folder}")

    for sub in ("stepEnd", "speciation", "Speciation", "CouplingHistory"):
        candidate = os.path.join(folder, sub)
        if os.path.isdir(candidate) and _outputFiles(candidate):
            return candidate, folder

    if _outputFiles(folder):
        return folder, os.path.dirname(folder)

    raise FileNotFoundError(
        f"pycte.postProcess : no output file (<name>_<step>.txt) in {folder} nor in its stepEnd/ or speciation/ "
        "subfolders. Are the outputs requested with pycte.output(...), and did the run complete ?"
    )


def _readOutput(path):
    """One output file -> DataFrame (tab-separated, whitespace for old files)."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        header = next((line for line in handle if line.strip() and not line.lstrip().startswith("%")), "")
    return pd.read_csv(path, sep="\t" if "\t" in header else r"\s+", comment="%", header=0)


def _loadOutputs(outputFolder):
    """Output files sorted by step -> [(step, file name, DataFrame)]."""
    files = sorted(_outputFiles(outputFolder), key=_stepNumber)

    outputs = []
    for position, fileName in enumerate(files, start=1):
        path = os.path.join(outputFolder, fileName)
        try:
            frame = _readOutput(path)
        except Exception as error:
            warnings.warn(f"pycte.postProcess : cannot read {fileName} ({error})")
            continue
        if frame.empty:
            continue
        step = _stepNumber(fileName)
        if step == float("inf"):
            step = position
        outputs.append((step, fileName, frame))

    return outputs


def _readTimeSteps(runFolder, logName=_LOG_NAME):
    """(times, unit) written in warning.log, or (None, None)."""
    logPath = os.path.join(runFolder, logName)
    if not os.path.isfile(logPath):
        return None, None

    with open(logPath, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.read().splitlines()

    for index, line in enumerate(lines):
        header = re.search(r"Time steps\s*\(([^)]*)\)", line)
        if not header:
            continue
        unit = header.group(1).strip()
        chunks = []
        for nextLine in lines[index + 1:]:
            stripped = nextLine.strip()
            if not stripped:
                break
            if not re.fullmatch(r"[-+0-9eE.,\s]+", stripped):
                break
            chunks.append(stripped.rstrip("."))
            if stripped.endswith("."):
                break
        try:
            times = [float(v) for v in ",".join(chunks).split(",") if v.strip()]
        except ValueError:
            return None, None
        return (times or None), unit

    return None, None


def _physicalTime(times, step, offset):
    """Physical time of output file n, or None."""
    if not times:
        return None
    if offset and step == 0:
        return 0.0
    index = step - offset
    if 0 <= index < len(times):
        return times[index]
    return None


def _selectRow(frame, cell, x):
    """Selects the cell used for the breakthrough."""
    if x is not None:
        if "x" not in frame.columns:
            raise KeyError(
                "pycte.postProcess : x= given but the output files have "
                "no 'x' column."
            )
        position = (frame["x"] - float(x)).abs().idxmin()
        return frame.loc[position]
    if not isinstance(cell, int) or isinstance(cell, bool):
        raise TypeError(
            f"pycte.postProcess : cell= expects an integer (got {cell!r})."
        )
    if not -len(frame) <= cell < len(frame):
        raise IndexError(
            f"pycte.postProcess : cell {cell} does not exist. "
            f"Available: 0 to {len(frame) - 1} (or -1 to {-len(frame)})."
        )
    return frame.iloc[cell]


def _unpackRequest(value):
    """breakthrough= / columnProfile= value -> (species, folder, n)."""
    if value is None:
        return None, None, None
    if isinstance(value, str):
        return [value], None, None
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            species = list(value[0])
            folder = value[1] if len(value) > 1 else None
            step = value[2] if len(value) > 2 else None
            if len(value) > 3:
                raise TypeError(
                    "pycte.postProcess : at most 3 elements expected "
                    "([species], folder, t or cell)."
                )
            return species, folder, step
        return list(value), None, None
    raise TypeError(
        f"pycte.postProcess : invalid value {value!r}. Expected a list "
        f"of species, or a tuple ([species], folder[, t or cell])."
    )


def _resolveSpecies(requested, columns):
    """Validates the requested species list against the available columns."""
    available = [c for c in columns if c not in _COORD_COLS]
    if requested is None:
        return available
    if isinstance(requested, str):
        requested = [requested]
    species = list(requested)
    unknown = [s for s in species if s not in columns]
    if unknown:
        raise KeyError(
            f"pycte.postProcess : unknown species {unknown}. "
            f"Available: {available}"
        )
    return species


def _applyRename(frame, rename):
    """Renames the columns of the result (dict {old: new})."""
    if not rename:
        return frame
    if not isinstance(rename, dict):
        raise TypeError(
            "pycte : rename= expects a dictionary {'old name': 'new name'}."
        )
    unknown = [k for k in rename if k not in frame.columns]
    if unknown:
        warnings.warn(
            f"pycte : rename= ignores {unknown}, not in the result. "
            f"Available columns: {list(frame.columns)}"
        )
    return frame.rename(columns=rename)


def _uniqueColumns(new, base, key=None):
    """Suffixes (_2, _3, ...) the columns of `new` already present in `base`."""
    renaming = {}
    for column in new.columns:
        if column == key or column not in base.columns:
            continue
        candidate, index = column, 2
        while candidate in base.columns or candidate in renaming.values():
            candidate = f"{column}_{index}"
            index += 1
        renaming[column] = candidate
    if renaming:
        warnings.warn(
            f"pycte : column(s) already present in the target DataFrame, "
            f"renamed: {renaming}. Use rename= to choose the names."
        )
    return new.rename(columns=renaming)


def _appendColumns(base, new):
    """Adds the columns of `new` to `base`, aligned on 't' or 'x' when possible."""
    if not isinstance(base, pd.DataFrame):
        raise TypeError(
            "pycte.postProcess : append= expects an existing DataFrame "
            f"(got {type(base).__name__})."
        )

    keys = [k for k in ("t", "x") if k in base.columns and k in new.columns]
    key = keys[0] if keys else None

    if key is None:
        new = _uniqueColumns(new, base)
        return pd.concat([base.reset_index(drop=True),
                          new.reset_index(drop=True)], axis=1)

    baseIndexed = base.set_index(key)
    newIndexed = new.set_index(key)
    common = baseIndexed.index.intersection(newIndexed.index)
    redundant = []
    for column in new.columns:
        if column == key or column not in base.columns:
            continue
        if len(common) and baseIndexed.loc[common, column].reset_index(drop=True).equals(
            newIndexed.loc[common, column].reset_index(drop=True)
        ):
            redundant.append(column)
    new = new.drop(columns=redundant)

    new = _uniqueColumns(new, base, key=key)
    merged = base.merge(new, on=key, how="outer").sort_values(key)
    return merged.reset_index(drop=True)


TOTAL_PREFIX = "Tot!_"

_ELEMENTS = frozenset((
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co "
    "Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te "
    "I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir "
    "Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr"
).split())
_NOT_SPECIES = frozenset((
    "t", "time", "ph", "pe", "eh", "chargebalance", "charge", "mu", "ionicstrength",
    "ionic_strength", "temperature", "temp", "tc", "porosity", "saturation", "alkalinity",
))
_SOLVENT = frozenset(("H2O", "e-"))
_DATABASE_EXTENSIONS = (".dat", ".inp", ".phrq", ".txt")
_databaseCache = {}


def _formulaElements(name):
    """Element composition of a species formula, or None."""
    formula = re.sub(r"\(aq\)$", "", str(name).strip())
    formula = re.sub(r"(\+\d+|-\d+|\++|-+)$", "", formula)
    if not formula:
        return None
    stack = [{}]
    index = 0
    while index < len(formula):
        char = formula[index]
        if char in "([":
            stack.append({})
            index += 1
            continue
        if char in ")]":
            if len(stack) == 1:
                return None
            digits = re.match(r"\d*", formula[index + 1:]).group()
            factor = int(digits) if digits else 1
            group = stack.pop()
            for element, count in group.items():
                stack[-1][element] = stack[-1].get(element, 0) + count * factor
            index += 1 + len(digits)
            continue
        match = re.match(r"([A-Z][a-z]?)(\d*)", formula[index:])
        if not match:
            return None
        element, digits = match.groups()
        if element not in _ELEMENTS:
            return None
        count = int(digits) if digits else 1
        stack[-1][element] = stack[-1].get(element, 0) + count
        index += len(element) + len(digits)
    if len(stack) != 1 or not stack[0]:
        return None
    return stack[0]


def _readText(path):
    """Content of a database file, whatever its encoding."""
    with open(path, "rb") as handle:
        raw = handle.read()
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _phreeqcAqueous(text):
    """Aqueous species of a PhreeqC database (SOLUTION_SPECIES and masters)."""
    from .extractDB import _PHREEQC_KEYWORDS, _parse_reaction_side

    species = set()
    block = None
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        if tokens[0].upper() in _PHREEQC_KEYWORDS and "=" not in line:
            block = tokens[0].upper()
            continue
        if block == "SOLUTION_MASTER_SPECIES" and len(tokens) >= 2:
            species.add(tokens[1])
        elif block == "SOLUTION_SPECIES" and "=" in line:
            left, right = line.split("=", 1)
            for _, name in _parse_reaction_side(left) + _parse_reaction_side(right):
                species.add(name)
    return species - _SOLVENT


def _loadDatabase(path):
    """PhreeqC .dat or ORCHESTRA .inp database for the totals, or None."""
    path = os.path.abspath(path)
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return None
    if key in _databaseCache:
        return _databaseCache[key]

    text = _readText(path)
    database = None
    if "@species" in text or "@logKreaction" in text:
        from . import parse_reactions

        declared = {split[0].strip() for split in (
            parse_reactions.split_top_level(call)
            for call in parse_reactions.parse_tagged_calls(text, "species")) if split}
        sorbed = set(parse_reactions.build_surface_species(path))
        components = parse_reactions.build_components(path)
        database = {
            "path": path,
            "kind": "ORCHESTRA",
            "aqueous": declared - sorbed - _SOLVENT,
            "orchestra": (parse_reactions.build_reaction_dict(path), components),
        }
    elif re.search(r"^\s*SOLUTION_SPECIES\b", text, re.MULTILINE | re.IGNORECASE):
        database = {
            "path": path,
            "kind": "PhreeqC",
            "aqueous": _phreeqcAqueous(text),
            "orchestra": None,
        }
    _databaseCache[key] = database
    return database


def _loggedDatabase(runFolder, logName=_LOG_NAME):
    """Database path written in warning.log by the run, or None."""
    logPath = os.path.join(runFolder, logName)
    if not os.path.isfile(logPath):
        return None
    with open(logPath, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = re.match(r"^pycte : Database : (.+)$", line.strip())
            if match:
                return match.group(1).strip()
    return None


def _findDatabase(database, runFolder, columns):
    """Database for the totals: argument, warning.log, then run folder."""
    if database is not None:
        candidates = [database, os.path.join(runFolder, str(database))]
        for candidate in candidates:
            if os.path.isfile(candidate):
                loaded = _loadDatabase(candidate)
                if loaded is None:
                    raise ValueError(
                        f"pycte.postProcess : {candidate} is neither a PhreeqC "
                        f"(SOLUTION_SPECIES) nor an ORCHESTRA (@species) database."
                    )
                return loaded
        raise FileNotFoundError(f"pycte.postProcess : database not found: {database}")

    logged = _loggedDatabase(runFolder)
    if logged:
        for candidate in (logged, os.path.join(runFolder, logged)):
            if os.path.isfile(candidate):
                loaded = _loadDatabase(candidate)
                if loaded is not None:
                    return loaded

    scored = []
    if os.path.isdir(runFolder):
        for fileName in sorted(os.listdir(runFolder)):
            path = os.path.join(runFolder, fileName)
            if not fileName.lower().endswith(_DATABASE_EXTENSIONS) or not os.path.isfile(path):
                continue
            loaded = _loadDatabase(path)
            if loaded is None:
                continue
            score = len(loaded["aqueous"].intersection(columns))
            if score:
                scored.append((score, loaded))
    if not scored:
        return None
    bestScore = max(score for score, _ in scored)
    best = [loaded for score, loaded in scored if score == bestScore]
    if len(best) > 1:
        warnings.warn(
            f"pycte.postProcess : several databases fit the outputs "
            f"{[os.path.basename(d['path']) for d in best]}, "
            f"{os.path.basename(best[0]['path'])} is used. Use database= to choose."
        )
    return best[0]


def _looksAqueous(column):
    """Without database: is this output column an aqueous species?"""
    if column in _COORD_COLS or column in _SOLVENT or "_" in column:
        return False
    if column.lower() in _NOT_SPECIES:
        return False
    return _formulaElements(column) is not None


def _totalTarget(name, components):
    """Component summed by Tot!_<name>."""
    if name in components:
        return name
    elements = _formulaElements(name)
    if elements:
        others = [e for e in elements if e not in ("H", "O")] or list(elements)
        if len(others) == 1 and others[0] in components:
            return others[0]
    raise KeyError(
        f"pycte.postProcess : {TOTAL_PREFIX}{name} not understood. Give an "
        f"element or a component, e.g. {TOTAL_PREFIX}Ca, {TOTAL_PREFIX}C, "
        f"{TOTAL_PREFIX}CO3-2. Available: {sorted(components)}"
    )


def _totalCoefficients(totals, columns, database):
    """{'Tot!_X': {aqueous species: stoichiometric coefficient of X}}."""
    if database is not None:
        aqueous = [c for c in columns if c in database["aqueous"]]
    else:
        aqueous = [c for c in columns if _looksAqueous(c)]

    if database is not None and database["kind"] == "ORCHESTRA":
        from . import parse_reactions

        reactions, components = database["orchestra"]
        stoich = parse_reactions.build_stoich_matrix(
            reactions, components, aqueous, filepath=database["path"], verbose=False)
        composition = {s: {c: v for c, v in stoich.loc[s].items() if v} for s in aqueous}
    else:
        composition = {}
        unreadable = []
        for specie in aqueous:
            elements = _formulaElements(specie)
            if elements is None:
                unreadable.append(specie)
            else:
                composition[specie] = elements
        if unreadable:
            warnings.warn(
                f"pycte.postProcess : formula of {unreadable} not readable, "
                f"left out of the {TOTAL_PREFIX} totals."
            )

    components = {c for elements in composition.values() for c in elements}
    coefficients = {}
    for total in totals:
        target = _totalTarget(total[len(TOTAL_PREFIX):], components)
        contributions = {s: float(elements[target]) for s, elements in composition.items()
                         if elements.get(target)}
        coefficients[total] = contributions
    return coefficients


def _addTotals(outputs, totals, database):
    """Adds the Tot!_X columns to every output frame."""
    columns = list(dict.fromkeys(c for _, _, frame in outputs for c in frame.columns))
    coefficients = _totalCoefficients(totals, columns, database)
    completed = []
    for step, fileName, frame in outputs:
        frame = frame.copy()
        for total, contributions in coefficients.items():
            species = list(contributions)
            values = frame.reindex(columns=species, fill_value=0.0).to_numpy(dtype=float)
            frame[total] = values @ [contributions[s] for s in species] if species else 0.0
        completed.append((step, fileName, frame))
    return completed, coefficients


def availableSpecies(folder=None):
    """Lists the species (columns other than x/y/z) present in the outputs."""
    outputFolder, _ = _resolveFolders(folder)
    outputs = _loadOutputs(outputFolder)
    if not outputs:
        return []
    return [c for c in outputs[0][2].columns if c not in _COORD_COLS]


def postProcess(species=None, folder=None, breakthrough=None,
                columnProfile=None, t=None, cell=-1, x=None,
                rename=None, append=None, database=None):
    """Outputs of a run -> DataFrame: breakthrough (one cell, every step, default) or columnProfile (every cell, step t).

        df = pycte.postProcess(['Ca+2', 'Tot!_C'], folder, cell=-1)
        df = pycte.postProcess(columnProfile=(['Ca+2'], folder, 12))"""
    if breakthrough is not None and columnProfile is not None:
        raise TypeError(
            "pycte.postProcess : breakthrough= and columnProfile= are "
            "mutually exclusive, choose one mode."
        )

    mode = "profile" if columnProfile is not None else "breakthrough"
    request = columnProfile if mode == "profile" else breakthrough

    requestedSpecies, requestedFolder, requestedStep = _unpackRequest(request)
    if requestedSpecies is None:
        requestedSpecies = species
    if requestedFolder is None:
        requestedFolder = folder
    if requestedStep is not None:
        if mode == "profile":
            t = requestedStep
        else:
            cell = requestedStep

    outputFolder, runFolder = _resolveFolders(requestedFolder)
    outputs = _loadOutputs(outputFolder)
    if not outputs:
        raise FileNotFoundError(
            f"pycte.postProcess : no usable output file in "
            f"{outputFolder}."
        )

    requestedList = [requestedSpecies] if isinstance(requestedSpecies, str) else requestedSpecies
    totals = list(dict.fromkeys(
        s for s in (requestedList or []) if str(s).startswith(TOTAL_PREFIX)))
    usedDatabase, totalsDetail = None, {}
    if totals:
        columns = list(dict.fromkeys(c for _, _, frame in outputs for c in frame.columns))
        usedDatabase = _findDatabase(database, runFolder, columns)
        if usedDatabase is None:
            warnings.warn(
                f"pycte.postProcess : no database found for {totals}, the aqueous "
                f"species are guessed from their formulas. Use database= to give it."
            )
        outputs, totalsDetail = _addTotals(outputs, totals, usedDatabase)

    speciesList = _resolveSpecies(requestedSpecies, list(outputs[0][2].columns))
    times, timeUnit = _readTimeSteps(runFolder)
    steps = [step for step, _, _ in outputs]
    offset = 0 if os.path.basename(outputFolder) == "CouplingHistory" else 1

    if mode == "breakthrough":
        result = _breakthroughFrame(outputs, speciesList, times, offset, cell, x)
        selection = cell if x is None else f"x~{x}"
    else:
        result, selection = _profileFrame(outputs, speciesList, times, offset, t)

    result.attrs["folder"] = runFolder
    result.attrs["outputFolder"] = outputFolder
    result.attrs["timeUnit"] = timeUnit
    result.attrs["mode"] = mode
    result.attrs["selection"] = selection
    if totals:
        result.attrs["database"] = usedDatabase["path"] if usedDatabase else None
        result.attrs["totals"] = totalsDetail

    result = _applyRename(result, rename)
    if append is not None:
        result = _appendColumns(append, result)
    return result


def _breakthroughFrame(outputs, speciesList, times, offset, cell, x):
    """Value of one cell for each time step."""
    rows = []
    for step, fileName, frame in outputs:
        missing = [s for s in speciesList if s not in frame.columns]
        if missing:
            warnings.warn(
                f"pycte.postProcess : {fileName} does not contain {missing}, skipped."
            )
            continue
        selected = _selectRow(frame, cell, x)
        row = {"t": step}
        physical = _physicalTime(times, step, offset)
        if physical is not None:
            row["time"] = physical
        for specie in speciesList:
            row[specie] = selected[specie]
        rows.append(row)

    result = pd.DataFrame(rows)
    ordered = ["t"] + (["time"] if "time" in result.columns else []) + speciesList
    return result[ordered].sort_values("t").reset_index(drop=True)


def _profileFrame(outputs, speciesList, times, offset, t):
    """Spatial profile: all cells, at one (or several) time steps."""
    byStep = {step: (fileName, frame) for step, fileName, frame in outputs}
    steps = sorted(byStep)

    single = not isinstance(t, (list, tuple, set))
    wanted = [t] if single else list(t)
    if wanted == [None]:
        wanted = [steps[-1]]

    resolved = []
    for value in wanted:
        if value is None:
            resolved.append(steps[-1])
        elif value in byStep:
            resolved.append(value)
        elif isinstance(value, int) and value < 0 and -value <= len(steps):
            resolved.append(steps[value])
        else:
            raise KeyError(
                f"pycte.postProcess : time step {value!r} not found. "
                f"Available: {steps[0]} to {steps[-1]}."
            )

    frames = []
    for step in resolved:
        fileName, frame = byStep[step]
        missing = [s for s in speciesList if s not in frame.columns]
        if missing:
            raise KeyError(
                f"pycte.postProcess : {fileName} does not contain {missing}."
            )
        block = pd.DataFrame()
        if not single:
            block["t"] = [step] * len(frame)
            physical = _physicalTime(times, step, offset)
            if physical is not None:
                block["time"] = physical
        block["x"] = frame["x"] if "x" in frame.columns else range(len(frame))
        for specie in speciesList:
            block[specie] = frame[specie].values
        frames.append(block)

    result = pd.concat(frames, ignore_index=True)
    return result, (resolved[0] if single else resolved)


_SPECIATION_SOLVERS = ["PhreeqC", "xGEMS", "ORCHESTRA", "nativeKinetics"]
_TRANSPORT_SOLVERS = ["COMSOL", "nativeTransport", "PFLOTRAN"]

_SOLVER_LABELS = {}
for _name in _SPECIATION_SOLVERS:
    _SOLVER_LABELS[_name] = "speciation"
for _name in _TRANSPORT_SOLVERS:
    _SOLVER_LABELS[_name] = "transport"
del _name

_KNOWN_ROLES = set(_SOLVER_LABELS.values())

_UNIT_TO_SEC = {
    "msec": 1.0 / 1000.0,
    "sec": 1.0,
    "min": 60.0,
    "h": 3600.0,
    "d": 86400.0,
}
_VALUE_RE = re.compile(r"([\d]+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(msec|sec|min|h|d)\b")


def _normalizeText(text):
    """Replaces any known solver name with its generic label."""
    for name, label in sorted(_SOLVER_LABELS.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(rf"\b{re.escape(name)}\b", label, text)
    return text


_SHORT_PATTERNS = [
    (r"total coupling calculation time", "TotalCoupling"),
    (r"communication time between", "Communication"),
    (r"wall clock time of total .*calculation", "WallClockTotal"),
    (r"processor time of total .*calculation", "ProcTimeTotal"),
    (r"wall clock time of interfacing data", "WallClockInterf"),
    (r"processor time of interfacing data", "ProcTimeInterf"),
    (r"licence waiting time", "LicenceWaiting"),
    (r"database extraction", "DBextraction"),
    (r"initialisation time", "Init"),
    (r"^total (speciation|transport) time$", "Total"),
]

_SHORT_STOPWORDS = {"time", "of", "the", "a", "an", "in", "for", "during", "total"}


def _shortColumn(description, module):
    """Short column name of a warning.log time label."""
    role = module.capitalize() if module in _KNOWN_ROLES else ""
    text = description.strip().lower()

    stem = None
    for pattern, replacement in _SHORT_PATTERNS:
        if re.search(pattern, text):
            stem = replacement
            break

    if stem is None:
        words = [w for w in re.split(r"[^0-9A-Za-z]+", description) if w]
        words = [w for w in words if w.lower() not in _SHORT_STOPWORDS]
        stem = "".join(w[:1].upper() + w[1:] for w in words) or "Time"

    return f"{stem}{role}(sec)"


def _parseLogFile(path, short=True):
    """warning.log -> (metrics, warnings count, errors, coupling completed)."""
    rawMetrics = []
    warningsCount = None
    errors = []
    couplingCompleted = False

    lines = []
    inSettings = False
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if inSettings:
                inSettings = bool(stripped)
            elif stripped.startswith(_SETTINGS_HEADER):
                inSettings = True
            elif stripped:
                lines.append(stripped)

    for line in lines:
        head = re.match(r"^(.*?)\s:\s(.*)$", line)
        if not head:
            continue
        module = _normalizeText(head.group(1).strip())
        remainder = head.group(2).strip()
        if remainder.startswith("Database :"):
            continue

        value = _VALUE_RE.search(remainder)
        if value:
            valueSec = float(value.group(1)) * _UNIT_TO_SEC[value.group(2)]
            description = remainder[: value.start()].strip().rstrip(":").strip()
            rawMetrics.append((module, _normalizeText(description), valueSec))
            continue

        description = _normalizeText(remainder)
        if "coupling completed" in description.lower():
            couplingCompleted = True
        warningMatch = re.search(r"(\d+)\s+warning", description, re.IGNORECASE)
        if warningMatch:
            warningsCount = int(warningMatch.group(1))
        if "error" in description.lower():
            errors.append(f"{module} : {description}")

    descriptionCounts = {}
    for module, description, _ in rawMetrics:
        if module in _KNOWN_ROLES:
            continue
        descriptionCounts[description] = descriptionCounts.get(description, 0) + 1

    metrics = {}
    for module, description, valueSec in rawMetrics:
        if short:
            column = _shortColumn(description, module)
        elif module in _KNOWN_ROLES or descriptionCounts.get(description, 0) > 1:
            column = f"{description} ({module}) (sec)"
        else:
            column = f"{description} (sec)"
        while column in metrics:
            column = column.replace("(sec)", f"({module})(sec)")
            if column in metrics:
                column = column.replace("(sec)", "_bis(sec)")
        metrics[column] = valueSec

    return metrics, warningsCount, errors, couplingCompleted


def getTimes(folder=None, logName=_LOG_NAME, short=True,
             rename=None, append=None, label=None):
    """Computation times of a run, read from warning.log -> one-row DataFrame."""
    if folder in (None, "", "."):
        folder = os.getcwd()
    folder = os.path.abspath(os.path.expanduser(str(folder)))

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"pycte.getTimes : folder not found: {folder}")

    logPath = os.path.join(folder, logName)
    if not os.path.isfile(logPath):
        parentLog = os.path.join(os.path.dirname(folder), logName)
        if os.path.isfile(parentLog):
            folder = os.path.dirname(folder)
            logPath = parentLog
        else:
            raise FileNotFoundError(
                f"pycte.getTimes : {logName} not found in {folder}."
            )

    metrics, warningsCount, errors, couplingCompleted = _parseLogFile(logPath, short=short)

    row = dict(metrics)
    row["warnings_count"] = warningsCount
    row["errors_count"] = len(errors)
    row["errors_detail"] = "; ".join(errors) if errors else ""
    row["coupling_completed"] = couplingCompleted

    result = pd.DataFrame([row])
    if label is not None:
        result.insert(0, "run", label)
    result.attrs["folder"] = folder

    result = _applyRename(result, rename)
    if append is not None:
        if not isinstance(append, pd.DataFrame):
            raise TypeError(
                "pycte.getTimes : append= expects an existing DataFrame "
                f"(got {type(append).__name__})."
            )
        result = pd.concat([append, result], ignore_index=True, sort=False)
    return result
