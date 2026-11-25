"""
pycte.postProcess / pycte.getTimes
==================================

Post-traitement des sorties d'un run pycte, sans Streamlit : tout est
rendu sous forme de pandas.DataFrame.

    import pycte

    # percee : une maille, tous les pas de temps
    df = pycte.postProcess(breakthrough=(['Ca+2', 'Cl-'], nomFolder))
    df = pycte.postProcess(['Ca+2', 'Cl-'], nomFolder)      # equivalent

    # profil spatial : toutes les mailles, un pas de temps
    df = pycte.postProcess(columnProfile=(['Ca+2', 'Cl-'], nomFolder))
    df = pycte.postProcess(columnProfile=(['Ca+2'], nomFolder, 12))

    # temps de calcul lus dans warning.log
    dft = pycte.getTimes()          # dossier courant par defaut

`folder` est auto-detecte : on accepte aussi bien le dossier de run
(celui qui contient le sous-dossier `Speciation/` et `warning.log`) que
directement un dossier de sorties (`Speciation/`, `CouplingHistory/`,
`PrimarySpecies/`, ...). Par defaut, c'est le dossier courant.
"""

import os
import re
import warnings

import pandas as pd

__all__ = ["postProcess", "getTimes", "availableSpecies"]

_COORD_COLS = ("x", "y", "z")
_LOG_NAME = "warning.log"


def _stepNumber(fileName):
    """Numero de pas de temps contenu dans le nom de fichier (PhreeqC_12.txt -> 12)."""
    match = re.search(r"(\d+)", os.path.splitext(os.path.basename(fileName))[0])
    return int(match.group(1)) if match else float("inf")


def _resolveFolders(folder=None):
    """
    Retourne (outputFolder, runFolder).

    `folder` peut etre :
      - None / "" / "."   -> dossier courant
      - le dossier de run -> on utilise son sous-dossier speciation/, a defaut coupling/
        (anciens calculs : Speciation/, puis CouplingHistory/)
      - un dossier de sorties (speciation, coupling, transport, ...) -> tel quel
    """
    if folder in (None, "", "."):
        folder = os.getcwd()
    folder = os.path.abspath(os.path.expanduser(str(folder)))

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"pycte.postProcess : dossier introuvable : {folder}")

    for sub in ("speciation", "coupling", "Speciation", "CouplingHistory"):
        candidate = os.path.join(folder, sub)
        if os.path.isdir(candidate) and any(
            f.endswith(".txt") for f in os.listdir(candidate)
        ):
            return candidate, folder

    if any(f.endswith(".txt") for f in os.listdir(folder)):
        return folder, os.path.dirname(folder)

    raise FileNotFoundError(
        f"pycte.postProcess : aucun fichier .txt trouve dans {folder} "
        f"ni dans ses sous-dossiers speciation/ ou coupling/. Les sorties "
        f"sont-elles demandees par le mot-cle output ?"
    )


def _loadOutputs(outputFolder):
    """Charge tous les .txt de sortie, tries par numero de pas de temps.

    Retourne une liste de tuples (step, fileName, DataFrame).
    """
    files = sorted(
        (f for f in os.listdir(outputFolder) if f.endswith(".txt")),
        key=_stepNumber,
    )

    outputs = []
    for position, fileName in enumerate(files, start=1):
        path = os.path.join(outputFolder, fileName)
        try:
            frame = pd.read_csv(path, sep=r"\s+", comment="%", header=0)
        except Exception as error:
            warnings.warn(f"pycte.postProcess : lecture impossible de {fileName} ({error})")
            continue
        if frame.empty:
            continue
        step = _stepNumber(fileName)
        if step == float("inf"):
            step = position
        outputs.append((step, fileName, frame))

    return outputs


def _readTimeSteps(runFolder, logName=_LOG_NAME):
    """Lit la liste des temps cumules (et leur unite) ecrite dans warning.log.

    Retourne (times, unit) ou (None, None) si le log est absent/illisible.
    """
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
    """Temps physique associe a un pas de temps, ou None.

    Fichier n : temps dtpycte[n - 1] (n >= 1), temps 0 pour n = 0 (etat initial). offset = 0 : anciens fichiers
    CouplingHistory/Coupling_k.txt, numerotes a partir de 0 (fichier k : temps dtpycte[k]).
    """
    if not times:
        return None
    if offset and step == 0:
        return 0.0
    index = step - offset
    if 0 <= index < len(times):
        return times[index]
    return None


def _selectRow(frame, cell, x):
    """Selectionne la maille utilisee pour la percee."""
    if x is not None:
        if "x" not in frame.columns:
            raise KeyError(
                "pycte.postProcess : argument x= fourni mais les fichiers de "
                "sortie n'ont pas de colonne 'x'."
            )
        position = (frame["x"] - float(x)).abs().idxmin()
        return frame.loc[position]
    return frame.iloc[cell]


def _unpackRequest(value):
    """
    Decode la valeur passee a breakthrough= / columnProfile=.

    Formes acceptees :
        ['Ca+2', 'Cl-']                  -> (especes, None, None)
        'Ca+2'                           -> (especes, None, None)
        (['Ca+2'], folder)               -> (especes, folder, None)
        (['Ca+2'], folder, t)            -> (especes, folder, t)
    """
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
                    "pycte.postProcess : au plus 3 elements attendus "
                    "([especes], folder, t)."
                )
            return species, folder, step
        return list(value), None, None
    raise TypeError(
        f"pycte.postProcess : valeur invalide {value!r}. Attendu une liste "
        f"d'especes, ou un tuple ([especes], folder[, t])."
    )


def _resolveSpecies(requested, columns):
    """Valide la liste d'especes demandee contre les colonnes disponibles."""
    available = [c for c in columns if c not in _COORD_COLS]
    if requested is None:
        return available
    if isinstance(requested, str):
        requested = [requested]
    species = list(requested)
    unknown = [s for s in species if s not in columns]
    if unknown:
        raise KeyError(
            f"pycte.postProcess : espece(s) inconnue(s) {unknown}. "
            f"Disponibles : {available}"
        )
    return species


def _applyRename(frame, rename):
    """Renomme les colonnes du resultat (dict {ancien: nouveau})."""
    if not rename:
        return frame
    if not isinstance(rename, dict):
        raise TypeError(
            "pycte : rename= attend un dictionnaire {'ancien nom': 'nouveau nom'}."
        )
    unknown = [k for k in rename if k not in frame.columns]
    if unknown:
        warnings.warn(
            f"pycte : rename= ignore {unknown}, absent(s) du resultat. "
            f"Colonnes disponibles : {list(frame.columns)}"
        )
    return frame.rename(columns=rename)


def _uniqueColumns(new, base, key=None):
    """Suffixe (_2, _3, ...) les colonnes de `new` deja presentes dans `base`."""
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
            f"pycte : colonne(s) deja presente(s) dans le DataFrame d'accueil, "
            f"renommee(s) : {renaming}. Utilise rename= pour choisir les noms."
        )
    return new.rename(columns=renaming)


def _appendColumns(base, new):
    """Ajoute les colonnes de `new` a un DataFrame existant.

    L'alignement se fait sur 't' (percee) ou 'x' (profil) si la colonne est
    presente des deux cotes, sinon simplement position par position.
    """
    if not isinstance(base, pd.DataFrame):
        raise TypeError(
            "pycte.postProcess : append= attend un DataFrame existant "
            f"(recu {type(base).__name__})."
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


def availableSpecies(folder=None):
    """Liste les especes (colonnes hors x/y/z) presentes dans les sorties."""
    outputFolder, _ = _resolveFolders(folder)
    outputs = _loadOutputs(outputFolder)
    if not outputs:
        return []
    return [c for c in outputs[0][2].columns if c not in _COORD_COLS]


def postProcess(species=None, folder=None, breakthrough=None,
                columnProfile=None, t=None, cell=-1, x=None,
                rename=None, append=None):
    """
    Post-traitement d'un run pycte -> pandas.DataFrame.

    Deux modes, choisis par le mot-cle utilise :

    percee (defaut) : une maille, tous les pas de temps
        df = pycte.postProcess(breakthrough=(['Ca+2', 'Cl-'], nomFolder))
        df = pycte.postProcess(['Ca+2', 'Cl-'], nomFolder)   # equivalent
        -> colonnes : t, time (si warning.log lisible), puis une par espece

    profil de colonne : toutes les mailles, un pas de temps
        df = pycte.postProcess(columnProfile=(['Ca+2', 'Cl-'], nomFolder))
        df = pycte.postProcess(columnProfile=(['Ca+2'], nomFolder, 12))
        -> colonnes : x, puis une par espece
        -> t peut aussi etre une liste de pas : df contient alors les
           colonnes t, time, x, puis une par espece.

    Parametres
    ----------
    species, folder :
        utilisables en positionnel, ex. postProcess(['Ca+2'], nomFolder).
        Ils sont ignores si breakthrough= ou columnProfile= fournit
        deja ses propres valeurs.
    breakthrough : list | str | tuple
        Mode percee. [especes] ou ([especes], folder[, t]).
    columnProfile : list | str | tuple
        Mode profil spatial. [especes] ou ([especes], folder[, t]).
    t : int | list[int] | None
        Mode profil : pas de temps voulu (defaut : le dernier ; -1 = le
        dernier, -2 l'avant-dernier, ...). Ignore en mode percee.
    cell : int
        Mode percee : position de la maille (defaut -1, derniere maille,
        c'est-a-dire la sortie de colonne ; 0 = premiere maille).
    x : float | None
        Mode percee : alternative a `cell`, abscisse voulue ; la maille la
        plus proche est utilisee.
    rename : dict | None
        Renomme les colonnes du resultat, ex.
        rename={'Ca+2': 'Calcium', 'x': 'Position (m)'}.
    append : pandas.DataFrame | None
        Ajoute les colonnes du resultat a ce DataFrame existant au lieu
        d'en rendre un nouveau, ex.
        df = pycte.postProcess(breakthrough=(['Cl-'], run2), append=df).
        L'alignement se fait sur 't' (percee) ou 'x' (profil) ; une
        colonne homonyme deja presente est suffixee _2, _3, ...

    Les especes valent None par defaut = toutes les colonnes disponibles
    (hors coordonnees x/y/z).
    """
    if breakthrough is not None and columnProfile is not None:
        raise TypeError(
            "pycte.postProcess : breakthrough= et columnProfile= sont "
            "exclusifs, choisis un seul mode."
        )

    mode = "profile" if columnProfile is not None else "breakthrough"
    request = columnProfile if mode == "profile" else breakthrough

    requestedSpecies, requestedFolder, requestedStep = _unpackRequest(request)
    if requestedSpecies is None:
        requestedSpecies = species
    if requestedFolder is None:
        requestedFolder = folder
    if requestedStep is not None:
        t = requestedStep

    outputFolder, runFolder = _resolveFolders(requestedFolder)
    outputs = _loadOutputs(outputFolder)
    if not outputs:
        raise FileNotFoundError(
            f"pycte.postProcess : aucun fichier de sortie exploitable dans "
            f"{outputFolder}."
        )

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

    result = _applyRename(result, rename)
    if append is not None:
        result = _appendColumns(append, result)
    return result


def _breakthroughFrame(outputs, speciesList, times, offset, cell, x):
    """Valeur d'une maille pour chaque pas de temps."""
    rows = []
    for step, fileName, frame in outputs:
        missing = [s for s in speciesList if s not in frame.columns]
        if missing:
            warnings.warn(
                f"pycte.postProcess : {fileName} ne contient pas {missing}, ignore."
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
    """Profil spatial : toutes les mailles, a un (ou plusieurs) pas de temps."""
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
                f"pycte.postProcess : pas de temps {value!r} absent. "
                f"Disponibles : {steps[0]} a {steps[-1]}."
            )

    frames = []
    for step in resolved:
        fileName, frame = byStep[step]
        missing = [s for s in speciesList if s not in frame.columns]
        if missing:
            raise KeyError(
                f"pycte.postProcess : {fileName} ne contient pas {missing}."
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
    """Remplace tout nom de solveur connu par son etiquette generique."""
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
    """Nom de colonne court a partir de la description du log.

    "Wall clock time of total speciation calculation" + module "speciation"
    -> "WallClockTotalSpeciation(sec)".
    """
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
    """Parse un warning.log pycte.

    Retourne (metrics, warningsCount, errors, couplingCompleted).
    """
    rawMetrics = []
    warningsCount = None
    errors = []
    couplingCompleted = False

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    for line in lines:
        head = re.match(r"^(.*?)\s:\s(.*)$", line)
        if not head:
            continue
        module = _normalizeText(head.group(1).strip())
        remainder = head.group(2).strip()

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
    """
    Temps de calcul d'un run pycte, lus dans son fichier warning.log.

    Parametres
    ----------
    folder : str | None
        Dossier du run (defaut : dossier courant). Un dossier de sorties
        du run est accepte aussi : on remonte alors d'un cran.
    logName : str
        Nom du fichier de log (defaut "warning.log").
    short : bool
        True (defaut) : noms de colonnes courts, ex.
        "WallClockTotalSpeciation(sec)". False : libelles complets du log.
    rename : dict | None
        Renomme les colonnes du resultat.
    append : pandas.DataFrame | None
        Empile la ligne obtenue sous ce DataFrame existant, pour accumuler
        plusieurs runs, ex.
        dft = pycte.getTimes(run2, append=dft).
    label : str | None
        Ajoute une colonne 'run' avec cette etiquette, pratique pour
        distinguer les lignes quand on empile plusieurs runs.

    Retour
    ------
    pandas.DataFrame d'UNE ligne : une colonne par temps trouve (en
    secondes), plus warnings_count, errors_count, errors_detail et
    coupling_completed.
    """
    if folder in (None, "", "."):
        folder = os.getcwd()
    folder = os.path.abspath(os.path.expanduser(str(folder)))

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"pycte.getTimes : dossier introuvable : {folder}")

    logPath = os.path.join(folder, logName)
    if not os.path.isfile(logPath):
        parentLog = os.path.join(os.path.dirname(folder), logName)
        if os.path.isfile(parentLog):
            folder = os.path.dirname(folder)
            logPath = parentLog
        else:
            raise FileNotFoundError(
                f"pycte.getTimes : {logName} introuvable dans {folder}."
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
                "pycte.getTimes : append= attend un DataFrame existant "
                f"(recu {type(append).__name__})."
            )
        result = pd.concat([append, result], ignore_index=True, sort=False)
    return result
