"""
Sorties intermediaires de pycte, pilotees par le mot-cle output.

    output = {'coupling':   [1, 5, 10],      # pas a ecrire, par cle ; chaque cle cree un dossier du meme nom
              'speciation': [0, 10],
              'transport':  range(1, 11)}

  coupling   -> coupling/Coupling_n.txt                 etat de fin de pas de couplage (commMtrx)
  speciation -> speciation/<module>_n.txt               speciation complete (PhreeqC, xGEMS, ORCHESTRA, ...)
                speciation/primarySpecies/PrimarySpecies_n.txt
  transport  -> transport/nativeTransport_n.txt, transport/pflotran_n.txt, transport/<tag>/COMSOL_n.txt (COMSOL)

n = numero du pas de couplage (1 = fin du premier pas, comme dans la liste) ; 0 = etat initial : conditions
initiales (coupling) ou speciation initiale de firstStepEquilibrium (speciation).
output absent : {'coupling': [dernier pas]}. Une cle absente : ni sortie ni dossier.
Au demarrage, les dossiers de sortie connus de pycte (noms actuels et anciens : CouplingHistory, Speciation,
PrimarySpecies, Transport, outputTransport*, outputVTU*, renouvellement) sont supprimes du dossier de calcul,
puis seuls les dossiers demandes sont recrees (vides).
"""
import difflib
import os
import shutil
from pathlib import Path

OUTPUT_KEYS = ('coupling', 'speciation', 'transport')
LEGACY_FOLDERS = ('CouplingHistory', 'Speciation', 'PrimarySpecies', 'Transport', 'renouvellement')
LEGACY_PREFIXES = ('outputTransport', 'outputVTU')


def normalize(output, nSteps):
    """output (dict, None ou False) -> {cle: ensemble de numeros de pas} ; ValueError si mal forme."""
    if output is None:
        return {'coupling': {nSteps}}
    if output is False:
        return {}
    if not isinstance(output, dict):
        raise ValueError(f"output must be a dict {{'coupling' | 'speciation' | 'transport': [steps]}} (got {output!r})")
    result = {}
    for key, steps in output.items():
        if key not in OUTPUT_KEYS:
            hint = difflib.get_close_matches(str(key), OUTPUT_KEYS, n=1)
            raise ValueError(f"output : unknown key {key!r}" + (f" (did you mean '{hint[0]}' ?)" if hint else "")
                             + f" : use {', '.join(OUTPUT_KEYS)}")
        if isinstance(steps, (int, float)) and not isinstance(steps, bool):
            steps = [steps]
        try:
            steps = [int(s) for s in steps if int(s) == s]
        except (TypeError, ValueError):
            raise ValueError(f"output['{key}'] must be a list of time step numbers (got {steps!r})")
        bad = [s for s in steps if not 0 <= s <= nSteps]
        if bad:
            raise ValueError(f"output['{key}'] : step(s) {bad} outside [0, {nSteps}] ({nSteps} coupling steps, "
                             "0 = initial state)")
        result[key] = set(steps)
    return result


def stepNumber(centralDict):
    """Numero de sortie du pas en cours : 0 pendant la speciation initiale (firstStepEquilibrium), sinon lStep + 1."""
    return 0 if centralDict.get('firstStepEquilibrium') else centralDict['lStep'] + 1


def wanted(centralDict, key, step=None):
    """Faut-il ecrire la sortie `key` au pas en cours (ou au pas `step`) ?"""
    steps = (centralDict.get('output') or {}).get(key)
    step = stepNumber(centralDict) if step is None else step
    return bool(steps) and step in steps and key in (centralDict.get('paths') or {})


def filePath(centralDict, pathKey, prefix, step=None):
    """Chemin du fichier <paths[pathKey]>/<prefix>_<n>.txt du pas en cours (ou du pas `step`)."""
    step = stepNumber(centralDict) if step is None else step
    return os.path.join(centralDict['paths'][pathKey], f"{prefix}_{step}.txt")


def folders(runFolder, output, comsolTags=()):
    """Dossiers de sortie demandes : {cle de centralDict['paths']: chemin}."""
    run = Path(runFolder)
    paths = {}
    if 'coupling' in output:
        paths['coupling'] = str(run / 'coupling')
    if 'speciation' in output:
        paths['speciation'] = str(run / 'speciation')
        paths['primarySpecies'] = str(run / 'speciation' / 'primarySpecies')
    if 'transport' in output:
        paths['transport'] = str(run / 'transport')
        for tag in comsolTags:
            paths[f'Transport{tag}'] = str(run / 'transport' / str(tag))
            paths[f'TransportVTU{tag}'] = str(run / 'transport' / f"{tag}_VTU")
    return paths


def prepareFolders(runFolder, paths):
    """Supprime les dossiers de sortie connus de pycte, puis recree (vides) ceux de `paths`."""
    run = Path(runFolder)
    known = set(OUTPUT_KEYS) | set(LEGACY_FOLDERS)
    for entry in run.iterdir():
        if entry.is_dir() and (entry.name in known or entry.name.startswith(LEGACY_PREFIXES)):
            shutil.rmtree(entry)
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)
