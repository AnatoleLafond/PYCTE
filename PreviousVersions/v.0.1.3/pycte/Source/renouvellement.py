"""
Renouvellement periodique de mailles (reservoirs d'electrolyte, solutions renouvelees, ...).

Mot-cle d'entree : renouvellement = liste de groupes (ou un seul groupe), chaque groupe :
    {'cells'         : 0 ou [0, 1, ...]      indices de maille = lignes du fichier de conditions initiales
                                             (negatif : depuis la fin, -1 = derniere maille)
     'times'         : [t1, t2, ...]         temps de renouvellement de ce groupe, dans l'unite timeUnit
     'concentration' : {espece: valeur}}     optionnel : solution fraiche. Les especes de systemSpecies non
                                             listees sont mises a 0. Absent ou None : les mailles reprennent
                                             leurs valeurs du fichier de conditions initiales (toutes les
                                             especes de systemSpeciation).
Exemple :
    renouvellement = [
        {'cells': 0,  'times': [86400, 172800, 259200], 'concentration': {'Cl-': 0.08, 'Na+': 0.08}},
        {'cells': -1, 'times': [604800]},                         # conditions initiales
    ]

Un renouvellement prevu au temps t est applique a la fin du premier pas de couplage dont le temps
tStep >= t (apres transport et chimie) : le pas suivant part de la solution renouvelee. Si plusieurs temps
d'un groupe tombent dans le meme pas de couplage, le groupe n'est renouvele qu'une fois (avertissement dans
warning.log). Les groupes sont appliques dans l'ordre de la liste.

Sortie : renouvellement/renouvellement.txt (tabulations), concentrations des mailles juste avant chaque
renouvellement, une ligne par maille et par renouvellement :
    tRenouvellement (temps demande), tStep (temps effectif), groupe, maille, puis toutes les colonnes de commMtrx.
"""
import difflib
from pathlib import Path

import numpy as np
import pandas as pd

from . import warningManager

OUTPUT_FILE = "renouvellement.txt"


def _error(message):
    raise ValueError(f"renouvellement : {message}")


def _asList(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    return [value]


def setup(centralDict, groups):
    """Verifie et normalise le mot-cle renouvellement ; garde une copie des conditions initiales."""
    comm = centralDict['commMtrx']
    nNodes = len(comm)
    if isinstance(groups, dict):
        groups = [groups]
    if not isinstance(groups, (list, tuple)) or not groups:
        _error("give a list of groups {'cells': ..., 'times': [...], 'concentration': {...}}")
    if all(isinstance(g, (int, float, np.number)) for g in groups):
        _error("the former format (a list of times) is no longer supported : give a list of groups "
               "{'cells': ..., 'times': [...], 'concentration': {...}} (see renouvellement.py)")
    coords = set(centralDict.get('coord') or [])
    speciesCols = [s for s in centralDict['systemSpeciation'] if s in comm.columns]
    zeroCols = [s for s in centralDict['systemSpecies'] if s in comm.columns]
    normalized = []
    for i, g in enumerate(groups):
        name = f"group {i}"
        if not isinstance(g, dict):
            _error(f"{name} must be a dict {{'cells': ..., 'times': [...], 'concentration': {{...}}}}")
        unknown = set(g) - {'cells', 'times', 'concentration'}
        if unknown:
            _error(f"{name} : unknown key(s) {sorted(unknown)} (use 'cells', 'times', 'concentration')")
        if 'cells' not in g or 'times' not in g:
            _error(f"{name} : 'cells' and 'times' are required")
        cells = []
        for c in _asList(g['cells']):
            if isinstance(c, bool) or not isinstance(c, (int, np.integer)) or not -nNodes <= c < nNodes:
                _error(f"{name} : cell {c!r} is not a node index in [-{nNodes}, {nNodes - 1}] "
                       f"({nNodes} nodes in the initial conditions)")
            cells.append(int(c) % nNodes)
        if len(set(cells)) != len(cells):
            _error(f"{name} : a cell is listed twice ({cells})")
        times = [float(t) for t in _asList(g['times'])]
        if not times or not np.all(np.isfinite(times)):
            _error(f"{name} : 'times' must be a non-empty list of numbers")
        if np.any(np.diff(times) <= 0):
            _error(f"{name} : 'times' must be strictly increasing (got {times})")
        conc = g.get('concentration')
        if conc is not None:
            if not isinstance(conc, dict):
                _error(f"{name} : 'concentration' must be a dict {{species: value}} or None")
            for sp, val in conc.items():
                if sp not in comm.columns or sp in coords:
                    hint = difflib.get_close_matches(str(sp), speciesCols, n=1)
                    _error(f"{name} : '{sp}' is not a species column of the initial conditions"
                           + (f" (did you mean '{hint[0]}' ?)" if hint else ""))
                if not np.isfinite(float(val)):
                    _error(f"{name} : non-finite concentration for '{sp}'")
            conc = {sp: float(v) for sp, v in conc.items()}
        normalized.append({'cells': cells, 'times': times, 'next': 0, 'concentration': conc})
    folder = Path(centralDict['inputPath']) / "renouvellement"
    centralDict['paths']['renouvellement'] = str(folder)
    return {'renouvellement': normalized,
            'rnvInitial': comm.copy(),
            'rnvSpeciesCols': speciesCols,
            'rnvZeroCols': zeroCols,
            'rnvFile': folder / OUTPUT_FILE}


def apply(centralDict):
    """Renouvelle les groupes dont un temps est echu a la fin du pas ; sauvegarde l'etat avant renouvellement."""
    t = float(centralDict['tStep'])
    comm = centralDict['commMtrx']
    initial = centralDict['rnvInitial']
    records = []
    for i, g in enumerate(centralDict['renouvellement']):
        pending = g['times'][g['next']:]
        due = [tr for tr in pending if t >= tr - 1e-9 * max(abs(tr), 1.0)]
        if not due:
            continue
        g['next'] += len(due)
        if len(due) > 1:
            message = (f"renouvellement, time = {t}{centralDict['timeUnit']} : {len(due)} renewal times of group {i} "
                       f"({due}) fall within the same coupling step : the cells are renewed once")
            warningManager.warn(message)
        rows = g['cells']
        before = comm.iloc[rows].copy()
        before.insert(0, 'maille', rows)
        before.insert(0, 'groupe', i)
        before.insert(0, 'tStep', t)
        before.insert(0, 'tRenouvellement', due[-1])
        records.append(before)
        if g['concentration'] is None:
            cols = comm.columns.get_indexer(centralDict['rnvSpeciesCols'])
            comm.iloc[rows, cols] = initial.iloc[rows, initial.columns.get_indexer(centralDict['rnvSpeciesCols'])].to_numpy()
        else:
            comm.iloc[rows, comm.columns.get_indexer(centralDict['rnvZeroCols'])] = 0.0
            for sp, val in g['concentration'].items():
                comm.iloc[rows, comm.columns.get_loc(sp)] = val
        print(f"renouvellement : group {i}, cells {rows} (requested t = {due[-1]}{centralDict['timeUnit']})", flush=True)
    if records:
        out = pd.concat(records, ignore_index=True)
        path = Path(centralDict['rnvFile'])
        out.to_csv(path, mode='a', header=not path.exists(), index=False, sep='\t')
    return centralDict
