"""Periodic renewal of cells: renewal = [{'cells': ..., 'times': [...], 'concentration': {...} or None}]."""
import bisect
import difflib
from pathlib import Path

import numpy as np
import pandas as pd

from . import warningManager
from . import display

OUTPUT_FILE = "renewal.txt"


def _error(message):
    raise ValueError(f"renewal : {message}")


def _asList(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    return [value]


WATER_PER_KG = 55.50868


def setup(centralDict, groups):
    """Checks the renewal keyword (before the MultiCompoundTransport conversion)."""
    comm = centralDict['commMtrx']
    nNodes = len(comm)
    totalsLater = bool(centralDict.get('MultiCompoundTransport'))
    if isinstance(groups, dict):
        groups = [groups]
    if not isinstance(groups, (list, tuple)) or not groups:
        _error("give a list of groups {'cells': ..., 'times': [...], 'concentration': {...}}")
    if all(isinstance(g, (int, float, np.number)) for g in groups):
        _error("the former format (a list of times) is no longer supported : give a list of groups "
               "{'cells': ..., 'times': [...], 'concentration': {...}} (see renewal.py)")
    coords = set(centralDict.get('coord') or [])
    speciesCols = [s for s in centralDict['coupledVariables'] if s in comm.columns]
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
        # a time after the last step (another unit than timeUnit, a typo) would never be applied
        end = float(centralDict['dtpycte'][-1]) if len(centralDict.get('dtpycte') or []) else None
        late = [t for t in times if end is not None and t > end + 1e-9 * max(abs(end), 1.0)]
        if late:
            warningManager.warn(f"renewal : {name} : time(s) {late} after the end of the run ({end:g} "
                                f"{centralDict['timeUnit']}) : never applied (the times are in timeUnit = "
                                f"'{centralDict['timeUnit']}')")
        conc = g.get('concentration')
        if conc is not None:
            if not isinstance(conc, dict):
                _error(f"{name} : 'concentration' must be a dict {{species: value}} or None")
            for sp, val in conc.items():
                if sp in coords or (sp not in comm.columns and not totalsLater):
                    hint = difflib.get_close_matches(str(sp), speciesCols, n=1)
                    _error(f"{name} : '{sp}' is not a species column of the initial conditions"
                           + (f" (did you mean '{hint[0]}' ?)" if hint else ""))
                if not np.isfinite(float(val)):
                    _error(f"{name} : non-finite concentration for '{sp}'")
            conc = {sp: float(v) for sp, v in conc.items()}
            if 'H2O' in zeroCols and 'H2O' not in conc:
                conc['H2O'] = WATER_PER_KG
        water = conc['H2O'] / WATER_PER_KG if conc and 'H2O' in conc else 1.0
        normalized.append({'cells': cells, 'times': times, 'next': 0, 'concentration': conc, 'water': water})
    folder = Path(centralDict['inputPath']) / "renewal"
    centralDict['paths']['renewal'] = str(folder)
    return {'renewal': normalized,
            'renewalFile': folder / OUTPUT_FILE}


def finalize(centralDict):
    """Stores the initial conditions and checks the fresh solutions on the final commMtrx columns."""
    comm = centralDict['commMtrx']
    coords = set(centralDict.get('coord') or [])
    speciesCols = [s for s in centralDict['coupledVariables'] if s in comm.columns]
    for i, g in enumerate(centralDict['renewal']):
        conc = g['concentration'] or {}
        absent = [sp for sp in conc if sp not in comm.columns or sp in coords]
        if absent:
            _error(f"group {i} : {absent} are not columns of the coupled variables ({', '.join(speciesCols)})")
    water = centralDict.get('cellWaterMass') if centralDict.get('molesStorage') else None
    return {'renewalInitial': comm.copy(),
            'renewalSpeciesCols': speciesCols,
            'renewalZeroCols': [s for s in centralDict['systemSpecies'] if s in comm.columns],
            'renewalWaterMass': None if water is None else np.asarray(water, dtype=float).copy()}


def _positions(frame, names):
    """Positions of the columns `names` in `frame`; error if one is missing."""
    positions = frame.columns.get_indexer(names)
    missing = [name for name, position in zip(names, positions) if position < 0]
    if missing:
        _error(f"column(s) {missing} are not in the coupled variables")
    return positions


def apply(centralDict):
    """Renews the groups due at the end of the step and logs the state before renewal."""
    t = float(centralDict['tStep'])
    comm = centralDict['commMtrx']
    initial = centralDict['renewalInitial']
    records = []
    for i, g in enumerate(centralDict['renewal']):
        # times strictly increasing (setup): a time tr >= t + 4e-9.max(|t|, 1) is never due (tr - 1e-9.max(|tr|, 1)
        # stays above t by ~3e-9.max(|t|, 1), far beyond rounding), so only the times below that bound are tested,
        # with the same test: the same times due, without going through every pending time at every step
        times, start = g['times'], g['next']
        stop = bisect.bisect_left(times, t + 4e-9 * max(abs(t), 1.0), start)
        due = [tr for tr in times[start:stop] if t >= tr - 1e-9 * max(abs(tr), 1.0)]
        if not due:
            continue
        g['next'] += len(due)
        if len(due) > 1:
            message = (f"renewal, time = {t}{centralDict['timeUnit']} : {len(due)} renewal times of group {i} "
                       f"({due}) fall within the same coupling step : the cells are renewed once")
            warningManager.warn(message)
        rows = g['cells']
        before = comm.iloc[rows].copy()
        before.insert(0, 'cell', rows)
        before.insert(0, 'group', i)
        before.insert(0, 'tStep', t)
        before.insert(0, 'tRenewal', due[-1])
        records.append(before)
        if g['concentration'] is None:
            cols = _positions(comm, centralDict['renewalSpeciesCols'])
            comm.iloc[rows, cols] = initial.iloc[rows, _positions(initial, centralDict['renewalSpeciesCols'])].to_numpy()
        else:
            comm.iloc[rows, _positions(comm, centralDict['renewalZeroCols'])] = 0.0
            for sp, position in zip(g['concentration'], _positions(comm, list(g['concentration']))):
                comm.iloc[rows, position] = g['concentration'][sp]
        if centralDict.get('renewalWaterMass') is not None and centralDict.get('cellWaterMass') is not None:
            waterMass = np.asarray(centralDict['cellWaterMass'], dtype=float).copy()
            waterMass[rows] = centralDict['renewalWaterMass'][rows] if g['concentration'] is None else g['water']
            centralDict['cellWaterMass'] = waterMass
        display.stepPrint(f"renewal : group {i}, cells {rows} (requested t = {due[-1]}{centralDict['timeUnit']})", flush=True)
    if records:
        out = pd.concat(records, ignore_index=True)
        path = Path(centralDict['renewalFile'])
        out.to_csv(path, mode='a', header=not path.exists(), index=False, sep='\t')
    return centralDict
