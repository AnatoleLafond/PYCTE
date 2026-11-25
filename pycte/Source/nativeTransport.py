import bisect
import operator
import numpy as np
import pandas as pd
import time
import importlib.util
import os
import concurrent.futures
import time
import traceback
import re
try:
    from . import outputManager
    from . import warningManager
    from . import display
    from . import timeUnits
    from . import nahed
    from . import frames
except ImportError:
    import outputManager
    import warningManager
    import display
    import timeUnits
    import nahed
    import frames

FARADAY = 96485.33212
GAS_CONSTANT = 8.314462618
NP_SPARSE_BANDWIDTH = 80
NP_NEG_TOL = 1e-14         # mol/L : largest negative concentration clipped by the implicit Nernst-Planck Newton
NP_POS_FRACTION = 0.9      # a concentration loses at most this fraction of its value per Newton iteration
NP_COUPLING_SHARE = 1e-6   # implicit Nernst-Planck : a species carrying less than this share of the conductance of every
                           # face is solved after the potential (block-triangular Newton matrix, same converged solution) ;
                           # None : every species in the coupled block
NP_REFACTOR_THETA = 0.2    # implicit Nernst-Planck : LU recomputed when the Newton contraction rate exceeds this value
NP_HOLD_STEP = (0.7, 1.6)  # implicit Nernst-Planck : step kept (and its LU reused) while the proposed factor is within
NP_NEWTON_TOL = 1e-9       # implicit Nernst-Planck : estimated relative error of the Newton iterate at convergence
ADE_TVD_TOL = 1e-10        # implicit van Leer : relative change of the fixed point on the limiter correction
ADE_TVD_MAX_ITER = 50      # implicit van Leer : iterations before the step is halved
ADE_TVD_MAX_SPLIT = 12     # implicit van Leer : halvings of a step before the run is stopped
ADE_TIME_SCHEMES = {'euler': 1.0, 'impliciteuler': 1.0, 'backwardeuler': 1.0,
                    'cranknicolson': 0.5, 'cn': 0.5}

# implicit ADE : LU factorisations of the band matrices, kept from one transport call to the next. The key is the
# matrix itself (step, operator, boundary rows), so that a change of any coefficient gives a new factorisation
_FACTOR_CACHE = {}
_FACTOR_CACHE_SIZE = 64


# operators built from the settings only (ADE operator and matrix, Nernst-Planck transmissivities and storages): the
# same at every call of a run, kept by frames.constant for the same setting objects; emptied at the end of a run
_operators = {}
_OPERATORS_SIZE = 16


def cachedFactorization(key, factorize):
    solve = _FACTOR_CACHE.pop(key, None)
    if solve is None:
        solve = factorize()
        if len(_FACTOR_CACHE) >= _FACTOR_CACHE_SIZE:
            _FACTOR_CACHE.pop(next(iter(_FACTOR_CACHE)))
    _FACTOR_CACHE[key] = solve
    return solve

def speciationModule():
    try:
        from . import nativeSpeciation
    except ImportError:
        import nativeSpeciation
    return nativeSpeciation
_activityParams = {}
def activityParameters(path):
    key = str(path)
    if key not in _activityParams:
        db = speciationModule().Database.from_phreeqc(key)
        _activityParams[key] = {name: (sp.dh_a, sp.dh_b) for name, sp in db.species.items()
                                if sp.kind == 'aq' and sp.dh_a is not None}
    return _activityParams[key]

class TransportError(Exception):
    pass

BC_CONDITIONS = ('constant', 'closed', 'flux')
_parsedConditions = {}          # condition text -> (electrode, condition): parseCondition of a text, kept


def parseCondition(bc):
    # a pure function of the text, called for every species and side at every transport call: the answer kept per
    # text (a tuple), an error raised again each time as before
    if type(bc) is str:
        found = _parsedConditions.get(bc)
        if found is not None:
            return found
    parts = [p.strip() for p in bc.split('+', 1)] if isinstance(bc, str) else [bc]
    electrode = parts[0] == 'electrode'
    cond = (parts[1] if len(parts) > 1 else 'closed') if electrode else bc
    if cond not in BC_CONDITIONS:
        raise ValueError(f"unknown boundary condition {bc!r} : use 'constant' (Dirichlet), 'closed' (Neumann), "
                         "'flux' (Cauchy), or 'electrode+constant' / 'electrode+closed' / 'electrode+flux' "
                         "('electrode' = 'electrode+closed')")
    if type(bc) is str:
        if len(_parsedConditions) >= 256:
            _parsedConditions.clear()
        _parsedConditions[bc] = (electrode, cond)
    return electrode, cond
def dirichletMask(centralDict, species, n):
    mask = np.zeros((n, len(species)), dtype=bool)
    bc = centralDict.get('bcActive') or centralDict['boundaryConditions']
    for side, node in ((0, 0), (1, n - 1)):
        if parseCondition(bc[side])[1] != 'constant':
            continue
        values = centralDict['firstBoundary' if side == 0 else 'secondBoundary'] or {}
        for j, sp in enumerate(species):
            if sp in values:
                mask[node, j] = True
    return mask
def boundarySchedule(value):
    if isinstance(value, (str, dict)) or not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"boundaryConditions must be a list of two items [x = 0, x = L] (got {value!r})")
    schedule = []
    for side, item in enumerate(value):
        where = f"boundaryConditions[{side}]"
        if isinstance(item, str):
            entries = [(0.0, item)]
        elif isinstance(item, dict) and item:
            entries = []
            for key, val in item.items():
                t, cond = (val, key) if isinstance(key, str) else (key, val)
                try:
                    t = float(t)
                except (TypeError, ValueError):
                    raise ValueError(f"{where} : {key!r}: {val!r} -> give {{condition: start time}} or "
                                     "{start time: condition}")
                entries.append((t, cond))
        else:
            raise ValueError(f"{where} must be a condition or a dict {{condition: start time}} (got {item!r})")
        for t, cond in entries:
            try:
                parseCondition(cond)
            except ValueError as err:
                raise ValueError(f"{where} : {err}")
            if not np.isfinite(t) or t < 0:
                raise ValueError(f"{where} : start time must be >= 0 (got {t})")
        entries.sort()
        times = [t for t, _ in entries]
        if len(set(times)) != len(times):
            raise ValueError(f"{where} : two conditions start at the same time ({times})")
        if times[0] != 0.0:
            raise ValueError(f"{where} : the condition at time 0 is required (first start time = {times[0]})")
        schedule.append(entries)
    return schedule
_startTime = operator.itemgetter(0)


def conditionAt(entries, t):
    # entries sorted by start time, all distinct (boundarySchedule): those starting by t + tol are the first ones, the
    # condition is that of the last of them, found by bisection instead of going through every entry at every call
    tol = 1e-9 * max(1.0, abs(t))
    found = bisect.bisect_right(entries, t + tol, key=_startTime)
    if not found:
        return [][-1]                                   # no entry: the IndexError of the list of none
    return entries[found - 1][1]


def _startsBetween(entries, low, high):
    """Start times ts of entries (sorted by start time) with low < ts < high."""
    first = bisect.bisect_right(entries, low, key=_startTime)
    last = bisect.bisect_left(entries, high, key=_startTime)
    return [ts for ts, _ in entries[first:last]]


def transportSegments(centralDict):
    # a transport call is split at the start times of the boundary conditions and at the area changes (areaSchedule)
    schedule = centralDict.get('boundarySchedule')
    areaEntries = centralDict.get('areaSchedule') or ()
    dtU, t0 = centralDict['dtStep'], centralDict.get('tTransportStart')
    if not (schedule or areaEntries) or t0 is None:
        return [(centralDict['boundaryConditions'], dtU, t0)]
    t1 = t0 + dtU
    tol = 1e-9 * max(1.0, abs(t1))
    # every schedule sorted by start time (boundarySchedule, areaSchedule): its start times inside the call found by
    # bisection, the same set as from every start time
    starts = set()
    for entries in schedule or ():
        starts.update(_startsBetween(entries, t0 + tol, t1 - tol))
    starts.update(_startsBetween(areaEntries, t0 + tol, t1 - tol))
    cuts = sorted(starts)
    bounds = [t0] + cuts + [t1]
    def conditions(a):
        return [conditionAt(entries, a) for entries in schedule] if schedule else centralDict['boundaryConditions']
    return [(conditions(a), b - a, a) for a, b in zip(bounds[:-1], bounds[1:])]


# areaSchedule = [{'cells': [node indices], 'times': [t1, t2, ...], 'area': [area from t1, area from t2, ...]}] (or one
# group) : the cross-section area (m2) of these nodes changes at these times (unit timeUnit, > 0) and stays constant in
# between ; before the first time, the area of the 'area' setting. An area value is a number (every cell of the group)
# or one value per cell. The area enters the node volume (nodeSize x area) and the conductance of the two half-cells of
# the node. Cartesian coordinates only.
#  - area decrease : the removed part of the node leaves the 1D domain with its solution : the concentrations are
#    unchanged, the solutes of the removed volume are lost ;
#  - area increase : the added volume is pure water : amounts kept, concentrations x A_before / A_after ; the water
#    columns (H2O, and the H and O totals of the PhreeqC totals mode) get the water of the added volume (WATER_TOTALS).
AREA_SCHEDULE_KEYS = ('cells', 'times', 'area')
WATER_TOTALS = {'H2O': 55.50868, 'O': 55.50868, 'H': 2 * 55.50868}     # mol per kg of pure water


def areaSchedule(value, base=None):
    """Checks areaSchedule ; with base (area of every node, m2), returns [(t, area of every node from t), ...]."""
    groups = [value] if isinstance(value, dict) else value
    if not isinstance(groups, (list, tuple)) or not groups or not all(isinstance(g, dict) for g in groups):
        raise ValueError("expected a group or a list of groups {'cells': [node indices], 'times': [t1, t2, ...], "
                         "'area': [area from t1 (m2), area from t2, ...]}")
    nNodes = None if base is None else len(base)
    changes, owner = {}, {}
    for k, g in enumerate(groups):
        name = f"group {k}"
        unknown = sorted(set(g) - set(AREA_SCHEDULE_KEYS))
        if unknown or any(key not in g for key in AREA_SCHEDULE_KEYS):
            raise ValueError(f"{name} : keys 'cells', 'times' and 'area' are required"
                             + (f", unknown {unknown}" if unknown else ""))
        cells = list(g['cells']) if isinstance(g['cells'], (list, tuple, range, np.ndarray)) else [g['cells']]
        if not cells:
            raise ValueError(f"{name} : 'cells' is empty")
        for c in cells:
            if isinstance(c, (bool, np.bool_)) or not isinstance(c, (int, np.integer)):
                raise ValueError(f"{name} : cell {c!r} is not a node index (integer)")
            if nNodes is not None and not -nNodes <= c < nNodes:
                raise ValueError(f"{name} : cell {c} is not a node index in [-{nNodes}, {nNodes - 1}] ({nNodes} nodes "
                                 "in the initial conditions)")
        if nNodes is not None:
            cells = [int(c) % nNodes for c in cells]
        if len(set(cells)) != len(cells):
            raise ValueError(f"{name} : a cell is listed twice ({cells})")
        times = list(g['times']) if isinstance(g['times'], (list, tuple, np.ndarray)) else [g['times']]
        areas = list(g['area']) if isinstance(g['area'], (list, tuple, np.ndarray)) else [g['area']]
        if not times or any(isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, float, np.number))
                            or not np.isfinite(t) or t <= 0 for t in times):
            raise ValueError(f"{name} : 'times' must be a list of times > 0 (unit timeUnit) (got {g['times']!r})")
        times = [float(t) for t in times]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError(f"{name} : 'times' must be strictly increasing (got {times})")
        if len(areas) != len(times):
            raise ValueError(f"{name} : 'area' has {len(areas)} values for {len(times)} times (one area per time)")
        for j, a in enumerate(areas):
            try:
                vals = np.atleast_1d(np.asarray(a, dtype=float))
            except (TypeError, ValueError):
                vals = np.array([np.nan])
            if vals.ndim != 1 or len(vals) not in (1, len(cells)) or not np.all(np.isfinite(vals) & (vals > 0)):
                raise ValueError(f"{name} : area {a!r} at t = {times[j]} : expected a number > 0 (m2) or one value "
                                 f"> 0 per cell ({len(cells)})")
            areas[j] = vals
        if nNodes is None:
            continue
        for c in cells:
            if c in owner:
                raise ValueError(f"cell {c} is in group {owner[c]} and in group {k} : one group per cell")
            owner[c] = k
        for t, vals in zip(times, areas):
            changes.setdefault(t, []).append((np.array(cells), vals))
    if base is None:
        return None
    timeline, current = [], np.array(base, dtype=float)
    for t in sorted(changes):
        current = current.copy()
        for cells, vals in changes[t]:
            current[cells] = vals
        timeline.append((t, current))
    return timeline


def areaAt(centralDict, t, before=False):
    """Area of every node at time t (unit timeUnit) ; before : just before t (a change at t excluded)."""
    area = centralDict['areaBase']
    tol = 1e-9 * max(1.0, abs(t))
    schedule = centralDict['areaSchedule']
    if isinstance(schedule, list):
        # sorted by time, distinct (areaSchedule): the changes in effect are the first ones, found by bisection
        found = (bisect.bisect_left(schedule, t - tol, key=_startTime) if before
                 else bisect.bisect_right(schedule, t + tol, key=_startTime))
        return schedule[found - 1][1] if found else area
    for ts, a in schedule:
        if not ((ts < t - tol) if before else (ts <= t + tol)):
            break
        area = a
    return area


def areaChange(C, species, before, after):
    """Concentrations C after the areas of the nodes change from before to after : pure water in the added volume."""
    grow = np.flatnonzero(after > before)
    if not len(grow):
        return C
    keep = before[grow] / after[grow]
    C = C.copy()
    for sp in species:
        values = C[sp].to_numpy(dtype=float, copy=True)
        values[grow] = values[grow] * keep + WATER_TOTALS.get(sp, 0.0) * (1.0 - keep)
        C[sp] = values
    return C
DONNAN_PSI_MAX = 60.0
def nodeVolumes(centralDict):
    size = np.asarray(centralDict['nodeSize'], dtype=float)
    n = len(size)
    coord = str(centralDict.get('coordinateSystem') or 'cartesian').lower()
    if coord == 'cartesian':
        return size * np.broadcast_to(np.asarray(centralDict['area'], dtype=float), (n,))
    comm = centralDict.get('commMtrx')
    r = (comm['x'].to_numpy(dtype=float) if comm is not None and 'x' in comm.columns
         else np.asarray(centralDict['x'], dtype=float))
    if coord == 'cylindrical':
        height = np.asarray(1.0 if centralDict.get('cylinderHeight') is None else centralDict['cylinderHeight'],
                            dtype=float)
        omega, p = 2.0 * np.pi * np.broadcast_to(height, (n,)), 1
    else:
        omega, p = np.full(n, 4.0 * np.pi), 2
    dxh = np.asarray(centralDict['dxHalfCell'], dtype=float)
    w = dxh[:-1] / (dxh[:-1] + dxh[1:])
    rFace = r[:-1] + w * (r[1:] - r[:-1])
    edges = np.concatenate(([max(rFace[0] - size[0], 0.0)], rFace, [rFace[-1] + size[-1]]))
    return omega * (edges[1:] ** (p + 1) - edges[:-1] ** (p + 1)) / (p + 1)


def saturationFromGas(centralDict):
    column = centralDict['gasVolumeColumn']
    gas = centralDict['commMtrx'][column].to_numpy(dtype=float)
    pores = np.asarray(centralDict['porosity'], dtype=float) * nodeVolumes(centralDict)
    if not np.all(np.isfinite(gas)) or np.any(gas < -1e-12 * pores):
        bad = np.flatnonzero(~(np.isfinite(gas) & (gas >= -1e-12 * pores)))
        raise ValueError(f"pycte : gasVolume '{column}' must be >= 0 (m3 per node) : node(s) {bad[:10].tolist()} "
                         f"(values {gas[bad[:10]].tolist()})")
    saturation = 1.0 - np.maximum(gas, 0.0) / pores
    if np.any(saturation <= 0):
        bad = np.flatnonzero(saturation <= 0)
        raise ValueError(f"pycte : gasVolume '{column}' fills the whole pore volume (porosity x node volume) at "
                         f"node(s) {bad[:10].tolist()} (gas {gas[bad[:10]].tolist()} m3, pores "
                         f"{pores[bad[:10]].tolist()} m3)")
    return saturation


def _expClip(x):
    return np.exp(np.minimum(np.maximum(x, -700.0), 700.0))
def _asShape(a, shape):
    a = np.asarray(a, dtype=float)
    return a if a.shape == shape else np.broadcast_to(a, shape)
def donnanPotential(c, z, sigma, phi=None, psi0=None, tol=1e-12, maxIter=200):
    c = np.maximum(np.asarray(c, dtype=float), 0.0)
    z = np.asarray(z, dtype=float)[:, None]
    phi = np.ones_like(c) if phi is None else np.asarray(phi, dtype=float)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), c.shape[1:]).astype(float)
    zc = z * phi * c
    minusZ = -z
    zzc = z * zc
    absZc = np.abs(zc)
    def g(psi):
        e = _expClip(minusZ * psi[None, :])
        return (zc * e).sum(axis=0) + sigma, -(zzc * e).sum(axis=0), (absZc * e).sum(axis=0)
    lo = np.full(sigma.shape, -DONNAN_PSI_MAX)
    hi = np.full(sigma.shape, DONNAN_PSI_MAX)
    gLo, gHi = g(lo)[0], g(hi)[0]
    ok = (gLo >= 0) & (gHi <= 0)
    bound = np.where(gLo < 0, -DONNAN_PSI_MAX, DONNAN_PSI_MAX)
    if psi0 is None:
        Is = 0.5 * (z * z * phi * c).sum(axis=0)
        psi0 = np.arcsinh(np.where(Is > 0, sigma / (2.0 * np.maximum(Is, 1e-300)), 0.0))
    psi = np.clip(np.asarray(psi0, dtype=float) * np.ones(sigma.shape), lo, hi)
    for _ in range(maxIter):
        f, df, scale = g(psi)
        lo = np.where(f > 0, psi, lo)
        hi = np.where(f <= 0, psi, hi)
        with np.errstate(divide='ignore', invalid='ignore'):
            step = np.where(df < 0, f / df, 0.0)
        new = psi - step
        bad = ~((new > lo) & (new < hi)) | (df >= 0)
        new = np.where(bad, 0.5 * (lo + hi), new)
        done = (np.abs(f) <= tol * (scale + np.abs(sigma))) | (hi - lo < 1e-14)
        psi = np.where(done, psi, new)
        if np.all(done):
            break
    return np.where(ok, psi, bound), ok
def donnanFromTotals(Ctot, z, thF, thDL, sigma, phi=None, fixed=None, fixVal=None, psi0=None, tol=1e-12, maxIter=200):
    Ctot = np.asarray(Ctot, dtype=float)
    z = np.asarray(z, dtype=float)[:, None]
    phi = np.ones_like(Ctot) if phi is None else np.asarray(phi, dtype=float)
    thF = _asShape(thF, Ctot.shape[1:])
    thDL = _asShape(thDL, Ctot.shape[1:])
    sigma = _asShape(sigma, Ctot.shape[1:])
    fixed = np.zeros(Ctot.shape, dtype=bool) if fixed is None else np.asarray(fixed, dtype=bool)
    fixVal = np.zeros_like(Ctot) if fixVal is None else np.asarray(fixVal, dtype=float)
    active = thDL > 0
    anyFixed = bool(fixed.any())
    minusZ = -z
    zPhi = z * phi
    thDLPhi = thDL[None, :] * phi
    CtotPos = np.maximum(Ctot, 0.0)
    def state(psi):
        e = _expClip(minusZ * psi[None, :])
        den = thF[None, :] + thDLPhi * e
        c = np.where(fixed, fixVal, CtotPos / den) if anyFixed else CtotPos / den
        w = np.where(fixed, 1.0, thF[None, :] / den) if anyFixed else thF[None, :] / den
        zpe = zPhi * e * c
        return c, zpe.sum(axis=0) + sigma, -(z * zpe * w).sum(axis=0), np.abs(zpe).sum(axis=0)
    lo = np.full(thF.shape, -DONNAN_PSI_MAX)
    hi = np.full(thF.shape, DONNAN_PSI_MAX)
    hLo, hHi = state(lo)[1], state(hi)[1]
    ok = (hLo >= 0) & (hHi <= 0) | ~active
    bound = np.where(hLo < 0, -DONNAN_PSI_MAX, DONNAN_PSI_MAX)
    psi = np.zeros(thF.shape) if psi0 is None else np.clip(np.asarray(psi0, dtype=float) * np.ones(thF.shape), lo, hi)
    for _ in range(maxIter):
        c, f, df, scale = state(psi)
        lo = np.where(f > 0, psi, lo)
        hi = np.where(f <= 0, psi, hi)
        with np.errstate(divide='ignore', invalid='ignore'):
            step = np.where(df < 0, f / df, 0.0)
        new = psi - step
        bad = ~((new > lo) & (new < hi)) | (df >= 0)
        new = np.where(bad, 0.5 * (lo + hi), new)
        done = ~active | (np.abs(f) <= tol * (scale + np.abs(sigma))) | (hi - lo < 1e-14)
        psi = np.where(done, psi, new)
        if np.all(done):
            break
    psi = np.where(active, np.where(ok, psi, bound), 0.0)
    return state(psi)[0], psi, ok
def donnanStorage(c, z, thF, thDL, sigma, phi=None, psi=None):
    c = np.asarray(c, dtype=float)
    zz = np.asarray(z, dtype=float)[:, None]
    thF = _asShape(thF, c.shape[1:])
    thDL = _asShape(thDL, c.shape[1:])
    phi = np.ones_like(c) if phi is None else np.asarray(phi, dtype=float)
    if psi is None:
        psi = donnanPotential(c, z, sigma, phi)[0]
    e = _expClip(-zz * psi[None, :])
    thEff = thF[None, :] + thDL[None, :] * phi * e
    v = zz * phi * e
    u = v * np.maximum(c, 0.0)
    G = (zz * u).sum(axis=0)
    return thEff * c, thEff, u, v, G, psi

toSeconds = timeUnits.toSeconds

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
def bernoulli(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-6
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        np.divide(x, np.expm1(x), out=out)
    if small.any():
        out = np.where(small, 1.0 - x / 2.0 + x ** 2 / 12.0, out)
    return out
def bernoulliDeriv(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-4
    out[small] = -0.5 + x[small] / 6.0
    xs = x[~small]
    out[~small] = bernoulli(xs) * (1.0 - bernoulli(-xs)) / xs
    return out
def bernoulliPair(x, deriv=False):
    x = np.asarray(x, dtype=float)
    Bp, Bm = bernoulli(x), bernoulli(-x)
    if not deriv:
        return Bp, Bm
    small = np.abs(x) < 1e-4
    with np.errstate(divide='ignore', invalid='ignore'):
        dBp = np.where(small, -0.5 + x / 6.0, Bp * (1.0 - Bm) / x)
        dBm = np.where(small, -0.5 - x / 6.0, Bm * (1.0 - Bp) / (-x))
    return Bp, Bm, dBp, dBm
def transportCore(centralDict, commMtrx, trsptedSpecies, transportLog):
    reactionIntegral = {}
    def logWarning(where, message):
        for entry in transportLog:
            if entry['level'] == 'WARNING' and entry['where'] == where and entry['message'] == message:
                entry['count'] += 1
                return
        transportLog.append({'level': 'WARNING', 'where': where, 'message': message, 'count': 1})
    def logError(where, message):
        raise TransportError(where, message)
    def meshProp(value, lo, hi):
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(max(hi - lo, 0), float(arr))
        return np.asarray(arr[lo:hi], dtype=float)
    def areaRef():
        if isRadial:
            return float(geom['nodeArea'][0])
        return float(meshProp(centralDict['area'], 0, 1)[0])
    def radialGeometry(r):
        p = 1 if coordSystem == 'cylindrical' else 2
        r = np.asarray(r, dtype=float)
        nG = len(r)
        if coordSystem == 'cylindrical':
            height = centralDict.get('cylinderHeight')
            height = np.asarray(1.0 if height is None else height, dtype=float)
            if height.ndim > 1 or (height.ndim == 1 and len(height) != nG):
                logError('radialGeometry', f"cylinderHeight must be a number or one value per node "
                         f"(got {height.size} values for {nG} nodes)")
            height = meshProp(height, 0, nG)
            if not np.all(np.isfinite(height)) or np.any(height <= 0):
                k = int(np.argmin(np.where(np.isfinite(height), height, -np.inf)))
                logError('radialGeometry', f"cylinderHeight must be > 0 (got {height[k]} at node {k})")
            omega = 2.0 * np.pi * height
        else:
            omega = np.full(nG, 4.0 * np.pi)
        dxh = meshProp(centralDict['dxHalfCell'], 0, nG)
        size = meshProp(centralDict['nodeSize'], 0, nG)
        if nG < 2 or not np.all(np.isfinite(r)) or r[0] < 0 or np.any(np.diff(r) <= 0):
            logError('radialGeometry', f"coordinateSystem = '{coordSystem}' : x is the radius, it must be >= 0 "
                     "and strictly increasing (at least 2 nodes)")
        if len(dxh) != nG or len(size) != nG or np.any(dxh <= 0) or np.any(size <= 0):
            logError('radialGeometry', f"dxHalfCell ({len(dxh)}) and nodeSize ({len(size)}) must be > 0 and have "
                     f"one value per node ({nG})")
        w = dxh[:-1] / (dxh[:-1] + dxh[1:])
        rFace = r[:-1] + w * (r[1:] - r[:-1])
        edges = np.concatenate(([max(rFace[0] - size[0], 0.0)], rFace, [rFace[-1] + size[-1]]))
        volume = omega * (edges[1:] ** (p + 1) - edges[:-1] ** (p + 1)) / (p + 1)
        nodeArea = omega * r ** p
        return {'volume': volume, 'nodeArea': nodeArea, 'rFace': rFace, 'edges': edges,
                'aRight': np.concatenate((omega[:-1] * rFace ** p, nodeArea[-1:])),
                'aLeft': np.concatenate((nodeArea[:1], omega[1:] * rFace ** p))}
    def cellVolume(lo, hi):
        if isRadial:
            return np.array(geom['volume'][lo:hi], dtype=float)
        return meshProp(centralDict['nodeSize'], lo, hi) * meshProp(centralDict['area'], lo, hi)
    def halfCellAreas(lo, hi):
        if isRadial:
            return np.array(geom['aRight'][lo:hi], dtype=float), np.array(geom['aLeft'][lo:hi], dtype=float)
        area = meshProp(centralDict['area'], lo, hi)
        return area, area
    def radialSubSteps(nGiven, T_right, T_left, storage):
        if not isRadial:
            return nGiven
        rate = (np.asarray(T_right) + np.asarray(T_left)) / (1000.0 * np.asarray(storage))
        need = int(np.ceil(dt * float(np.max(rate)))) if rate.size else 1
        return max(int(nGiven), need, 1)
    def npSubSteps(duration):
        hMax = float(centralDict.get('npMaxSubStep', np.inf))
        if not (np.isfinite(hMax) and hMax > 0):
            return 1
        return max(int(np.ceil(duration / hMax)), 1)
    BC_KIND = {'constant': 'dirichlet', 'closed': 'neumann', 'flux': 'cauchy'}
    # boundaryConditions, firstBoundary and secondBoundary do not change during a call (basicTransport sets the
    # conditions of the segment before it): the condition of a side and the boundary of a species, asked for every
    # species and side (~500 times per call with 120 species), are read once per call and kept (not the errors)
    sideConditions = {}
    speciesBoundaries = {}
    def sideSetup(side):
        found = sideConditions.get(side)
        if found is not None:
            return found
        bcList = centralDict['boundaryConditions']
        if isinstance(bcList, str) or len(bcList) != 2:
            logError('boundaries', f"boundaryConditions must be a list of two conditions [x = 0, x = L] (got {bcList!r})")
        try:
            found = sideConditions[side] = parseCondition(bcList[side])
            return found
        except ValueError as err:
            logError('boundaries', f"boundaryConditions[{side}] : {err}")
    def isElectrode(side):
        return sideSetup(side)[0]
    # the electrode nodes and the current mask depend on electrodeNodes and the number of nodes only: asked at every
    # sub-step (~2 000 times per call with an imposed current), built once per call (only read by the callers)
    electrodeKept = {}
    def electrodeNodes():
        found = electrodeKept.get('nodes')
        if found is None:
            found = electrodeKept['nodes'] = electrodeNodesOf()
        return found
    def currentMask():
        found = electrodeKept.get('mask')
        if found is None:
            found = electrodeKept['mask'] = currentMaskOf()
        return found
    def electrodeNodesOf():
        # electrodeNodes = [i0, iL] : nodes carrying the electrode reactions (default : the two boundary nodes).
        # The current flows only between them ; the nodes outside (e.g. a reservoir behind the electrode) are
        # coupled by diffusion only (zero current on their interfaces).
        nodes = centralDict.get('electrodeNodes')
        if nodes is None:
            return 0, n - 1
        if not isinstance(nodes, (list, tuple)) or len(nodes) != 2:
            logError('electrodeNodes', f"electrodeNodes must be a list of two node indices [i0, iL] (got {nodes!r})")
        i0, iL = (int(k) + n if int(k) < 0 else int(k) for k in nodes)
        if not 0 <= i0 < iL <= n - 1:
            logError('electrodeNodes', f"electrodeNodes = {nodes!r} : need 0 <= i0 < iL <= {n - 1} "
                     "(electrode of x = 0 first, negative indices counted from the end)")
        return i0, iL
    def currentMaskOf():
        i0, iL = electrodeNodes()
        mask = np.zeros(n - 1)
        mask[i0:iL] = 1.0
        return mask
    def boundarySpec(sp, side):
        found = speciesBoundaries.get((sp, side))
        if found is not None:
            return found
        kind = BC_KIND[sideSetup(side)[1]]
        values = centralDict['firstBoundary' if side == 0 else 'secondBoundary'] or {}
        if kind == 'neumann' or sp not in values:
            found = 'neumann', None
        else:
            found = kind, float(values[sp])
        speciesBoundaries[(sp, side)] = found
        return found
    def boundaryTransfer():
        h = centralDict.get('boundaryTransferCoeff')
        h = np.ravel(np.asarray(0.0 if h is None else h, dtype=float))
        if h.size == 1:
            h = np.repeat(h, 2)
        if h.size != 2 or not np.all(np.isfinite(h)) or np.any(h < 0):
            logError('boundaries', "boundaryTransferCoeff must be >= 0 [m/s] : a number or [h_0, h_L]")
        areaR, areaL = halfCellAreas(0, n)
        return float(h[0]) * float(areaL[0]), float(h[1]) * float(areaR[-1])
    def cauchyArrays(spc, noInflow):
        K = np.zeros((2, len(spc)))
        cExt = np.zeros((2, len(spc)))
        mask = np.zeros((2, len(spc)), dtype=bool)
        Kb = boundaryTransfer()
        for side in (0, 1):
            for i, sp in enumerate(spc):
                kind, val = boundarySpec(sp, side)
                if kind == 'cauchy':
                    mask[side, i], K[side, i], cExt[side, i] = True, Kb[side], val
            if noInflow and mask[side].any() and Kb[side] == 0:
                logWarning('boundaries', f"boundaryConditions[{side}] = 'flux' (Cauchy) without inflow and with "
                           "boundaryTransferCoeff = 0 : nothing enters, the boundary behaves as 'closed'")
        return K, cExt, mask
    def allBoundaries(df):
        # Dirichlet values of every transported species j (column j) at its boundary nodes, the values of a side
        # written at once (one iloc per side instead of one df.iat per species and side, ~25 us each): the same values
        # in the same cells, side 0 before side 1 as before (one node: side 1 wins in both)
        for side, node in ((0, 0), (1, -1)):
            positions, values = [], []
            for j, sp in enumerate(trsptedSpecies):
                kind, val = boundarySpec(sp, side)
                if kind == 'dirichlet':
                    positions.append(j)
                    values.append(val)
            if positions:
                df.iloc[node, positions] = values
        return df
    def sourceArray(spc, key='transportSource'):
        src = centralDict.get(key)
        if src is None:
            return None
        out = np.zeros((n, len(spc)))
        table = (src.to_numpy() if isinstance(src, pd.DataFrame) and src.columns.is_unique
                 and frames.allFloat64(src) else None)
        for j, sp in enumerate(spc):
            if table is not None:
                if sp not in src.columns:
                    continue
                val = table[:, src.columns.get_loc(sp)]
            elif isinstance(src, pd.DataFrame):
                if sp not in src.columns:
                    continue
                val = src[sp].to_numpy(dtype=float)
            elif hasattr(src, 'get'):
                if src.get(sp) is None:
                    continue
                val = np.asarray(src[sp], dtype=float)
            else:
                logError(key, f"{key} must be a DataFrame (one column per species, one row per node) or a dict "
                         "{species: number or one value per node}")
            if val.ndim > 1 or (val.ndim == 1 and len(val) != n):
                logError(key, f"{key}['{sp}'] has {val.size} values, expected {n} (one per node)")
            if not np.all(np.isfinite(val)):
                logError(key, f"{key}['{sp}'] contains non-finite values")
            if key == 'transportSink' and np.any(val < 0):
                logError(key, f"{key}['{sp}'] must be >= 0 (first-order sink coefficient)")
            out[:, j] = val
        if not out.any():
            return None
        return out / toSeconds(1.0, centralDict['timeUnit'])
    def reactionTerms(spc):
        return sourceArray(spc), sourceArray(spc, 'transportSink')
    def storeIntegral(spc, I):
        reactionIntegral['species'], reactionIntegral['I'] = list(spc), I
    # coupling matrix of Nahed & de Lamare (nahed.py, setting nahed) : every mode gives, after each of its time steps,
    # the Jacobian J of the equation of the step (mol/s ; analytic, or by nahed.fdJacobian of the residual) and the
    # increment of the unknowns. Measurement only : nothing of the solver is changed.
    def couplingMonitor(names):
        monitor = nahed.monitorFor(centralDict)
        if monitor is not None:
            monitor.initial(names)
        return monitor
    def couplingStep(monitor, rows, cols, vals, du, blockOf, tEnd, h, check=None):
        monitor.afterStep({'rows': np.asarray(rows, dtype=np.int64), 'cols': np.asarray(cols, dtype=np.int64),
                           'vals': np.asarray(vals, dtype=float), 'du': np.asarray(du, dtype=float),
                           'blockOf': np.asarray(blockOf, dtype=np.int64)}, t=tGlobalStart + tEnd, h=h)
        # tests only : centralDict['_nahedCheck'] = [] receives nahed.checkJacobian of every step ; check = (residual,
        # uNew, k, width) with the unknowns ranged node by node (rows, cols and du in that order)
        if check is not None and isinstance(centralDict.get('_nahedCheck'), list):
            residual, uNew, k, width = check
            centralDict['_nahedCheck'].append(nahed.checkJacobian(residual, uNew, du, rows, cols, vals, k, width))
    def couplingStepFD(monitor, residual, uPrev, uNew, k, width, tEnd, h):
        # unknowns ranged node by node (k per node) ; J at the end of the step
        rows, cols, vals = nahed.fdJacobian(residual, uNew, k, width)
        couplingStep(monitor, rows, cols, vals, uNew - uPrev, np.arange(len(uNew)) % k, tEnd, h,
                     (residual, uNew, k, width))
    def couplingSummary(monitor):
        return f", coupling matrix : {monitor.describe()}" if monitor is not None else ""
    def checkSourceCharge(S, K, c, Z):
        if S is None and K is None:
            return
        rate = (0.0 if S is None else S) - (0.0 if K is None else K * c)
        net = np.abs(rate @ Z)
        scale = np.abs(rate) @ np.abs(Z)
        if np.any(net > 1e-6 * np.maximum(scale, 1e-300)):
            k = int(np.argmax(net))
            logWarning('iterative coupling', f"the reaction source term is not electroneutral (node {k} : sum z.S = "
                       f"{net[k] * toSeconds(1.0, centralDict['timeUnit']):.3g} per {centralDict['timeUnit']}) : "
                       "the Nernst-Planck closure does not compensate this charge (check the charge balance of the "
                       "chemistry)")
    def adeFlowRate():
        v = centralDict.get('velocity') or 0.0
        if not v:
            return 0.0
        if isRadial:
            return float(geom['flowRate'])
        return float(v) * float(meshProp(centralDict['area'], 0, 1)[0])
    def activitySetup(species):
        ns = speciationModule()
        charges = centralDict.get('speciesValence') or {}
        z = np.array([float(charges[sp]) if sp in charges else float(ns.parse_charge(sp)) for sp in species])
        path = centralDict.get('activityDatabase')
        params = {}
        if path:
            if not os.path.isfile(str(path)):
                logError('activityGradient', f"activityDatabase not found : {path}")
            params = activityParameters(path)
        missing = [sp for sp, zi in zip(species, z) if zi != 0 and sp not in params]
        if path and missing:
            logWarning('activityGradient', f"no '-gamma' parameters in activityDatabase for {', '.join(missing)} : "
                       "Davies equation used")
        tk = float(centralDict.get('temperature', 298.15))
        return {'ns': ns, 'z': z, 'tk': tk,
                'dh_a': np.array([params[sp][0] if sp in params else np.nan for sp in species], dtype=float),
                'dh_b': np.array([params[sp][1] if sp in params else 0.0 for sp in species], dtype=float)}
    def lnGamma(act, c):
        I = 0.5 * ((act['z'] ** 2)[:, None] * np.maximum(c, 0.0)).sum(axis=0)
        return np.log(10.0) * act['ns'].log_gamma(I, act['z'], act['dh_a'], act['dh_b'], act['tk']).T
    def dLnGamma(act, c):
        g = lnGamma(act, c)
        return g[:, 1:] - g[:, :-1]
    def operatorSources():
        # the settings the transport operator is made of: porosity, diffusion coefficient, mesh, area (or the radial
        # geometry of the call), dispersivity, velocity. trspt and basicTransport give centralDict new objects when
        # they change (saturation, water mass, Millington-Quirk, areaSchedule): the same objects, the same operator
        return (centralDict.get('porosity'), centralDict.get('diffCoeff'), centralDict.get('dxHalfCell'),
                centralDict.get('nodeSize'), centralDict.get('area'), centralDict.get('dispersivity'),
                centralDict.get('velocity'), geom)
    def adeOperator():
        # the same at every call for the same settings: built once (only read by the callers ; an error raised again)
        return frames.constant(('adeOperator', n, coordSystem), operatorSources(), adeOperatorOf,
                               cache=_operators, size=_OPERATORS_SIZE)
    def adeOperatorOf():
        theta = meshProp(centralDict['porosity'], 0, n)
        poreD = meshProp(centralDict['diffCoeff'], 0, n)
        dxh = meshProp(centralDict['dxHalfCell'], 0, n)
        volume = cellVolume(0, n)
        for name, arr in (('porosity', theta), ('diffCoeff', poreD), ('dxHalfCell', dxh), ('cell volume', volume)):
            if len(arr) != n:
                logError('adeOperator', f"{name} has {len(arr)} values, expected {n} (one per node)")
            if not np.all(np.isfinite(arr)):
                logError('adeOperator', f"{name} contains non-finite values")
        if np.any(poreD < 0):
            logError('adeOperator', "diffCoeff must be >= 0")
        storage = volume * theta
        if np.any(storage <= 0) or np.any(dxh <= 0):
            k = int(np.argmin(np.minimum(storage, dxh)))
            logError('adeOperator', f"porosity, cell volume and dxHalfCell must be > 0 (node {k})")
        alphaL = float(centralDict.get('dispersivity') or 0.0)
        if alphaL < 0:
            logError('adeOperator', f"dispersivity must be >= 0 (got {alphaL})")
        Q = adeFlowRate()
        areaR, areaL = halfCellAreas(0, n)
        def interfaceConductance(disp):
            R_half = []
            for area in (areaR, areaL):
                cond = theta * area * poreD + disp
                with np.errstate(divide='ignore', invalid='ignore'):
                    R = dxh / np.where(cond > 0, cond, 1.0)
                R[cond <= 0] = np.inf
                R_half.append(R)
            R_sum = R_half[0][:-1] + R_half[1][1:]
            T = np.zeros(n - 1)
            finite_R = np.isfinite(R_sum) & (R_sum > 0)
            T[finite_R] = 1.0 / R_sum[finite_R]
            return T
        T = interfaceConductance(alphaL * abs(Q))
        Tmol = interfaceConductance(0.0) if (alphaL and Q) else T
        return storage, T, Tmol, Q
    def adeAdvection(Q):
        dxh = meshProp(centralDict['dxHalfCell'], 0, n)
        def downstream(Qa, dxh):
            A = np.zeros((5, n))
            A[2, -1] += Qa
            A[1, 1] -= Qa
            if n > 2:
                k = np.arange(1, n - 1)
                w = dxh[k] / (dxh[k - 1] + dxh[k]) if advScheme == 'lud' else np.zeros(n - 2)
                A[2, k] += Qa * (1.0 + w)
                A[1, k] -= Qa * w
                A[1, k + 1] -= Qa * (1.0 + w)
                A[0, k + 1] += Qa * w
            return A
        if Q > 0:
            return downstream(Q, dxh)
        if Q < 0:
            return downstream(-Q, dxh[::-1])[::-1, ::-1]
        return np.zeros((5, n))
    def adeMatrix(T, Tmol, Q, dG=None):
        T = T[:, None] if T.ndim == 1 else T
        Tmol = Tmol[:, None] if Tmol.ndim == 1 else Tmol
        if dG is None:
            a = b = T
        else:
            Bp, Bm = bernoulliPair(dG)
            Tdisp = T - Tmol
            a = Tdisp + Tmol * Bm
            b = Tdisp + Tmol * Bp
        Lb = np.zeros((5, n, a.shape[1]))
        Lb[1, 1:] = -b
        Lb[3, :-1] = -a
        Lb[2, :-1] += b
        Lb[2, 1:] += a
        Lb += adeAdvection(Q)[:, :, None]
        return Lb
    def adeApplyL(C, Lb, far=(True, True)):
        LC = Lb[2] * C
        LC[:-1] += Lb[3, :-1] * C[1:]
        LC[1:] += Lb[1, 1:] * C[:-1]
        if far[1]:
            LC[:-2] += Lb[4, :-2] * C[2:]
        if far[0]:
            LC[2:] += Lb[0, 2:] * C[:-2]
        return LC
    def farBands(Lb):
        return bool(Lb[0, 2:].any()), bool(Lb[4, :-2].any())
    vanLeerGeometry = {}
    def vanLeerCorrection(C, Q):
        if not vanLeerGeometry:
            dxh = meshProp(centralDict['dxHalfCell'], 0, n)
            for key, d in (('forward', dxh), ('backward', dxh[::-1])):
                vanLeerGeometry[key] = (d[1:-1][:, None], (d[:-2] + d[1:-1])[:, None], (d[1:-1] + d[2:])[:, None])
        if Q < 0:
            return vanLeerDownstream(C[::-1], -Q, vanLeerGeometry['backward'])[::-1]
        return vanLeerDownstream(C, Q, vanLeerGeometry['forward'])
    def vanLeerDownstream(C, Qa, geometry):
        out = np.zeros_like(C)
        if n < 3 or Qa == 0:
            return out
        dxhMid, gapUp, gapDn = geometry
        dUp = C[1:-1] - C[:-2]
        dDn = C[2:] - C[1:-1]
        gUp = dUp / gapUp
        gDn = dDn / gapDn
        prod = gUp * gDn
        with np.errstate(divide='ignore', invalid='ignore'):
            s = np.where(prod > 0, 2.0 * prod / (gUp + gDn), 0.0)
        delta = dxhMid * s
        delta = np.sign(delta) * np.minimum(np.abs(delta), np.minimum(np.abs(dUp), np.abs(dDn)))
        out[1:-1] += Qa * delta
        out[2:] -= Qa * delta
        return out
    def adeResidual(X, Xprev, h, theta, storage, Lb, bDiag, bSrc, fixed, fixVal, Q, tvd, S=None, K=None):
        # equation of an ADE step (coupling matrix, mol/s) : M (X - Xprev) / h + theta.A(X) + (1 - theta).A(Xprev)
        # - sources, A = transport operator + Cauchy rows + sink ; theta = 1 for the explicit scheme (equivalent
        # implicit Euler residual) ; Dirichlet rows : X - value
        def A(Y):
            AY = adeApplyL(Y, Lb)
            if tvd:
                AY += vanLeerCorrection(Y, Q)
            AY[0] += bDiag[0] * Y[0]
            AY[-1] += bDiag[1] * Y[-1]
            if K is not None:
                AY += storage[:, None] * K * Y
            return AY
        R = storage[:, None] * (X - Xprev) / h + theta * A(X)
        if theta < 1:
            R += (1.0 - theta) * A(Xprev)
        R[0] -= bSrc[0]
        R[-1] -= bSrc[1]
        if S is not None:
            R -= storage[:, None] * S
        R = 1000.0 * R
        R[fixed] = (X - fixVal)[fixed]
        return R
    def adeExplicit(C, storage, Lb, fixed, fixVal, bDiag, bSrc, Q, S=None, K=None):
        safety = float(centralDict.get('cflSafety') or 0.9)
        nMax = int(centralDict.get('maxSubCycling') or 100000)
        tvd = advScheme == 'vanleer' and Q != 0
        heun = (advScheme == 'lud' or tvd) and Q != 0
        kappa = 0.75 if (heun and not tvd) else 1.0
        free = ~np.all(fixed, axis=1)
        diagB = np.broadcast_to(Lb[2], C.shape).copy()
        diagB[0] += bDiag[0]
        diagB[-1] += bDiag[1]
        if tvd:
            diagB += abs(Q)
        rate = np.max(diagB[free] / storage[free, None], axis=1)
        rateMax = float(np.max(rate)) if rate.size else 0.0
        nSub = max(1, int(np.ceil(dt * rateMax / (safety * kappa)))) if rateMax > 0 else 1
        if nSub > nMax:
            k = int(np.flatnonzero(free)[np.argmax(rate)])
            logError('adeExplicit', f"{nSub} sub-steps needed (> maxSubCycling = {nMax}, dt_sub = "
                     f"{writeTime(dt / nSub, 3)}, limiting node {k}) : use adeSolver = 'implicit' "
                     "or increase maxSubCycling")
        h = dt / nSub
        far = farBands(Lb)
        storageCol = storage[:, None]
        def rhs(C):
            LC = adeApplyL(C, Lb, far)
            if tvd:
                LC += vanLeerCorrection(C, Q)
            LC[0] += bDiag[0] * C[0] - bSrc[0]
            LC[-1] += bDiag[1] * C[-1] - bSrc[1]
            if S is not None:
                return -LC / storageCol + S
            return -LC / storageCol
        C = C.copy()
        fixIdx = np.nonzero(fixed)
        fixVals = fixVal[fixIdx]
        C[fixIdx] = fixVals
        I = np.zeros_like(C)
        sinkFactor = None if K is None else 1.0 + h * K
        monitor = couplingMonitor(list(trsptedSpecies))
        width = 2 if (far[0] or far[1] or tvd) else 1
        for iSub in range(nSub):
            C1 = C + h * rhs(C)
            C1[fixIdx] = fixVals
            if heun:
                C1 = 0.5 * (C + C1 + h * rhs(C1))
                C1[fixIdx] = fixVals
            if sinkFactor is not None:
                C1 = C1 / sinkFactor
                C1[fixIdx] = fixVals
            if monitor is not None:
                couplingStepFD(monitor, lambda u, Cp=C: adeResidual(u.reshape(Cp.shape), Cp, h, 1.0, storage, Lb, bDiag,
                                                                     bSrc, fixed, fixVal, Q, tvd, S, K).ravel(),
                               C.ravel(), C1.ravel(), C.shape[1], width, (iSub + 1) * h, h)
            I += h * C1
            C = C1
        storeIntegral(trsptedSpecies, I)
        return C, nSub
    def tridiagonalSolver(a):
        from scipy.linalg import solve_banded
        if n < 3:
            return lambda rhs: solve_banded((1, 1), a, rhs, overwrite_b=True, check_finite=False)
        from scipy.linalg.lapack import dgttrf, dgttrs
        dl, d, du, du2, ipiv, info = dgttrf(a[2, :-1], a[1], a[0, 1:])
        if info > 0:
            return lambda rhs: solve_banded((1, 1), a, rhs, overwrite_b=True, check_finite=False)
        if info < 0:
            raise ValueError(f'illegal value in {-info}-th argument of internal gbsv/gtsv')
        def solve(rhs):
            x, info = dgttrs(dl, d, du, du2, ipiv, rhs, overwrite_b=1)
            if info != 0:
                raise ValueError(f'illegal value in {-info}-th argument of internal gbsv/gtsv')
            return x
        return solve
    def pentadiagonalSolver(ab):
        from scipy.linalg import solve_banded
        from scipy.linalg.lapack import dgbtrf, dgbtrs
        lab = np.zeros((7, n))
        lab[2:] = ab
        lu, piv, info = dgbtrf(lab, 2, 2)
        if info != 0:
            return lambda rhs: solve_banded((2, 2), ab, rhs, check_finite=False)
        def solve(rhs):
            x, info = dgbtrs(lu, 2, 2, rhs, piv)
            if info != 0:
                raise ValueError(f'illegal value in {-info}-th argument of internal gbtrs')
            return x
        return solve
    def adeTheta():
        scheme = centralDict.get('adeTimeScheme') or 'euler'
        key = str(scheme).lower().replace('-', '').replace('_', '').replace(' ', '')
        if key not in ADE_TIME_SCHEMES:
            logError('adeImplicit', f"unknown adeTimeScheme '{scheme}' : use 'euler' (implicit Euler, first order) "
                     "or 'crank-nicolson' (second order)")
        return ADE_TIME_SCHEMES[key]
    # implicit ADE, theta scheme over each of the adeSubSteps steps h :
    #   (M/h + theta.A) C_new = (M/h - (1 - theta).A) C_old + sources,  A = transport operator + Cauchy rows + sink
    # theta = 1 : implicit Euler (first order in time), theta = 1/2 : Crank-Nicolson (second order in time).
    # Upwind : tridiagonal matrix ; LUD : pentadiagonal (second upstream band) ; van Leer : upwind matrix plus the
    # flux-limited correction (TVD), iterated at the new time level with the same LU (fixed point), the step being
    # halved when the iteration does not converge. The LU of each matrix is kept (cachedFactorization).
    def adeImplicit(C, storage, Lb, fixed, fixVal, bDiag, bSrc, Q, S=None, K=None):
        try:
            from scipy.linalg import solve_banded
        except ImportError:
            logError('adeImplicit', "adeSolver = 'implicit' needs scipy : pip install scipy")
        nSteps = centralDict.get('adeSubSteps') or 1
        if int(nSteps) != nSteps or nSteps < 1:
            logError('adeImplicit', f"adeSubSteps must be an integer >= 1 (got {nSteps})")
        nSteps = int(nSteps)
        theta = adeTheta()
        tvd = advScheme == 'vanleer' and Q != 0
        far = farBands(Lb)
        penta = far[0] or far[1]
        stats = {'iterations': 0, 'splits': 0, 'positivity': False}
        if theta < 1:
            # Crank-Nicolson : the explicit half (1 - theta).A is positive only if (1 - theta).h.max(A/M) <= 1 (<= 1/2
            # with the van Leer limiter) ; the number of steps is raised to keep it (no oscillation, no negative c)
            diag = np.broadcast_to(Lb[2], C.shape).copy()
            diag[0] += bDiag[0]
            diag[-1] += bDiag[1]
            if K is not None:
                diag += storage[:, None] * K
            rate = float(np.max((diag / storage[:, None])[~fixed], initial=0.0))
            need = int(np.ceil((1.0 - theta) * dt * rate / (0.5 if tvd else 1.0)))
            if need > nSteps:
                nSteps, stats['positivity'] = need, True
            if penta:
                logWarning('adeImplicit', "advectionScheme = 'LUD' is not monotone : with adeTimeScheme = "
                           "'crank-nicolson' it can oscillate at steep fronts (use 'vanLeer' to avoid it)")
        h = dt / nSteps
        perSpecies = Lb.shape[2] > 1
        groups = {}
        for j in range(C.shape[1]):
            key = (bool(fixed[0, j]), bool(fixed[-1, j]), float(bDiag[0, j]), float(bDiag[1, j]),
                   j if perSpecies else 0, j if K is not None else -1)
            groups.setdefault(key, []).append(j)
        C = C.copy()
        C[fixed] = fixVal[fixed]
        I = np.zeros_like(C)
        # coupling matrix : the species groups are solved one after the other over the whole call ; the state of every
        # species at the end of each step is kept, the matrices are computed after the last group
        monitor = couplingMonitor(list(trsptedSpecies))
        traj = None if monitor is None else np.empty((nSteps,) + C.shape)
        C0 = None if monitor is None else C.copy()
        for (fixL, fixR, b0, bL, jOp, jK), cols in groups.items():
            L = Lb[:, :, jOp]
            Lg = Lb[:, :, jOp:jOp + 1]
            sinkDiag = None if jK < 0 else storage * K[:, jK]
            solvers = {}
            def solver(hs):
                if hs not in solvers:
                    solvers[hs] = (buildSolver(hs), (storage / hs)[:, None])
                return solvers[hs]
            def buildSolver(hs):
                Mh = storage / hs
                if penta:
                    ab = np.zeros((5, n))
                    ab[0, 2:] = theta * L[4, :-2]
                    ab[1, 1:] = theta * L[3, :-1]
                    ab[2, :] = Mh + theta * L[2]
                    if sinkDiag is not None:
                        ab[2, :] += theta * sinkDiag
                    ab[3, :-1] = theta * L[1, 1:]
                    ab[4, :-2] = theta * L[0, 2:]
                    ab[2, 0] += theta * b0
                    ab[2, -1] += theta * bL
                    if fixL:
                        ab[2, 0], ab[1, 1], ab[0, 2] = 1.0, 0.0, 0.0
                    if fixR:
                        ab[2, -1], ab[3, -2], ab[4, -3] = 1.0, 0.0, 0.0
                    return cachedFactorization((n, 5, ab.tobytes()), lambda: pentadiagonalSolver(ab))
                a = np.zeros((3, n))
                a[0, 1:] = theta * L[3, :-1]
                a[1, :] = Mh + theta * L[2]
                if sinkDiag is not None:
                    a[1, :] += theta * sinkDiag
                a[2, :-1] = theta * L[1, 1:]
                a[1, 0] += theta * b0
                a[1, -1] += theta * bL
                if fixL:
                    a[1, 0], a[0, 1] = 1.0, 0.0
                if fixR:
                    a[1, -1], a[2, -2] = 1.0, 0.0
                return cachedFactorization((n, 3, a.tobytes()), lambda: tridiagonalSolver(a))
            Cg = C[:, cols]
            source = None if S is None else storage[:, None] * S[:, cols]
            src0, srcL = bSrc[0, cols], bSrc[1, cols]
            fix0, fixLast = fixVal[0, cols], fixVal[-1, cols]
            def boundaryRows(rhs):
                if fixL:
                    rhs[0] = fix0
                if fixR:
                    rhs[-1] = fixLast
                return rhs
            def explicitPart(X):
                AX = adeApplyL(X, Lg, far)
                AX[0] += b0 * X[0]
                AX[-1] += bL * X[-1]
                if sinkDiag is not None:
                    AX += sinkDiag[:, None] * X
                if tvd:
                    AX += vanLeerCorrection(X, Q)
                return (1.0 - theta) * AX
            def integral(X, Xn, hs):
                return hs * Xn if theta == 1.0 else hs * (theta * Xn + (1.0 - theta) * X)
            def advance(X, hs, depth):
                solve, MhCol = solver(hs)
                rhs0 = MhCol * X
                if source is not None:
                    rhs0 += source
                rhs0[0] += src0
                rhs0[-1] += srcL
                if theta < 1:
                    rhs0 -= explicitPart(X)
                if not tvd:
                    Xn = solve(boundaryRows(rhs0))
                    return Xn, integral(X, Xn, hs)
                Y = X
                for _ in range(ADE_TVD_MAX_ITER):
                    Yn = solve(boundaryRows(rhs0 - theta * vanLeerCorrection(Y, Q)))
                    stats['iterations'] += 1
                    change, scale = float(np.max(np.abs(Yn - Y))), float(np.max(np.abs(Yn)))
                    Y = Yn
                    if change <= ADE_TVD_TOL * scale:
                        return Y, integral(X, Y, hs)
                if depth >= ADE_TVD_MAX_SPLIT:
                    logError('adeImplicit', f"advectionScheme = 'vanLeer' : the limiter correction does not converge "
                             f"(step {writeTime(hs, 3)} after {ADE_TVD_MAX_SPLIT} halvings) : use advectionScheme = "
                             "'LUD' or 'upwind', or adeSolver = 'explicit'")
                stats['splits'] += 1
                X1, I1 = advance(X, hs / 2, depth + 1)
                X2, I2 = advance(X1, hs / 2, depth + 1)
                return X2, I1 + I2
            Ig = np.zeros((n, len(cols)))
            for s in range(nSteps):
                Cg, dI = advance(Cg, h, 0)
                Ig += dI
                if traj is not None:
                    traj[s][:, cols] = Cg
            I[:, cols] = Ig
            C[:, cols] = Cg
        if monitor is not None:
            width = 2 if (penta or tvd) else 1
            for s in range(nSteps):
                Cp = C0 if s == 0 else traj[s - 1]
                couplingStepFD(monitor, lambda u, Cp=Cp: adeResidual(u.reshape(Cp.shape), Cp, h, theta, storage, Lb,
                                                                     bDiag, bSrc, fixed, fixVal, Q, tvd, S, K).ravel(),
                               Cp.ravel(), traj[s].ravel(), C.shape[1], width, (s + 1) * h, h)
        storeIntegral(trsptedSpecies, I)
        return C, nSteps, stats
    # coupling = 'free' (default) : the chemistry sees the free water only, the double layer (DL) is kept by the
    #            transport from one step to the next (created at t0 from the Donnan equilibrium with the pore water).
    # coupling = 'totals' : at the start of every transport step the DL counter-ions (charge -sigma.thetaDL) are
    #            borrowed from the exchangers of the chemistry (columns <model>_X<z>-<element> of commMtrx, read only),
    #            with the composition of the Donnan equilibrium with the free water ; at the end the chemistry gets
    #            back all the mobile ions (free water + DL) minus what was borrowed (trspt) : no mass created, the
    #            chemistry (exchange, buffering of the DL H+) acts on the totals. counterIons = {element: species}
    #            overrides the default counter-ion of an exchanger element (element + '+z').
    DONNAN_KEYS = {'porosityDL', 'thickness', 'specificSurface', 'CEC', 'bulkDensity', 'chargeDL', 'stern',
                   'tortuosityDL', 'activityDL', 'coupling', 'counterIons'}
    def donnanSetup(spc, Df, reservoirNodes=()):
        cfg = centralDict.get('donnan')
        if not isinstance(cfg, dict):
            logError('donnan', "donnan must be a dict (see nativeTransport.py : 'porosityDL' or 'thickness', 'CEC' + "
                     "'bulkDensity' or 'chargeDL', 'stern', 'tortuosityDL', 'activityDL')")
        unknown = set(cfg) - DONNAN_KEYS
        if unknown:
            logError('donnan', f"unknown key(s) {sorted(unknown)} : use {sorted(DONNAN_KEYS)}")
        def nodeValues(key, default=None, positive=False):
            val = cfg.get(key, default)
            if val is None:
                return None
            arr = np.asarray(val, dtype=float)
            if arr.ndim > 1 or (arr.ndim == 1 and len(arr) != n):
                logError('donnan', f"'{key}' must be a number or one value per node (got {arr.size} values, {n} nodes)")
            arr = meshProp(arr, 0, n)
            if not np.all(np.isfinite(arr)) or np.any(arr < 0) or (positive and np.any(arr <= 0)):
                logError('donnan', f"'{key}' must be {'> 0' if positive else '>= 0'}")
            return arr
        theta = meshProp(centralDict['porosity'], 0, n)
        thickness = cfg.get('thickness')
        if ('porosityDL' in cfg) == (thickness is not None):
            logError('donnan', "give either 'porosityDL' (fraction of the porosity) or 'thickness' ('debye', n)")
        if thickness is not None:
            if (not isinstance(thickness, (list, tuple)) or len(thickness) != 2 or thickness[0] != 'debye'
                    or not float(thickness[1]) > 0):
                logError('donnan', f"'thickness' must be ('debye', n) with n > 0 (got {thickness!r})")
            if 'chargeDL' in cfg or 'specificSurface' not in cfg or 'CEC' not in cfg or 'bulkDensity' not in cfg:
                logError('donnan', "'thickness' needs 'specificSurface', 'CEC' and 'bulkDensity' (not 'chargeDL')")
        fDL = nodeValues('porosityDL')
        if fDL is not None and np.any(fDL >= 1):
            logError('donnan', "porosityDL must be < 1 : the free (electroneutral) porosity theta_f must stay > 0")
        stern = nodeValues('stern', 0.0)
        if np.any(stern > 1):
            logError('donnan', "stern must be in [0, 1]")
        if 'chargeDL' in cfg:
            if 'CEC' in cfg:
                logError('donnan', "give either 'chargeDL' or 'CEC' + 'bulkDensity', not both")
            chargeDL = np.asarray(cfg['chargeDL'], dtype=float)
            chargeDL = meshProp(chargeDL, 0, n)
            qBulk = None
        else:
            if 'CEC' not in cfg or 'bulkDensity' not in cfg:
                logError('donnan', "the charge of the double layer needs 'CEC' [mol_c/kg] and 'bulkDensity' [kg/m3] "
                         "(or 'chargeDL' [mol_c/L of double-layer water])")
            qBulk = -(1.0 - stern) * nodeValues('CEC') * nodeValues('bulkDensity', positive=True)
            chargeDL = None
        tau = cfg.get('tortuosityDL', 1.0)
        if isinstance(tau, dict):
            tauD = np.array([float(tau.get(sp, 1.0)) for sp in spc])
        else:
            tauD = np.full(len(spc), float(tau))
        if np.any(tauD < 0):
            logError('donnan', "tortuosityDL must be >= 0")
        actDL = str(cfg.get('activityDL', 'same')).lower()
        if actDL not in ('same', 'ideal'):
            logError('donnan', f"activityDL must be 'same' or 'ideal' (got {cfg.get('activityDL')!r})")
        coupling = str(cfg.get('coupling', 'free')).lower()
        if coupling not in ('free', 'totals'):
            logError('donnan', f"coupling must be 'free' or 'totals' (got {cfg.get('coupling')!r})")
        counterIons = cfg.get('counterIons') or {}
        if not isinstance(counterIons, dict):
            logError('donnan', "counterIons must be a dict {exchanger element: aqueous species}")
        charges = centralDict.get('speciesValence') or {}
        z = np.array([float(charges[sp]) if sp in charges else float(speciationModule().parse_charge(sp)) for sp in spc])
        tortF = meshProp(centralDict.get('tortuosity', 1.0), 0, n)
        DD = tauD[:, None] * Df / tortF[None, :]
        aS = nodeValues('specificSurface') if thickness is not None else None
        rhoB = nodeValues('bulkDensity') if thickness is not None else None
        nDebye = float(thickness[1]) if thickness is not None else None
        noDL = np.zeros(n, dtype=bool)
        noDL[list(reservoirNodes)] = True
        act = activitySetup(spc) if actDL == 'ideal' else None
        warned = {}
        def params(c):
            if thickness is not None:
                I = 0.5 * ((z ** 2)[:, None] * np.maximum(c, 0.0)).sum(axis=0)
                thDL = aS * rhoB * nDebye * 0.304e-9 / np.sqrt(np.maximum(I, 1e-12))
                cap = 0.95 * theta
                if np.any(thDL > cap) and not warned.get('cap'):
                    logWarning('donnan', "the double-layer porosity from 'thickness' exceeds 0.95.theta (low ionic "
                               "strength) : it is capped at 0.95.theta")
                    warned['cap'] = True
                thDL = np.minimum(thDL, cap)
            else:
                thDL = fDL * theta
            thDL = np.where(noDL, 0.0, thDL)
            thF = theta - thDL
            if qBulk is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    sigma = np.where(thDL > 0, qBulk / (1000.0 * thDL), 0.0)
            else:
                sigma = np.where(thDL > 0, chargeDL, 0.0)
            phi = np.exp(lnGamma(act, c)) if act else np.ones_like(c)
            return thF, thDL, sigma, phi
        return {'spc': list(spc), 'z': z, 'Df': Df, 'DD': DD, 'params': params, 'theta': theta,
                'coupling': coupling, 'counterIons': counterIons}
    def donnanConductance(dn, psi, par, disp=0.0, reservoirFaces=()):
        thF, thDL, sigma, phi = par
        e = _expClip(-dn['z'][:, None] * psi[None, :])
        dEff = thF[None, :] * dn['Df'] + thDL[None, :] * dn['DD'] * phi * e
        dxh = meshProp(centralDict['dxHalfCell'], 0, n)
        areaR, areaL = halfCellAreas(0, n)
        R = []
        for area in (areaR, areaL):
            cond = area[None, :] * dEff + disp
            with np.errstate(divide='ignore', invalid='ignore'):
                Rh = dxh[None, :] / np.where(cond > 0, cond, 1.0)
            Rh[cond <= 0] = np.inf
            R.append(Rh)
        Rsum = R[0][:, :-1] + R[1][:, 1:]
        with np.errstate(divide='ignore', invalid='ignore'):
            T = np.where(np.isfinite(Rsum) & (Rsum > 0), 1.0 / Rsum, 0.0)
        for face, smp in reservoirFaces:
            Rs = R[1][:, smp] if face == 0 else R[0][:, smp]
            T[:, face] = np.where(np.isfinite(Rs), 1.0 / Rs, 0.0)
        return T
    def donnanBorrow(dn, c, par):
        # coupling 'totals' : DL counter-ions borrowed from the exchangers, mol/L of medium (species x nodes)
        thF, thDL, sigma, phi = par
        spc, z = dn['spc'], dn['z']
        q = np.where(thDL > 0, -sigma * thDL, 0.0)
        avail = np.zeros(c.shape)
        pattern = re.compile(r'^(.+?)_X(\d*)-(.+)$')
        for col in commMtrx.columns:
            m = pattern.match(str(col))
            if not m:
                continue
            zx = int(m.group(2) or 1)
            sp = dn['counterIons'].get(m.group(3), m.group(3) + ('+' if zx == 1 else f'+{zx}'))
            if sp not in spc:
                logWarning('donnan', f"coupling 'totals' : no transported species '{sp}' for the exchanger column "
                           f"'{col}' (give counterIons = {{'{m.group(3)}': species}}) : not borrowed")
                continue
            avail[spc.index(sp)] += np.maximum(commMtrx[col].to_numpy(dtype=float), 0.0)
        side = np.sign(q)[None, :] * z[:, None] > 0
        avail = np.where(side, avail, 0.0)
        psi0 = donnanPotential(c, z, sigma, phi)[0]
        w = thDL[None, :] * phi * np.maximum(c, 0.0) * _expClip(-z[:, None] * psi0[None, :])
        w = np.where(avail > 0, w + 1e-12 * avail, 0.0)
        zAbs = np.abs(z)[:, None]
        def charge(s):
            return (zAbs * np.minimum(s[None, :] * w, avail)).sum(axis=0)
        qAbs = np.abs(q)
        short = charge(np.full(n, 1e300)) < qAbs * (1.0 - 1e-9)
        if np.any(short & (qAbs > 0)):
            logWarning('donnan', "coupling 'totals' : the exchangers cannot supply the double-layer charge at "
                       f"node(s) {np.flatnonzero(short & (qAbs > 0))[:10].tolist()} : all their counter-ions are borrowed")
        lo_, hi_ = np.full(n, -300.0), np.full(n, 300.0)
        for _ in range(200):
            mid = 0.5 * (lo_ + hi_)
            low = charge(10.0 ** mid) < qAbs
            lo_, hi_ = np.where(low, mid, lo_), np.where(low, hi_, mid)
        T = np.minimum((10.0 ** hi_)[None, :] * w, avail)
        got = (zAbs * T).sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            T = T * np.where((got > 0) & ~short, qAbs / got, 1.0)[None, :]
        return np.where(qAbs[None, :] > 0, np.minimum(T, avail), 0.0)
    def donnanStart(dn, c, fixed=None, fixVal=None):
        par = dn['params'](c)
        thF, thDL, sigma, phi = par
        state = centralDict.get('donnanState')
        if dn['coupling'] == 'totals' and centralDict.pop('_donnanBorrow', False):
            # chemistry output = whole pore water (theta) : the DL is rebuilt from the exchangers
            theta = meshProp(centralDict['porosity'], 0, n)
            T = donnanBorrow(dn, c, par)
            prev = centralDict.get('donnanPrevious')
            if prev and prev['species'] == dn['spc'] and prev['c'].shape == c.shape:
                # nodes untouched since the previous transport (no chemistry in between, e.g. operatorSplitting =
                # 'alternative') : same borrowing, so that theta.c + T gives back exactly the previous totals
                same = np.all(c == prev['c'], axis=0)
                T = np.where(same[None, :], prev['T'], T)
            centralDict['donnanBorrowed'] = {'species': list(dn['spc']), 'T': T, 'theta': theta}
            Ctot = theta[None, :] * c + T
            psi0 = state['psi'] if state and state['psi'].shape == thF.shape else None
        elif state and state['species'] == dn['spc'] and state['Ctot'].shape == c.shape:
            Ctot = state['Ctot'] + state['thF'][None, :] * (c - state['cf'])
            psi0 = state['psi']
        else:
            Ctot = donnanStorage(c, dn['z'], thF, thDL, sigma, phi)[0]
            psi0 = None
        cf, psi, ok = donnanFromTotals(Ctot, dn['z'], thF, thDL, sigma, phi, fixed, fixVal, psi0)
        if not np.all(ok):
            logWarning('donnan', "the double-layer charge cannot be compensated (no counter-ion) at node(s) "
                       f"{np.flatnonzero(~ok)[:10].tolist()} : Psi_D is bounded")
        Ctot = donnanStorage(cf, dn['z'], thF, thDL, sigma, phi, psi)[0]
        return cf, Ctot, psi, par
    def donnanEnd(dn, cf, Ctot, psi, par, temperature):
        thF, thDL, sigma, phi = par
        centralDict['donnanState'] = {'species': dn['spc'], 'Ctot': Ctot.copy(), 'cf': cf.copy(), 'thF': thF.copy(),
                                      'psi': psi.copy()}
        cD = phi * cf * _expClip(-dn['z'][:, None] * psi[None, :])
        out = {'psiD_V': psi * GAS_CONSTANT * temperature / FARADAY, 'thetaDL': thDL.copy()}
        for i, sp in enumerate(dn['spc']):
            out[f"{sp}_DL"] = np.where(thDL > 0, cD[i], 0.0)
        centralDict['donnanOutput'] = out
    def adeExplicitDonnan(C, fixed, fixVal, bDiag, bSrc, Q, dG, S=None, K=None):
        spc = list(trsptedSpecies)
        safety = float(centralDict.get('cflSafety') or 0.9)
        nMax = int(centralDict.get('maxSubCycling') or 100000)
        tvd = advScheme == 'vanleer' and Q != 0
        heun = (advScheme == 'lud' or tvd) and Q != 0
        kappa = 0.75 if (heun and not tvd) else 1.0
        dn = donnanSetup(spc, np.tile(meshProp(centralDict['diffCoeff'], 0, n), (len(spc), 1)))
        V = cellVolume(0, n)
        alphaL = float(centralDict.get('dispersivity') or 0.0)
        disp = alphaL * abs(Q)
        cf, Ctot, psi, par = donnanStart(dn, C.T, fixed.T, fixVal.T)
        free = ~np.all(fixed, axis=1)
        I = np.zeros(C.shape)
        def operator(cf, psi, par):
            T = donnanConductance(dn, psi, par, disp).T
            Tmol = donnanConductance(dn, psi, par, 0.0).T if disp else T
            return adeMatrix(T, Tmol, Q, dG)
        def rhs(cfT, Lb, src):
            LC = adeApplyL(cfT, Lb)
            if tvd:
                LC += vanLeerCorrection(cfT, Q)
            LC[0] += bDiag[0] * cfT[0] - bSrc[0]
            LC[-1] += bDiag[1] * cfT[-1] - bSrc[1]
            if src is not None:
                return -LC / V[:, None] + src
            return -LC / V[:, None]
        def speciate(CtotT, par, psi0):
            thF, thDL, sigma, phi = par
            c, p, _ = donnanFromTotals(CtotT.T, dn['z'], thF, thDL, sigma, phi, fixed.T, fixVal.T, psi0)
            Cc = donnanStorage(c, dn['z'], thF, thDL, sigma, phi, p)[0]
            return c, Cc, p
        M = len(spc)
        eyeM = np.eye(M)
        def eigenRate(thEff, u, v, w, out, nodes):
            fx = fixed[nodes]
            J = (thEff.T[nodes][:, :, None] * eyeM[None]
                 - w[nodes][:, None, None] * u.T[nodes][:, :, None] * v.T[nodes][:, None, :])
            J = np.where(fx[:, :, None] | fx[:, None, :], eyeM[None], J)
            A = np.linalg.solve(J, eyeM[None] * out[nodes][:, None, :])
            return np.max(np.abs(np.linalg.eigvals(A)), axis=1)
        def localRate(cf, psi, par, diagB):
            thF, thDL, sig, phi = par
            _, thEff, u, v, G, _ = donnanStorage(cf, dn['z'], thF, thDL, sig, phi, psi)
            with np.errstate(divide='ignore', invalid='ignore'):
                w = np.where(G > 0, thDL / G, 0.0)
            out = np.where(fixed, 0.0, diagB) / V[:, None]
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                th = np.where(fixed, 1.0, thEff.T)
                uf = np.where(fixed, 0.0, u.T)
                vf = np.where(fixed, 0.0, v.T)
                delta = out / th
                den = 1.0 - w * (vf * uf / th).sum(axis=1)
                beta = w / den
                a = uf * vf * delta / th
            ok = (np.all(np.isfinite(delta) & (delta >= 0) & (th > 0) & np.isfinite(a) & (a >= 0), axis=1)
                  & np.isfinite(beta) & (beta >= 0) & (den > 0))
            rate = np.empty(n)
            if not np.all(ok):
                rate[~ok] = eigenRate(thEff, u, v, w, out, ~ok)
            dmax = delta.max(axis=1)
            rate[ok] = dmax[ok]
            pos = a > 0
            coupled = ok & np.any(pos, axis=1) & (beta > 0)
            if np.any(coupled):
                rate[coupled] = secularRoot(delta[coupled], np.where(pos[coupled], a[coupled], 0.0),
                                            pos[coupled], beta[coupled], dmax[coupled])
            return rate
        def secularRoot(delta, a, pos, beta, dmax):
            dstar = np.where(pos, delta, -np.inf).max(axis=1)
            gamma = np.where(pos, dstar[:, None] - delta, 1.0)
            star = pos & (gamma == 0)
            mu = beta * np.where(star, a, 0.0).sum(axis=1)
            active = np.ones(len(mu), dtype=bool)
            for _ in range(100):
                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    r = a[active] / (mu[active][:, None] + gamma[active])
                    s1 = r.sum(axis=1)
                    s2 = (r / (mu[active][:, None] + gamma[active])).sum(axis=1)
                    step = (beta[active] * s1 - 1.0) * s1 / s2
                stop = ~np.isfinite(step) | (step <= 4e-16 * mu[active])
                idx = np.flatnonzero(active)
                mu[idx[~stop]] += step[~stop]
                active[idx[stop]] = False
                if not np.any(active):
                    break
            return np.maximum(dmax, dstar + mu)
        def donnanResidual(cT, CtotOld, h, par, psi0):
            # equation of the step (coupling matrix, mol/s) : equivalent implicit Euler residual of the free
            # concentrations, as adeImplicitDonnan, the parameters of the step and the lagged terms frozen
            thF, thDL, sig, phi = par
            c = cT.T
            psiD = donnanPotential(c, dn['z'], sig, phi, psi0=psi0)[0]
            Cb = donnanStorage(c, dn['z'], thF, thDL, sig, phi, psiD)[0]
            R = V[:, None] * (Cb - CtotOld).T / h + adeApplyL(cT, operator(c, psiD, par))
            if tvd:
                R += vanLeerCorrection(cT, Q)
            R[0] += bDiag[0] * cT[0] - bSrc[0]
            R[-1] += bDiag[1] * cT[-1] - bSrc[1]
            if S is not None:
                R -= (V * thF)[:, None] * S
            if K is not None:
                R += (V * thF)[:, None] * K * cT
            R = 1000.0 * R
            R[fixed] = (cT - fixVal)[fixed]
            return R
        monitor = couplingMonitor(spc)
        width = 2 if heun else 1
        t, nSub = 0.0, 0
        while dt - t > 1e-12 * dt:
            par = dn['params'](cf)
            cfPrev, CtotPrev = cf, Ctot
            Lb = operator(cf, psi, par)
            diagB = np.broadcast_to(Lb[2], C.shape).copy()
            diagB[0] += bDiag[0]
            diagB[-1] += bDiag[1]
            if tvd:
                diagB += abs(Q)
            rate = localRate(cf, psi, par, diagB)[free]
            rateMax = float(np.max(rate)) if rate.size else 0.0
            h = (dt - t) if rateMax <= 0 else (dt - t) / max(1, int(np.ceil((dt - t) * rateMax / (safety * kappa))))
            src = None if S is None else par[0][:, None] * S
            k1 = rhs(cf.T, Lb, src)
            c1, C1, p1 = speciate(Ctot.T + h * k1, par, psi)
            if heun:
                k2 = rhs(c1.T, operator(c1, p1, par), src)
                cf, Ctot, psi = speciate(0.5 * (Ctot.T + C1.T + h * k2), par, p1)
            else:
                cf, Ctot, psi = c1, C1, p1
            if K is not None:
                cKeep = cf / (1.0 + h * K.T)
                cf, Ctot, psi = speciate((Ctot - par[0][None, :] * (cf - cKeep)).T, par, psi)
                I += h * cKeep.T
            else:
                I += h * cf.T
            if monitor is not None:
                couplingStepFD(monitor, lambda u, Co=CtotPrev, hs=h, pa=par, p0=psi:
                               donnanResidual(u.reshape(n, M), Co, hs, pa, p0).ravel(),
                               cfPrev.T.ravel(), cf.T.ravel(), M, width, t + h, h)
            t += h
            nSub += 1
            if nSub > nMax:
                k = int(np.flatnonzero(free)[np.argmax(rate)])
                logError('adeExplicitDonnan', f"more than {nMax} sub-steps needed (dt_sub = {writeTime(h, 3)}, limiting "
                         f"node {k}) : increase maxSubCycling")
        donnanEnd(dn, cf, Ctot, psi, par, float(centralDict.get('temperature', 298.15)))
        storeIntegral(spc, I)
        return cf.T, nSub
    def adeImplicitDonnan(C, fixed, fixVal, bDiag, bSrc, Q, dG, S=None, K=None):
        try:
            from scipy.linalg import solve_banded
        except ImportError:
            logError('adeImplicitDonnan', "adeSolver = 'implicit' needs scipy : pip install scipy")
        nSteps = centralDict.get('adeSubSteps') or 1
        if int(nSteps) != nSteps or nSteps < 1:
            logError('adeImplicitDonnan', f"adeSubSteps must be an integer >= 1 (got {nSteps})")
        h0 = dt / int(nSteps)
        maxIter = int(centralDict.get('npImplicitMaxIter') or 30)
        spc = list(trsptedSpecies)
        M = len(spc)
        eyeM = np.eye(M)
        dn = donnanSetup(spc, np.tile(meshProp(centralDict['diffCoeff'], 0, n), (M, 1)))
        z = dn['z']
        V = cellVolume(0, n)
        disp = float(centralDict.get('dispersivity') or 0.0) * abs(Q)
        cf, Ctot, psi, par = donnanStart(dn, C.T, fixed.T, fixVal.T)
        I = np.zeros(C.shape)
        idx = np.arange(n * M).reshape(n, M)
        ii, jj = np.meshgrid(np.arange(M), np.arange(M), indexing='ij')
        rows = np.concatenate([idx[:, ii.ravel()].ravel(), idx[1:].ravel(), idx[:-1].ravel()])
        cols = np.concatenate([idx[:, jj.ravel()].ravel(), idx[:-1].ravel(), idx[1:].ravel()])
        fixedFlat = fixed.ravel()
        keep = ~fixedFlat[rows]
        rowsK, colsK = rows[keep], cols[keep]
        fixedIdx = np.flatnonzero(fixedFlat)
        bandPos = (M + rowsK - colsK, colsK)
        bandFlat = bandPos[0] * (n * M) + bandPos[1]
        bandUnique = len(np.unique(bandFlat)) == len(bandFlat)
        def system(c, CtotOld, h, par, psi0, needJac):
            thF, thDL, sig, phi = par
            psiD = donnanPotential(c, z, sig, phi, psi0=psi0)[0]
            Cb, thEff, u, v, G, _ = donnanStorage(c, z, thF, thDL, sig, phi, psiD)
            T = donnanConductance(dn, psiD, par, disp).T
            Tmol = donnanConductance(dn, psiD, par, 0.0).T if disp else T
            Lb = adeMatrix(T, Tmol, Q, dG)
            cT = c.T
            R = V[:, None] * (Cb - CtotOld).T / h + adeApplyL(cT, Lb)
            R[0] += bDiag[0] * cT[0] - bSrc[0]
            R[-1] += bDiag[1] * cT[-1] - bSrc[1]
            if S is not None:
                R -= (V * thF)[:, None] * S
            if K is not None:
                R += (V * thF)[:, None] * K * cT
            R[fixed] = (cT - fixVal)[fixed]
            if not needJac:
                return R, psiD, None
            with np.errstate(divide='ignore', invalid='ignore'):
                w = np.where(G > 0, thDL / G, 0.0)
            blk = (V / h)[:, None, None] * (thEff.T[:, :, None] * eyeM[None]
                                             - w[:, None, None] * u.T[:, :, None] * v.T[:, None, :])
            dia = np.broadcast_to(Lb[2], (n, M)).copy()
            dia[0] += bDiag[0]
            dia[-1] += bDiag[1]
            if K is not None:
                dia += (V * thF)[:, None] * K
            blk[:, np.arange(M), np.arange(M)] += dia
            vals = np.concatenate([blk.reshape(n, M * M).ravel(),
                                   np.broadcast_to(Lb[1][1:], (n - 1, M)).ravel(),
                                   np.broadcast_to(Lb[3][:-1], (n - 1, M)).ravel()])
            ab = np.zeros((2 * M + 1, n * M))
            if bandUnique:
                ab.reshape(-1)[bandFlat] = vals[keep] + 0.0
            else:
                np.add.at(ab, bandPos, vals[keep])
            ab[M, fixedIdx] = 1.0
            return R, psiD, ab
        def jacobianCheck(c, CtotOld, h, par, psi0, ab):
            N = n * M
            Aan = np.zeros((N, N))
            for r in range(N):
                for col in range(max(0, r - M), min(N, r + M + 1)):
                    Aan[r, col] = ab[M + r - col, col]
            Afd = np.zeros((N, N))
            for p in range(N):
                k, i = divmod(p, M)
                step = 1e-7 * max(abs(c[i, k]), 1e-12)
                cp, cm = c.copy(), c.copy()
                cp[i, k] += step
                cm[i, k] -= step
                Afd[:, p] = (system(cp, CtotOld, h, par, psi0, False)[0].ravel()
                             - system(cm, CtotOld, h, par, psi0, False)[0].ravel()) / (2 * step)
            return float(np.max(np.abs(Aan - Afd)) / np.max(np.abs(Afd)))
        stats = {'steps': 0, 'newton': 0, 'split': 0, 'maxIter': 0, 'factor': 0}
        from scipy.linalg.lapack import dgbtrf, dgbtrs
        def factorize(ab):
            a2 = np.zeros((3 * M + 1, n * M))
            a2[M:] = ab
            lu, piv, info = dgbtrf(a2, M, M, overwrite_ab=1)
            stats['factor'] += 1
            return (lu, piv) if info == 0 else None
        def newton(c0, CtotOld, h, par, psi0):
            c = c0.copy()
            c[fixed.T] = fixVal.T[fixed.T]
            psiD = psi0
            floor = 1e-14 * max(float(np.max(np.abs(c))), 1e-300)
            LU, lastChange, refactor = None, None, True
            for it in range(1, maxIter + 1):
                needJac = refactor or LU is None
                R, psiD, ab = system(c, CtotOld, h, par, psiD, needJac)
                if needJac:
                    if centralDict.get('_donnanJacobianCheck') is True:
                        centralDict['_donnanJacobianCheck'] = jacobianCheck(c, CtotOld, h, par, psiD, ab)
                    LU = factorize(ab)
                    if LU is None:
                        return None, it
                dx, info = dgbtrs(LU[0], M, M, -R.ravel(), LU[1])
                if info != 0 or not np.all(np.isfinite(dx)):
                    if needJac:
                        return None, it
                    refactor = True
                    continue
                cNew = np.maximum(c + dx.reshape(n, M).T, 0.0)
                change = float(np.max(np.abs(cNew - c) / (np.abs(cNew) + floor)))
                c = cNew
                if change < 1e-10:
                    psiD = donnanPotential(c, z, par[2], par[3], psi0=psiD)[0]
                    return (c, psiD), it
                refactor = lastChange is not None and change > 0.1 * lastChange
                lastChange = change
            return None, maxIter
        monitor = couplingMonitor(spc)
        t, h = 0.0, h0
        while dt - t > 1e-12 * dt:
            hTry = min(h, dt - t)
            par = dn['params'](cf)
            cfPrev, CtotPrev = cf, Ctot
            sol, it = newton(cf, Ctot, hTry, par, psi)
            stats['newton'] += it
            if sol is None:
                stats['split'] += 1
                h = hTry / 2.0
                if h < 1e-10 * dt:
                    logError('adeImplicitDonnan', f"Newton did not converge (time step below {writeTime(h, 3)}) : "
                             "check the Donnan parameters or increase npImplicitMaxIter")
                continue
            cf, psi = sol
            if monitor is not None:
                # coupling matrix : Jacobian of the step at its converged solution (mol/s), Dirichlet rows : identity.
                # By differences (fdJacobian) : the Newton matrix leaves out the dependence of the conductances on the
                # Donnan potential of the neighbour nodes (a coupling between the nodes and the species)
                def stepResidual(u, Co=CtotPrev, hs=hTry, pa=par, p0=psi):
                    R = system(u.reshape(n, M).T, Co, hs, pa, p0, False)[0]
                    return np.where(fixed, R, 1000.0 * R).ravel()
                couplingStepFD(monitor, stepResidual, cfPrev.T.ravel(), cf.T.ravel(), M, 1, t + hTry, hTry)
            Ctot = donnanStorage(cf, z, *par[:3], par[3], psi)[0]
            I += hTry * cf.T
            t += hTry
            stats['steps'] += 1
            stats['maxIter'] = max(stats['maxIter'], it)
            h = min(h0, 2.0 * hTry)
        centralDict['adeDonnanStats'] = stats
        donnanEnd(dn, cf, Ctot, psi, par, float(centralDict.get('temperature', 298.15)))
        storeIntegral(spc, I)
        return cf.T, stats
    # ADE with cross-diffusion (crossDiffusion = {target: {driver: K}}) : pure diffusion of coupled fields with a full
    # (non-diagonal) diffusion matrix, e.g. Soret (driver = temperature), Dufour (target = temperature), or the heat
    # equation itself (temperature as a transported "species" : speciesDiffCoeff = lambda / rhoCp, speciesPorosity = 1).
    #   phi_i dC_i/dt = div( phi_i D_i grad C_i + sum_j phi_i K_ij grad C_j ) + phi_i (S_i - Ksink_i C_i)
    # phi_i, D_i : speciesPorosity / speciesDiffCoeff of the target (porosity / diffCoeff by default) ; K_ij is a pore
    # coefficient like D_i [m2/s x unit of C_i / unit of C_j], of any sign ; it may vary from node to node.
    # Finite volumes on the nodes of the mesh (same faces, volumes and half-cell distances as the ADE), face
    # conductances in series (harmonic mean). All the transported species are solved together, theta scheme
    # (adeTimeScheme : implicit Euler or Crank-Nicolson) on adeSubSteps equal steps, sparse LU of the block matrix.
    # Boundaries : 'constant' (Dirichlet) or 'closed' (no total flux, i.e. no direct nor cross flux). No advection.
    # A driver that is not transported by this call (e.g. the temperature, transported by another instance of
    # solverInstances) is frozen at its value at the start of the call : its cross flux is an explicit source
    # (partitioned coupling, first order in the coupling).
    def adeCrossDiffusionTransport(C_old):
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import splu
        spc = list(trsptedSpecies)
        m = len(spc)
        C = C_old[spc].to_numpy(dtype=float).copy()
        if float(centralDict.get('velocity') or 0.0) != 0.0:
            logError('crossDiffusion', "crossDiffusion is implemented for pure diffusion : velocity must be 0")
        if useDonnan or useActivity:
            logError('crossDiffusion', "crossDiffusion cannot be combined with donnan or activityGradient")
        cross = {t: d for t, d in (centralDict.get('crossDiffusion') or {}).items() if t in spc}
        external = sorted({str(d) for drivers in cross.values() for d in drivers if d not in spc})
        unknown = [d for d in external if d not in commMtrx.columns]
        if unknown:
            logError('crossDiffusion', f"driving species {unknown} : not transported by this call (transportedSpecies : "
                     f"{spc}) and not a column of the coupled variables")
        fixed = np.zeros((n, m), dtype=bool)
        fixVal = np.zeros((n, m))
        for j, sp in enumerate(spc):
            for side, node in ((0, 0), (1, -1)):
                kind, val = boundarySpec(sp, side)
                if kind == 'cauchy':
                    logError('crossDiffusion', "boundary 'flux' (Cauchy) is not implemented with crossDiffusion : "
                             "use 'constant' or 'closed'")
                if kind == 'dirichlet':
                    fixed[node, j], fixVal[node, j] = True, val
        S, Ksink = reactionTerms(spc)
        if n < 2:
            logError('crossDiffusion', "crossDiffusion needs at least 2 nodes")
        D_sp = resolve_species_property('speciesDiffCoeff', 'diffCoeff')
        phi_sp = resolve_species_property('speciesPorosity', 'porosity')
        dxh = meshProp(centralDict['dxHalfCell'], 0, n)
        volume = cellVolume(0, n)
        areaR, areaL = halfCellAreas(0, n)
        phi = np.array([phi_sp[sp] for sp in spc])                    # (m, n)
        if np.any(phi <= 0) or np.any(volume <= 0) or np.any(dxh <= 0):
            logError('crossDiffusion', "porosity, cell volume and dxHalfCell must be > 0")
        storage = phi * volume[None, :]                               # (m, n)

        def faceConductance(phiT, coef):
            # conductance of the n - 1 faces for a flux phiT.coef.grad, half cells in series ; sign of coef kept
            mag = np.abs(coef)
            R = []
            for area in (areaR, areaL):
                cond = phiT * area * mag
                with np.errstate(divide='ignore', invalid='ignore'):
                    r = dxh / np.where(cond > 0, cond, 1.0)
                r[cond <= 0] = np.inf
                R.append(r)
            Rs = R[0][:-1] + R[1][1:]
            G = np.zeros(n - 1)
            ok = np.isfinite(Rs) & (Rs > 0)
            G[ok] = 1.0 / Rs[ok]
            sign = np.sign(coef[:-1] + coef[1:])
            return G * sign

        # operator A (flux balance, A.C = - sum of the fluxes entering each node) as COO triplets
        rows, cols, vals = [], [], []
        def addBlock(i, j, G):
            k = np.arange(n - 1)
            for a, b, v in ((k, k, G), (k + 1, k + 1, G), (k, k + 1, -G), (k + 1, k, -G)):
                rows.append(i * n + a)
                cols.append(j * n + b)
                vals.append(v)
        extSrc = np.zeros((m, n))          # explicit cross fluxes of the drivers not transported by this call
        for i, sp in enumerate(spc):
            D = meshProp(D_sp[sp], 0, n)
            if np.any(D < 0):
                logError('crossDiffusion', f"diffusion coefficient of {sp} must be >= 0")
            addBlock(i, i, faceConductance(phi[i], D))
            for driver, K in (cross.get(sp) or {}).items():
                G = faceConductance(phi[i], meshProp(K, 0, n))
                if driver in spc:
                    addBlock(i, spc.index(driver), G)
                    continue
                Cj = commMtrx[driver].to_numpy(dtype=float)
                flux = G * (Cj[1:] - Cj[:-1])            # entering node k from node k + 1
                extSrc[i, :-1] += flux
                extSrc[i, 1:] -= flux
        N = m * n
        A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)).tocsr()
        Mdiag = storage.ravel()
        sink = np.zeros(N) if Ksink is None else Ksink.T.ravel()
        src = (np.zeros(N) if S is None else S.T.ravel()) + (extSrc / storage).ravel()
        th = adeTheta()
        nSub = int(centralDict.get('adeSubSteps') or 1)
        h = dt / nSub
        from scipy.sparse import diags
        # theta scheme written on the increment dX = X_new - X : (M/h + th.B) dX = - B X + M S, B = A + M.Ksink.
        # Unknowns and equations scaled by the magnitude of each field (Sc^-1 L Sc) : the fields may differ by many
        # orders of magnitude (temperature ~ 300 K, trace element ~ 1e-7 mol/l) and the LU is only accurate in norm ;
        # without the scaling (and the increments) the round-off of the large fields pollutes the small ones.
        B = (A + diags(Mdiag * sink)).tocsr()
        scale = np.array([np.max(np.abs(C[:, i])) or 1.0 for i in range(m)])
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        Sc = np.repeat(scale, n)
        lhs = (diags(1.0 / Sc) @ (diags(Mdiag / h) + th * B) @ diags(Sc)).tolil()
        fixedFlat = fixed.T.ravel()
        for r in np.flatnonzero(fixedFlat):
            lhs.rows[r], lhs.data[r] = [int(r)], [1.0]
        lhs = lhs.tocsc()
        key = ('crossDiffusion', N, h, lhs.data.tobytes(), lhs.indices.tobytes(), lhs.indptr.tobytes())
        solve = cachedFactorization(key, lambda: splu(lhs).solve)
        X = C.T.ravel()
        X[fixedFlat] = fixVal.T.ravel()[fixedFlat]
        integral = np.zeros(N)
        monitor = couplingMonitor(spc)
        if monitor is not None:
            # coupling matrix : matrix of the step M/h + th.B (mol/s), Dirichlet rows : identity ; given with the
            # unknowns ranged node by node (index l.m + i of the unknown i.n + l of X)
            nm = (np.arange(N) % n) * m + np.arange(N) // n
            Jc = (diags(Mdiag / h) + th * B).tocoo()
            keepJ = ~fixedFlat[Jc.row]
            fixIdx = np.flatnonzero(fixedFlat)
            jRows = nm[np.concatenate([Jc.row[keepJ], fixIdx])]
            jCols = nm[np.concatenate([Jc.col[keepJ], fixIdx])]
            jVals = np.concatenate([1000.0 * Jc.data[keepJ], np.ones(len(fixIdx))])
            fixFlatVal = fixVal.T.ravel()
            def toNodes(Y):
                out = np.empty(N)
                out[nm] = Y
                return out
            def crossResidual(u, Xp):
                Y = u[nm]
                R = 1000.0 * (Mdiag * (Y - Xp) / h + th * (B @ Y) + (1.0 - th) * (B @ Xp) - Mdiag * src)
                R[fixedFlat] = (Y - fixFlatVal)[fixedFlat]
                return toNodes(R)
        for iSub in range(nSub):
            b = (Mdiag * src - B @ X) / Sc
            b[fixedFlat] = 0.0
            Xn = X + Sc * solve(b)
            integral += h * (th * Xn + (1 - th) * X)
            if monitor is not None:
                couplingStep(monitor, jRows, jCols, jVals, toNodes(Xn - X), np.arange(N) % m, (iSub + 1) * h, h,
                             (lambda u, Xp=X: crossResidual(u, Xp), toNodes(Xn), m, 1))
            X = Xn
        if not np.all(np.isfinite(X)):
            logError('crossDiffusion', "non-finite values after the transport step")
        storeIntegral(spc, integral.reshape(m, n).T)
        display.stepPrint(f"[cross-diffusion ADE ({'Crank-Nicolson' if th < 1 else 'Euler'}) : {nSub} step(s), "
                          f"{m} coupled fields]", end=" ", flush=True)
        return pd.DataFrame(X.reshape(m, n).T, columns=spc, index=C_old.index)
    def adeTransport(C_old):
        if centralDict.get('crossDiffusion'):
            return adeCrossDiffusionTransport(C_old)
        C = (C_old.to_numpy(dtype=float) if C_old.columns.is_unique and C_old.columns.tolist() == list(trsptedSpecies)
             else C_old[trsptedSpecies].to_numpy(dtype=float))
        fixed = np.zeros(C.shape, dtype=bool)
        fixVal = np.zeros(C.shape)
        for j, sp in enumerate(trsptedSpecies):
            for side, node in ((0, 0), (1, -1)):
                kind, val = boundarySpec(sp, side)
                if kind == 'dirichlet':
                    fixed[node, j], fixVal[node, j] = True, val
        S, Ksink = reactionTerms(list(trsptedSpecies))
        if n < 2:
            if S is not None:
                C = C + dt * S
            if Ksink is not None:
                C = C / (1.0 + dt * Ksink)
            C[fixed] = fixVal[fixed]
            storeIntegral(trsptedSpecies, dt * C)
            return pd.DataFrame(C, columns=trsptedSpecies, index=C_old.index)
        storage, T, Tmol, Q = adeOperator()
        dG = None
        if useActivity:
            allSp = list(centralDict.get('transportedSpecies') or trsptedSpecies)
            cAll = commMtrx[allSp].to_numpy(dtype=float).T
            for i, sp in enumerate(allSp):
                for side, node in ((0, 0), (1, -1)):
                    kind, val = boundarySpec(sp, side)
                    if kind == 'dirichlet':
                        cAll[i, node] = val
            idx = [allSp.index(sp) for sp in trsptedSpecies]
            dG = dLnGamma(activitySetup(allSp), cAll)[idx].T
        if dG is None:
            # from the operator (kept above), the flow rate and the advection scheme only: built once, only read
            Lb = frames.constant(('adeMatrix', n, Q, advScheme), (T, Tmol, centralDict.get('dxHalfCell')),
                                 lambda: adeMatrix(T, Tmol, Q, None), cache=_operators, size=_OPERATORS_SIZE)
        else:
            Lb = adeMatrix(T, Tmol, Q, dG)
        K, cExt, mask = cauchyArrays(list(trsptedSpecies), noInflow=(Q == 0))
        Qin = np.array([max(Q, 0.0), max(-Q, 0.0)])
        bDiag = np.where(mask, K + Qin[:, None], 0.0)
        bSrc = bDiag * cExt
        if useDonnan and adeSolver == 'implicit':
            C, st = adeImplicitDonnan(C, fixed, fixVal, bDiag, bSrc, Q, dG, S, Ksink)
            display.stepPrint(f"[implicit ADE ({advScheme}, Donnan) : {st['steps']} step(s), {st['newton']} Newton iterations, "
                  f"{st['factor']} LU" + (f", {st['split']} split" if st['split'] else "") + "]", end=" ", flush=True)
        elif useDonnan:
            C, nSub = adeExplicitDonnan(C, fixed, fixVal, bDiag, bSrc, Q, dG, S, Ksink)
            display.stepPrint(f"[{nSub} sub-steps ({advScheme}, Donnan)]", end=" ", flush=True)
        elif adeSolver == 'implicit':
            C, nSub, st = adeImplicit(C, storage, Lb, fixed, fixVal, bDiag, bSrc, Q, S, Ksink)
            scheme = 'Crank-Nicolson' if adeTheta() < 1 else 'Euler'
            display.stepPrint(f"[implicit ADE ({advScheme}, {scheme}) : {nSub} step(s)"
                  + (" (raised for positivity)" if st['positivity'] else "")
                  + (f", {st['iterations']} limiter iterations" if st['iterations'] else "")
                  + (f", {st['splits']} step halving(s)" if st['splits'] else "") + "]", end=" ", flush=True)
        else:
            C, nSub = adeExplicit(C, storage, Lb, fixed, fixVal, bDiag, bSrc, Q, S, Ksink)
            display.stepPrint(f"[{nSub} sub-steps ({advScheme})]", end=" ", flush=True)
        if not np.all(np.isfinite(C)):
            logError('adeTransport', "non-finite concentrations after the transport step")
        return pd.DataFrame(C, columns=trsptedSpecies, index=C_old.index)
    def resolve_species_property(dict_key, fallback_key):
        per_species = centralDict.get(dict_key) or {}
        if not hasattr(per_species, 'get'):
            logError('resolve_species_property', f"{dict_key} must be a dict " + "{species: value}, got a "
                     + type(per_species).__name__ + " -> a missing ':' inside the braces makes it a set")
        shared = centralDict[fallback_key]
        return {sp: meshProp(per_species.get(sp, shared), 0, n) for sp in trsptedSpecies}
    def precompute_species_transmissivity(lo, hi, poreD_sp, poro_sp):
        poro = meshProp(poro_sp, lo, hi)
        areaR, areaL = halfCellAreas(lo, hi)
        poreD = meshProp(poreD_sp, lo, hi)
        dx_half = meshProp(centralDict['dxHalfCell'], lo, hi)
        R_half = []
        for area in (areaR, areaL):
            cond = poro * area * poreD
            with np.errstate(divide='ignore', invalid='ignore'):
                R = dx_half / np.where(cond > 0, cond, 1.0)
            R[cond <= 0] = np.inf
            R_half.append(R)
        n_trim = len(cond)
        R_sum = R_half[0][:-1] + R_half[1][1:]
        T_interface = np.zeros(n_trim - 1)
        finite_R = np.isfinite(R_sum) & (R_sum > 0)
        T_interface[finite_R] = 1000.0 / R_sum[finite_R]
        return T_interface
    def logMeanConc(c1, c2):
        eps = 1e-30
        c1 = np.maximum(np.asarray(c1), eps)
        c2 = np.maximum(np.asarray(c2), eps)
        close = np.abs(c2 - c1) <= (1e-9 * np.maximum(c1, c2) + eps)
        with np.errstate(divide='ignore', invalid='ignore'):
            cbar = np.where(close, 0.5 * (c1 + c2), (c2 - c1) / (np.log(c2) - np.log(c1)))
        return cbar

    def logMeanConcRows(c):
        eps = 1e-30
        cPos = np.maximum(c, eps)
        c1, c2 = cPos[:, :-1], cPos[:, 1:]
        gap = c2 - c1
        close = np.abs(gap) <= (1e-9 * np.maximum(c1, c2) + eps)
        logC = np.log(cPos)
        with np.errstate(divide='ignore', invalid='ignore'):
            cbar = np.where(close, 0.5 * (c1 + c2), gap / (logC[:, 1:] - logC[:, :-1]))
        return cbar
    def nernstPlanckStep(c, h, T, Zc, charged, condBar, storage, dG=None, Z2Cond=None, ZCond=None):
        cbar = logMeanConcRows(c)
        flux_fick = T * (c[:, 1:] - c[:, :-1])
        if dG is not None:
            flux_fick = flux_fick + T * cbar * dG
        allCharged = charged is True
        numerator = (Zc * flux_fick if allCharged else (Zc * flux_fick)[charged]).sum(axis=0)
        weighted = ((Zc ** 2) * condBar if Z2Cond is None else Z2Cond) * cbar
        denominator = (weighted if allCharged else weighted[charged]).sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(denominator > 0, numerator / np.where(denominator > 0, denominator, 1.0), 0.0)
        total_flux_interface = flux_fick - (Zc * condBar if ZCond is None else ZCond) * cbar * ratio
        diff = np.zeros_like(c)
        diff[:, :-1] += total_flux_interface
        diff[:, 1:] -= total_flux_interface
        return c + (h * diff / storage) / 1000
    def nernstPlanckRate(c, T, Zc, charged, condBar, storage, dG=None, Z2Cond=None, ZCond=None, K=None):
        # largest rate (1/s) of the explicit step at the state c : diffusion T, migration in the diffusion potential
        # |z.condBar.ratio| (same ratio as nernstPlanckStep), activity gradient |T.dG|, Cauchy exchange K ; the explicit
        # step is stable (and keeps c >= 0) for h.rate <= 1
        cbar = logMeanConcRows(c)
        flux_fick = T * (c[:, 1:] - c[:, :-1])
        if dG is not None:
            flux_fick = flux_fick + T * cbar * dG
        allCharged = charged is True
        numerator = (Zc * flux_fick if allCharged else (Zc * flux_fick)[charged]).sum(axis=0)
        weighted = ((Zc ** 2) * condBar if Z2Cond is None else Z2Cond) * cbar
        denominator = (weighted if allCharged else weighted[charged]).sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(denominator > 0, numerator / np.where(denominator > 0, denominator, 1.0), 0.0)
        face = T + np.abs((Zc * condBar if ZCond is None else ZCond) * ratio)
        if dG is not None:
            face = face + np.abs(T * dG)
        out = np.zeros_like(c)
        out[:, :-1] += face
        out[:, 1:] += face
        out = out / 1000.0
        if K is not None:
            out[:, 0] += K[0]
            out[:, -1] += K[1]
        return float(np.max(out / storage))
    def nernstPlanckOperator():
        # transmissivities and storages of the species: from the settings only (the same objects, the same arrays),
        # built once and only read by nernstPlanckTransport (an error raised again at every call)
        sources = (centralDict.get('speciesDiffCoeff'), centralDict.get('diffCoeff'), centralDict.get('speciesPorosity'),
                   centralDict.get('porosity'), centralDict.get('dxHalfCell'), centralDict.get('nodeSize'),
                   centralDict.get('area'), geom, trsptedSpecies)
        return frames.constant(('nernstPlanckOperator', n, coordSystem), sources, nernstPlanckOperatorOf,
                               cache=_operators, size=_OPERATORS_SIZE)
    def nernstPlanckOperatorOf():
        speciesDiffCoeff = resolve_species_property('speciesDiffCoeff', 'diffCoeff')
        speciesPorosity = resolve_species_property('speciesPorosity', 'porosity')
        lo, hi = 0, n
        dxHalfCell_trim = meshProp(centralDict['dxHalfCell'], lo, hi)
        dx_total = dxHalfCell_trim[:-1] + dxHalfCell_trim[1:]
        T_by_sp = {sp: precompute_species_transmissivity(lo, hi, speciesDiffCoeff[sp], speciesPorosity[sp]) for sp in trsptedSpecies}
        storage_by_sp = {}
        volume_trim = cellVolume(lo, hi)
        for sp in trsptedSpecies:
            poro_sp = meshProp(speciesPorosity[sp], lo, hi)
            storage = volume_trim * poro_sp
            storage_by_sp[sp] = np.where(storage > 0, storage, 1.0)
        spc = list(trsptedSpecies)
        T = np.array([np.asarray(T_by_sp[sp], dtype=float) for sp in spc])
        storage = np.array([storage_by_sp[sp] for sp in spc])
        return dx_total, T_by_sp, storage_by_sp, T, storage
    def nernstPlanckTransport(C_old, C_new):
        speciesValence = centralDict['speciesValence']
        dx_total, T_by_sp, storage_by_sp, T, storage = nernstPlanckOperator()
        z_by_sp = {sp: speciesValence[sp] for sp in trsptedSpecies}
        C_state = {sp: C_old[sp].to_numpy() for sp in trsptedSpecies}
        nSub = npSubSteps(dt)
        if isRadial:            # radialSubSteps gives nSub back unchanged otherwise
            for sp in trsptedSpecies:
                T_sp = np.asarray(T_by_sp[sp])
                nSub = radialSubSteps(nSub, np.append(T_sp, 0.0), np.insert(T_sp, 0, 0.0), storage_by_sp[sp])
        dirichlet = dirichletNodes()
        spc = list(trsptedSpecies)
        Z = np.array([float(z_by_sp[sp]) for sp in spc])
        c = np.array([np.asarray(C_state[sp], dtype=float) for sp in spc])
        fixes = [(i, node, val) for i, sp in enumerate(spc) if sp in dirichlet
                 for node, val in ((0, dirichlet[sp][0]), (-1, dirichlet[sp][1])) if val is not None]
        K, cExt, _ = cauchyArrays(spc, noInflow=True)
        hasCauchy = bool(K.any())
        if hasCauchy:
            Tn = np.zeros_like(storage)
            Tn[:, :-1] += T
            Tn[:, 1:] += T
            rateB = (Tn[:, [0, -1]] / 1000.0 + K.T) / storage[:, [0, -1]]
            nSub = max(int(nSub), int(np.ceil(dt * float(np.max(rateB)))))
        act = activitySetup(spc) if useActivity else None
        S, Ksink = reactionTerms(spc)
        checkSourceCharge(S, Ksink, c.T, Z)
        S = None if S is None else S.T
        Ksink = None if Ksink is None else Ksink.T
        I = np.zeros_like(c)
        h = dt / nSub
        Zc, charged, condBar = Z[:, None], Z != 0, T * dx_total
        Z2Cond, ZCond = (Zc ** 2) * condBar, Zc * condBar
        if charged.all():
            charged = True
        # npMaxSubStep (engine.py) only bounds the diffusion : before every sub-step the stability rate is evaluated
        # with the migration in the diffusion potential and the activity gradient ; if h.rate exceeds the safety
        # factor, the rest of the call is split into more, shorter sub-steps (unchanged sub-steps otherwise)
        safety = float(centralDict.get('npCflSafety') or 0.9)
        # coupling matrix : the potential eliminated by the zero-current condition is restored as a block (psi_0 = 0,
        # psi_l+1 - psi_l = ratio.dx of the face) ; equation of the step : equivalent implicit Euler residual, the
        # activity gradient of the step frozen ; current rows : zero current through each face (charged species)
        monitor = couplingMonitor(spc + ['psi'])
        Mn, kb = len(spc), len(spc) + 1
        chargedMask = Z != 0
        def zeroCurrentPsi(c, dG):
            cbar = logMeanConcRows(c)
            ff = T * (c[:, 1:] - c[:, :-1])
            if dG is not None:
                ff = ff + T * cbar * dG
            num = (Zc * ff)[chargedMask].sum(axis=0)
            den = (Z2Cond * cbar)[chargedMask].sum(axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
            return np.concatenate(([0.0], np.cumsum(ratio * dx_total)))
        def zeroCurrentResidual(u, cPrev, h, dG):
            U = u.reshape(n, kb)
            c, psi = U[:, :Mn].T, U[:, Mn]
            cbar = logMeanConcRows(c)
            dpsi = psi[1:] - psi[:-1]
            flux = T * (c[:, 1:] - c[:, :-1]) - Zc * T * cbar * dpsi
            if dG is not None:
                flux = flux + T * cbar * dG
            diff = np.zeros_like(c)
            diff[:, :-1] += flux
            diff[:, 1:] -= flux
            R = 1000.0 * storage * (c - cPrev) / h - diff
            if hasCauchy:
                R[:, 0] -= 1000.0 * K[0] * (cExt[0] - c[:, 0])
                R[:, -1] -= 1000.0 * K[1] * (cExt[1] - c[:, -1])
            if S is not None:
                R -= 1000.0 * storage * S
            if Ksink is not None:
                R += 1000.0 * storage * Ksink * c
            cur = (Zc * flux)[chargedMask].sum(axis=0)
            den = (Z2Cond * cbar)[chargedMask].sum(axis=0)
            out = np.empty((n, kb))
            out[:, :Mn] = R.T
            out[0, Mn] = psi[0]
            out[1:, Mn] = np.where(den > 0, cur, dpsi)
            for i, node, val in fixes:
                out[node, i] = c[i, node] - val
            return out.ravel()
        def pack(c, psi):
            return np.column_stack((c.T, psi)).ravel()
        tDone = 0.0
        left, refined, nDone = nSub, 0, 0
        while left > 0:
            cPrev = c
            dG = dLnGamma(act, c) if act else None
            rate = nernstPlanckRate(c, T, Zc, charged, condBar, storage, dG, Z2Cond, ZCond,
                                    K if hasCauchy else None)
            if h * rate > safety * (1.0 + 1e-12):
                rest = h * left
                left = int(np.ceil(rest * rate / safety))
                if nDone + left > int(centralDict.get('maxSubCycling') or 100000):
                    logError('nernstPlanckTransport', f"{nDone + left} sub-steps needed by the migration (> "
                             f"maxSubCycling) : use npSolver = 'implicit' or a finer mesh")
                h = rest / left
                refined += 1
            left -= 1
            nDone += 1
            c = nernstPlanckStep(c, h, T, Zc, charged, condBar, storage, dG, Z2Cond, ZCond)
            if S is not None:
                c = c + h * S
            if hasCauchy:
                c[:, 0] += h * K[0] * (cExt[0] - cPrev[:, 0]) / storage[:, 0]
                c[:, -1] += h * K[1] * (cExt[1] - cPrev[:, -1]) / storage[:, -1]
            if Ksink is not None:
                c = c / (1.0 + h * Ksink)
            for i, node, val in fixes:
                c[i, node] = val
            I += h * c
            tDone += h
            if monitor is not None:
                couplingStepFD(monitor, lambda u, cp=cPrev, hs=h, g=dG: zeroCurrentResidual(u, cp, hs, g),
                               pack(cPrev, zeroCurrentPsi(cPrev, dG)), pack(c, zeroCurrentPsi(c, dG)), kb, 1, tDone, h)
        if refined:
            logWarning('nernstPlanckTransport', "explicit Nernst-Planck : sub-steps shortened for stability (migration "
                       "in the diffusion potential faster than diffusion)")
        display.stepPrint(f"[explicit Nernst-Planck : {nDone} sub-step(s)" + (", shortened for stability" if refined else "") + "]",
              end=" ", flush=True)
        storeIntegral(spc, I.T)
        C_state = {sp: c[i] for i, sp in enumerate(spc)}
        return _writeColumns(C_new, trsptedSpecies, C_state)

    def solve_current_from_voltage(numerator0, denominator_safe, dx_total, targetVoltage, temperature):
        RT_over_F = GAS_CONSTANT * temperature / FARADAY
        mask = currentMask()
        sum_a = np.sum(mask * (numerator0 / denominator_safe) * dx_total)
        sum_b = np.sum(mask * dx_total / denominator_safe)
        I_mol = (-(targetVoltage / RT_over_F) - sum_a) / (sum_b + electrodeResistanceTotal() * FARADAY / RT_over_F)
        return I_mol
    def nernstPlanckCurrentTerms(C, T_by_sp, z_by_sp, dx_total):
        flux_fick = {}
        cbar = {}
        cond_bar = {}
        for sp in trsptedSpecies:
            c = C[sp]
            flux_fick[sp] = T_by_sp[sp] * (c[1:] - c[:-1])
            cbar[sp] = logMeanConc(c[:-1], c[1:])
            cond_bar[sp] = T_by_sp[sp] * dx_total
        numerator0 = np.zeros_like(dx_total)
        denominator = np.zeros_like(dx_total)
        for sp in trsptedSpecies:
            z = z_by_sp[sp]
            if z == 0:
                continue
            numerator0 = numerator0 + z * flux_fick[sp]
            denominator = denominator + (z ** 2) * cond_bar[sp] * cbar[sp]
        return flux_fick, cbar, cond_bar, numerator0, denominator
    def resolve_imposed_current(numerator0, denominator_safe, dx_total, closureMode, closureValue, temperature):
        if closureMode == 'current':
            return closureValue / FARADAY
        elif closureMode == 'currentDensity':
            return (closureValue * areaRef()) / FARADAY
        elif closureMode == 'voltage':
            return solve_current_from_voltage(numerator0, denominator_safe, dx_total, closureValue, temperature)
        return 0.0
    def nernstPlanckCurrentStep(C, dt_sub, T_by_sp, z_by_sp, dx_total, storage_by_sp,
                                 closureMode, closureValue, temperature):
        flux_fick, cbar, cond_bar, numerator0, denominator = nernstPlanckCurrentTerms(C, T_by_sp, z_by_sp, dx_total)
        denominator_safe = np.where(denominator > 0, denominator, 1.0)
        I_mol = resolve_imposed_current(numerator0, denominator_safe, dx_total, closureMode, closureValue, temperature)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(denominator > 0, (numerator0 + I_mol * currentMask()) / denominator_safe, 0.0)
        C_new_sp = {}
        for sp in trsptedSpecies:
            z = z_by_sp[sp]
            total_flux_interface = flux_fick[sp] - z * cond_bar[sp] * cbar[sp] * ratio
            n_trim = len(C[sp])
            flux_right = np.zeros(n_trim)
            flux_left = np.zeros(n_trim)
            flux_right[:-1] = total_flux_interface
            flux_left[1:] = -total_flux_interface
            diff = flux_left + flux_right
            c_new = C[sp] + (dt_sub * diff / storage_by_sp[sp]) / 1000
            C_new_sp[sp] = c_new
        return C_new_sp
    def dirichletNodes():
        dirichlet = {}
        for sp in trsptedSpecies:
            v0, vL = (val if kind == 'dirichlet' else None
                      for kind, val in (boundarySpec(sp, side) for side in (0, 1)))
            if v0 is not None or vL is not None:
                dirichlet[sp] = (v0, vL)
        return dirichlet
    def electrodeSetup(z_by_sp):
        if not (isElectrode(0) or isElectrode(1)):
            return None
        reactions = centralDict.get('electrodeReactions') or {
            'anode':   [{'name': 'O2 evolution', 'n': 4, 'stoich': {'H+': 4}}],
            'cathode': [{'name': 'H2 evolution', 'n': 2, 'stoich': {'OH-': 2}}],
        }
        cfg = {}
        for role, sign in (('anode', 1), ('cathode', -1)):
            lst = reactions.get(role) or []
            if not lst:
                logError('electrodeReactions', f"no reaction given for the {role} : electrodeReactions['{role}'] "
                         "must contain at least one reaction")
            checked = []
            for k, r in enumerate(lst):
                name = r.get('name', f"{role} reaction {k + 1}")
                for sp in r['stoich']:
                    if sp not in z_by_sp:
                        logError('electrodeReactions', f"'{sp}' ({name}) is not a transported species : "
                                 "it cannot be produced or consumed at the electrode")
                q = sum(nu * z_by_sp[sp] for sp, nu in r['stoich'].items())
                if abs(q - sign * r['n']) > 1e-9:
                    logError('electrodeReactions', f"{name} is not charge balanced (Sigma nu.z = {q}, "
                             f"expected {sign * r['n']} at the {role}) : check 'stoich' and 'n'")
                checked.append({'name': name, 'n': r['n'], 'stoich': r['stoich'], 'eff': r.get('efficiency')})
            defaults = [r for r in checked if r['eff'] is None]
            if len(defaults) > 1:
                logError('electrodeReactions', f"only one {role} reaction can have efficiency None "
                         "(it carries the rest of the current)")
            for r in checked:
                r['default'] = (r is defaults[0]) if defaults else (r is checked[-1])
            if sum(r['eff'] for r in checked if not r['default'] and r['eff'] is not None) > 1 + 1e-9:
                logError('electrodeReactions', f"{role} efficiencies sum to more than 1 : "
                         "the reactions would carry more charge than the current")
            cfg[role] = checked
        return cfg
    def applyElectrodeReactions(C, I_mol, h, storage_by_sp, cfg):
        balance = centralDict.setdefault('electrodeBalance', {})
        iEl = electrodeNodes()
        for side, node, qIn in ((0, iEl[0], I_mol), (1, iEl[1], -I_mol)):
            if not isElectrode(side) or qIn == 0:
                continue
            role = 'anode' if qIn > 0 else 'cathode'
            total = abs(qIn)
            rates = []
            carried = 0.0
            for r in cfg[role]:
                if r['default']:
                    rates.append(None)
                    continue
                rate = (r['eff'] or 0.0) * total / r['n']
                for sp, nu in r['stoich'].items():
                    if nu < 0:
                        avail = max(C[sp][node], 0.0) * storage_by_sp[sp][node] * 1000.0
                        rate = min(rate, avail / (-nu * h))
                rates.append(rate)
                carried += rate * r['n']
            for k, r in enumerate(cfg[role]):
                if r['default']:
                    rate = max(total - carried, 0.0) / r['n']
                    for sp, nu in r['stoich'].items():
                        if nu < 0:
                            avail = max(C[sp][node], 0.0) * storage_by_sp[sp][node] * 1000.0
                            if rate * (-nu) * h > avail:
                                rate = avail / (-nu * h)
                                logWarning('electrodeReactions', f"{'x=0' if side == 0 else 'x=L'} : the default "
                                           f"{role} reaction '{r['name']}' is limited by '{sp}' (exhausted) -> part of "
                                           "the current is carried by no reaction, charge balance not respected")
                    rates[k] = rate
            key = 'x=0' if side == 0 else 'x=L'
            bal = balance.setdefault(key, {'charge_C': 0.0})
            bal['charge_C'] += total * FARADAY * h
            for r, rate in zip(cfg[role], rates):
                bal[r['name']] = bal.get(r['name'], 0.0) + rate * h
                for sp, nu in r['stoich'].items():
                    C[sp][node] = C[sp][node] + nu * rate * h / (1000.0 * storage_by_sp[sp][node])
        return C
    def applyWaterEquilibrium(C):
        Kw = centralDict.get('Kw', 1.0e-14)
        H, OH = C['H+'], C['OH-']
        x = 0.5 * ((H + OH) - np.sqrt((H - OH) ** 2 + 4.0 * Kw))
        C['H+'] = H - x
        C['OH-'] = OH - x
        return C
    def electrodeSetupBV(z_by_sp):
        if not (isElectrode(0) or isElectrode(1)):
            return None
        reactions = centralDict.get('electrodeReactions') or {
            'anode':   [{'name': 'O2 evolution', 'n': 4, 'stoich': {'H+': 4},  'E0': 1.229,  'i0': 1e-5}],
            'cathode': [{'name': 'H2 evolution', 'n': 2, 'stoich': {'OH-': 2}, 'E0': -0.828, 'i0': 1.0}],
        }
        prev = centralDict.get('electrodePotential') or {}
        cfg = {'phi': [prev.get('x=0'), prev.get('x=L')]}
        for role, sign in (('anode', 1), ('cathode', -1)):
            lst = reactions.get(role) or []
            if not lst:
                logError('electrodeReactions', f"no reaction given for the {role} : electrodeReactions['{role}'] "
                         "must contain at least one reaction")
            checked = []
            for k, r in enumerate(lst):
                name = r.get('name', f"{role} reaction {k + 1}")
                for key in ('n', 'stoich', 'E0', 'i0'):
                    if key not in r:
                        logError('electrodeReactions', f"'{key}' is missing for {name} : "
                                 "required by electrodeKinetics = 'butler-volmer'")
                for sp in r['stoich']:
                    if sp not in z_by_sp:
                        logError('electrodeReactions', f"'{sp}' ({name}) is not a transported species : "
                                 "it cannot be produced or consumed at the electrode")
                q = sum(nu * z_by_sp[sp] for sp, nu in r['stoich'].items())
                if abs(q - sign * r['n']) > 1e-9:
                    logError('electrodeReactions', f"{name} is not charge balanced (Sigma nu.z = {q}, "
                             f"expected {sign * r['n']} at the {role}) : check 'stoich' and 'n'")
                aa, ac = r.get('alpha_a', 0.5), r.get('alpha_c', 0.5)
                if r['i0'] <= 0 or aa <= 0 or ac <= 0:
                    logError('electrodeReactions', f"i0, alpha_a and alpha_c must be > 0 ({name} : "
                             f"i0 = {r['i0']}, alpha_a = {aa}, alpha_c = {ac})")
                orders = r.get('orders') or {}
                for sp in orders:
                    if sp not in r['stoich']:
                        logError('electrodeReactions', f"order given for '{sp}', which is not in the "
                                 f"stoichiometry of {name}")
                nuOx = {sp: sign * nu for sp, nu in r['stoich'].items()}
                checked.append({'name': name, 'n': r['n'], 'sign': sign, 'nuOx': nuOx, 'rev': bool(r.get('reversible', False)),
                                'E0': r['E0'], 'i0': r['i0'], 'aa': aa, 'ac': ac, 'cref': r.get('cref', 1.0),
                                'red': [(sp, orders.get(sp, -nu)) for sp, nu in nuOx.items() if nu < 0],
                                'ox':  [(sp, orders.get(sp, nu)) for sp, nu in nuOx.items() if nu > 0]})
            cfg[role] = checked
        return cfg
    def applyElectrodeReactionsBV(C, I_mol, h, storage_by_sp, cfg):
        balance = centralDict.setdefault('electrodeBalance', {})
        potential = centralDict.setdefault('electrodePotential', {})
        f = FARADAY / (GAS_CONSTANT * centralDict.get('temperature', 298.15))
        areaEl = centralDict.get('electrodeArea') or [None, None]
        areaDom = geom['nodeArea'] if isRadial else np.ravel(centralDict.get('area', 1.0))
        iEl = electrodeNodes()
        for side, node, qIn in ((0, iEl[0], I_mol), (1, iEl[1], -I_mol)):
            if not isElectrode(side) or qIn == 0:
                continue
            role = 'anode' if qIn > 0 else 'cathode'
            key = 'x=0' if side == 0 else 'x=L'
            A = areaEl[side] or float(areaDom[node])
            if isRadial and not A > 0:
                logError('electrodeReactions', f"{key} : the electrode node is at r = 0 (no surface) : "
                         "give electrodeArea [m2] for this electrode")
            par = []
            for r in cfg[role]:
                k0 = A * r['i0'] / (r['n'] * FARADAY)
                cR = cO = 1.0
                for sp, o in r['red']:
                    cR *= (max(C[sp][node], 0.0) / r['cref']) ** o
                for sp, o in r['ox']:
                    cO *= (max(C[sp][node], 0.0) / r['cref']) ** o
                avail = {sp: max(C[sp][node], 0.0) * storage_by_sp[sp][node] * 1000.0 for sp in r['nuOx']}
                rOx = min([avail[sp] / (-nu * h) for sp, nu in r['nuOx'].items() if nu < 0], default=np.inf)
                rRed = min([avail[sp] / (nu * h) for sp, nu in r['nuOx'].items() if nu > 0], default=np.inf)
                kR = k0 * cR if (r['rev'] or r['sign'] > 0) else 0.0
                kO = k0 * cO if (r['rev'] or r['sign'] < 0) else 0.0
                par.append((kR, kO, r['aa'] * f, r['ac'] * f, r['E0'], rOx, rRed, r['n']))
            def rates(E):
                out, g, dg = [], -qIn, 0.0
                for kR, kO, a, c, E0, rOx, rRed, nk in par:
                    fa = kR * np.exp(min(a * (E - E0), 700.0))
                    fc = kO * np.exp(min(-c * (E - E0), 700.0))
                    rk, drk = fa - fc, a * fa + c * fc
                    if rk > rOx:
                        rk, drk = rOx, 0.0
                    elif rk < -rRed:
                        rk, drk = -rRed, 0.0
                    out.append(rk)
                    g += nk * rk
                    dg += nk * drk
                return out, g, dg
            E = cfg['phi'][side]
            if E is None:
                E = max(cfg[role], key=lambda r: r['i0'])['E0']
            _, g, _ = rates(E)
            lo, hi = (E, None) if g < 0 else (None, E)
            step = 0.05
            while (lo is None or hi is None) and step < 100.0:
                trial = E + step if hi is None else E - step
                if rates(trial)[1] < 0:
                    lo = trial
                else:
                    hi = trial
                step *= 2.0
            solved = lo is not None and hi is not None
            if not solved:
                E = lo if hi is None else hi
                logWarning('electrodeReactions', f"{key} : the {role} reactions cannot carry the current "
                           "(reactants exhausted, no unlimited reaction such as water electrolysis) -> "
                           "charge balance not respected")
            else:
                for _ in range(100):
                    _, g, dg = rates(E)
                    if abs(g) <= 1e-10 * abs(qIn):
                        break
                    if g < 0:
                        lo = E
                    else:
                        hi = E
                    En = E - g / dg if dg > 0 else lo
                    E = En if lo < En < hi else 0.5 * (lo + hi)
                    if hi - lo < 1e-12:
                        break
            E = float(E)
            out, g, _ = rates(E)
            free = [k for k, p in enumerate(par) if -p[6] < out[k] < p[5]]
            if free:
                j = max(free, key=lambda k: abs(par[k][7] * out[k]))
                out[j] -= g / par[j][7]
            out = [float(rk) for rk in out]
            if solved:
                cfg['phi'][side] = E
                potential[key] = E
            bal = balance.setdefault(key, {'charge_C': 0.0})
            bal['charge_C'] += abs(qIn) * FARADAY * h
            for r, rk in zip(cfg[role], out):
                bal[r['name']] = bal.get(r['name'], 0.0) + r['sign'] * rk * h
                for sp, nu in r['nuOx'].items():
                    C[sp][node] = C[sp][node] + nu * rk * h / (1000.0 * storage_by_sp[sp][node])
        phiEl = [cfg['phi'][s] if isElectrode(s) else 0.0 for s in (0, 1)]
        if None not in phiEl:
            centralDict['electrodeDeltaPhi'] = phiEl[1] - phiEl[0]
        return C
    def sgSolveField(C, T_by_sp, z_by_sp, dx_total, closureMode, closureValue, temperature, phi0, I0, dG=None):
        spc = list(trsptedSpecies)
        return sgSolveFieldArr(np.array([np.asarray(C[sp], dtype=float) for sp in spc]),
                               np.array([np.asarray(T_by_sp[sp], dtype=float) for sp in spc]),
                               np.array([float(z_by_sp[sp]) for sp in spc]),
                               dx_total, closureMode, closureValue, temperature, phi0, I0, dG)
    def electrodeDrop(V):
        dPhiEl = centralDict.get('electrodeDeltaPhi')
        if dPhiEl is not None:
            return dPhiEl
        dV = centralDict.get('voltageDropValue')
        if dV is not None:
            return np.sign(V) * min(dV, abs(V))
        return np.sign(V) * (centralDict.get('electrodeVoltageDrop') or 0.0)
    def electrodeResistanceTotal():
        R = centralDict.get('electrodeResistance')
        return 0.0 if R is None else float(np.sum(np.atleast_1d(np.asarray(R, dtype=float))))
    def cellVoltage(psi, I_A, V=None):
        iEl = electrodeNodes()
        dPsi = float(psi[iEl[1]] - psi[iEl[0]])
        dPhiEl = electrodeDrop(V) if V is not None else centralDict.get('electrodeDeltaPhi')
        if dPhiEl is None:
            dV = centralDict.get('voltageDropValue')
            dPhiEl = -np.sign(I_A) * (dV if dV is not None else (centralDict.get('electrodeVoltageDrop') or 0.0))
        target = centralDict.get('targetVoltage')
        if target is None or V is not None or not I_A:
            ohmic = -electrodeResistanceTotal() * I_A
            return {'cellVoltage_V': dPsi + float(dPhiEl) + ohmic, 'electrodeOhmic_V': ohmic}
        Vt = -np.sign(I_A) * abs(float(target))
        R = (dPsi + float(dPhiEl) - Vt) / I_A
        if R < 0:
            logWarning('targetVoltage', f"cell voltage without resistance {abs(dPsi + float(dPhiEl)):.4g} V > "
                       f"targetVoltage = {abs(float(target)):.4g} V : electrode resistance set to 0")
            R = 0.0
        ohmic = -R * I_A
        return {'cellVoltage_V': dPsi + float(dPhiEl) + ohmic, 'electrodeOhmic_V': ohmic, 'electrodeResistance_Ohm': R}
    def voltageAt(tAbs, closureMode, closureValue):
        '''Imposed voltage at the absolute time tAbs (s): rampPotential = [V_start, V_end, duration in timeUnit] goes from
        V_start to V_end, then stays at V_end; closureValue without a ramp or out of the voltage mode.'''
        ramp = centralDict.get('rampPotential')
        if not ramp or closureMode != 'voltage':
            return closureValue
        vStart, vEnd, tRamp = ramp
        tRamp = toSeconds(float(tRamp), centralDict['timeUnit'])
        if tRamp <= 0:
            return vEnd
        frac = min(max(tAbs / tRamp, 0.0), 1.0)
        return vStart + frac * (vEnd - vStart)
    def updateVoltageDrop(C, closureMode, closureValue, temperature):
        centralDict.pop('voltageDropValue', None)
        centralDict.pop('voltageDropInfo', None)
        if not centralDict.get('voltageDrop'):
            return
        if centralDict.get('electrodeKinetics') == 'butler-volmer':
            logWarning('voltageDrop', "voltageDrop is ignored with electrodeKinetics = 'butler-volmer' "
                       "(the electrode potentials come from the Butler-Volmer equations)")
            return
        if not (isElectrode(0) and isElectrode(1)):
            logError('voltageDrop', "voltageDrop needs an electrode at both boundaries : "
                     "boundaryConditions = ['electrode', 'electrode']")
        if 'H+' not in C:
            logError('voltageDrop', "voltageDrop needs 'H+' in transportedSpecies (pH = -log10[H+])")
        if centralDict.get('electrodeVoltageDrop'):
            logWarning('voltageDrop', "voltageDrop = True : electrodeVoltageDrop is ignored")
        V = closureValue if closureMode == 'voltage' else -closureValue
        ramp = centralDict.get('rampPotential')
        if ramp and closureMode == 'voltage':
            V = ramp[1] if ramp[1] else ramp[0]
        if not V:
            return
        cH = np.asarray(C['H+'], dtype=float)
        iEl = list(electrodeNodes())
        if not (cH[iEl[0]] > 0 and cH[iEl[1]] > 0):
            logError('voltageDrop', f"[H+] <= 0 at an electrode node ({cH[iEl[0]]:.3g}, {cH[iEl[1]]:.3g} mol/L) : "
                     "pH undefined")
        aH = cH[iEl]
        if useActivity:
            spc = list(C)
            cEnds = np.array([np.asarray(C[sp], dtype=float)[iEl] for sp in spc])
            aH = aH * np.exp(lnGamma(activitySetup(spc), cEnds)[spc.index('H+')])
        pH0, pHL = -np.log10(aH[0]), -np.log10(aH[1])
        pHanode, pHcathode = (pHL, pH0) if V > 0 else (pH0, pHL)
        slope = np.log(10.0) * GAS_CONSTANT * temperature / FARADAY
        dV = abs(-1.23 - slope * (pHcathode - pHanode))
        centralDict['voltageDropValue'] = float(dV)
        centralDict['voltageDropInfo'] = {'pH_anode': float(pHanode), 'pH_cathode': float(pHcathode), 'dV': float(dV)}
    def sgSolveFieldArr(c, T, Z, dx_total, closureMode, closureValue, temperature, phi0, I0, dG=None, adv=None):
        RT_over_F = GAS_CONSTANT * temperature / FARADAY
        phi = np.array(phi0, dtype=float)
        kR = 0.0
        if closureMode == 'voltage':
            I_mol = float(I0)
            dPhiEl = electrodeDrop(closureValue)
            Vn = (closureValue - dPhiEl) / RT_over_F
            kR = electrodeResistanceTotal() * FARADAY / RT_over_F
        else:
            I_mol = resolve_imposed_current(None, None, dx_total, closureMode, closureValue, temperature)
        charged = Z != 0
        Zc = Z[charged][:, None]
        Tc = T[charged]
        cL, cR = c[charged][:, :-1], c[charged][:, 1:]
        cLp, cRp = np.maximum(cL, 0.0), np.maximum(cR, 0.0)
        dGc = 0.0 if dG is None else dG[charged]
        Q = float(adv['Q']) if adv is not None else 0.0
        qF = adv.get('qFace', 0.0) if adv is not None else 0.0
        mask = currentMask()
        for it in range(100):
            if adv is None:
                Te, u = Tc, Zc * phi[None, :] + dGc
            else:
                Qf = Q + qF
                Te = Tc + adv['dispG'][None, :] * abs(Qf)
                Te = np.where(Te > 0, Te, 1e-300)
                r = Tc / Te
                u = Zc * phi[None, :] * r - 1000.0 * Qf / Te + dGc * r
            Bp, Bm, dBp, dBm = bernoulliPair(u, deriv=True)
            gS = Bm * cR - Bp * cL
            g = (Zc * Te * gS).sum(axis=0)
            gp = -(dBm * cRp + dBp * cLp)
            dg = ((Zc ** 2) * Tc * gp).sum(axis=0)
            dg = np.maximum(dg, 1e-300)
            R = g + I_mol * mask
            RV = (np.sum(mask * phi) - Vn - kR * I_mol) if closureMode == 'voltage' else 0.0
            dQ = 0.0
            if adv is not None:
                dTdQ = adv['dispG'] * np.sign(Qf)
                dudQ = (-1000.0 - u * dTdQ) / Te
                bQ = (Zc * (dTdQ * gS + Te * gp * dudQ)).sum(axis=0)
                RQ = Q + float(np.dot(adv['betaPsi'], phi)) + adv['dpScaled']
                if closureMode == 'voltage':
                    U, V = np.column_stack((np.ones_like(dg), bQ)), np.vstack((np.ones_like(dg), adv['betaPsi']))
                    D, rb = np.array([[-kR, 0.0], [0.0, 1.0]]), np.array([RV, RQ])
                else:
                    U, V, D, rb = bQ[:, None], adv['betaPsi'][None, :], np.array([[1.0]]), np.array([RQ])
                XU = U / dg[:, None]
                try:
                    db = np.linalg.solve(D - V @ XU, -rb + V @ (R / dg))
                except np.linalg.LinAlgError:
                    db = np.full(len(rb), np.nan)
                if not np.all(np.isfinite(db)):
                    logError('sgSolveField', "Nernst-Planck with water flow : singular field / water flow system "
                             f"(closure '{closureMode}', Q = {Q:.4g} m3/s) : check permeability, "
                             "electroOsmoticPermeability and boundaryPressure")
                dphi = -R / dg - XU @ db
                dI = db[0] if closureMode == 'voltage' else 0.0
                dQ = db[-1]
            else:
                dI = (RV - np.sum(mask * R / dg)) / (np.sum(mask / dg) + kR) if closureMode == 'voltage' else 0.0
                dphi = -(R + dI * mask) / dg
            cap = np.maximum(10.0, 0.5 * np.abs(phi))
            scale = min(1.0, float(np.min(cap / np.maximum(np.abs(dphi), 1e-300))))
            phi = phi + scale * dphi
            if closureMode == 'voltage':
                I_mol = I_mol + scale * dI
            Q = Q + scale * dQ
            if np.max(np.abs(dphi)) * scale < 1e-9 * max(1.0, np.max(np.abs(phi))):
                break
        else:
            iWorst = int(np.argmax(np.abs(dphi)))
            logError('sgSolveField', "Nernst-Planck (Scharfetter-Gummel) : the electric field did not converge "
                     f"in 100 Newton iterations (closure '{closureMode}', last correction |F dpsi/RT| = "
                     f"{np.max(np.abs(dphi)) * scale:.3g} at interface {iWorst}, I = {I_mol * FARADAY:.4g} A"
                     + (f", water flow Q = {Q:.4g} m3/s" if adv is not None else "") + ") : "
                     "local field too strong for the damped Newton (depleted zone, closed or inconsistent boundary)")
        if adv is not None:
            return phi, I_mol, Q
        return phi, I_mol
    def nernstPlanckCurrentAdaptive(C, dt_total, T_by_sp, z_by_sp, dx_total, storage_by_sp,
                                    closureMode, closureValue, temperature, eo=None, ale=None):
        safety = centralDict.get('cflSafety', 0.9)
        nMin = npSubSteps(dt_total)
        nMax = int(centralDict.get('maxSubCycling', 100000))
        dirichlet = dirichletNodes()
        kineticsBV = centralDict.get('electrodeKinetics') == 'butler-volmer'
        zeroCurrent = closureMode == 'current' and not closureValue
        electrodeCfg = None if zeroCurrent else (electrodeSetupBV(z_by_sp) if kineticsBV else electrodeSetup(z_by_sp))
        applyElectrode = applyElectrodeReactionsBV if kineticsBV else applyElectrodeReactions
        waterEq = centralDict.get('waterEquilibrium')
        if waterEq is None:
            waterEq = ('H+' in trsptedSpecies) and ('OH-' in trsptedSpecies)
        flux_fick, cbar, cond_bar, numerator0, denominator = nernstPlanckCurrentTerms(C, T_by_sp, z_by_sp, dx_total)
        denominator_safe = np.where(denominator > 0, denominator, 1.0)
        # closure of each sub-step: rampPotential followed in time (voltageAt), as by nernstPlanckImplicit
        closureAt = lambda tSub: voltageAt(tGlobalStart + tSub, closureMode, closureValue)
        I_mol = resolve_imposed_current(numerator0, denominator_safe, dx_total, closureMode, closureAt(0.0), temperature)
        with np.errstate(divide='ignore', invalid='ignore'):
            phi = -np.where(denominator > 0, (numerator0 + I_mol * currentMask()) / denominator_safe, 0.0) * dx_total
        spc = list(trsptedSpecies)
        T = np.array([np.asarray(T_by_sp[sp], dtype=float) for sp in spc])
        Z = np.array([float(z_by_sp[sp]) for sp in spc])
        Zc = Z[:, None]
        storage = np.array([np.asarray(storage_by_sp[sp], dtype=float) for sp in spc])
        c = np.array([np.asarray(C[sp], dtype=float) for sp in spc])
        if ale is not None and ale[2]:
            c = c / ale[0][None, :]
        fixes = [(i, node, val) for i, sp in enumerate(spc) if sp in dirichlet
                 for node, val in ((0, dirichlet[sp][0]), (-1, dirichlet[sp][1])) if val is not None]
        eoOpen = eo is not None and eo['mode'] == 'open'
        K, cExt, cauMask = cauchyArrays(spc, noInflow=not eoOpen)
        KB = 1000.0 * K
        Q, eoVolume, adv = 0.0, 0.0, None
        if eoOpen:
            RT_over_F = GAS_CONSTANT * temperature / FARADAY
            adv = {'betaPsi': eo['beta'] * RT_over_F / eo['RhTot'],
                   'dpScaled': flowDrive(eo),
                   'dispG': 1000.0 * eo['alphaL'] / eo['Ld'],
                   'qFace': eo.get('qFace', 0.0)}
            Q = -(float(np.dot(adv['betaPsi'], phi)) + adv['dpScaled'])
        act = activitySetup(spc) if useActivity else None
        srcRate, Ksink = reactionTerms(spc)
        checkSourceCharge(srcRate, Ksink, c.T, Z)
        srcRate = None if srcRate is None else srcRate.T
        Ksink = None if Ksink is None else Ksink.T
        I = np.zeros_like(c)
        dG = None
        # coupling matrix : unknowns c and psi (psi_0 = 0, psi_l+1 - psi_l = phi of the face) ; equation of the step :
        # equivalent implicit Euler residual with the current equation of sgSolveFieldArr, the current I, the water
        # flow Q and the activity gradient frozen at the end of the step. The field of the end of a step is solved at
        # the start of the next one (or after the loop) : the matrix of a step is computed there (pending)
        monitor = couplingMonitor(spc + ['psi'])
        Mn, kb = len(spc), len(spc) + 1
        chargedMask = Z != 0
        maskI = currentMask()
        pending = None
        def currentResidual(u, cPrev, h, dG, I_mol, Q, r0, r1):
            U = u.reshape(n, kb)
            cc, psi = U[:, :Mn].T, U[:, Mn]
            dpsi = psi[1:] - psi[:-1]
            if eoOpen:
                Qf = Q + adv['qFace']
                Teff = T + adv['dispG'][None, :] * abs(Qf)
                Teff = np.where(Teff > 0, Teff, 1e-300)
                r = T / Teff
                uu = Zc * dpsi[None, :] * r - 1000.0 * Qf / Teff
                if dG is not None:
                    uu = uu + dG * r
            else:
                Teff = T
                uu = Zc * dpsi[None, :] + (0.0 if dG is None else dG)
            Bp, Bm = bernoulliPair(uu)
            J = Teff * (Bm * cc[:, 1:] - Bp * cc[:, :-1])
            diff = np.zeros_like(cc)
            diff[:, :-1] += J
            diff[:, 1:] -= J
            diff[:, 0] += KB[0] * (cExt[0] - cc[:, 0])
            diff[:, -1] += KB[1] * (cExt[1] - cc[:, -1])
            if eoOpen:
                QL = Q + eo.get('qOut', 0.0)
                diff[:, 0] += 1000.0 * Q * np.where(cauMask[0] & (Q > 0), cExt[0], cc[:, 0])
                diff[:, -1] -= 1000.0 * QL * np.where(cauMask[1] & (QL < 0), cExt[1], cc[:, -1])
            if r0 is None:
                R = 1000.0 * storage * (cc - cPrev) / h - diff
            else:
                R = 1000.0 * storage * (r1[None, :] * cc - r0[None, :] * cPrev) / h - diff
            if srcRate is not None:
                R -= 1000.0 * storage * srcRate
            if Ksink is not None:
                R += 1000.0 * storage * Ksink * cc
            out = np.empty((cc.shape[1], kb))
            out[:, :Mn] = R.T
            out[0, Mn] = psi[0]
            out[1:, Mn] = (Zc * J)[chargedMask].sum(axis=0) + I_mol * maskI
            for i, node, val in fixes:
                out[node, i] = cc[i, node] - val
            return out.ravel()
        def pack(cc, phiFaces):
            return np.column_stack((cc.T, np.concatenate(([0.0], np.cumsum(phiFaces))))).ravel()
        def measurePending(phiNew, I_new, Q_new, dGNew):
            p = pending
            couplingStepFD(monitor, lambda u: currentResidual(u, p['cPrev'], p['h'], dGNew, I_new, Q_new,
                                                              p['r0'], p['r1']),
                           pack(p['cPrev'], p['phiPrev']), pack(p['cNew'], phiNew), kb, 1, p['tEnd'], p['h'])
        t, nSub = 0.0, 0
        while dt_total - t > 1e-12 * dt_total:
            if act:
                dG = dLnGamma(act, c)
            if eoOpen:
                phi, I_mol, Q = sgSolveFieldArr(c, T, Z, dx_total, closureMode, closureAt(t), temperature, phi, I_mol,
                                                dG, dict(adv, Q=Q))
                Qf = Q + adv['qFace']
                Teff = T + adv['dispG'][None, :] * abs(Qf)
                Teff = np.where(Teff > 0, Teff, 1e-300)
                r = T / Teff
                u = Zc * phi[None, :] * r - 1000.0 * Qf / Teff
                if dG is not None:
                    u = u + dG * r
                Bp, Bm = bernoulliPair(u)
            else:
                phi, I_mol = sgSolveFieldArr(c, T, Z, dx_total, closureMode, closureAt(t), temperature, phi, I_mol, dG)
                Teff = T
                Bp, Bm = bernoulliPair(Zc * phi[None, :] + (0.0 if dG is None else dG))
            if pending is not None:
                measurePending(phi, I_mol, Q, dG)
                pending = None
            cStart, phiStart = c, phi
            J = Teff * (Bm * c[:, 1:] - Bp * c[:, :-1])
            out = np.zeros_like(c)
            out[:, :-1] += Teff * Bp
            out[:, 1:] += Teff * Bm
            out[:, 0] += KB[0]
            out[:, -1] += KB[1]
            if eoOpen:
                QL = Q + eo.get('qOut', 0.0)
                out[:, 0] += 1000.0 * max(-Q, 0.0)
                out[:, -1] += 1000.0 * max(QL, 0.0)
            rate = np.max(out / (1000.0 * storage), axis=0)
            rateMax = float(np.max(rate))
            h = min(dt_total - t, dt_total / nMin)
            if rateMax > 0:
                h = min(h, safety / rateMax)
            diff = np.zeros_like(c)
            diff[:, :-1] += J
            diff[:, 1:] -= J
            diff[:, 0] += KB[0] * (cExt[0] - c[:, 0])
            diff[:, -1] += KB[1] * (cExt[1] - c[:, -1])
            if eoOpen:
                cB0 = np.where(cauMask[0] & (Q > 0), cExt[0], c[:, 0])
                cBL = np.where(cauMask[1] & (QL < 0), cExt[1], c[:, -1])
                diff[:, 0] += 1000.0 * Q * cB0
                diff[:, -1] -= 1000.0 * QL * cBL
                eoVolume += Q * h
            if ale is None:
                c = c + (h * diff / storage) / 1000
            else:
                r0 = ale[0] + (ale[1] - ale[0]) * (t / dt_total)
                r1 = ale[0] + (ale[1] - ale[0]) * ((t + h) / dt_total)
                c = (r0[None, :] * c + (h * diff / storage) / 1000) / r1[None, :]
            if srcRate is not None:
                c = c + h * srcRate
            if Ksink is not None:
                c = c / (1.0 + h * Ksink)
            I += h * c
            if electrodeCfg or waterEq:
                rows = [c[i] for i in range(len(spc))]
                C_new_sp = dict(zip(spc, rows))
                if electrodeCfg:
                    C_new_sp = applyElectrode(C_new_sp, I_mol, h, storage_by_sp, electrodeCfg)
                if waterEq:
                    C_new_sp = applyWaterEquilibrium(C_new_sp)
                for i, sp in enumerate(spc):
                    if C_new_sp[sp] is not rows[i]:
                        c[i] = C_new_sp[sp]
            for i, node, val in fixes:
                c[i, node] = val
            t += h
            nSub += 1
            if monitor is not None:
                pending = {'cPrev': cStart, 'phiPrev': phiStart, 'cNew': c.copy(), 'h': h, 'tEnd': t,
                           'r0': None if ale is None else r0, 'r1': None if ale is None else r1}
            if nSub >= nMax:
                iNode = int(np.argmax(rate))
                logError('nernstPlanckCurrentAdaptive', f"more than {nMax} sub-steps needed (dt_sub = {writeTime(h, 3)}, "
                         f"limiting node {iNode}, local |F dpsi/RT| = {abs(phi[min(iNode, len(phi)-1)]):.3g}). "
                         "The local electric field is too strong for the explicit scheme "
                         "(depleted zone / closed boundary ?). Increase maxSubCycling or check boundary conditions.")
        dG = dLnGamma(act, c) if act else None
        if eoOpen:
            phi, I_mol, Q = sgSolveFieldArr(c, T, Z, dx_total, closureMode, closureAt(t), temperature, phi, I_mol,
                                            dG, dict(adv, Q=Q))
        else:
            phi, I_mol = sgSolveFieldArr(c, T, Z, dx_total, closureMode, closureAt(t), temperature, phi, I_mol, dG)
        if pending is not None:
            measurePending(phi, I_mol, Q, dG)
            pending = None
        if eo is not None:
            electroOsmosisDiagnostics(eo, Q, phi, eoVolume, temperature)
        storeIntegral(spc, I.T)
        C = {sp: c[i] for i, sp in enumerate(spc)}
        return C, nSub, phi, I_mol
    def flowSetup(electrodeVolume):
        Qimp = adeFlowRate()
        loss = centralDict.get('_waterLossRate')
        if not (eoRequested or Qimp or loss is not None):
            return None
        if not eoRequested and loss is None:
            alphaL = float(centralDict.get('dispersivity') or 0.0)
            if alphaL < 0:
                logError('basicTransport', f"dispersivity must be >= 0 (got {alphaL})")
            dxh = meshProp(centralDict['dxHalfCell'], 0, n)
            Ld = dxh[:-1] + dxh[1:]
            ev = electrodeVolume or [None, None]
            if isElectrode(0) and ev[0]:
                Ld[0] = dxh[1]
            if isElectrode(1) and ev[1]:
                Ld[-1] = dxh[-2]
            return {'mode': 'open', 'electroOsmosis': False, 'beta': np.zeros(n - 1), 'RhTot': 1.0, 'Ld': Ld,
                    'alphaL': alphaL, 'p0': 0.0, 'pL': 0.0, 'Qimp': Qimp}
        eo = electroOsmosisSetup(electrodeVolume)
        if eo['mode'] == 'closed' and Qimp:
            logError('electroOsmosis', "velocity (imposed water flow) is incompatible with hydraulicBoundary = "
                     "'closed' (no water flow) : use hydraulicBoundary = 'open'")
        if eo['mode'] == 'closed' and loss is not None:
            logError('saturation', "the water lost when the saturation decreases is expelled as a Darcy flow : "
                     "use hydraulicBoundary = 'open'")
        eo.update({'electroOsmosis': True, 'Qimp': Qimp})
        if loss is not None:
            w = np.asarray(loss, dtype=float) * cellVolume(0, n)
            eo['qFace'] = np.cumsum(w)[:-1]
            eo['qOut'] = float(np.sum(w))
            centralDict['waterExpelled'] = w
        return eo
    def flowDrive(eo):
        drive = (eo['pL'] - eo['p0']) / eo['RhTot'] - eo['Qimp']
        if 'qFace' in eo:
            drive += float(np.dot(eo['Rh'], eo['qFace'])) / eo['RhTot']
        return drive
    def electroOsmosisSetup(electrodeVolume):
        def nodeValues(key, required=True, default=None):
            val = centralDict.get(key)
            if val is None:
                if required:
                    logError('electroOsmosis', f"'{key}' is required with electroOsmoticPermeability")
                val = default
            arr = np.asarray(val, dtype=float)
            if arr.ndim > 1 or (arr.ndim == 1 and len(arr) != n):
                logError('electroOsmosis', f"'{key}' must be a number or one value per node "
                         f"(got {arr.size} values for {n} nodes)")
            arr = meshProp(arr, 0, n)
            if not np.all(np.isfinite(arr)):
                logError('electroOsmosis', f"'{key}' contains non-finite values")
            return arr
        k_eo = nodeValues('electroOsmoticPermeability')
        perm = nodeValues('permeability')
        if np.any(perm <= 0):
            logError('electroOsmosis', f"permeability must be > 0 (node {int(np.argmin(perm))})")
        mu = centralDict.get('viscosity')
        mu = 8.9e-4 if mu is None else float(mu)
        if not mu > 0:
            logError('electroOsmosis', f"viscosity must be > 0 (got {mu})")
        mode = str(centralDict.get('hydraulicBoundary') or 'open').lower()
        if mode not in ('open', 'closed'):
            logError('electroOsmosis', f"unknown hydraulicBoundary '{mode}' : use 'open' or 'closed'")
        pB = centralDict.get('boundaryPressure')
        pB = [0.0, 0.0] if pB is None else list(np.ravel(np.asarray(pB, dtype=float)))
        if len(pB) != 2 or not np.all(np.isfinite(pB)):
            logError('electroOsmosis', "boundaryPressure must be [p_0, p_L] in Pa")
        alphaL = float(centralDict.get('dispersivity') or 0.0)
        if alphaL < 0:
            logError('electroOsmosis', f"dispersivity must be >= 0 (got {alphaL})")
        dxh = meshProp(centralDict['dxHalfCell'], 0, n)
        areaR, areaL = halfCellAreas(0, n)
        with np.errstate(divide='ignore', invalid='ignore'):
            RhR = mu * dxh / (perm * areaR)
            RhL = mu * dxh / (perm * areaL)
        Rh = RhR[:-1] + RhL[1:]
        wEo = mu * dxh * k_eo / perm
        beta = (wEo[:-1] + wEo[1:]) / (dxh[:-1] + dxh[1:])
        Ld = dxh[:-1] + dxh[1:]
        ev = electrodeVolume or [None, None]
        if isElectrode(0) and ev[0]:
            Rh[0], beta[0], Ld[0] = RhL[1], mu * k_eo[1] / perm[1], dxh[1]
        if isElectrode(1) and ev[1]:
            Rh[-1], beta[-1], Ld[-1] = RhR[-2], mu * k_eo[-2] / perm[-2], dxh[-2]
        if not np.all(np.isfinite(Rh)) or np.any(Rh <= 0):
            logError('electroOsmosis', "hydraulic resistance undefined (zero face area ?)")
        return {'mode': mode, 'Rh': Rh, 'beta': beta, 'RhTot': float(np.sum(Rh)), 'Ld': Ld, 'alphaL': alphaL,
                'p0': float(pB[0]), 'pL': float(pB[1])}
    def electroOsmosisDiagnostics(eo, Q, phi, volume, temperature):
        if not eo['electroOsmosis']:
            return
        RT_over_F = GAS_CONSTANT * temperature / FARADAY
        dp = -eo['Rh'] * (Q + eo.get('qFace', 0.0) - eo['Qimp']) - eo['beta'] * RT_over_F * np.asarray(phi, dtype=float)
        p = eo['p0'] + np.concatenate(([0.0], np.cumsum(dp)))
        aRef = areaRef()
        prevVolume = (centralDict.get('electroOsmosis') or {}).get('volume_m3', 0.0)
        centralDict['electroOsmosis'] = {'mode': eo['mode'], 'waterFlow_m3s': float(Q),
                                         'darcyVelocity_ms': float(Q) / aRef if aRef > 0 else np.nan,
                                         'pressure_Pa': p, 'volume_m3': prevVolume + float(volume)}
    def nernstPlanckImplicit(C, dt_total, T_by_sp, z_by_sp, dx_total, storage_by_sp,
                             closureMode, closureValue, temperature, eo=None, dn=None, ale=None):
        try:
            import copy
            from scipy.linalg.lapack import dgbtrf, dgbtrs, dgttrf, dgttrs
            from scipy.sparse import csc_matrix, csr_matrix
            from scipy.sparse.linalg import splu
        except ImportError:
            logError('nernstPlanckImplicit', "npSolver = 'implicit' needs scipy (LAPACK band solver) : pip install scipy")
        maxChange = float(centralDict.get('npImplicitMaxChange') or 0.1)
        atol = float(centralDict.get('npImplicitAtol') or 1e-6)
        maxIter = int(centralDict.get('npImplicitMaxIter') or 30)
        nMax = int(centralDict.get('maxSubCycling', 100000))
        RT_over_F = GAS_CONSTANT * temperature / FARADAY
        kR = electrodeResistanceTotal() * FARADAY / RT_over_F if closureMode == 'voltage' else 0.0
        def voltageNow(tAbs):
            return voltageAt(tAbs, closureMode, closureValue)
        dirichlet = dirichletNodes()
        kineticsBV = centralDict.get('electrodeKinetics') == 'butler-volmer'
        zeroCurrent = closureMode == 'current' and not closureValue
        electrodeCfg = None if zeroCurrent else (electrodeSetupBV(z_by_sp) if kineticsBV else electrodeSetup(z_by_sp))
        applyElectrode = applyElectrodeReactionsBV if kineticsBV else applyElectrodeReactions
        waterEq = centralDict.get('waterEquilibrium')
        if waterEq is None:
            waterEq = ('H+' in trsptedSpecies) and ('OH-' in trsptedSpecies)
        spc = list(trsptedSpecies)
        M, N = len(spc), len(C[spc[0]])
        nb = M + 1
        voltage = closureMode == 'voltage'
        iEl, iMask = electrodeNodes(), currentMask()
        nU = N * nb
        icIdx = np.arange(N)[None, :] * nb + np.arange(M)[:, None]
        ipIdx = np.arange(N) * nb + M
        Z = np.array([float(z_by_sp[sp]) for sp in spc])
        Zc = Z[:, None]
        T = np.array([np.asarray(T_by_sp[sp], dtype=float) for sp in spc])
        ST = np.array([1000.0 * np.asarray(storage_by_sp[sp], dtype=float) for sp in spc])
        pair = bool(waterEq) and ('H+' in spc) and ('OH-' in spc)
        Kw = float(centralDict.get('Kw', 1.0e-14))
        iH, iO = (spc.index('H+'), spc.index('OH-')) if pair else (None, None)
        pairRows = np.zeros((M, N), dtype=bool)
        if pair:
            pairRows[[iH, iO]] = True
        def waterPair(P):
            s = np.sqrt(P * P + 4.0 * Kw)
            with np.errstate(divide='ignore', invalid='ignore'):
                cH = np.where(P >= 0, 0.5 * (P + s), 2.0 * Kw / (s - P))
                cO = np.where(P >= 0, 2.0 * Kw / (s + P), 0.5 * (s - P))
            return cH, cO, s
        fixed = np.zeros((M, N), dtype=bool)
        fixVal = np.zeros((M, N))
        for i, sp in enumerate(spc):
            if sp in dirichlet:
                v0, vL = dirichlet[sp]
                if v0 is not None:
                    fixed[i, 0], fixVal[i, 0] = True, v0
                if vL is not None:
                    fixed[i, -1], fixVal[i, -1] = True, vL
        if pair:
            for k in (0, N - 1):
                if fixed[iH, k] or fixed[iO, k]:
                    vH = fixVal[iH, k] if fixed[iH, k] else Kw / max(fixVal[iO, k], 1e-300)
                    vO = fixVal[iO, k] if fixed[iO, k] else Kw / max(vH, 1e-300)
                    fixed[[iH, iO], k], fixVal[iH, k], fixVal[iO, k] = True, vH, vO
        fixedRow = np.zeros(nU, dtype=bool)
        fixedRow[icIdx[fixed]] = True
        anyFixed = bool(fixed.any())
        dnState = None
        if dn is not None:
            DV = 1000.0 * dn['V']
            cf0, Ctot0, psiD0, par0 = donnanStart(dn, np.array([np.asarray(C[sp], dtype=float) for sp in spc]),
                                                  fixed, fixVal)
            dnState = {'par': par0, 'psi': psiD0, 'psiIter': psiD0, 'Sold': DV[None, :] * Ctot0, 'jac': None}
            def dnT(psiD, par):
                return 1000.0 * donnanConductance(dn, psiD, par, 0.0, dn['resFaces'])
            T = dnT(psiD0, par0)
            C = {sp: cf0[i] for i, sp in enumerate(spc)}
            for i, sp in enumerate(spc):
                T_by_sp[sp] = T[i].copy()
        eoOpen = eo is not None and eo['mode'] == 'open'
        Kcau, cExtB, cauMask = cauchyArrays(spc, noInflow=not eoOpen)
        KB = 1000.0 * Kcau
        if eoOpen:
            betaPsi = eo['beta'] * RT_over_F / eo['RhTot']
            dpScaled = flowDrive(eo)
            dispG = 1000.0 * eo['alphaL'] / eo['Ld']
            qF, qOut = eo.get('qFace', 0.0), eo.get('qOut', 0.0)
            vQ = np.zeros(N * (M + 1))
            vQ[np.arange(N) * (M + 1) + M] = np.concatenate(([0.0], betaPsi)) - np.concatenate((betaPsi, [0.0]))
        cInit = np.array([np.abs(np.asarray(C[sp], dtype=float)) for sp in spc])
        gScale = float(np.max(((Zc ** 2) * T * 0.5 * (cInit[:, :-1] + cInit[:, 1:])).sum(axis=0))) if N > 1 else 0.0
        if gScale <= 0 and N > 1:
            gScale = float(np.max(((Zc ** 2) * T).sum(axis=0)))
        # regularisation of the current equation of each face (epsG.dPsi carries current without ions) : relative to
        # the conductance of that face, not to the largest one of the mesh (reservoir faces), otherwise it carries a
        # real share of the current in a depleted frit and creates charge there
        if gScale > 0:
            gFace = ((Zc ** 2) * T * 0.5 * (cInit[:, :-1] + cInit[:, 1:])).sum(axis=0)
            epsG = 1e-12 * np.maximum(gFace, 1e-12 * gScale)
        else:
            epsG = np.ones(max(N - 1, 0))
        a, b = icIdx[:, :-1], icIdx[:, 1:]
        pa = np.broadcast_to(ipIdx[:-1], a.shape)
        pb = np.broadcast_to(ipIdx[1:], a.shape)
        rowsV = np.concatenate([np.ravel(x) for x in (icIdx, a, a, a, a, b, b, b, b, pb, pb, pb, pb)])
        colsV = np.concatenate([np.ravel(x) for x in (icIdx, a, b, pb, pa, a, b, pb, pa, a, b, pb, pa)])
        if dn is not None:
            ii, jj = np.meshgrid(np.arange(M), np.arange(M), indexing='ij')
            rowsV = np.concatenate([rowsV, np.ravel(icIdx[ii.ravel()])])
            colsV = np.concatenate([colsV, np.ravel(icIdx[jj.ravel()])])
        keepV = ~fixedRow[rowsV]
        rowsV, colsV = rowsV[keepV], colsV[keepV]
        fixedIdx = np.flatnonzero(fixedRow)
        rowsK = np.concatenate([fixedIdx, [ipIdx[0]], ipIdx[1:], ipIdx[1:]]).astype(np.int64)
        colsK = np.concatenate([fixedIdx, [ipIdx[0]], ipIdx[1:], ipIdx[:-1]]).astype(np.int64)
        valsK = np.concatenate([np.ones(len(fixedIdx)), [1.0], epsG, -epsG])
        rows = np.concatenate([rowsV, rowsK]).astype(np.int64)
        cols = np.concatenate([colsV, colsK]).astype(np.int64)
        if pair:
            rP, rO = icIdx[iH], icIdx[iO]
            rowIsO = (rows % nb == iO)
            colIsH = (cols % nb == iH)
            colIsO = (cols % nb == iO)
            colNode = cols // nb
            rowSign = np.where(rowIsO, -1.0, 1.0)
            rows = np.where(rowIsO, rows - iO + iH, rows)
            cols = np.where(colIsO, cols - iO + iH, cols)
            rows = np.concatenate([rows, rO])
            cols = np.concatenate([cols, rO])
            valsK = np.concatenate([valsK, np.ones(N)])
            nodeH, nodeO = colNode[colIsH], colNode[colIsO]
        def matrixPattern(r, cl, shape, csr=False):
            # matrix of the entries (r, cl), duplicates summed : LAPACK band storage for a square matrix of small
            # bandwidth, compressed sparse columns (rows with csr) otherwise
            if shape[0] == shape[1] and not csr:
                kl = int(max(np.max(r - cl, initial=0), 0))
                ku = int(max(np.max(cl - r, initial=0), 0))
                if kl + ku <= NP_SPARSE_BANDWIDTH:
                    ldab = 2 * kl + ku + 1
                    return {'band': (kl, ku, ldab), 'n': shape[0], 'flat': (kl + ku + r - cl) + cl * ldab}
            major, minor, nMajor, nMinor = (r, cl, shape[0], shape[1]) if csr else (cl, r, shape[1], shape[0])
            keys, pos = np.unique(major * nMinor + minor, return_inverse=True)
            ptr = np.concatenate(([0], np.cumsum(np.bincount(keys // nMinor, minlength=nMajor))))
            return {'band': None, 'shape': shape, 'csr': csr, 'pos': pos, 'size': len(keys), 'indices': keys % nMinor,
                    'indptr': ptr}
        def assemble(pat, vals):
            if pat['band'] is not None:
                ldab, n = pat['band'][2], pat['n']
                return np.bincount(pat['flat'], weights=vals, minlength=ldab * n).reshape((ldab, n), order='F')
            data = np.bincount(pat['pos'], weights=vals, minlength=pat['size'])
            return (csr_matrix if pat['csr'] else csc_matrix)((data, pat['indices'], pat['indptr']), shape=pat['shape'])
        # Newton matrix, one layout per set of coupled species (cplSp, None : every species). Block-triangular layout :
        # psi and the coupled species are factorised together, every other species is a tridiagonal chain over the
        # nodes, solved once the coupled block is known. Left out of the matrix, not of the residual (the converged
        # solution is unchanged, only the Newton convergence rate can be) : the current carried by these species, their
        # Donnan terms in the rows of the coupled species and the Donnan terms between them
        lay, layouts, keepLU = {}, {}, {'LU': None, 'h': None, 'ucolQ': None}
        lastJac = {'vals': None}   # entries of the last Newton matrix assembled (coupling matrix, nahed.py)
        def setLayout(cplSp):
            keepLU['LU'] = None
            key = None if cplSp is None or cplSp.all() else cplSp.tobytes()
            if key not in layouts:
                if key is None:
                    layouts[key] = {'cpl': None, 'nCoupled': M, 'pat': matrixPattern(rows, cols, (nU, nU))}
                else:
                    cpl = np.zeros(nU, dtype=bool)
                    cpl[ipIdx] = True
                    cpl[icIdx[cplSp]] = True
                    rail = icIdx[~cplSp].ravel()
                    nC, nR = int(cpl.sum()), len(rail)
                    newC = np.full(nU, -1, dtype=np.int64)
                    newC[cpl] = np.arange(nC)
                    newR = np.full(nU, -1, dtype=np.int64)
                    newR[rail] = np.arange(nR)
                    rowC, colC = cpl[rows], cpl[cols]
                    selC = np.flatnonzero(rowC & colC)
                    selB = np.flatnonzero(~rowC & colC)
                    selD = np.flatnonzero(~rowC & ~colC & (rows % nb == cols % nb))
                    side = (cols[selD] - rows[selD]) // nb
                    posD = newR[rows[selD]]
                    # exact : only neutral species out of the coupled block (no current, no Donnan term)
                    layouts[key] = {'cpl': cpl, 'rail': rail, 'nR': nR, 'cplSp': cplSp.copy(),
                                    'exact': not np.any(~cplSp & (Z != 0)),
                                    'nCoupled': int(cplSp.sum()), 'selC': selC, 'selB': selB,
                                    'pat': matrixPattern(newC[rows[selC]], newC[cols[selC]], (nC, nC)),
                                    'patB': matrixPattern(newR[rows[selB]], newC[cols[selB]], (nR, nC), csr=True),
                                    'tri': [(selD[side == k], posD[side == k] + min(k, 0)) for k in (-1, 0, 1)]}
            lay.clear()
            lay.update(layouts[key])
        def assembleLayout(vals):
            if lay['cpl'] is None:
                return assemble(lay['pat'], vals)
            dl, d, du = (np.bincount(p, weights=vals[i], minlength=lay['nR']) for i, p in lay['tri'])
            return assemble(lay['pat'], vals[lay['selC']]), (dl[:-1], d, du[:-1]), assemble(lay['patB'], vals[lay['selB']])
        def coupledSpecies(c, keep=None):
            # species of the coupled block : the charged species carrying more than NP_COUPLING_SHARE of the
            # conductance z^2.T.c of a face, H+ and OH- of the water pair, and the species already kept (keep)
            if NP_COUPLING_SHARE is None or N < 2:
                return None
            cAbs = np.abs(c)
            g = (Zc ** 2) * T * 0.5 * (cAbs[:, :-1] + cAbs[:, 1:])
            with np.errstate(divide='ignore', invalid='ignore'):
                share = np.max(np.where(g > 0, g / g.sum(axis=0), 0.0), axis=1)
            cpl = (Z != 0) & (share > NP_COUPLING_SHARE)
            if pair:
                cpl[[iH, iO]] = True
            return cpl if keep is None else cpl | keep
        def jacobian(c, psi, h, Bm, Bp, dJ, Teff, Q):
            TBp, TBm = Teff * Bp, Teff * Bm
            if dn is not None:
                thEff, uD, vD, GD, thDLj = dnState['jac']
                STh = DV[None, :] * thEff / h
                if Ksink is not None:
                    STh = STh + (DV * aleThF(dnState['par'][0], aleR['r1']))[None, :] * Ksink
            else:
                STh = ST / h if aleR['r1'] is None else ST * aleR['r1'][None, :] / h
                if Ksink is not None:
                    STh = STh + ST * Ksink
            STh[:, 0] += KB[0]
            STh[:, -1] += KB[1]
            if eoOpen:
                QL = Q + qOut
                STh[:, 0] -= 1000.0 * Q * np.where(cauMask[0] & (Q > 0), 0.0, 1.0)
                STh[:, -1] += 1000.0 * QL * np.where(cauMask[1] & (QL < 0), 0.0, 1.0)
            parts = [STh, TBp, -TBm, -dJ, dJ,
                     -TBp, TBm, dJ, -dJ,
                     -Zc * TBp, Zc * TBm, Zc * dJ, -Zc * dJ]
            if dn is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    w = np.where(GD > 0, DV * thDLj / (GD * h), 0.0)
                parts.append(-(uD[:, None, :] * vD[None, :, :]).reshape(M * M, N) * w[None, :])
            vals = np.concatenate([np.ravel(x) for x in parts])
            vals = np.concatenate([vals[keepV], valsK])
            if pair:
                s = c[iH] + c[iO]
                scale = rowSign.copy()
                scale[colIsH] *= (c[iH] / s)[nodeH]
                scale[colIsO] *= -(c[iO] / s)[nodeO]
                vals[:len(scale)] *= scale
            lastJac['vals'] = vals
            return assembleLayout(vals)
        def aleThF(thF, r):
            # ALE with donnan : the gas displaces the free water only (thF.r), not the double layer
            return thF if r is None else thF * r
        def residual(c, psi, I, Q, cOld, h, Vn, sigma, needJac):
            if dn is not None:
                thF, thDL, sigD, phiD = dnState['par']
                thF = aleThF(thF, aleR['r1'])
                psiD = donnanPotential(c, Z, sigD, phiD, psi0=dnState['psiIter'])[0]
                dnState['psiIter'] = psiD
                Cb, thEff, uD, vD, GD, _ = donnanStorage(c, Z, thF, thDL, sigD, phiD, psiD)
                Tr = dnT(psiD, dnState['par'])
                if needJac:
                    dnState['jac'] = (thEff, uD, vD, GD, thDL)
            else:
                Tr = T
            if eoOpen:
                Qf, QL = Q + qF, Q + qOut
                Teff = Tr + dispG[None, :] * abs(Qf)
                Teff = np.where(Teff > 0, Teff, 1e-300)
                rT = Tr / Teff
                u = Zc * (psi[1:] - psi[:-1])[None, :] * rT - 1000.0 * Qf / Teff
                if lag['dG'] is not None:
                    u = u + lag['dG'] * rT
            else:
                Teff = Tr
                u = Zc * (psi[1:] - psi[:-1])[None, :]
                if lag['dG'] is not None:
                    u = u + lag['dG']
            if needJac:
                Bp, Bm, dBp, dBm = bernoulliPair(u, deriv=True)
            else:
                Bp, Bm = bernoulliPair(u)
            g = Bm * c[:, 1:] - Bp * c[:, :-1]
            J = Teff * g
            if dn is not None:
                Fm = (DV[None, :] * Cb - dnState['Sold']) / h - sigma * I
                if srcRate is not None:
                    Fm -= (DV * thF)[None, :] * srcRate
                if Ksink is not None:
                    Fm += (DV * thF)[None, :] * Ksink * c
            elif aleR['r1'] is None:
                Fm = ST * (c - cOld) / h - sigma * I
            else:
                Fm = (ST * aleR['r1'][None, :] * c - ST * aleR['r0'][None, :] * cOld) / h - sigma * I
                if srcRate is not None:
                    Fm -= ST * srcRate
                if Ksink is not None:
                    Fm += ST * Ksink * c
            Fm[:, 0] += KB[0] * (c[:, 0] - cExtB[0])
            Fm[:, -1] += KB[1] * (c[:, -1] - cExtB[1])
            if eoOpen:
                cB0 = np.where(cauMask[0] & (Q > 0), cExtB[0], c[:, 0])
                cBL = np.where(cauMask[1] & (QL < 0), cExtB[1], c[:, -1])
                Fm[:, 0] -= 1000.0 * Q * cB0
                Fm[:, -1] += 1000.0 * QL * cBL
            Fm[:, :-1] -= J
            Fm[:, 1:] += J
            if anyFixed:
                Fm[fixed] = (c - fixVal)[fixed]
            res = np.empty(nU)
            res[icIdx] = Fm
            res[ipIdx[0]] = psi[0]
            res[ipIdx[1:]] = (Zc * J).sum(axis=0) + I * iMask + epsG * (psi[1:] - psi[:-1])
            if pair:
                res[rP] -= res[rO]
                res[rO] = 0.0
            resI = (psi[iEl[1]] - psi[iEl[0]] - Vn - kR * I) if voltage else 0.0
            resQ = (Q + float(np.dot(betaPsi, psi[1:] - psi[:-1])) + dpScaled) if eoOpen else 0.0
            if not needJac:
                return res, resI, resQ, None, None
            gp = -dBm * c[:, 1:] - dBp * c[:, :-1]
            dJ = (Tr if eoOpen else Teff) * Zc * gp
            ucolQ = None
            if eoOpen:
                dTdQ = dispG[None, :] * np.sign(Qf)
                dudQ = (-1000.0 - u * dTdQ) / Teff
                dJdQ = dTdQ * g + Teff * gp * dudQ
                dF = np.zeros((M, N))
                dF[:, :-1] -= dJdQ
                dF[:, 1:] += dJdQ
                dF[:, 0] -= 1000.0 * cB0
                dF[:, -1] += 1000.0 * cBL
                if anyFixed:
                    dF[fixed] = 0.0
                ucolQ = np.zeros(nU)
                ucolQ[icIdx] = dF
                ucolQ[ipIdx[1:]] = (Zc * dJdQ).sum(axis=0)
                if pair:
                    ucolQ[rP] -= ucolQ[rO]
                    ucolQ[rO] = 0.0
            return res, resI, resQ, jacobian(c, psi, h, Bm, Bp, dJ, Teff, Q), ucolQ
        def borderColumn(sigma):
            ucol = np.zeros(nU)
            ucol[ipIdx[1:]] = iMask
            src = (sigma != 0) & ~fixed
            ucol[icIdx[src]] -= sigma[src]
            if pair:
                ucol[rP] -= ucol[rO]
                ucol[rO] = 0.0
            return ucol
        def factorMatrix(pat, A):
            if pat['band'] is None:
                try:
                    return splu(A, permc_spec='MMD_AT_PLUS_A')
                except RuntimeError:
                    return None
            lu, piv, info = dgbtrf(A, *pat['band'][:2], overwrite_ab=1)
            return (lu, piv) if info == 0 else None
        def solveMatrix(pat, LU, rhs):
            if pat['band'] is None:
                return LU.solve(rhs), 0
            return dgbtrs(LU[0], *pat['band'][:2], rhs, LU[1])
        def factorize(A):
            if lay['cpl'] is None:
                return factorMatrix(lay['pat'], A)
            AC, tri, B = A
            LU = factorMatrix(lay['pat'], AC)
            dl, d, du, du2, ipiv, info = dgttrf(*tri)
            return None if LU is None or info != 0 else (LU, (dl, d, du, du2, ipiv), B)
        def factorSolve(LU, rhs):
            if lay['cpl'] is None:
                return solveMatrix(lay['pat'], LU, rhs)
            LU, tri, B = LU
            cpl, rail = lay['cpl'], lay['rail']
            xC, info = solveMatrix(lay['pat'], LU, np.ascontiguousarray(rhs[cpl]))
            if info != 0:
                return xC, info
            xR, info = dgttrs(*tri, rhs[rail] - B @ xC)
            x = np.empty(rhs.shape)
            x[cpl], x[rail] = xC, xR
            return x, info
        def linearSolve(LU, res, resI, ucol, resQ=0.0, ucolQ=None):
            if eoOpen:
                cols = [-res] + ([ucol] if voltage else []) + [ucolQ]
                x, info = factorSolve(LU, np.column_stack(cols))
                if info != 0:
                    return None, None, None
                xr, XU = x[:, 0], x[:, 1:]
                rowsV, D, rb = [], [], []
                if voltage:
                    vI = np.zeros(nU)
                    vI[ipIdx[iEl[1]]], vI[ipIdx[iEl[0]]] = 1.0, -1.0
                    rowsV.append(vI); D.append([-kR, 0.0]); rb.append(resI)
                rowsV.append(vQ); D.append([0.0, 1.0] if voltage else [1.0]); rb.append(resQ)
                V = np.array(rowsV)
                S = np.array(D) - V @ XU
                try:
                    db = np.linalg.solve(S, -np.array(rb) - V @ xr)
                except np.linalg.LinAlgError:
                    return None, None, None
                return xr - XU @ db, (db[0] if voltage else 0.0), db[-1]
            if voltage:
                x, info = factorSolve(LU, np.column_stack((-res, ucol)))
                if info != 0:
                    return None, None, None
                vx1 = x[ipIdx[iEl[1]], 0] - x[ipIdx[iEl[0]], 0]
                vx2 = x[ipIdx[iEl[1]], 1] - x[ipIdx[iEl[0]], 1]
                if vx2 + kR == 0 or not np.isfinite(vx2):
                    return None, None, None
                dI = (vx1 + resI) / (vx2 + kR)
                return x[:, 0] - x[:, 1] * dI, dI, 0.0
            x, info = factorSolve(LU, -res)
            return (x, 0.0, 0.0) if info == 0 else (None, None, None)
        stats = {'factor': 0, 'solve': 0, 'wholeMatrix': 0}
        def newton(cStart, cOld, psi, I, Q, h, Vn, sigma):
            c = cStart.copy()
            if anyFixed:
                c[fixed] = fixVal[fixed]
            ucol = borderColumn(sigma) if voltage else None
            # the LU of the previous steps is kept while the step is unchanged (simplified Newton), and recomputed as
            # soon as the contraction rate exceeds NP_REFACTOR_THETA
            if keepLU['LU'] is not None and abs(keepLU['h'] - h) <= 1e-12 * h:
                LU, ucolQ, refactor = keepLU['LU'], keepLU['ucolQ'], False
            else:
                LU, ucolQ, refactor = None, None, True
            lastNorm = None
            for it in range(1, maxIter + 1):
                needJac = refactor or LU is None
                res, resI, resQ, A, uQ = residual(c, psi, I, Q, cOld, h, Vn, sigma, needJac=needJac)
                if needJac:
                    LU, ucolQ = factorize(A), uQ
                    stats['factor'] += 1
                    keepLU.update(LU=LU, h=h, ucolQ=ucolQ)
                    if LU is None:
                        return None, it
                delta, dI, dQ = linearSolve(LU, res, resI, ucol, resQ, ucolQ)
                stats['solve'] += 1
                if delta is None or not np.all(np.isfinite(delta)):
                    keepLU['LU'] = None
                    return None, it
                dC, dPsi = delta[icIdx], delta[ipIdx]
                mP = float(np.max(np.abs(dPsi)))
                norm = max(float(np.max(np.abs(dC) / (np.abs(c) + atol))), mP / max(1.0, float(np.max(np.abs(psi)))))
                theta = (norm / lastNorm if lastNorm > 0 else 0.0) if lastNorm is not None else None
                refactor = theta is not None and theta > NP_REFACTOR_THETA
                lastNorm = norm
                capPsi = max(10.0, 0.5 * float(np.max(np.abs(psi))))
                alpha = min(1.0, capPsi / mP) if mP > 0 else 1.0
                # positivity : the step is damped so that no concentration goes below -NP_NEG_TOL (a concentration
                # can lose at most NP_POS_FRACTION of its value per iteration) ; clipping a larger negative value
                # would create mass (and charge) and the step-size test below would accept it as converged
                neg = (c + alpha * dC < -NP_NEG_TOL) & ~pairRows
                if np.any(neg):
                    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                        lim = (NP_POS_FRACTION * np.maximum(c, 0.0) + NP_NEG_TOL) / (-dC)
                    alpha = min(alpha, float(np.min(lim[neg])))
                if alpha < 1.0:
                    refactor = True
                cPrev = c
                c = c + alpha * dC
                if pair:
                    P = cPrev[iH] - cPrev[iO] + alpha * dC[iH]
                    c[iH], c[iO], _ = waterPair(P)
                clipped = float(np.max(np.where(pairRows, 0.0, -c), initial=0.0))
                c = np.maximum(c, 0.0)
                psi = psi + alpha * dPsi
                if voltage:
                    I = I + alpha * dI
                if eoOpen:
                    Q = Q + alpha * dQ
                # error left in the iterate : theta / (1 - theta) times the last correction (the correction itself
                # before the contraction rate is known)
                est = theta / (1.0 - theta) if theta is not None and theta < 0.9 else 1.0
                if (alpha == 1.0 and clipped <= NP_NEG_TOL
                        and est * float(np.max(np.abs(c - cPrev) / (np.abs(c) + atol))) < NP_NEWTON_TOL
                        and est * mP < NP_NEWTON_TOL * max(1.0, float(np.max(np.abs(psi))))):
                    return (c, psi, I, Q), it
            keepLU['LU'] = None
            return None, maxIter
        bookKeys = ('electrodeBalance', 'electrodePotential', 'electrodeDeltaPhi')
        def snapshot():
            state = {k: copy.deepcopy(centralDict.get(k)) for k in bookKeys}
            state['phi'] = list(electrodeCfg['phi']) if (electrodeCfg and 'phi' in electrodeCfg) else None
            return state
        def restore(state):
            for k in bookKeys:
                if state[k] is None:
                    centralDict.pop(k, None)
                else:
                    centralDict[k] = copy.deepcopy(state[k])
            if state['phi'] is not None:
                electrodeCfg['phi'][:] = state['phi']
        def electrodeSources(c, I, h):
            sigma = np.zeros((M, N))
            if not electrodeCfg or I == 0:
                return sigma, None
            before = snapshot()
            Cd = applyElectrode({sp: c[i].copy() for i, sp in enumerate(spc)}, I, h, storage_by_sp, electrodeCfg)
            after = snapshot()
            restore(before)
            for i, sp in enumerate(spc):
                sigma[i] = ST[i] * (Cd[sp] - c[i]) / (h * I)
            return sigma, (before, after, I)
        def commitBalance(book, I):
            if book is None:
                return
            before, after, I0 = book
            restore(after)
            ratio = I / I0
            bal0 = before['electrodeBalance'] or {}
            for key, d in (centralDict.get('electrodeBalance') or {}).items():
                for name in d:
                    v0 = bal0.get(key, {}).get(name, 0.0)
                    d[name] = v0 + (d[name] - v0) * ratio
        act = activitySetup(spc) if useActivity else None
        srcRate, Ksink = reactionTerms(spc)
        checkSourceCharge(srcRate, Ksink, np.array([np.asarray(C[sp], dtype=float) for sp in spc]).T, Z)
        srcRate = None if srcRate is None else srcRate.T
        Ksink = None if Ksink is None else Ksink.T
        Iint = np.zeros((M, N))
        lag = {'dG': None}
        flux_fick, cbar, cond_bar, numerator0, denominator = nernstPlanckCurrentTerms(C, T_by_sp, z_by_sp, dx_total)
        denominator_safe = np.where(denominator > 0, denominator, 1.0)
        I_mol = resolve_imposed_current(numerator0, denominator_safe, dx_total, closureMode, voltageNow(tGlobalStart), temperature)
        with np.errstate(divide='ignore', invalid='ignore'):
            phi = -np.where(denominator > 0, (numerator0 + I_mol * currentMask()) / denominator_safe, 0.0) * dx_total
        phi, I_mol = sgSolveField(C, T_by_sp, z_by_sp, dx_total, closureMode, voltageNow(tGlobalStart), temperature, phi, I_mol,
                                  dLnGamma(act, np.array([np.asarray(C[sp], dtype=float) for sp in spc])) if act else None)
        psi = np.concatenate(([0.0], np.cumsum(phi)))
        c = np.array([np.asarray(C[sp], dtype=float) for sp in spc])
        aleR = {'r0': None, 'r1': None}
        if ale is not None:
            if ale[2]:
                c = c / ale[0][None, :]
            elif dn is not None:
                thF0, thDL0, sig0, phi0 = dnState['par']
                dnState['Sold'] = DV[None, :] * donnanStorage(c, Z, aleThF(thF0, ale[0]), thDL0, sig0, phi0,
                                                              dnState['psi'])[0]
        if pair:
            c[iH], c[iO], _ = waterPair(c[iH] - c[iO])
        if anyFixed:
            c[fixed] = fixVal[fixed]
        setLayout(coupledSpecies(c))
        monitor = nahed.monitorFor(centralDict)
        if monitor is not None:
            # coupling matrix of Nahed & de Lamare (nahed.py) : blocks = species i (block i), potential (block M)
            blockOf = np.arange(nU) % nb
            monitor.initial(spc + ['psi'])
        h = min(dt_total, float(centralDict.get('npImplicitStep') or dt_total / 1000.0))
        hMin = 1e-10 * dt_total
        t, nStep, nNewton, nReject = 0.0, 0, 0, 0
        prev = None
        Q = -(float(np.dot(betaPsi, psi[1:] - psi[:-1])) + dpScaled) if eoOpen else 0.0
        eoVolume = 0.0
        while dt_total - t > 1e-12 * dt_total:
            # end of the call reached in one step, or in two equal ones rather than a full step and a sliver
            rest = dt_total - t
            hTry = rest if rest <= h else (0.5 * rest if rest < 2.0 * h else h)
            Vn = None
            if voltage:
                dPhiEl = electrodeDrop(voltageNow(tGlobalStart + t))
                Vn = (voltageNow(tGlobalStart + t) - dPhiEl) / RT_over_F
            if act:
                lag['dG'] = dLnGamma(act, c)
            if dn is not None:
                dnState['par'] = dn['params'](c)
                dnState['psiIter'] = dnState['psi']
            sigma, book = electrodeSources(c, I_mol, hTry)
            if ale is not None:
                aleR['r0'] = ale[0] + (ale[1] - ale[0]) * (t / dt_total)
                aleR['r1'] = ale[0] + (ale[1] - ale[0]) * ((t + hTry) / dt_total)
            cStart, psiStart, IStart = c, psi, I_mol
            if prev is not None:
                w = hTry / prev[3]
                if pair:
                    Pp = (c[iH] - c[iO]) + w * ((c[iH] - c[iO]) - (prev[0][iH] - prev[0][iO]))
                cStart = np.maximum(c + w * (c - prev[0]), 0.0)
                if pair:
                    cStart[iH], cStart[iO], _ = waterPair(Pp)
                psiStart = psi + w * (psi - prev[1])
                IStart = I_mol + w * (I_mol - prev[2]) if voltage else I_mol
                QStart = Q + w * (Q - prev[4]) if eoOpen else Q
            else:
                QStart = Q
            sol, it = newton(cStart, c, psiStart, IStart, QStart, hTry, Vn, sigma)
            if sol is None and cStart is not c:
                nNewton += it
                sol, it = newton(c, c, psi, I_mol, Q, hTry, Vn, sigma)
            if sol is None and lay['cpl'] is not None and not lay['exact']:
                # no convergence with the block-triangular matrix : step retried with the whole matrix, kept for the
                # rest of the call if it converges (the charged species out of the coupled block were not negligible)
                nNewton += it
                stats['wholeMatrix'] += 1
                cplSp = lay['cplSp']
                setLayout(None)
                sol, it = newton(c, c, psi, I_mol, Q, hTry, Vn, sigma)
                if sol is None:
                    setLayout(cplSp)
            nNewton += it
            change = np.inf
            worstSp, worstNode = None, None
            if sol is not None:
                floor = np.maximum(atol, 0.01 * np.max(np.abs(c), axis=1))[:, None]
                relChange = np.abs(sol[0] - c) / (np.abs(c) + floor)
                change = float(np.max(relChange))
                iWorst, kWorst = np.unravel_index(np.argmax(relChange), relChange.shape)
                worstSp, worstNode = spc[iWorst], int(kWorst)
                if dn is not None:
                    psiNew = donnanPotential(sol[0], Z, dnState['par'][2], dnState['par'][3], psi0=dnState['psi'])[0]
                    dPsiD = np.abs(psiNew - dnState['psi'])
                    if maxChange * float(np.max(dPsiD)) > change:
                        change, worstSp, worstNode = maxChange * float(np.max(dPsiD)), 'Psi_D', int(np.argmax(dPsiD))
            if sol is None or change > 2.0 * maxChange:
                nReject += 1
                h = hTry * (0.25 if sol is None else max(0.2, 0.9 * maxChange / change))
                if h < hMin:
                    logError('nernstPlanckImplicit', f"time step below {writeTime(hMin, 3)} at t = {writeTime(t, 3)} "
                             f"(Newton {'did not converge' if sol is None else 'converged but concentrations change too fast'}"
                             f" after {nStep} steps, {nReject} rejected ; last attempt : h = {writeTime(hTry, 3)}, "
                             + (f"change = {change:.3g} for {worstSp} at node {worstNode}" if sol is not None else
                                (f"V = {voltageNow(tGlobalStart + t):.6g} V, " if voltage else "") + f"I = {I_mol * FARADAY:.4g} A")
                             + ") : check boundary conditions or increase npImplicitMaxIter / npImplicitMaxChange")
                continue
            prev = (c, psi, I_mol, hTry, Q)
            c, psi, I_mol, Q = sol
            if monitor is not None:
                # coupling matrix (eq. 38-39) : Jacobian [dR/du]^n of this step evaluated at its converged solution,
                # before the Donnan storage of the step is committed ; the Donnan state of the Newton is restored (the
                # Newton runs as without the coupling matrix). Increment : the pair H+/OH- is solved for
                # P = [H+] - [OH-], carried by the column of H+
                du = np.empty(nU)
                du[icIdx] = c - prev[0]
                du[ipIdx] = psi - prev[1]
                if pair:
                    du[icIdx[iH]] = (c[iH] - c[iO]) - (prev[0][iH] - prev[0][iO])
                    du[icIdx[iO]] = 0.0
                def stepResidual(u, cOld=prev[0], hs=hTry, Is=I_mol, Qs=Q):
                    kept = {key: dnState.get(key) for key in ('psiIter', 'jac')} if dn is not None else None
                    U = u.reshape(N, nb)
                    res = residual(U[:, :M].T.copy(), U[:, M].copy(), Is, Qs, cOld, hs, Vn, sigma, needJac=False)[0]
                    if kept is not None:
                        dnState.update(kept)
                    return res
                uNow = np.empty(nU)
                uNow[icIdx], uNow[ipIdx] = c, psi
                if dn is not None and not pair:
                    # with Donnan the Newton matrix leaves out the dependence of the conductances on the Donnan
                    # potential of the neighbour nodes : the Jacobian is computed by differences (fdJacobian)
                    couplingStepFD(monitor, stepResidual, uNow - du, uNow, nb, 1, t + hTry, hTry)
                else:
                    dnKept = {key: dnState.get(key) for key in ('psiIter', 'jac')} if dn is not None else None
                    residual(c, psi, I_mol, Q, prev[0], hTry, Vn, sigma, needJac=True)
                    if dnKept is not None:
                        dnState.update(dnKept)
                    couplingStep(monitor, rows, cols, lastJac['vals'], du, blockOf, t + hTry, hTry,
                                 None if pair else (stepResidual, uNow, nb, 1))
            Iint += hTry * c
            if dn is not None:
                parA = dnState['par']
                dnState['psi'] = donnanPotential(c, Z, parA[2], parA[3], psi0=dnState['psi'])[0]
                dnState['Sold'] = DV[None, :] * donnanStorage(c, Z, aleThF(parA[0], aleR['r1']), *parA[1:3], parA[3],
                                                              dnState['psi'])[0]
            eoVolume += Q * hTry
            commitBalance(book, I_mol)
            t += hTry
            nStep += 1
            if lay['cpl'] is not None:
                grown = coupledSpecies(c, lay['cplSp'])
                if not np.array_equal(grown, lay['cplSp']):
                    setLayout(grown)
            fac = min(2.0, max(0.5, 0.9 * maxChange / max(change, 1e-12)))
            if hTry < h:
                # last step(s) of the call, shortened : the next call starts from the full step
                if fac < 1.0:
                    h = hTry * fac
            elif not NP_HOLD_STEP[0] <= fac <= NP_HOLD_STEP[1]:
                h = hTry * fac
            if nStep >= nMax:
                logError('nernstPlanckImplicit', f"more than {nMax} implicit steps needed (h = {writeTime(hTry, 3)}) : "
                         "increase maxSubCycling or npImplicitMaxChange")
        centralDict['npImplicitStep'] = h
        centralDict['npImplicitStats'] = {'steps': nStep, 'rejected': nReject, 'newton': nNewton,
                                          'factorizations': stats['factor'], 'solves': stats['solve'],
                                          'coupledSpecies': lay['nCoupled'], 'wholeMatrixRetries': stats['wholeMatrix']}
        if monitor is not None:
            centralDict['npImplicitStats'].update(couplingMatrix=monitor.Phi, couplingBlocks=monitor.names)
        if dn is not None:
            Tfin = dnT(dnState['psi'], dnState['par'])
            for i, sp in enumerate(spc):
                T_by_sp[sp] = Tfin[i].copy()
            donnanEnd(dn, c, dnState['Sold'] / DV[None, :], dnState['psi'], dnState['par'], temperature)
        storeIntegral(spc, Iint.T)
        C = {sp: c[i] for i, sp in enumerate(spc)}
        if eoOpen:
            phi = psi[1:] - psi[:-1]
        else:
            phi, I_mol = sgSolveField(C, T_by_sp, z_by_sp, dx_total, closureMode, voltageNow(tGlobalStart + t), temperature,
                                      psi[1:] - psi[:-1], I_mol, dLnGamma(act, c) if act else None)
        if eo is not None:
            electroOsmosisDiagnostics(eo, Q, phi, eoVolume, temperature)
        display.stepPrint(f"[implicit : {nNewton} Newton iterations, {stats['factor']} LU, {nReject} rejected steps"
              + (f", {lay['nCoupled']} of {M} species in the coupled block" if lay['nCoupled'] < M else "")
              + (f", coupling matrix : {monitor.describe()}" if monitor is not None else "") + "]",
              end=" ", flush=True)
        return C, nStep, phi, I_mol
    def computeElectricalDiagnosticsSG(C, phi, I_mol, T_by_sp, z_by_sp, temperature):
        RT_over_F = GAS_CONSTANT * temperature / FARADAY
        psi = np.concatenate(([0.0], np.cumsum(RT_over_F * phi)))
        g_sp = {}
        spc = list(trsptedSpecies)
        dG = (dLnGamma(activitySetup(spc), np.array([np.asarray(C[sp], dtype=float) for sp in spc]))
              if useActivity else np.zeros((len(spc), len(phi))))
        for i, sp in enumerate(spc):
            z = z_by_sp[sp]
            u = z * phi + dG[i]
            g_sp[sp] = -(z ** 2) * T_by_sp[sp] * (bernoulliDeriv(-u) * np.maximum(C[sp][1:], 0.0)
                                                  + bernoulliDeriv(u) * np.maximum(C[sp][:-1], 0.0))
        g_tot = sum(g_sp.values())
        g_tot = np.where(g_tot > 0, g_tot, 1.0)
        area_ref = areaRef()
        return {
            'electricPotential': psi,
            'currentTotal': I_mol * FARADAY,
            'currentDensity': (I_mol * FARADAY) / area_ref if (area_ref > 0 or not isRadial) else np.nan,
            'transferenceNumbers': {sp: g_sp[sp] / g_tot for sp in trsptedSpecies},
        }
    def computeElectricalDiagnostics(C, T_by_sp, z_by_sp, dx_total, closureMode, closureValue, temperature):
        flux_fick, cbar, cond_bar, numerator0, denominator = nernstPlanckCurrentTerms(C, T_by_sp, z_by_sp, dx_total)
        denominator_safe = np.where(denominator > 0, denominator, 1.0)
        I_mol = resolve_imposed_current(numerator0, denominator_safe, dx_total, closureMode, closureValue, temperature)
        RT_over_F = GAS_CONSTANT * temperature / FARADAY
        Lambda = (numerator0 + I_mol * currentMask()) / denominator_safe
        dPsi = -RT_over_F * Lambda * dx_total
        psi = np.concatenate(([0.0], np.cumsum(dPsi)))
        transferenceNumbers = {}
        for sp in trsptedSpecies:
            z = z_by_sp[sp]
            transferenceNumbers[sp] = (z ** 2) * cond_bar[sp] * cbar[sp] / denominator_safe
        area_ref = areaRef()
        return {
            'electricPotential': psi,
            'currentTotal': I_mol * FARADAY,
            'currentDensity': (I_mol * FARADAY) / area_ref if (area_ref > 0 or not isRadial) else np.nan,
            'transferenceNumbers': transferenceNumbers,
        }
    def nernstPlanckCurrentTransport(C_old, C_new):
        speciesValence = centralDict['speciesValence']
        speciesDiffCoeff = resolve_species_property('speciesDiffCoeff', 'diffCoeff')
        speciesPorosity = resolve_species_property('speciesPorosity', 'porosity')
        lo, hi = 0, n
        dxHalfCell_trim = meshProp(centralDict['dxHalfCell'], lo, hi)
        dx_total = dxHalfCell_trim[:-1] + dxHalfCell_trim[1:]
        nodeSize_trim = meshProp(centralDict['nodeSize'], lo, hi)
        T_by_sp = {sp: precompute_species_transmissivity(lo, hi, speciesDiffCoeff[sp], speciesPorosity[sp]) for sp in trsptedSpecies}
        z_by_sp = {sp: speciesValence[sp] for sp in trsptedSpecies}
        storage_by_sp = {}
        volume_trim = cellVolume(lo, hi)
        for sp in trsptedSpecies:
            poro_sp = meshProp(speciesPorosity[sp], lo, hi)
            storage = volume_trim * poro_sp
            storage_by_sp[sp] = np.where(storage > 0, storage, 1.0)
        electrodeVolume = centralDict.get('electrodeVolume') or [None, None]
        areaR_trim, areaL_trim = halfCellAreas(lo, hi)
        for side, iRes, iSmp, iFace in ((0, 0, 1, 0), (1, n - 1, n - 2, -1)):
            if isElectrode(side) and electrodeVolume[side]:
                aSmp = areaL_trim[iSmp] if side == 0 else areaR_trim[iSmp]
                for sp in trsptedSpecies:
                    cond = speciesPorosity[sp][iSmp] * aSmp * speciesDiffCoeff[sp][iSmp]
                    T_by_sp[sp][iFace] = 1000.0 * cond / dxHalfCell_trim[iSmp] if cond > 0 else 0.0
                    storage_by_sp[sp][iRes] = electrodeVolume[side]
        temperature = centralDict.get('temperature', 298.15)
        if centralDict.get('imposedCurrent'):
            closureMode, closureValue = 'current', centralDict['imposedCurrent']
        elif centralDict.get('imposedCurrentDensity'):
            closureMode, closureValue = 'currentDensity', centralDict['imposedCurrentDensity']
        elif centralDict.get('imposedVoltage') is not None:      # 0 V is a short circuit, not an open circuit
            closureMode, closureValue = 'voltage', centralDict['imposedVoltage']
        elif centralDict.get('rampPotential'):                    # the ramp alone imposes the voltage
            closureMode, closureValue = 'voltage', centralDict['rampPotential'][1]
        else:
            closureMode, closureValue = 'current', 0.0
        if centralDict.get('rampPotential') and closureMode != 'voltage':
            logError('rampPotential', "rampPotential is a voltage ramp : it is not used with imposedCurrent / "
                     "imposedCurrentDensity (give imposedVoltage, or no closure, with rampPotential)")
        # an imposed current (or voltage) crosses the whole column: each end needs an electrode or an open condition
        if not (closureMode == 'current' and not closureValue):
            closed = [end for side, end in ((0, 'x = 0'), (1, 'x = L'))
                      if not isElectrode(side) and sideSetup(side)[1] == 'closed']
            if closed:
                logError('boundaries', f"imposed {closureMode} ({closureValue}) : no charge can cross the closed "
                         f"boundary at {' and '.join(closed)} (boundaryConditions = {centralDict['boundaryConditions']}) "
                         ": give an electrode there ('electrode', 'electrode+constant', 'electrode+flux') or an open "
                         "condition ('constant', 'flux')")
        C_state = {sp: C_old[sp].to_numpy() for sp in trsptedSpecies}
        updateVoltageDrop(C_state, closureMode, closureValue, temperature)
        npSolve =nernstPlanckImplicit if npSolver == 'implicit' else nernstPlanckCurrentAdaptive
        flow = flowSetup(electrodeVolume)
        if electrodeNodes() != (0, n - 1):
            if flow is not None or useDonnan or any(electrodeVolume):
                logError('electrodeNodes', "interior electrode nodes (electrodeNodes) are not implemented with water "
                         "flow / electro-osmosis, donnan or electrodeVolume")
            if isRadial:
                logError('electrodeNodes', "interior electrode nodes (electrodeNodes) are implemented in cartesian "
                         "coordinates only")
        eoArg = {'eo': flow} if flow is not None else {}
        ale = centralDict.get('_ale')
        if ale is not None:
            sA = ale['S']
            sB = sA - ale['rate'] * dt
            ale['S'] = sB
            eoArg['ale'] = (sA / ale['end'], sB / ale['end'], ale.pop('first', False))
        if useDonnan:
            spc = list(trsptedSpecies)
            if centralDict.get('speciesPorosity'):
                logWarning('donnan', "speciesPorosity is ignored with donnan : the porosity is split into free water "
                           "and double layer (anion exclusion is computed by the Donnan equilibrium)")
            resNodes, resFaces = [], []
            V = cellVolume(lo, hi).copy()
            thetaAll = meshProp(centralDict['porosity'], lo, hi)
            for side, iRes, iSmp, iFace in ((0, 0, 1, 0), (1, n - 1, n - 2, n - 2)):
                if isElectrode(side) and electrodeVolume[side]:
                    resNodes.append(iRes)
                    resFaces.append((iFace, iSmp))
                    V[iRes] = electrodeVolume[side] / thetaAll[iRes]
            dnCtx = donnanSetup(spc, np.array([speciesDiffCoeff[sp] for sp in spc]), resNodes)
            dnCtx.update({'V': V, 'resFaces': resFaces})
            thF0 = dnCtx['params'](np.array([C_state[sp] for sp in spc]))[0]
            for sp in spc:
                storage_by_sp[sp] = np.where(V * thF0 > 0, V * thF0, 1.0)
            eoArg['dn'] = dnCtx
        C_state, nSub, phi, I_mol = npSolve(
            C_state, dt, T_by_sp, z_by_sp, dx_total, storage_by_sp,
            closureMode, closureValue, temperature, **eoArg)
        display.stepPrint(f"[{nSub} sub-steps]", end=" ", flush=True)
        if centralDict.get('voltageDropInfo'):
            info = centralDict['voltageDropInfo']
            display.stepPrint(f"[dV = {info['dV']:.4g} V, pH anode {info['pH_anode']:.3g}, pH cathode {info['pH_cathode']:.3g}]",
                  end=" ", flush=True)
        diagnostics =computeElectricalDiagnosticsSG(C_state, phi, I_mol, T_by_sp, z_by_sp, temperature)
        centralDict['electricPotential'] = diagnostics['electricPotential']
        centralDict['currentTotal'] = diagnostics['currentTotal']
        centralDict['currentDensity'] = diagnostics['currentDensity']
        centralDict['transferenceNumbers'] = diagnostics['transferenceNumbers']
        centralDict.pop('cellVoltageInfo', None)
        target = centralDict.get('targetVoltage')
        if target is not None:
            if closureMode == 'voltage':
                logError('targetVoltage', "targetVoltage is used with imposedCurrent / imposedCurrentDensity : with "
                         "imposedVoltage give electrodeResistance")
            if centralDict.get('electrodeResistance') is not None:
                logError('targetVoltage', "targetVoltage computes the electrode resistance : give targetVoltage or "
                         "electrodeResistance, not both")
        if closureMode != 'current' or closureValue:
            if (centralDict.get('electrodeResistance') is not None or target is not None
                    or centralDict.get('voltageDropInfo')):
                info = cellVoltage(diagnostics['electricPotential'], diagnostics['currentTotal'],
                                   closureValue if closureMode == 'voltage' else None)
                centralDict['cellVoltageInfo'] = info
                display.stepPrint(f"[cell V = {info['cellVoltage_V']:.4g} V, R.I = {info['electrodeOhmic_V']:.4g} V, "
                      + (f"R = {info['electrodeResistance_Ohm']:.4g} ohm, " if 'electrodeResistance_Ohm' in info else "")
                      + f"I = {diagnostics['currentTotal']:.4g} A]", end=" ", flush=True)
        return _writeColumns(C_new, trsptedSpecies, C_state)


    dt = toSeconds(centralDict['dtStep'], centralDict['timeUnit'])
    if centralDict.get('tTransportStart') is not None:
        tGlobalStart = toSeconds(float(centralDict['tTransportStart']), centralDict['timeUnit'])
    else:
        tGlobalStart = centralDict.get('simElapsedTime', 0.0)
    centralDict['simElapsedTime'] = tGlobalStart + dt
    if centralDict.get('FickDiffusion') or centralDict.get('advection'):
        logError('basicTransport', "FickDiffusion and advection have been removed from nativeTransport : use "
                 "ADE = True (velocity = 0 : Fick diffusion ; diffCoeff = 0 and dispersivity = 0 : pure advection)")
    useADE = bool(centralDict.get('ADE'))
    useNernstPlanck = bool(centralDict.get('NernstPlanck')) and not useADE
    useActivity = bool(centralDict.get('activityGradient'))
    useDonnan = centralDict.get('donnan') is not None
    centralDict.pop('donnanOutput', None)
    if not (useADE or useNernstPlanck):
        logError('basicTransport', "no transport process selected : set ADE or NernstPlanck in the input file")
    if useADE and centralDict.get('NernstPlanck'):
        logWarning('basicTransport', "ADE and NernstPlanck are both set : ADE is solved, NernstPlanck is ignored")
    velocity = centralDict.get('velocity') or 0.0
    adeSolver = str(centralDict.get('adeSolver') or 'explicit').lower()
    advScheme = str(centralDict.get('advectionScheme') or 'upwind').lower()
    if useADE and advScheme not in ('upwind', 'lud', 'vanleer'):
        logError('basicTransport', f"unknown advectionScheme '{advScheme}' : use 'upwind' (first order), "
                 "'LUD' (linear upwind differencing, second order) or 'vanLeer' (TVD, second order, no oscillation)")
    if useADE and useDonnan and adeSolver == 'implicit' and advScheme in ('lud', 'vanleer'):
        logError('basicTransport', f"advectionScheme = '{centralDict.get('advectionScheme')}' with donnan is "
                 "implemented for the explicit ADE only : set adeSolver = 'explicit' or advectionScheme = 'upwind'")
    if useADE and useDonnan and adeSolver == 'implicit' and adeTheta() != 1.0:
        logError('basicTransport', "adeTimeScheme = 'crank-nicolson' is not implemented with donnan : use "
                 "adeTimeScheme = 'euler'")
    if useADE and adeSolver not in ('explicit', 'implicit'):
        logError('basicTransport', f"unknown adeSolver '{adeSolver}' : use 'explicit' or 'implicit'")
    if velocity and centralDict.get('dispersivity') and not centralDict.get('dispersionInTransport'):
        logError('basicTransport', "engine.py adds velocity x dispersivity to diffCoeff, but the dispersion is now "
                 "computed by nativeTransport (theta.D_meca = alphaL.|q|, q = Q / A(r) in radial) : remove that "
                 "addition from engine.py and set 'dispersionInTransport' : True in centralDict")
    coordSystem = str(centralDict.get('coordinateSystem') or 'cartesian').lower()
    if coordSystem not in ('cartesian', 'cylindrical', 'spherical'):
        logError('coordinateSystem', f"unknown coordinateSystem '{coordSystem}' : use 'cartesian', "
                 "'cylindrical' or 'spherical'")
    isRadial = coordSystem != 'cartesian'
    geom = None
    if isRadial:
        if 'x' in commMtrx.columns:
            rNodes = commMtrx['x'].to_numpy(dtype=float)
        elif centralDict.get('x') is not None:
            rNodes = np.asarray(centralDict['x'], dtype=float)
        else:
            logError('coordinateSystem', f"coordinateSystem = '{coordSystem}' needs the radius of the nodes "
                     "(column 'x' of the initial conditions)")
        geom = radialGeometry(rNodes)
        if not np.all(meshProp(centralDict['area'], 0, len(rNodes)) == 1):
            logWarning('coordinateSystem', f"area is not used with coordinateSystem = '{coordSystem}' : the surface "
                       "is 2.pi.r.cylinderHeight (cylindrical) or 4.pi.r^2 (spherical)")
        geom['flowRate'] = velocity * geom['nodeArea'][0] if velocity else 0.0
        if velocity and not geom['nodeArea'][0] > 0:
            logError('coordinateSystem', "velocity is the Darcy velocity at r = x[0] ; x[0] = 0 (axis / centre) has "
                     "no surface, the flow rate is undefined : start the mesh at x[0] > 0")
        usesCurrentDensity = (useNernstPlanck and not centralDict.get('imposedCurrent')
                              and centralDict.get('imposedCurrentDensity'))
        if usesCurrentDensity and not geom['nodeArea'][0] > 0:
            logError('coordinateSystem', "imposedCurrentDensity is given at r = x[0] ; x[0] = 0 (axis / centre) has "
                     "no surface : use imposedCurrent or start the mesh at x[0] > 0")
    C_old = _takeColumns(commMtrx, trsptedSpecies)
    C_new = None if useADE else _takeColumns(commMtrx, trsptedSpecies)
    n = len(C_old)
    hasImposedClosure = (bool(centralDict.get('imposedCurrent')) or bool(centralDict.get('imposedCurrentDensity'))
                         or centralDict.get('imposedVoltage') is not None or bool(centralDict.get('rampPotential')))
    useNernstPlanckCurrent = useNernstPlanck and hasImposedClosure
    useNernstPlanckZeroCurrent = useNernstPlanck and not hasImposedClosure
    npSolver = str(centralDict.get('npSolver') or 'explicit').lower()
    if useNernstPlanck and npSolver not in ('explicit', 'implicit'):
        logError('basicTransport', f"unknown npSolver '{npSolver}' : use 'explicit' or 'implicit'")
    eoValue = centralDict.get('electroOsmoticPermeability')
    eoRequested = eoValue is not None and bool(np.any(np.asarray(eoValue, dtype=float) != 0))
    if eoRequested and useADE:
        logWarning('electroOsmosis', "electroOsmoticPermeability is only used with NernstPlanck : ignored with ADE "
                   "(the Darcy velocity is given by 'velocity')")
    if useNernstPlanck and advScheme in ('lud', 'vanleer'):
        logWarning('basicTransport', f"advectionScheme = '{centralDict.get('advectionScheme')}' applies to the explicit ADE only : ignored by "
                   "Nernst-Planck")
    if useDonnan and useNernstPlanck and npSolver != 'implicit':
        logError('donnan', "donnan with Nernst-Planck is implemented for the implicit solver (npSolver = 'implicit') ; "
                 "with ADE : adeSolver = 'explicit' or 'implicit'")
    hasElectrode = isElectrode(0) or isElectrode(1)
    waterLoss = centralDict.get('_waterLossRate') is not None
    if useADE and waterLoss:
        logError('saturation', "a saturation written by the chemistry (saturation = column name) expels the water "
                 "lost as a Darcy flow : implemented with NernstPlanck only (not with ADE)")
    if useADE and hasElectrode:
        logWarning('boundaries', "'electrode' has no effect with ADE (no electric current) : only the transport "
                   "condition of the electrode boundary is applied ('electrode' = 'electrode+closed')")

    if useNernstPlanckCurrent:
        C_old = allBoundaries(C_old)
        C_new = nernstPlanckCurrentTransport(C_old, C_new)
        C_new = allBoundaries(C_new)

    elif useNernstPlanckZeroCurrent:
        C_old = allBoundaries(C_old)
        if npSolver == 'implicit' or hasElectrode or eoRequested or velocity or waterLoss:
            C_new = nernstPlanckCurrentTransport(C_old, C_new)
        else:
            C_new = nernstPlanckTransport(C_old, C_new)
        C_new = allBoundaries(C_new)

    else:
        C_new = adeTransport(C_old)
    if centralDict.get('nahed') and not (useNernstPlanck and npSolver == 'implicit'):
        monitor = centralDict.get('nahedMonitor')
        if monitor is not None:
            display.stepPrint(f"[coupling matrix : {monitor.describe()}]", end=" ", flush=True)
    if reactionIntegral and (centralDict.get('transportSource') is not None
                             or centralDict.get('transportSink') is not None):
        centralDict['_reactionIntegral'] = pd.DataFrame(reactionIntegral['I'] / toSeconds(1.0, centralDict['timeUnit']),
                                                        index=C_old.index, columns=reactionIntegral['species'])
    return C_new


def _dtypes(df):
    try:
        return df._mgr.get_dtypes()
    except AttributeError:
        return df.dtypes.to_numpy()


def _floatColumns(df):
    return df.columns.is_unique and frames.allFloat64(df)


def _takeColumns(df, names):
    """df[names].copy() (same values, same array layout), built from the array for unique float64 columns."""
    names = list(names)
    if len(set(names)) == len(names) and _floatColumns(df):
        positions = frames.indexer(df.columns, names)
        if np.all(positions >= 0):
            return pd.DataFrame(df.to_numpy().T[positions].T, index=df.index, columns=frames.columnIndex(names))
    return df[names].copy()


def _mixedNumericColumns(df, transported):
    """Names of the non-float64 columns of df when df has unique columns, every column a numpy integer or float
    column, and none of those non-float columns transported (nor 'x'); None otherwise."""
    if not df.columns.is_unique or 'x' not in df.columns:
        return None
    names, found = df.columns.tolist(), []
    for name, dtype in zip(names, _dtypes(df)):
        if not isinstance(dtype, np.dtype) or dtype.kind not in 'iuf':
            return None
        if dtype != np.float64:
            if name in transported or name == 'x':
                return None
            found.append(name)
    return found or None


_presentLists = {}              # (id(columns), names) -> (columns, names found)


def _presentIn(columns, names):
    """[s for s in names if s in columns] (a new list). For an Index of str labels and str names, membership in the
    Index is membership in the set of its labels (hash, then equality): the list is made from that set once, then
    kept for the same Index object (immutable) and the same names, as frames.indexer keeps its positions; one
    `in Index` costs ~0.4 us (~70 us for 170 species at every call)."""
    names = list(names)
    if not all(type(s) is str for s in names):
        return [s for s in names if s in columns]
    key = (id(columns), tuple(names))
    found = _presentLists.get(key)
    if found is not None and found[0] is columns:
        return list(found[1])
    labels = columns.tolist()
    if not all(type(c) is str for c in labels):
        return [s for s in names if s in columns]
    known = set(labels)
    present = [s for s in names if s in known]
    if len(_presentLists) >= 64:
        _presentLists.clear()
    _presentLists[key] = (columns, present)
    return list(present)


def _writeColumns(df, names, arrays):
    """df after `df[name] = arrays[name]` for each name, in order (the loop below): built at once, one block of the same
    values, when the names are the columns of df in their order (distinct), df has float64 columns only and each array
    is float64 with one value per row. Every column then holds a copy of its array, as after the loop (pandas copies
    the array of `df[name] = array`, and writes no warning for a column replaced): a frame equal to the loop's, without
    its one block per column (~60 us per column, then to_numpy() gathering the blocks again)."""
    names = list(names)
    if (len(set(names)) == len(names) and df.columns.is_unique and df.columns.tolist() == names
            and _floatColumns(df)):
        found = [arrays[name] for name in names]
        if all(type(a) is np.ndarray and a.dtype == np.float64 and a.ndim == 1 and len(a) == len(df) for a in found):
            values = np.empty((len(names), len(df)))
            for j, a in enumerate(found):
                values[j] = a
            out = pd.DataFrame(values.T, index=df.index, columns=df.columns)
            if df.attrs:
                out.attrs = dict(df.attrs)
            return out
    for name in names:
        df[name] = arrays[name]
    return df


def _allIn(columns, names):
    """all(name in columns for name in names): for unique columns and names (strings), from the positions kept by
    frames.indexer (-1 for a missing name)."""
    if columns.is_unique and all(type(name) is str for name in names):
        return bool(np.all(frames.indexer(columns, names) >= 0))
    return all(name in columns for name in names)


class _TransportedValues:
    """The transported columns of a single-process transport call before they become a frame: values (nodes x
    species, float64), the index and the names. frame() is the frame basicTransport returned before (same values,
    same layout, same labels); trspt copies the values into its snapshot without it."""

    def __init__(self, values, index, names):
        self.values, self.index, self.names = values, index, names
        self.columns = frames.columnIndex(names)

    def to_numpy(self):
        return self.values

    def frame(self):
        return pd.DataFrame(self.values.T.copy().T, index=self.index, columns=frames.columnIndex(self.names))


def basicTransport(centralDict, commMtrx, trsptedSpecies, valuesOnly=False):
    if 'consoleDisplay' in centralDict:
        display.configure(centralDict['consoleDisplay'])
    transportLog = []
    # transported columns as they were, returned when the transport fails (then trspt stops the run): taken only in
    # that case (the transport never writes into commMtrx), instead of a copy at every call; taken here when a name
    # is missing, so that its KeyError is raised at the same place as before
    columns = commMtrx.columns
    C_unchanged = None if _allIn(columns, trsptedSpecies) else _takeColumns(commMtrx, trsptedSpecies)
    saved = {k: centralDict.get(k) for k in ('boundaryConditions', 'dtStep', 'tTransportStart', 'area')}
    integral = None
    calcWall = 0.0
    try:
        C = commMtrx
        segments = transportSegments(centralDict)
        # areaSchedule : area of the segment, from the area of the end of the previous call (areaActive) ; the volume
        # added to a node is pure water (areaChange, on a copy : commMtrx is never written)
        areaPrev = centralDict.get('areaActive') if centralDict.get('areaSchedule') else None
        for k, (bc, dtSeg, tSeg) in enumerate(segments):
            if list(bc) != list(centralDict.get('bcActive') or bc):
                display.stepPrint(f"\n[boundary conditions from t = {tSeg}{centralDict['timeUnit']} : {bc[0]!r} (x = 0), "
                      f"{bc[1]!r} (x = L)]", end=" ", flush=True)
            centralDict['bcActive'] = list(bc)
            if areaPrev is not None:
                if tSeg is None:
                    raise TransportError('areaSchedule', "the start time of the transport call is unknown "
                                         "(tTransportStart) : areaSchedule cannot be applied")
                areaSeg = areaAt(centralDict, tSeg)
                changed = np.flatnonzero(areaSeg != areaPrev)
                if len(changed):
                    display.stepPrint(f"\n[area from t = {tSeg}{centralDict['timeUnit']} : node(s) "
                                      f"{changed[:10].tolist()}{' ...' if len(changed) > 10 else ''} "
                                      f"{areaPrev[changed[:3]].tolist()} -> {areaSeg[changed[:3]].tolist()} m2]",
                                      end=" ", flush=True)
                    C = areaChange(C, trsptedSpecies, areaPrev, areaSeg)
                centralDict['area'] = areaSeg
                areaPrev = areaSeg
            centralDict.update({'boundaryConditions': list(bc), 'dtStep': dtSeg, 'tTransportStart': tSeg})
            ref = time.perf_counter()
            try:
                C_seg = transportCore(centralDict, C, trsptedSpecies, transportLog)
            finally:
                calcWall += time.perf_counter() - ref
            values = (C_seg.to_numpy() if C_seg.columns.is_unique and C_seg.columns.tolist() == list(trsptedSpecies)
                      else C_seg[trsptedSpecies].to_numpy())
            seg = centralDict.pop('_reactionIntegral', None)
            if seg is not None:
                integral = seg if integral is None else integral + seg
            if k == len(segments) - 1 and values.dtype == np.float64 and C.columns.is_unique:
                if valuesOnly:              # single process (trspt): the frame built only if trspt needs it
                    return _TransportedValues(values, C.index, trsptedSpecies), False, transportLog, integral, calcWall
                return (pd.DataFrame(values.T.copy().T, index=C.index, columns=frames.columnIndex(trsptedSpecies)), False,
                        transportLog, integral, calcWall)
            C = C.copy()
            C[trsptedSpecies] = values
        return C[trsptedSpecies].copy(), False, transportLog, integral, calcWall
    except TransportError as err:
        where, message = err.args
    except Exception as err:
        where = 'nativeTransport'
        message = f"unexpected {type(err).__name__} : {err}\n" + traceback.format_exc()
    finally:
        centralDict.update(saved)
    transportLog.append({'level': 'ERROR', 'where': where, 'message': message, 'count': 1})
    if C_unchanged is None:
        C_unchanged = _takeColumns(commMtrx, trsptedSpecies)
    return C_unchanged, True, transportLog, None, calcWall


def writeWarningLog(centralDict, transportLog):
    merged = []
    for entry in transportLog:
        same = [e for e in merged if (e['level'], e['where'], e['message']) == (entry['level'], entry['where'], entry['message'])]
        if same:
            same[0]['count'] += entry['count']
        else:
            merged.append(dict(entry))
    transportLog = merged
    if not transportLog:
        return
    # console display by the main process only (basicTransport may run in a worker), above the progress bar, where
    # a warning is shown once per origin: its values (a cell voltage, ...) change from one step to the next
    for entry in transportLog:
        text = f"{entry['level']} ({entry['where']}) : {entry['message'].rstrip()}"
        if entry['level'] == 'WARNING' and not display.stepByStep():
            display.message(text + " (next ones in warning.log only)", key=('transport', entry['where']))
        else:
            display.message(text)
    nErr = sum(entry['level'] == 'ERROR' for entry in transportLog)
    report = [f"nativeTransport, time = {centralDict['tStep']}{centralDict['timeUnit']}, "
              f"time step n°{centralDict['lStep']+1} : {nErr} error(s), {len(transportLog) - nErr} warning(s)\n"]
    for entry in transportLog:
        count = f" [x{entry['count']}]" if entry['count'] > 1 else ""
        lines = entry['message'].rstrip('\n').split('\n')
        report.append(f"    {entry['level']} ({entry['where']}){count} : {lines[0]}\n")
        report.extend(f"        {line}\n" for line in lines[1:])
    report.append("\n")
    warningManager.block("".join(report), len(transportLog))



_PROCESS_CALL_COST = 0.005
_PROCESS_START_COST = 2.0
# choice between worker processes and a single process, per instance (solverInstances; None: single nativeTransport)
_transportProcesses = {}


def resetProcesses():
    """End of run: the choice between worker processes and a single process is made again at the next run; the
    operators kept for the settings of the run are released."""
    _transportProcesses.clear()
    _operators.clear()


# transportThreads > 1: keys of centralDict that the worker code (basicTransport and what it runs with centralDict)
# never reads, left out of the dictionary pickled for each worker at every call: the time steps of the run (dtpycte,
# 370 kB for 100 000 steps), the output settings, data of the chemistry. tests/test_worker_dicts.py checks that the
# worker code never names them
WORKER_SKIP = frozenset({'dtpycte', 'output', 'stoichPhreeqC', 'stoichReduced', 'reactions', 'primToSecSpecies',
                         'nativeSpeciationU', 'transportReactionIntegral'})


def _workerDict(centralDict):
    """centralDict for the transport workers: the same values (same objects), without the keys of WORKER_SKIP."""
    return {k: v for k, v in centralDict.items() if k not in WORKER_SKIP}


def _chooseProcesses(centralDict, calcWall):
    """After a single-process call: worker processes kept only if their gain over the remaining steps exceeds their cost."""
    n = centralDict['transportThreads']
    remaining = max(len(centralDict.get('dtpycte') or []) - centralDict.get('lStep', 0), 1)
    gain = remaining * (calcWall * (1.0 - 1.0 / n) - _PROCESS_CALL_COST) - _PROCESS_START_COST
    _transportProcesses[centralDict.get('solverInstance')] = gain > 0
    if gain > 0:
        # worker processes started now, during the speciation that follows, instead of at the next transport call
        warningManager.warm(n)
    else:
        warningManager.info(f"{centralDict.get('solverInstance') or 'nativeTransport'} : transportThreads = {n} not used : a transport call takes "
                            f"{1e3 * calcWall:.1f} ms in a single process, less than the cost of {n} worker processes "
                            f"(~{1e3 * _PROCESS_CALL_COST:.0f} ms per call and ~{_PROCESS_START_COST:.0f} s to start) : "
                            "single-process transport (same results)")


_transportColumns = [None]       # columns Index of the commMtrx built by the last call of trspt


def trspt(centralDict):
    startTransport = time.perf_counter()
    display.stepPrint("Native transport", end=" ", flush=True)
    if (centralDict.get('NernstPlanck') or centralDict.get('donnan') is not None
            or centralDict.get('crossDiffusion')) and centralDict['transportThreads'] > 1:
        warningManager.warn("nativeTransport (trspt) : transportThreads forced to 1 (Nernst-Planck / Donnan / "
                            "crossDiffusion couple all species)")
        centralDict['transportThreads'] = 1
    processes = _transportProcesses.get(centralDict.get('solverInstance'))
    threads = centralDict['transportThreads'] if processes else 1
    waterMass = centralDict.get('cellWaterMass') if centralDict.get('molesStorage') else None
    donnanTotals = (isinstance(centralDict.get('donnan'), dict)
                    and str(centralDict['donnan'].get('coupling', 'free')).lower() == 'totals')
    centralDict.pop('donnanBorrowed', None)
    centralDict.pop('_waterLossRate', None)
    centralDict.pop('_ale', None)
    centralDict.pop('waterExpelled', None)
    if centralDict.get('saturationColumn') and centralDict.get('electroOsmosis'):
        centralDict['electroOsmosis'].update({'waterFlow_m3s': 0.0, 'darcyVelocity_ms': 0.0,
                                              'pressure_Pa': np.zeros(len(centralDict['commMtrx']))})
    saturationColumn = centralDict.get('saturationColumn')
    if saturationColumn:
        centralDict['saturation'] = centralDict['commMtrx'][saturationColumn].to_numpy(dtype=float).copy()
    saturation = centralDict.get('saturation')
    if saturation is not None:
        saturation = np.asarray(saturation, dtype=float)
        if saturation.ndim == 0:
            saturation = np.full(len(centralDict['commMtrx']), float(saturation))
        if len(saturation) != len(centralDict['commMtrx']) or not np.all((saturation > 0) & (saturation <= 1)):
            raise ValueError("nativeTransport : saturation must be in ]0, 1], a number or one value per node")
        saturationNow = saturation
        if np.all(saturation == 1):
            saturation = None
    if donnanTotals:
        if waterMass is not None:
            raise ValueError("nativeTransport : donnan coupling = 'totals' is not implemented with molesStorage")
        centralDict['_donnanBorrow'] = True
    savedPorosity = None
    if saturationColumn:
        previous = centralDict.get('saturationPrevious')
        previous = saturationNow if previous is None else np.asarray(previous, dtype=float)
        dtCall = toSeconds(centralDict['dtStep'], centralDict['timeUnit'])
        loss = np.asarray(centralDict['porosity'], dtype=float) * (previous - saturationNow)
        if np.any(loss != 0) and dtCall > 0:
            centralDict['_waterLossRate'] = loss / dtCall
            centralDict['_ale'] = {'S': previous.copy(), 'end': saturationNow.copy(),
                                   'rate': (previous - saturationNow) / dtCall, 'first': True}
        centralDict['saturationPrevious'] = saturationNow.copy()
        if centralDict.get('tortuosityMode') == 'millington-quirk':
            savedPorosity = {k: centralDict.get(k) for k in ('porosity', 'speciesPorosity', 'diffCoeff',
                                                             'speciesDiffCoeff', 'tortuosity')}
            tau = np.asarray(centralDict['porosity'], dtype=float) ** (1.0 / 3.0) * saturationNow ** (7.0 / 3.0)
            centralDict['tortuosity'] = tau
            centralDict['diffCoeff'] = tau * np.asarray(centralDict['diffCoeffWater'], dtype=float)
            centralDict['speciesDiffCoeff'] = {spc: tau * np.asarray(val, dtype=float)
                                              for spc, val in centralDict['speciesDiffCoeffWater'].items()}
    if waterMass is not None or saturation is not None:
        if waterMass is not None:
            waterMass = np.asarray(waterMass, dtype=float)
        fill = waterMass if saturation is None else (saturation if waterMass is None else waterMass * saturation)
        if savedPorosity is None:
            savedPorosity = {k: centralDict.get(k) for k in ('porosity', 'speciesPorosity')}
        centralDict['porosity'] = np.asarray(centralDict['porosity'], dtype=float) * fill
        if centralDict.get('speciesPorosity'):
            centralDict['speciesPorosity'] = {s: np.asarray(v, dtype=float) * fill
                                             for s, v in centralDict['speciesPorosity'].items()}
    if waterMass is not None:
        moles = centralDict['commMtrx']
        waterCols = _presentIn(moles.columns, centralDict['transportedSpecies'])
        src = centralDict.get('transportSource')
        if src is not None:
            savedPorosity['transportSource'] = src
            if isinstance(src, pd.DataFrame):
                src = src.copy()
                cols = [s for s in waterCols if s in src.columns]
                src[cols] = src[cols].div(waterMass, axis=0)
            else:
                src = {s: (np.asarray(v, dtype=float) / waterMass if s in waterCols else v) for s, v in src.items()}
            centralDict['transportSource'] = src
        # the frame given to the transport, in one process or in the worker processes (same frame in both cases)
        if _floatColumns(moles):
            conc = moles.to_numpy(dtype=float, copy=True)
            iw = frames.indexer(moles.columns, waterCols)
            conc[:, iw] = conc[:, iw] / waterMass[:, None]
            centralDict['commMtrx'] = pd.DataFrame(conc, index=moles.index, columns=moles.columns)
        else:
            centralDict['commMtrx'] = pd.concat([moles[waterCols].div(waterMass, axis=0), moles.drop(columns=waterCols)],
                                                axis=1)[moles.columns]
    initialIndex = centralDict['commMtrx'].columns
    initialColumns = initialIndex.tolist()
    transportedSet = set(centralDict['transportedSpecies'])
    otherColumns = [s for s in initialColumns if s not in transportedSet]
    snapshot = None
    # donnan coupling 'totals' converts the frame after the transport to float64 (to_numpy(dtype=float), below): a
    # frame of numpy int and float columns (e.g. 'time', an int dt) is taken as a float snapshot too, with the same
    # values at the end. Its non-float columns not transported are put back as they were in the frame of the
    # 'transport' output, written before that conversion; without the conversion (no double layer borrowed), the frame
    # of the pandas path is built (mixedTotals: those columns, entryFrame: commMtrx as given)
    entryFrame = centralDict['commMtrx']
    mixedTotals = None
    if donnanTotals and waterMass is None and threads == 1 and not _floatColumns(entryFrame):
        mixedTotals = _mixedNumericColumns(entryFrame, transportedSet)
    if _floatColumns(centralDict['commMtrx']) or mixedTotals is not None:
        # x as before the transport: its column of the snapshot (taken before anything is written into it), the values
        # of commMtrx['x'].copy(); that Series is built only if the pandas path below needs it (xSaved)
        xPosition = initialIndex.get_loc('x')
        snapshot = centralDict['commMtrx'].to_numpy(dtype=float, copy=True)
        snapshotIndex = centralDict['commMtrx'].index
        xValues = snapshot[:, xPosition].copy()
        xSave = None
    else:
        xSave = centralDict["commMtrx"]['x'].copy()
        commMtrxOther = centralDict["commMtrx"][(otherColumns)].copy()

    def xSaved():
        """centralDict['commMtrx']['x'].copy() as it was before the transport."""
        return xSave if xSave is not None else pd.Series(xValues.copy(), index=snapshotIndex, name='x')

    def assemble(parts):
        '''commMtrx after the transport, on the snapshot: the transported columns of `parts` (frames of float64 columns
        on the nodes of the snapshot), x as before the transport, amounts again with the water mass. The same frame
        in one process (one part) and with worker processes (one part per process) as the pandas path below (concat,
        x, then water mass) would give, assembled at once.'''
        # pd.Index(initialColumns), the columns of the new commMtrx: the one built at the previous call when the
        # speciation gave it back (same labels, built the same way), so that the positions found on it are kept
        # from one step to the next (frames.indexer)
        last = _transportColumns[0]
        columnIndex = last if last is not None and initialIndex is last else pd.Index(initialColumns)
        _transportColumns[0] = columnIndex
        for part in parts:
            snapshot[:, frames.indexer(columnIndex, part.columns)] = part.to_numpy()
        snapshot[:, columnIndex.get_loc('x')] = xValues
        if waterMass is not None:
            iw = frames.indexer(columnIndex, waterCols)
            snapshot[:, iw] = snapshot[:, iw] * waterMass[:, None]
        return pd.DataFrame(snapshot, index=snapshotIndex, columns=columnIndex)
    if threads > 1:
        chunk_size = int(np.ceil(len(centralDict['transportedSpecies']) / threads))
        nChunks, remainder = divmod(len(centralDict['transportedSpecies']), chunk_size)
        lst_split = [
            centralDict['transportedSpecies'][i:i + chunk_size]
            for i in range(0, len(centralDict['transportedSpecies']), chunk_size)
        ]
        executor = warningManager.executor(threads)
        futures = []
        light = _workerDict(centralDict)
        for i,chunk in enumerate(lst_split):
            futures.append(warningManager.submit(executor, basicTransport, light, centralDict['commMtrx'], chunk))
        results = []
        results = [f.result() for f in futures]
        df1, Bool, logs, integrals, walls = zip(*results)
        calcWall, calcCpu = max(walls), sum(walls)
        abort = any(Bool)
        integrals = [x for x in integrals if x is not None]
        integral = pd.concat(integrals, axis=1) if integrals else None
        transportLog = [entry for log in logs for entry in log]
        if (snapshot is not None and not abort
                and all(_floatColumns(part) and part.index.equals(snapshotIndex) for part in df1)):
            commMtrxTrspt = assemble(df1)
        else:
            if snapshot is not None:
                commMtrxOther = pd.DataFrame(snapshot, index=snapshotIndex, columns=initialColumns)[otherColumns]
                snapshot = None
            commMtrxTrspt = pd.concat(df1,axis=1)
            commMtrxTrspt = pd.concat([commMtrxTrspt,commMtrxOther], axis=1)
            commMtrxTrspt = commMtrxTrspt[initialColumns]
    else:
        C_new, abort, transportLog, integral, calcWall = basicTransport(centralDict, centralDict['commMtrx'],
                                                                        centralDict['transportedSpecies'],
                                                                        valuesOnly=snapshot is not None)
        calcCpu = calcWall
        if centralDict['transportThreads'] > 1 and processes is None and not abort:
            _chooseProcesses(centralDict, calcWall)
        if mixedTotals is not None and (abort or centralDict.get('donnanBorrowed') is None):
            # no conversion to float64 at the end: the pandas path, on commMtrx as given
            snapshot = None
            xSave = entryFrame['x'].copy()
            commMtrxOther = entryFrame[otherColumns].copy()
            mixedTotals = None
        # _floatColumns of the frame: its values are float64, its columns those of the names
        if snapshot is not None and not abort and (C_new.columns.is_unique if isinstance(C_new, _TransportedValues)
                                                   else _floatColumns(C_new)):
            commMtrxTrspt = assemble([C_new])
        else:
            if isinstance(C_new, _TransportedValues):
                C_new = C_new.frame()
            if snapshot is not None:
                if mixedTotals is not None:
                    xSave = entryFrame['x'].copy()
                    commMtrxOther = entryFrame[otherColumns].copy()
                    mixedTotals = None
                else:
                    commMtrxOther = pd.DataFrame(snapshot, index=snapshotIndex, columns=initialColumns)[otherColumns]
                snapshot = None
            commMtrxTrspt = pd.concat([C_new,commMtrxOther], axis=1)
            commMtrxTrspt = commMtrxTrspt[initialColumns]


    writeWarningLog(centralDict, transportLog)
    if abort:
        raise RuntimeError(f"nativeTransport : fatal error at time step n°{centralDict['lStep'] + 1} (error report "
                           "above in warning.log)")
    if centralDict.get('areaSchedule') and centralDict.get('tTransportStart') is not None:
        # area of the last segment of the call : the next call starts from it (pure water added if it grows)
        centralDict['areaActive'] = areaAt(centralDict, float(centralDict['tTransportStart'])
                                           + float(centralDict['dtStep']), before=True)

    if snapshot is None:
        commMtrxTrspt['x'] = xSaved().copy()
    if savedPorosity is not None:
        centralDict.update(savedPorosity)
    if waterMass is not None:
        if snapshot is None:
            commMtrxTrspt = pd.concat([commMtrxTrspt[waterCols].mul(waterMass, axis=0),
                                       commMtrxTrspt.drop(columns=waterCols)], axis=1)[initialColumns]
        if integral is not None:
            cols = [s for s in waterCols if s in integral.columns]
            integral[cols] = integral[cols].mul(waterMass, axis=0)
    if integral is None:
        centralDict.pop('transportReactionIntegral', None)
    else:
        centralDict['transportReactionIntegral'] = integral
    
    if outputManager.wanted(centralDict, 'transport'):
        # extra columns gathered then joined at once: one insert per column fragments the frame
        # (pandas PerformanceWarning at every step)
        extra = {}
        if 'currentTotal' in centralDict:
            extra['psi'] = centralDict['electricPotential']
            extra['currentTotal_A'] = centralDict['currentTotal']
            if centralDict.get('voltageDropInfo'):
                extra['voltageDrop_V'] = centralDict['voltageDropInfo']['dV']
            extra.update(centralDict.get('cellVoltageInfo') or {})
            if centralDict.get('electroOsmosis'):
                extra['p_Pa'] = centralDict['electroOsmosis']['pressure_Pa']
                extra['waterFlow_m3s'] = centralDict['electroOsmosis']['waterFlow_m3s']
        extra.update(centralDict.get('donnanOutput') or {})
        if centralDict.get('waterExpelled') is not None:
            extra['waterExpelled_m3s'] = centralDict['waterExpelled']
        if centralDict.get('areaSchedule'):
            extra['area_m2'] = np.asarray(centralDict['areaActive'], dtype=float)
        if centralDict.get('donnanBorrowed'):
            bw = centralDict['donnanBorrowed']
            for i, sp in enumerate(bw['species']):
                if np.any(bw['T'][i] > 0):
                    extra[f"{sp}_borrowed"] = bw['T'][i]
        if extra:
            extraDf = pd.DataFrame(extra, index=commMtrxTrspt.index)
            if _floatColumns(commMtrxTrspt) and _floatColumns(extraDf):
                # the frame of the concat below, assembled at once (float64 columns only)
                kept = ~commMtrxTrspt.columns.isin(extraDf.columns)
                outDf = pd.DataFrame(np.hstack([commMtrxTrspt.to_numpy()[:, kept], extraDf.to_numpy()]),
                                     index=commMtrxTrspt.index,
                                     columns=commMtrxTrspt.columns[kept].tolist() + extraDf.columns.tolist())
            else:
                outDf = pd.concat([commMtrxTrspt.drop(columns=commMtrxTrspt.columns.intersection(extraDf.columns)),
                                   extraDf], axis=1)
        else:
            outDf = commMtrxTrspt           # outputManager.write copies it
        if mixedTotals:
            # the non-float columns as in the frame of the pandas path (their values, never transported, unchanged)
            restored = [c for c in mixedTotals if c in outDf.columns and not (extra and c in extra)]
            if restored:
                outDf = outDf.copy()
                for c in restored:
                    outDf[c] = entryFrame[c].to_numpy().copy()
        outputManager.write(centralDict, outDf, 'transport', 'nativeTransport')
    centralDict.pop('_donnanBorrow', None)
    borrowed = centralDict.pop('donnanBorrowed', None)
    if borrowed is not None:
        # donnan coupling 'totals' : the chemistry gets every mobile ion (free water + DL) minus the counter-ions
        # borrowed from the exchangers, as a concentration in the whole pore water (theta) ; the transport output
        # above keeps the free-water concentrations and the *_DL columns
        state = centralDict.pop('donnanState')
        if state['species'] != borrowed['species'] or state['Ctot'].shape != borrowed['T'].shape:
            raise ValueError("nativeTransport : donnan coupling = 'totals', inconsistent double-layer state")
        cols = commMtrxTrspt.columns.get_indexer(borrowed['species'])
        values = commMtrxTrspt.to_numpy(dtype=float, copy=True)
        cPrime = (state['Ctot'] - borrowed['T']) / borrowed['theta'][None, :]
        values[:, cols] = cPrime.T
        commMtrxTrspt = pd.DataFrame(values, index=commMtrxTrspt.index, columns=commMtrxTrspt.columns)
        centralDict['donnanPrevious'] = {'species': borrowed['species'], 'c': cPrime, 'T': borrowed['T']}


    elapsed = time.perf_counter() - startTransport
    centralDict.update({
        "commMtrx": commMtrxTrspt,
        "nativeTransportCalcTime_WallClock": centralDict["nativeTransportCalcTime_WallClock"] + calcWall,
        "nativeTransportCalcTime_ProcessorTime": centralDict["nativeTransportCalcTime_ProcessorTime"] + calcCpu,
        "nativeTransportInterfTime_WallClock": centralDict["nativeTransportInterfTime_WallClock"] + elapsed - calcWall,
        "nativeTransportTotalTime": centralDict["nativeTransportTotalTime"] + elapsed,
        })
    display.stepPrint(f"({writeTime(elapsed)})")
    return centralDict