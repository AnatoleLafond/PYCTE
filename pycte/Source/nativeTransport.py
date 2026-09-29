import numpy as np
import pandas as pd
import time
import importlib.util
import os
import concurrent.futures
import sys
import time
import traceback
try:
    from . import outputManager
    from . import warningManager
except ImportError:
    import outputManager
    import warningManager

FARADAY = 96485.33212
GAS_CONSTANT = 8.314462618

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
def parseCondition(bc):
    parts = [p.strip() for p in bc.split('+', 1)] if isinstance(bc, str) else [bc]
    electrode = parts[0] == 'electrode'
    cond = (parts[1] if len(parts) > 1 else 'closed') if electrode else bc
    if cond not in BC_CONDITIONS:
        raise ValueError(f"unknown boundary condition {bc!r} : use 'constant' (Dirichlet), 'closed' (Neumann), "
                         "'flux' (Cauchy), or 'electrode+constant' / 'electrode+closed' / 'electrode+flux' "
                         "('electrode' = 'electrode+closed')")
    return electrode, cond
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
def conditionAt(entries, t):
    tol = 1e-9 * max(1.0, abs(t))
    return [cond for ts, cond in entries if ts <= t + tol][-1]
def transportSegments(centralDict):
    schedule = centralDict.get('boundarySchedule')
    dtU, t0 = centralDict['dtStep'], centralDict.get('tTransportStart')
    if not schedule or t0 is None:
        return [(centralDict['boundaryConditions'], dtU, t0)]
    t1 = t0 + dtU
    tol = 1e-9 * max(1.0, abs(t1))
    cuts = sorted({ts for entries in schedule for ts, _ in entries if t0 + tol < ts < t1 - tol})
    bounds = [t0] + cuts + [t1]
    return [([conditionAt(entries, a) for entries in schedule], b - a, a) for a, b in zip(bounds[:-1], bounds[1:])]
DONNAN_PSI_MAX = 60.0
def _expClip(x):
    return np.exp(np.clip(x, -700.0, 700.0))
def donnanPotential(c, z, sigma, phi=None, psi0=None, tol=1e-12, maxIter=200):
    c = np.maximum(np.asarray(c, dtype=float), 0.0)
    z = np.asarray(z, dtype=float)[:, None]
    phi = np.ones_like(c) if phi is None else np.asarray(phi, dtype=float)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), c.shape[1:]).astype(float)
    zc = z * phi * c
    def g(psi):
        e = _expClip(-z * psi[None, :])
        return (zc * e).sum(axis=0) + sigma, -(z * zc * e).sum(axis=0), (np.abs(zc) * e).sum(axis=0)
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
    thF = np.broadcast_to(np.asarray(thF, dtype=float), Ctot.shape[1:])
    thDL = np.broadcast_to(np.asarray(thDL, dtype=float), Ctot.shape[1:])
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), Ctot.shape[1:])
    fixed = np.zeros(Ctot.shape, dtype=bool) if fixed is None else np.asarray(fixed, dtype=bool)
    fixVal = np.zeros_like(Ctot) if fixVal is None else np.asarray(fixVal, dtype=float)
    active = thDL > 0
    def state(psi):
        e = _expClip(-z * psi[None, :])
        den = thF[None, :] + thDL[None, :] * phi * e
        c = np.where(fixed, fixVal, np.maximum(Ctot, 0.0) / den)
        w = np.where(fixed, 1.0, thF[None, :] / den)
        zpe = z * phi * e * c
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
    thF = np.broadcast_to(np.asarray(thF, dtype=float), c.shape[1:])
    thDL = np.broadcast_to(np.asarray(thDL, dtype=float), c.shape[1:])
    phi = np.ones_like(c) if phi is None else np.asarray(phi, dtype=float)
    if psi is None:
        psi = donnanPotential(c, z, sigma, phi)[0]
    e = _expClip(-zz * psi[None, :])
    thEff = thF[None, :] + thDL[None, :] * phi * e
    v = zz * phi * e
    u = v * np.maximum(c, 0.0)
    G = (zz * u).sum(axis=0)
    return thEff * c, thEff, u, v, G, psi

def toSeconds(value, unit):
    factor = {'y': 3600 * 24 * 365.25, 'd': 3600 * 24, 'h': 3600, 'm': 60, 'min': 60}.get(unit, 1.0)
    return value * factor

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
    out[small] = 1.0 - x[small] / 2.0 + x[small] ** 2 / 12.0
    xs = x[~small]
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        out[~small] = xs / np.expm1(xs)
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
    def logWarning(where, message):
        for entry in transportLog:
            if entry['level'] == 'WARNING' and entry['where'] == where and entry['message'] == message:
                entry['count'] += 1
                return
        print(f"\nWARNING ({where}) : {message}")
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
    BC_KIND = {'constant': 'dirichlet', 'closed': 'neumann', 'flux': 'cauchy'}
    def sideSetup(side):
        bcList = centralDict['boundaryConditions']
        if isinstance(bcList, str) or len(bcList) != 2:
            logError('boundaries', f"boundaryConditions must be a list of two conditions [x = 0, x = L] (got {bcList!r})")
        try:
            return parseCondition(bcList[side])
        except ValueError as err:
            logError('boundaries', f"boundaryConditions[{side}] : {err}")
    def isElectrode(side):
        return sideSetup(side)[0]
    def boundarySpec(sp, side):
        kind = BC_KIND[sideSetup(side)[1]]
        values = centralDict['firstBoundary' if side == 0 else 'secondBoundary'] or {}
        if kind == 'neumann' or sp not in values:
            return 'neumann', None
        return kind, float(values[sp])
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
    def boundaries(df, col, sp):
        for side, node in ((0, 0), (1, -1)):
            kind, val = boundarySpec(sp, side)
            if kind == 'dirichlet':
                df.iloc[node, col] = val
        return df
    def adeFlowRate():
        v = centralDict.get('velocity') or 0.0
        if not v:
            return 0.0
        if isRadial:
            return float(geom['flowRate'])
        return float(v) * float(meshProp(centralDict['area'], 0, 1)[0])
    def activitySetup(species):
        ns = speciationModule()
        charges = centralDict.get('especeCharge') or {}
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
    def adeOperator():
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
    def adeApplyL(C, Lb):
        LC = Lb[2] * C
        LC[:-1] += Lb[3, :-1] * C[1:]
        LC[1:] += Lb[1, 1:] * C[:-1]
        LC[:-2] += Lb[4, :-2] * C[2:]
        LC[2:] += Lb[0, 2:] * C[:-2]
        return LC
    def vanLeerCorrection(C, Q):
        if Q < 0:
            return vanLeerDownstream(C[::-1], -Q, meshProp(centralDict['dxHalfCell'], 0, n)[::-1])[::-1]
        return vanLeerDownstream(C, Q, meshProp(centralDict['dxHalfCell'], 0, n))
    def vanLeerDownstream(C, Qa, dxh):
        out = np.zeros_like(C)
        if n < 3 or Qa == 0:
            return out
        k = np.arange(1, n - 1)
        dUp = C[k] - C[k - 1]
        dDn = C[k + 1] - C[k]
        gUp = dUp / (dxh[k - 1] + dxh[k])[:, None]
        gDn = dDn / (dxh[k] + dxh[k + 1])[:, None]
        prod = gUp * gDn
        with np.errstate(divide='ignore', invalid='ignore'):
            s = np.where(prod > 0, 2.0 * prod / (gUp + gDn), 0.0)
        delta = dxh[k][:, None] * s
        delta = np.sign(delta) * np.minimum(np.abs(delta), np.minimum(np.abs(dUp), np.abs(dDn)))
        out[k] += Qa * delta
        out[k + 1] -= Qa * delta
        return out
    def adeExplicit(C, storage, Lb, fixed, fixVal, bDiag, bSrc, Q):
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
        def rhs(C):
            LC = adeApplyL(C, Lb)
            if tvd:
                LC += vanLeerCorrection(C, Q)
            LC[0] += bDiag[0] * C[0] - bSrc[0]
            LC[-1] += bDiag[1] * C[-1] - bSrc[1]
            return -LC / storage[:, None]
        C = C.copy()
        C[fixed] = fixVal[fixed]
        for _ in range(nSub):
            C1 = C + h * rhs(C)
            C1[fixed] = fixVal[fixed]
            if heun:
                C1 = 0.5 * (C + C1 + h * rhs(C1))
                C1[fixed] = fixVal[fixed]
            C = C1
        return C, nSub
    def adeImplicit(C, storage, Lb, fixed, fixVal, bDiag, bSrc):
        try:
            from scipy.linalg import solve_banded
        except ImportError:
            logError('adeImplicit', "adeSolver = 'implicit' needs scipy : pip install scipy")
        nSteps = centralDict.get('adeSubSteps') or 1
        if int(nSteps) != nSteps or nSteps < 1:
            logError('adeImplicit', f"adeSubSteps must be an integer >= 1 (got {nSteps})")
        nSteps = int(nSteps)
        h = dt / nSteps
        Mh = storage / h
        perSpecies = Lb.shape[2] > 1
        groups = {}
        for j in range(C.shape[1]):
            key = (bool(fixed[0, j]), bool(fixed[-1, j]), float(bDiag[0, j]), float(bDiag[1, j]),
                   j if perSpecies else 0)
            groups.setdefault(key, []).append(j)
        C = C.copy()
        C[fixed] = fixVal[fixed]
        for (fixL, fixR, b0, bL, jOp), cols in groups.items():
            L = Lb[:, :, jOp]
            a = np.zeros((3, n))
            a[0, 1:] = L[3, :-1]
            a[1, :] = Mh + L[2]
            a[2, :-1] = L[1, 1:]
            a[1, 0] += b0
            a[1, -1] += bL
            if fixL:
                a[1, 0], a[0, 1] = 1.0, 0.0
            if fixR:
                a[1, -1], a[2, -2] = 1.0, 0.0
            Cg = C[:, cols]
            for _ in range(nSteps):
                rhs = Mh[:, None] * Cg
                rhs[0] += bSrc[0, cols]
                rhs[-1] += bSrc[1, cols]
                if fixL:
                    rhs[0] = fixVal[0, cols]
                if fixR:
                    rhs[-1] = fixVal[-1, cols]
                Cg = solve_banded((1, 1), a, rhs, overwrite_b=True, check_finite=False)
            C[:, cols] = Cg
        return C, nSteps
    DONNAN_KEYS = {'porosityDL', 'thickness', 'specificSurface', 'CEC', 'bulkDensity', 'chargeDL', 'stern',
                   'tortuosityDL', 'activityDL'}
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
        charges = centralDict.get('especeCharge') or {}
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
        return {'spc': list(spc), 'z': z, 'Df': Df, 'DD': DD, 'params': params, 'theta': theta}
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
    def donnanStart(dn, c, fixed=None, fixVal=None):
        par = dn['params'](c)
        thF, thDL, sigma, phi = par
        state = centralDict.get('donnanState')
        if state and state['species'] == dn['spc'] and state['Ctot'].shape == c.shape:
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
    def adeExplicitDonnan(C, fixed, fixVal, bDiag, bSrc, Q, dG):
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
        def operator(cf, psi, par):
            T = donnanConductance(dn, psi, par, disp).T
            Tmol = donnanConductance(dn, psi, par, 0.0).T if disp else T
            return adeMatrix(T, Tmol, Q, dG)
        def rhs(cfT, Lb):
            LC = adeApplyL(cfT, Lb)
            if tvd:
                LC += vanLeerCorrection(cfT, Q)
            LC[0] += bDiag[0] * cfT[0] - bSrc[0]
            LC[-1] += bDiag[1] * cfT[-1] - bSrc[1]
            return -LC / V[:, None]
        def speciate(CtotT, par, psi0):
            thF, thDL, sigma, phi = par
            c, p, _ = donnanFromTotals(CtotT.T, dn['z'], thF, thDL, sigma, phi, fixed.T, fixVal.T, psi0)
            Cc = donnanStorage(c, dn['z'], thF, thDL, sigma, phi, p)[0]
            return c, Cc, p
        M = len(spc)
        eyeM = np.eye(M)
        def localRate(cf, psi, par, diagB):
            thF, thDL, sig, phi = par
            _, thEff, u, v, G, _ = donnanStorage(cf, dn['z'], thF, thDL, sig, phi, psi)
            with np.errstate(divide='ignore', invalid='ignore'):
                w = np.where(G > 0, thDL / G, 0.0)
            J = thEff.T[:, :, None] * eyeM[None] - w[:, None, None] * u.T[:, :, None] * v.T[:, None, :]
            fx = fixed
            J = np.where(fx[:, :, None] | fx[:, None, :], eyeM[None], J)
            out = np.where(fx, 0.0, diagB) / V[:, None]
            A = np.linalg.solve(J, eyeM[None] * out[:, None, :])
            return np.max(np.abs(np.linalg.eigvals(A)), axis=1)
        t, nSub = 0.0, 0
        while dt - t > 1e-12 * dt:
            par = dn['params'](cf)
            Lb = operator(cf, psi, par)
            diagB = np.broadcast_to(Lb[2], C.shape).copy()
            diagB[0] += bDiag[0]
            diagB[-1] += bDiag[1]
            if tvd:
                diagB += abs(Q)
            rate = localRate(cf, psi, par, diagB)[free]
            rateMax = float(np.max(rate)) if rate.size else 0.0
            h = (dt - t) if rateMax <= 0 else (dt - t) / max(1, int(np.ceil((dt - t) * rateMax / (safety * kappa))))
            k1 = rhs(cf.T, Lb)
            c1, C1, p1 = speciate(Ctot.T + h * k1, par, psi)
            if heun:
                k2 = rhs(c1.T, operator(c1, p1, par))
                cf, Ctot, psi = speciate(0.5 * (Ctot.T + C1.T + h * k2), par, p1)
            else:
                cf, Ctot, psi = c1, C1, p1
            t += h
            nSub += 1
            if nSub > nMax:
                k = int(np.flatnonzero(free)[np.argmax(rate)])
                logError('adeExplicitDonnan', f"more than {nMax} sub-steps needed (dt_sub = {writeTime(h, 3)}, limiting "
                         f"node {k}) : increase maxSubCycling")
        donnanEnd(dn, cf, Ctot, psi, par, float(centralDict.get('temperature', 298.15)))
        return cf.T, nSub
    def adeImplicitDonnan(C, fixed, fixVal, bDiag, bSrc, Q, dG):
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
        idx = np.arange(n * M).reshape(n, M)
        ii, jj = np.meshgrid(np.arange(M), np.arange(M), indexing='ij')
        rows = np.concatenate([idx[:, ii.ravel()].ravel(), idx[1:].ravel(), idx[:-1].ravel()])
        cols = np.concatenate([idx[:, jj.ravel()].ravel(), idx[:-1].ravel(), idx[1:].ravel()])
        fixedFlat = fixed.ravel()
        keep = ~fixedFlat[rows]
        rowsK, colsK = rows[keep], cols[keep]
        fixedIdx = np.flatnonzero(fixedFlat)
        bandPos = (M + rowsK - colsK, colsK)
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
            blk[:, np.arange(M), np.arange(M)] += dia
            vals = np.concatenate([blk.reshape(n, M * M).ravel(),
                                   np.broadcast_to(Lb[1][1:], (n - 1, M)).ravel(),
                                   np.broadcast_to(Lb[3][:-1], (n - 1, M)).ravel()])
            ab = np.zeros((2 * M + 1, n * M))
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
        stats = {'steps': 0, 'newton': 0, 'split': 0, 'maxIter': 0}
        def newton(c0, CtotOld, h, par, psi0):
            c = c0.copy()
            c[fixed.T] = fixVal.T[fixed.T]
            psiD = psi0
            floor = 1e-14 * max(float(np.max(np.abs(c))), 1e-300)
            for it in range(1, maxIter + 1):
                R, psiD, ab = system(c, CtotOld, h, par, psiD, True)
                if centralDict.get('_donnanJacobianCheck') is True:
                    centralDict['_donnanJacobianCheck'] = jacobianCheck(c, CtotOld, h, par, psiD, ab)
                try:
                    dx = solve_banded((M, M), ab, -R.ravel(), check_finite=False)
                except (np.linalg.LinAlgError, ValueError):
                    return None, it
                if not np.all(np.isfinite(dx)):
                    return None, it
                cNew = np.maximum(c + dx.reshape(n, M).T, 0.0)
                change = float(np.max(np.abs(cNew - c) / (np.abs(cNew) + floor)))
                c = cNew
                if change < 1e-10:
                    psiD = donnanPotential(c, z, par[2], par[3], psi0=psiD)[0]
                    return (c, psiD), it
            return None, maxIter
        t, h = 0.0, h0
        while dt - t > 1e-12 * dt:
            hTry = min(h, dt - t)
            par = dn['params'](cf)
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
            Ctot = donnanStorage(cf, z, *par[:3], par[3], psi)[0]
            t += hTry
            stats['steps'] += 1
            stats['maxIter'] = max(stats['maxIter'], it)
            h = min(h0, 2.0 * hTry)
        centralDict['adeDonnanStats'] = stats
        donnanEnd(dn, cf, Ctot, psi, par, float(centralDict.get('temperature', 298.15)))
        return cf.T, stats
    def adeTransport(C_old):
        C = C_old[trsptedSpecies].to_numpy(dtype=float)
        fixed = np.zeros(C.shape, dtype=bool)
        fixVal = np.zeros(C.shape)
        for j, sp in enumerate(trsptedSpecies):
            for side, node in ((0, 0), (1, -1)):
                kind, val = boundarySpec(sp, side)
                if kind == 'dirichlet':
                    fixed[node, j], fixVal[node, j] = True, val
        if n < 2:
            C[fixed] = fixVal[fixed]
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
        Lb = adeMatrix(T, Tmol, Q, dG)
        K, cExt, mask = cauchyArrays(list(trsptedSpecies), noInflow=(Q == 0))
        Qin = np.array([max(Q, 0.0), max(-Q, 0.0)])
        bDiag = np.where(mask, K + Qin[:, None], 0.0)
        bSrc = bDiag * cExt
        if useDonnan and adeSolver == 'implicit':
            C, st = adeImplicitDonnan(C, fixed, fixVal, bDiag, bSrc, Q, dG)
            print(f"[implicit ADE ({advScheme}, Donnan) : {st['steps']} step(s), {st['newton']} Newton iterations"
                  + (f", {st['split']} split" if st['split'] else "") + "]", end=" ", flush=True)
        elif useDonnan:
            C, nSub = adeExplicitDonnan(C, fixed, fixVal, bDiag, bSrc, Q, dG)
            print(f"[{nSub} sub-steps ({advScheme}, Donnan)]", end=" ", flush=True)
        elif adeSolver == 'implicit':
            C, nSub = adeImplicit(C, storage, Lb, fixed, fixVal, bDiag, bSrc)
            print(f"[implicit ADE ({advScheme}) : {nSub} step(s)]", end=" ", flush=True)
        else:
            C, nSub = adeExplicit(C, storage, Lb, fixed, fixVal, bDiag, bSrc, Q)
            print(f"[{nSub} sub-steps ({advScheme})]", end=" ", flush=True)
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

    def nernstPlanckStep(c, h, T, Zc, charged, condBar, storage, dG=None):
        cbar = logMeanConc(c[:, :-1], c[:, 1:])
        flux_fick = T * (c[:, 1:] - c[:, :-1])
        if dG is not None:
            flux_fick = flux_fick + T * cbar * dG
        numerator = (Zc * flux_fick)[charged].sum(axis=0)
        denominator = ((Zc ** 2) * condBar * cbar)[charged].sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(denominator > 0, numerator / np.where(denominator > 0, denominator, 1.0), 0.0)
        total_flux_interface = flux_fick - Zc * condBar * cbar * ratio
        diff = np.zeros_like(c)
        diff[:, :-1] += total_flux_interface
        diff[:, 1:] -= total_flux_interface
        return c + (h * diff / storage) / 1000
    def nernstPlanckTransport(C_old, C_new):
        especeCharge = centralDict['especeCharge']
        especeDiffCoeff = resolve_species_property('especeDiffCoeff', 'diffCoeff')
        especePorosity = resolve_species_property('especePorosity', 'porosity')
        lo, hi = 0, n
        dxHalfCell_trim = meshProp(centralDict['dxHalfCell'], lo, hi)
        dx_total = dxHalfCell_trim[:-1] + dxHalfCell_trim[1:]
        T_by_sp = {sp: precompute_species_transmissivity(lo, hi, especeDiffCoeff[sp], especePorosity[sp]) for sp in trsptedSpecies}
        z_by_sp = {sp: especeCharge[sp] for sp in trsptedSpecies}
        storage_by_sp = {}
        volume_trim = cellVolume(lo, hi)
        for sp in trsptedSpecies:
            poro_sp = meshProp(especePorosity[sp], lo, hi)
            storage = volume_trim * poro_sp
            storage_by_sp[sp] = np.where(storage > 0, storage, 1.0)
        C_state = {sp: C_old[sp].to_numpy() for sp in trsptedSpecies}
        nSub = centralDict['subCyclingDiff']
        for sp in trsptedSpecies:
            T_sp = np.asarray(T_by_sp[sp])
            nSub = radialSubSteps(nSub, np.append(T_sp, 0.0), np.insert(T_sp, 0, 0.0), storage_by_sp[sp])
        dirichlet = dirichletNodes()
        spc = list(trsptedSpecies)
        T = np.array([np.asarray(T_by_sp[sp], dtype=float) for sp in spc])
        Z = np.array([float(z_by_sp[sp]) for sp in spc])
        storage = np.array([storage_by_sp[sp] for sp in spc])
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
        h = dt / nSub
        for _ in range(nSub):
            cPrev = c
            dG = dLnGamma(act, c) if act else None
            c = nernstPlanckStep(c, h, T, Z[:, None], Z != 0, T * dx_total, storage, dG)
            if hasCauchy:
                c[:, 0] += h * K[0] * (cExt[0] - cPrev[:, 0]) / storage[:, 0]
                c[:, -1] += h * K[1] * (cExt[1] - cPrev[:, -1]) / storage[:, -1]
            for i, node, val in fixes:
                c[i, node] = val
        C_state = {sp: c[i] for i, sp in enumerate(spc)}
        for sp in trsptedSpecies:
            C_new[sp] = C_state[sp]
        return C_new

    def solve_current_from_voltage(numerator0, denominator_safe, dx_total, targetVoltage, temperature):
        RT_over_F = GAS_CONSTANT * temperature / FARADAY
        sum_a = np.sum((numerator0 / denominator_safe) * dx_total)
        sum_b = np.sum(dx_total / denominator_safe)
        I_mol = (-(targetVoltage / RT_over_F) - sum_a) / sum_b
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
            ratio = np.where(denominator > 0, (numerator0 + I_mol) / denominator_safe, 0.0)
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
        for side, node, qIn in ((0, 0, I_mol), (1, -1, -I_mol)):
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
        for side, node, qIn in ((0, 0, I_mol), (1, -1, -I_mol)):
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
    def sgSolveFieldArr(c, T, Z, dx_total, closureMode, closureValue, temperature, phi0, I0, dG=None, adv=None):
        RT_over_F = GAS_CONSTANT * temperature / FARADAY
        phi = np.array(phi0, dtype=float)
        if closureMode == 'voltage':
            I_mol = float(I0)
            dPhiEl = centralDict.get('electrodeDeltaPhi')
            if dPhiEl is None:
                dPhiEl = np.sign(closureValue) * (centralDict.get('electrodeVoltageDrop') or 0.0)
            Vn = (closureValue - dPhiEl) / RT_over_F
        else:
            I_mol = resolve_imposed_current(None, None, dx_total, closureMode, closureValue, temperature)
        charged = Z != 0
        Zc = Z[charged][:, None]
        Tc = T[charged]
        cL, cR = c[charged][:, :-1], c[charged][:, 1:]
        cLp, cRp = np.maximum(cL, 0.0), np.maximum(cR, 0.0)
        dGc = 0.0 if dG is None else dG[charged]
        Q = float(adv['Q']) if adv is not None else 0.0
        for it in range(100):
            if adv is None:
                Te, u = Tc, Zc * phi[None, :] + dGc
            else:
                Te = Tc + adv['dispG'][None, :] * abs(Q)
                Te = np.where(Te > 0, Te, 1e-300)
                u = Zc * phi[None, :] - 1000.0 * Q / Te + dGc * Tc / Te
            Bp, Bm, dBp, dBm = bernoulliPair(u, deriv=True)
            gS = Bm * cR - Bp * cL
            g = (Zc * Te * gS).sum(axis=0)
            gp = -(dBm * cRp + dBp * cLp)
            dg = ((Zc ** 2) * Te * gp).sum(axis=0)
            dg = np.maximum(dg, 1e-300)
            R = g + I_mol
            RV = (np.sum(phi) - Vn) if closureMode == 'voltage' else 0.0
            dQ = 0.0
            if adv is not None:
                dTdQ = adv['dispG'] * np.sign(Q)
                dudQ = (-1000.0 + 1000.0 * Q * dTdQ / Te - dGc * Tc * dTdQ / Te) / Te
                bQ = (Zc * (dTdQ * gS + Te * gp * dudQ)).sum(axis=0)
                RQ = Q + float(np.dot(adv['betaPsi'], phi)) + adv['dpScaled']
                if closureMode == 'voltage':
                    U, V = np.column_stack((np.ones_like(dg), bQ)), np.vstack((np.ones_like(dg), adv['betaPsi']))
                    D, rb = np.array([[0.0, 0.0], [0.0, 1.0]]), np.array([RV, RQ])
                else:
                    U, V, D, rb = bQ[:, None], adv['betaPsi'][None, :], np.array([[1.0]]), np.array([RQ])
                XU = U / dg[:, None]
                try:
                    db = np.linalg.solve(D - V @ XU, -rb + V @ (R / dg))
                except np.linalg.LinAlgError:
                    db = np.full(len(rb), np.nan)
                if not np.all(np.isfinite(db)):
                    logError('sgSolveField', "Nernst-Planck with electro-osmosis : singular field / water flow system "
                             f"(closure '{closureMode}', Q = {Q:.4g} m3/s) : check permeability, "
                             "electroOsmoticPermeability and boundaryPressure")
                dphi = -R / dg - XU @ db
                dI = db[0] if closureMode == 'voltage' else 0.0
                dQ = db[-1]
            else:
                dI = (RV - np.sum(R / dg)) / np.sum(1.0 / dg) if closureMode == 'voltage' else 0.0
                dphi = -(R + dI) / dg
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
                     + (f", electro-osmotic water flow Q = {Q:.4g} m3/s" if adv is not None else "") + ") : "
                     "local field too strong for the damped Newton (depleted zone, closed or inconsistent boundary)")
        if adv is not None:
            return phi, I_mol, Q
        return phi, I_mol
    def nernstPlanckCurrentAdaptive(C, dt_total, T_by_sp, z_by_sp, dx_total, storage_by_sp,
                                    closureMode, closureValue, temperature, eo=None):
        safety = centralDict.get('cflSafety', 0.9)
        nMin = max(int(centralDict.get('subCyclingDiff', 1)), 1)
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
        I_mol = resolve_imposed_current(numerator0, denominator_safe, dx_total, closureMode, closureValue, temperature)
        with np.errstate(divide='ignore', invalid='ignore'):
            phi = -np.where(denominator > 0, (numerator0 + I_mol) / denominator_safe, 0.0) * dx_total
        spc = list(trsptedSpecies)
        T = np.array([np.asarray(T_by_sp[sp], dtype=float) for sp in spc])
        Z = np.array([float(z_by_sp[sp]) for sp in spc])
        Zc = Z[:, None]
        storage = np.array([np.asarray(storage_by_sp[sp], dtype=float) for sp in spc])
        c = np.array([np.asarray(C[sp], dtype=float) for sp in spc])
        fixes = [(i, node, val) for i, sp in enumerate(spc) if sp in dirichlet
                 for node, val in ((0, dirichlet[sp][0]), (-1, dirichlet[sp][1])) if val is not None]
        eoOpen = eo is not None and eo['mode'] == 'open'
        K, cExt, cauMask = cauchyArrays(spc, noInflow=not eoOpen)
        KB = 1000.0 * K
        Q, eoVolume, adv = 0.0, 0.0, None
        if eoOpen:
            RT_over_F = GAS_CONSTANT * temperature / FARADAY
            adv = {'betaPsi': eo['beta'] * RT_over_F / eo['RhTot'],
                   'dpScaled': (eo['pL'] - eo['p0']) / eo['RhTot'],
                   'dispG': 1000.0 * eo['alphaL'] / eo['Ld']}
            Q = -(float(np.dot(adv['betaPsi'], phi)) + adv['dpScaled'])
        act = activitySetup(spc) if useActivity else None
        dG = None
        t, nSub = 0.0, 0
        while dt_total - t > 1e-12 * dt_total:
            if act:
                dG = dLnGamma(act, c)
            if eoOpen:
                phi, I_mol, Q = sgSolveFieldArr(c, T, Z, dx_total, closureMode, closureValue, temperature, phi, I_mol,
                                                dG, dict(adv, Q=Q))
                Teff = T + adv['dispG'][None, :] * abs(Q)
                Teff = np.where(Teff > 0, Teff, 1e-300)
                u = Zc * phi[None, :] - 1000.0 * Q / Teff
                if dG is not None:
                    u = u + dG * T / Teff
                Bp, Bm = bernoulliPair(u)
            else:
                phi, I_mol = sgSolveFieldArr(c, T, Z, dx_total, closureMode, closureValue, temperature, phi, I_mol, dG)
                Teff = T
                Bp, Bm = bernoulliPair(Zc * phi[None, :] + (0.0 if dG is None else dG))
            J = Teff * (Bm * c[:, 1:] - Bp * c[:, :-1])
            out = np.zeros_like(c)
            out[:, :-1] += Teff * Bp
            out[:, 1:] += Teff * Bm
            out[:, 0] += KB[0]
            out[:, -1] += KB[1]
            if eoOpen:
                out[:, 0] += 1000.0 * max(-Q, 0.0)
                out[:, -1] += 1000.0 * max(Q, 0.0)
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
                cBL = np.where(cauMask[1] & (Q < 0), cExt[1], c[:, -1])
                diff[:, 0] += 1000.0 * Q * cB0
                diff[:, -1] -= 1000.0 * Q * cBL
                eoVolume += Q * h
            c = c + (h * diff / storage) / 1000
            if electrodeCfg or waterEq:
                C_new_sp = {sp: c[i] for i, sp in enumerate(spc)}
                if electrodeCfg:
                    C_new_sp = applyElectrode(C_new_sp, I_mol, h, storage_by_sp, electrodeCfg)
                if waterEq:
                    C_new_sp = applyWaterEquilibrium(C_new_sp)
                c = np.array([C_new_sp[sp] for sp in spc])
            for i, node, val in fixes:
                c[i, node] = val
            t += h
            nSub += 1
            if nSub >= nMax:
                iNode = int(np.argmax(rate))
                logError('nernstPlanckCurrentAdaptive', f"more than {nMax} sub-steps needed (dt_sub = {writeTime(h, 3)}, "
                         f"limiting node {iNode}, local |F dpsi/RT| = {abs(phi[min(iNode, len(phi)-1)]):.3g}). "
                         "The local electric field is too strong for the explicit scheme "
                         "(depleted zone / closed boundary ?). Increase maxSubCycling or check boundary conditions.")
        dG = dLnGamma(act, c) if act else None
        if eoOpen:
            phi, I_mol, Q = sgSolveFieldArr(c, T, Z, dx_total, closureMode, closureValue, temperature, phi, I_mol,
                                            dG, dict(adv, Q=Q))
        else:
            phi, I_mol = sgSolveFieldArr(c, T, Z, dx_total, closureMode, closureValue, temperature, phi, I_mol, dG)
        if eo is not None:
            electroOsmosisDiagnostics(eo, Q, phi, eoVolume, temperature)
        C = {sp: c[i] for i, sp in enumerate(spc)}
        return C, nSub, phi, I_mol
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
        RT_over_F = GAS_CONSTANT * temperature / FARADAY
        dp = -eo['Rh'] * Q - eo['beta'] * RT_over_F * np.asarray(phi, dtype=float)
        p = eo['p0'] + np.concatenate(([0.0], np.cumsum(dp)))
        aRef = areaRef()
        prevVolume = (centralDict.get('electroOsmosis') or {}).get('volume_m3', 0.0)
        centralDict['electroOsmosis'] = {'mode': eo['mode'], 'waterFlow_m3s': float(Q),
                                         'darcyVelocity_ms': float(Q) / aRef if aRef > 0 else np.nan,
                                         'pressure_Pa': p, 'volume_m3': prevVolume + float(volume)}
    def nernstPlanckImplicit(C, dt_total, T_by_sp, z_by_sp, dx_total, storage_by_sp,
                             closureMode, closureValue, temperature, eo=None, dn=None):
        try:
            import copy
            from scipy.linalg.lapack import dgbtrf, dgbtrs
        except ImportError:
            logError('nernstPlanckImplicit', "npSolver = 'implicit' needs scipy (LAPACK band solver) : pip install scipy")
        maxChange = float(centralDict.get('npImplicitMaxChange') or 0.1)
        atol = float(centralDict.get('npImplicitAtol') or 1e-6)
        maxIter = int(centralDict.get('npImplicitMaxIter') or 30)
        nMax = int(centralDict.get('maxSubCycling', 100000))
        RT_over_F = GAS_CONSTANT * temperature / FARADAY
        ramp = centralDict.get('rampPotential')
        print(f"[implicit debug] rampPotential lu = {ramp!r}  (None -> pas de rampe active, "
              f"verifier que engine.py lit bien 'rampPotential')  |  tGlobalStart = {tGlobalStart:.6g} s")
        def voltageNow(tAbs):
            if not ramp or closureMode != 'voltage':
                return closureValue
            vStart, vEnd, tRamp = ramp
            if tRamp <= 0:
                return vEnd
            frac = min(max(tAbs / tRamp, 0.0), 1.0)
            return vStart + frac * (vEnd - vStart)
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
            dpScaled = (eo['pL'] - eo['p0']) / eo['RhTot']
            dispG = 1000.0 * eo['alphaL'] / eo['Ld']
            vQ = np.zeros(N * (M + 1))
            vQ[np.arange(N) * (M + 1) + M] = np.concatenate(([0.0], betaPsi)) - np.concatenate((betaPsi, [0.0]))
        cInit = np.array([np.abs(np.asarray(C[sp], dtype=float)) for sp in spc])
        gScale = float(np.max(((Zc ** 2) * T * 0.5 * (cInit[:, :-1] + cInit[:, 1:])).sum(axis=0))) if N > 1 else 0.0
        epsG = 1e-12 * gScale if gScale > 0 else 0.0
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
        valsK = np.concatenate([np.ones(len(fixedIdx)), [1.0], np.full(N - 1, epsG), np.full(N - 1, -epsG)])
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
        kl = int(max(np.max(rows - cols), 0))
        ku = int(max(np.max(cols - rows), 0))
        ldab = 2 * kl + ku + 1
        flat = (kl + ku + rows - cols) + cols * ldab
        def jacobianBand(c, psi, h, Bm, Bp, dJ, Teff, Q):
            TBp, TBm = Teff * Bp, Teff * Bm
            if dn is not None:
                thEff, uD, vD, GD, thDLj = dnState['jac']
                STh = DV[None, :] * thEff / h
            else:
                STh = ST / h
            STh[:, 0] += KB[0]
            STh[:, -1] += KB[1]
            if eoOpen:
                STh[:, 0] -= 1000.0 * Q * np.where(cauMask[0] & (Q > 0), 0.0, 1.0)
                STh[:, -1] += 1000.0 * Q * np.where(cauMask[1] & (Q < 0), 0.0, 1.0)
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
            return np.bincount(flat, weights=vals, minlength=ldab * nU).reshape((ldab, nU), order='F')
        def residual(c, psi, I, Q, cOld, h, Vn, sigma, needJac):
            if dn is not None:
                thF, thDL, sigD, phiD = dnState['par']
                psiD = donnanPotential(c, Z, sigD, phiD, psi0=dnState['psiIter'])[0]
                dnState['psiIter'] = psiD
                Cb, thEff, uD, vD, GD, _ = donnanStorage(c, Z, thF, thDL, sigD, phiD, psiD)
                Tr = dnT(psiD, dnState['par'])
                if needJac:
                    dnState['jac'] = (thEff, uD, vD, GD, thDL)
            else:
                Tr = T
            if eoOpen:
                Teff = Tr + dispG[None, :] * abs(Q)
                Teff = np.where(Teff > 0, Teff, 1e-300)
                u = Zc * (psi[1:] - psi[:-1])[None, :] - 1000.0 * Q / Teff
                if lag['dG'] is not None:
                    u = u + lag['dG'] * Tr / Teff
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
            else:
                Fm = ST * (c - cOld) / h - sigma * I
            Fm[:, 0] += KB[0] * (c[:, 0] - cExtB[0])
            Fm[:, -1] += KB[1] * (c[:, -1] - cExtB[1])
            if eoOpen:
                cB0 = np.where(cauMask[0] & (Q > 0), cExtB[0], c[:, 0])
                cBL = np.where(cauMask[1] & (Q < 0), cExtB[1], c[:, -1])
                Fm[:, 0] -= 1000.0 * Q * cB0
                Fm[:, -1] += 1000.0 * Q * cBL
            Fm[:, :-1] -= J
            Fm[:, 1:] += J
            if anyFixed:
                Fm[fixed] = (c - fixVal)[fixed]
            res = np.empty(nU)
            res[icIdx] = Fm
            res[ipIdx[0]] = psi[0]
            res[ipIdx[1:]] = (Zc * J).sum(axis=0) + I + epsG * (psi[1:] - psi[:-1])
            if pair:
                res[rP] -= res[rO]
                res[rO] = 0.0
            resI = (psi[-1] - psi[0] - Vn) if voltage else 0.0
            resQ = (Q + float(np.dot(betaPsi, psi[1:] - psi[:-1])) + dpScaled) if eoOpen else 0.0
            if not needJac:
                return res, resI, resQ, None, None
            gp = -dBm * c[:, 1:] - dBp * c[:, :-1]
            dJ = Teff * Zc * gp
            ucolQ = None
            if eoOpen:
                dTdQ = dispG[None, :] * np.sign(Q)
                dudQ = -1000.0 / Teff + 1000.0 * Q * dTdQ / Teff ** 2
                if lag['dG'] is not None:
                    dudQ = dudQ - lag['dG'] * Tr * dTdQ / Teff ** 2
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
            return res, resI, resQ, jacobianBand(c, psi, h, Bm, Bp, dJ, Teff, Q), ucolQ
        def borderColumn(sigma):
            ucol = np.zeros(nU)
            ucol[ipIdx[1:]] = 1.0
            src = (sigma != 0) & ~fixed
            ucol[icIdx[src]] -= sigma[src]
            if pair:
                ucol[rP] -= ucol[rO]
                ucol[rO] = 0.0
            return ucol
        def factorize(ab):
            lu, piv, info = dgbtrf(ab, kl, ku, overwrite_ab=1)
            return (lu, piv) if info == 0 else None
        def linearSolve(LU, res, resI, ucol, resQ=0.0, ucolQ=None):
            lu, piv = LU
            if eoOpen:
                cols = [-res] + ([ucol] if voltage else []) + [ucolQ]
                x, info = dgbtrs(lu, kl, ku, np.column_stack(cols), piv)
                if info != 0:
                    return None, None, None
                xr, XU = x[:, 0], x[:, 1:]
                rowsV, D, rb = [], [], []
                if voltage:
                    vI = np.zeros(nU)
                    vI[ipIdx[-1]], vI[ipIdx[0]] = 1.0, -1.0
                    rowsV.append(vI); D.append([0.0, 0.0]); rb.append(resI)
                rowsV.append(vQ); D.append([0.0, 1.0] if voltage else [1.0]); rb.append(resQ)
                V = np.array(rowsV)
                S = np.array(D) - V @ XU
                try:
                    db = np.linalg.solve(S, -np.array(rb) - V @ xr)
                except np.linalg.LinAlgError:
                    return None, None, None
                return xr - XU @ db, (db[0] if voltage else 0.0), db[-1]
            if voltage:
                x, info = dgbtrs(lu, kl, ku, np.column_stack((-res, ucol)), piv)
                if info != 0:
                    return None, None, None
                vx1 = x[ipIdx[-1], 0] - x[ipIdx[0], 0]
                vx2 = x[ipIdx[-1], 1] - x[ipIdx[0], 1]
                if vx2 == 0 or not np.isfinite(vx2):
                    return None, None, None
                dI = (vx1 + resI) / vx2
                return x[:, 0] - x[:, 1] * dI, dI, 0.0
            x, info = dgbtrs(lu, kl, ku, -res, piv)
            return (x, 0.0, 0.0) if info == 0 else (None, None, None)
        stats = {'factor': 0, 'solve': 0}
        def newton(cStart, cOld, psi, I, Q, h, Vn, sigma):
            c = cStart.copy()
            if anyFixed:
                c[fixed] = fixVal[fixed]
            ucol = borderColumn(sigma) if voltage else None
            LU, lastNorm, refactor, ucolQ = None, None, True, None
            for it in range(1, maxIter + 1):
                needJac = refactor or LU is None
                res, resI, resQ, ab, uQ = residual(c, psi, I, Q, cOld, h, Vn, sigma, needJac=needJac)
                if needJac:
                    LU, ucolQ = factorize(ab), uQ
                    stats['factor'] += 1
                    if LU is None:
                        return None, it
                delta, dI, dQ = linearSolve(LU, res, resI, ucol, resQ, ucolQ)
                stats['solve'] += 1
                if delta is None or not np.all(np.isfinite(delta)):
                    return None, it
                dC, dPsi = delta[icIdx], delta[ipIdx]
                mP = float(np.max(np.abs(dPsi)))
                norm = max(float(np.max(np.abs(dC) / (np.abs(c) + atol))), mP / max(1.0, float(np.max(np.abs(psi)))))
                theta = (norm / lastNorm if lastNorm > 0 else 0.0) if lastNorm is not None else None
                refactor = theta is not None and theta > 0.1
                lastNorm = norm
                capPsi = max(10.0, 0.5 * float(np.max(np.abs(psi))))
                alpha = min(1.0, capPsi / mP) if mP > 0 else 1.0
                if alpha < 1.0:
                    refactor = True
                cPrev = c
                c = c + alpha * dC
                if pair:
                    P = cPrev[iH] - cPrev[iO] + alpha * dC[iH]
                    c[iH], c[iO], _ = waterPair(P)
                c = np.maximum(c, 0.0)
                psi = psi + alpha * dPsi
                if voltage:
                    I = I + alpha * dI
                if eoOpen:
                    Q = Q + alpha * dQ
                if (alpha == 1.0 and float(np.max(np.abs(c - cPrev) / (np.abs(c) + atol))) < 1e-9
                        and mP < 1e-9 * max(1.0, float(np.max(np.abs(psi))))):
                    return (c, psi, I, Q), it
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
        lag = {'dG': None}
        flux_fick, cbar, cond_bar, numerator0, denominator = nernstPlanckCurrentTerms(C, T_by_sp, z_by_sp, dx_total)
        denominator_safe = np.where(denominator > 0, denominator, 1.0)
        I_mol = resolve_imposed_current(numerator0, denominator_safe, dx_total, closureMode, voltageNow(tGlobalStart), temperature)
        with np.errstate(divide='ignore', invalid='ignore'):
            phi = -np.where(denominator > 0, (numerator0 + I_mol) / denominator_safe, 0.0) * dx_total
        phi, I_mol = sgSolveField(C, T_by_sp, z_by_sp, dx_total, closureMode, voltageNow(tGlobalStart), temperature, phi, I_mol,
                                  dLnGamma(act, np.array([np.asarray(C[sp], dtype=float) for sp in spc])) if act else None)
        psi = np.concatenate(([0.0], np.cumsum(phi)))
        c = np.array([np.asarray(C[sp], dtype=float) for sp in spc])
        if pair:
            c[iH], c[iO], _ = waterPair(c[iH] - c[iO])
        if anyFixed:
            c[fixed] = fixVal[fixed]
        h = min(dt_total, float(centralDict.get('npImplicitStep') or dt_total / 1000.0))
        hMin = 1e-10 * dt_total
        t, nStep, nNewton, nReject = 0.0, 0, 0, 0
        prev = None
        Q = -(float(np.dot(betaPsi, psi[1:] - psi[:-1])) + dpScaled) if eoOpen else 0.0
        eoVolume = 0.0
        while dt_total - t > 1e-12 * dt_total:
            hTry = min(h, dt_total - t)
            Vn = None
            if voltage:
                dPhiEl = centralDict.get('electrodeDeltaPhi')
                if dPhiEl is None:
                    dPhiEl = np.sign(voltageNow(tGlobalStart + t)) * (centralDict.get('electrodeVoltageDrop') or 0.0)
                Vn = (voltageNow(tGlobalStart + t) - dPhiEl) / RT_over_F
            if act:
                lag['dG'] = dLnGamma(act, c)
            if dn is not None:
                dnState['par'] = dn['params'](c)
                dnState['psiIter'] = dnState['psi']
            sigma, book = electrodeSources(c, I_mol, hTry)
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
                print(f"[implicit debug] reject #{nReject} : h={writeTime(hTry, 3)}, "
                      f"V={voltageNow(tGlobalStart + t):.6g} V, I_mol={I_mol:.4g} mol/s, "
                      + ("Newton n'a pas converge" if sol is None
                         else f"change={change:.3g} (espece {worstSp}, noeud {worstNode})"))
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
            if dn is not None:
                parA = dnState['par']
                dnState['psi'] = donnanPotential(c, Z, parA[2], parA[3], psi0=dnState['psi'])[0]
                dnState['Sold'] = DV[None, :] * donnanStorage(c, Z, *parA[:3], parA[3], dnState['psi'])[0]
            eoVolume += Q * hTry
            commitBalance(book, I_mol)
            t += hTry
            nStep += 1
            h = hTry * min(2.0, max(0.5, 0.9 * maxChange / max(change, 1e-12)))
            if nStep >= nMax:
                logError('nernstPlanckImplicit', f"more than {nMax} implicit steps needed (h = {writeTime(hTry, 3)}) : "
                         "increase maxSubCycling or npImplicitMaxChange")
        centralDict['npImplicitStep'] = h
        centralDict['npImplicitStats'] = {'steps': nStep, 'rejected': nReject, 'newton': nNewton,
                                          'factorizations': stats['factor'], 'solves': stats['solve']}
        if dn is not None:
            Tfin = dnT(dnState['psi'], dnState['par'])
            for i, sp in enumerate(spc):
                T_by_sp[sp] = Tfin[i].copy()
            donnanEnd(dn, c, dnState['Sold'] / DV[None, :], dnState['psi'], dnState['par'], temperature)
        C = {sp: c[i] for i, sp in enumerate(spc)}
        if eoOpen:
            phi = psi[1:] - psi[:-1]
        else:
            phi, I_mol = sgSolveField(C, T_by_sp, z_by_sp, dx_total, closureMode, voltageNow(tGlobalStart + t), temperature,
                                      psi[1:] - psi[:-1], I_mol, dLnGamma(act, c) if act else None)
        if eo is not None:
            electroOsmosisDiagnostics(eo, Q, phi, eoVolume, temperature)
        print(f"[implicit : {nNewton} Newton iterations, {stats['factor']} LU, {nReject} rejected steps]", end=" ", flush=True)
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
        Lambda = (numerator0 + I_mol) / denominator_safe
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
        especeCharge = centralDict['especeCharge']
        especeDiffCoeff = resolve_species_property('especeDiffCoeff', 'diffCoeff')
        especePorosity = resolve_species_property('especePorosity', 'porosity')
        lo, hi = 0, n
        dxHalfCell_trim = meshProp(centralDict['dxHalfCell'], lo, hi)
        dx_total = dxHalfCell_trim[:-1] + dxHalfCell_trim[1:]
        nodeSize_trim = meshProp(centralDict['nodeSize'], lo, hi)
        T_by_sp = {sp: precompute_species_transmissivity(lo, hi, especeDiffCoeff[sp], especePorosity[sp]) for sp in trsptedSpecies}
        z_by_sp = {sp: especeCharge[sp] for sp in trsptedSpecies}
        storage_by_sp = {}
        volume_trim = cellVolume(lo, hi)
        for sp in trsptedSpecies:
            poro_sp = meshProp(especePorosity[sp], lo, hi)
            storage = volume_trim * poro_sp
            storage_by_sp[sp] = np.where(storage > 0, storage, 1.0)
        electrodeVolume = centralDict.get('electrodeVolume') or [None, None]
        areaR_trim, areaL_trim = halfCellAreas(lo, hi)
        for side, iRes, iSmp, iFace in ((0, 0, 1, 0), (1, n - 1, n - 2, -1)):
            if isElectrode(side) and electrodeVolume[side]:
                aSmp = areaL_trim[iSmp] if side == 0 else areaR_trim[iSmp]
                for sp in trsptedSpecies:
                    cond = especePorosity[sp][iSmp] * aSmp * especeDiffCoeff[sp][iSmp]
                    T_by_sp[sp][iFace] = 1000.0 * cond / dxHalfCell_trim[iSmp] if cond > 0 else 0.0
                    storage_by_sp[sp][iRes] = electrodeVolume[side]
        temperature = centralDict.get('temperature', 298.15)
        if centralDict.get('imposedCurrent'):
            closureMode, closureValue = 'current', centralDict['imposedCurrent']
        elif centralDict.get('imposedCurrentDensity'):
            closureMode, closureValue = 'currentDensity', centralDict['imposedCurrentDensity']
        elif centralDict.get('imposedVoltage'):
            closureMode, closureValue = 'voltage', centralDict['imposedVoltage']
        else:
            closureMode, closureValue = 'current', 0.0
        C_state = {sp: C_old[sp].to_numpy() for sp in trsptedSpecies}
        npSolve = nernstPlanckImplicit if npSolver == 'implicit' else nernstPlanckCurrentAdaptive
        eoArg = {'eo': electroOsmosisSetup(electrodeVolume)} if eoRequested else {}
        if useDonnan:
            spc = list(trsptedSpecies)
            if centralDict.get('especePorosity'):
                logWarning('donnan', "especePorosity is ignored with donnan : the porosity is split into free water "
                           "and double layer (anion exclusion is computed by the Donnan equilibrium)")
            resNodes, resFaces = [], []
            V = cellVolume(lo, hi).copy()
            thetaAll = meshProp(centralDict['porosity'], lo, hi)
            for side, iRes, iSmp, iFace in ((0, 0, 1, 0), (1, n - 1, n - 2, n - 2)):
                if isElectrode(side) and electrodeVolume[side]:
                    resNodes.append(iRes)
                    resFaces.append((iFace, iSmp))
                    V[iRes] = electrodeVolume[side] / thetaAll[iRes]
            dnCtx = donnanSetup(spc, np.array([especeDiffCoeff[sp] for sp in spc]), resNodes)
            dnCtx.update({'V': V, 'resFaces': resFaces})
            thF0 = dnCtx['params'](np.array([C_state[sp] for sp in spc]))[0]
            for sp in spc:
                storage_by_sp[sp] = np.where(V * thF0 > 0, V * thF0, 1.0)
            eoArg['dn'] = dnCtx
        C_state, nSub, phi, I_mol = npSolve(
            C_state, dt, T_by_sp, z_by_sp, dx_total, storage_by_sp,
            closureMode, closureValue, temperature, **eoArg)
        print(f"[{nSub} sub-steps]", end=" ", flush=True)
        diagnostics = computeElectricalDiagnosticsSG(C_state, phi, I_mol, T_by_sp, z_by_sp, temperature)
        centralDict['electricPotential'] = diagnostics['electricPotential']
        centralDict['currentTotal'] = diagnostics['currentTotal']
        centralDict['currentDensity'] = diagnostics['currentDensity']
        centralDict['transferenceNumbers'] = diagnostics['transferenceNumbers']
        for sp in trsptedSpecies:
            C_new[sp] = C_state[sp]
        return C_new


    if centralDict['timeUnit'] == 'y':
        dt = centralDict['dtStep']*3600*24*365.25
    elif centralDict['timeUnit'] == 'd':
        dt = centralDict['dtStep']*3600*24
    elif centralDict['timeUnit'] == 'h':
        dt = centralDict['dtStep']*3600
    elif centralDict['timeUnit'] == 'm' or centralDict['timeUnit'] == 'min':
        dt = centralDict['dtStep']*60
    else: dt = centralDict['dtStep']
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
    if useNernstPlanck and velocity:
        logWarning('basicTransport', "velocity is ignored by Nernst-Planck (no advection) : use ADE for advection")
    adeSolver = str(centralDict.get('adeSolver') or 'explicit').lower()
    advScheme = str(centralDict.get('advectionScheme') or 'upwind').lower()
    if useADE and advScheme not in ('upwind', 'lud', 'vanleer'):
        logError('basicTransport', f"unknown advectionScheme '{advScheme}' : use 'upwind' (first order), "
                 "'LUD' (linear upwind differencing, second order) or 'vanLeer' (TVD, second order, no oscillation)")
    if useADE and advScheme in ('lud', 'vanleer') and adeSolver == 'implicit':
        logError('basicTransport', f"advectionScheme = '{centralDict.get('advectionScheme')}' is implemented for "
                 "the explicit ADE only : set adeSolver = 'explicit' or advectionScheme = 'upwind'")
    if useADE and adeSolver not in ('explicit', 'implicit'):
        logError('basicTransport', f"unknown adeSolver '{adeSolver}' : use 'explicit' or 'implicit'")
    if useADE and velocity and centralDict.get('dispersivity') and not centralDict.get('dispersionInTransport'):
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
        if useADE and velocity and not geom['nodeArea'][0] > 0:
            logError('coordinateSystem', "velocity is the Darcy velocity at r = x[0] ; x[0] = 0 (axis / centre) has "
                     "no surface, the flow rate is undefined : start the mesh at x[0] > 0")
        usesCurrentDensity = (useNernstPlanck and not centralDict.get('imposedCurrent')
                              and centralDict.get('imposedCurrentDensity'))
        if usesCurrentDensity and not geom['nodeArea'][0] > 0:
            logError('coordinateSystem', "imposedCurrentDensity is given at r = x[0] ; x[0] = 0 (axis / centre) has "
                     "no surface : use imposedCurrent or start the mesh at x[0] > 0")
    C_old = commMtrx[trsptedSpecies].copy()
    C_new = commMtrx[trsptedSpecies].copy()
    n = len(C_old)
    hasImposedClosure = bool(centralDict.get('imposedCurrent')) or bool(centralDict.get('imposedCurrentDensity')) or bool(centralDict.get('imposedVoltage'))
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
    if useADE and hasElectrode:
        logWarning('boundaries', "'electrode' has no effect with ADE (no electric current) : only the transport "
                   "condition of the electrode boundary is applied ('electrode' = 'electrode+closed')")

    if useNernstPlanckCurrent:
        for sp, col in enumerate(trsptedSpecies):
            C_old = boundaries(C_old, sp, col)
        C_new = nernstPlanckCurrentTransport(C_old, C_new)
        for sp, col in enumerate(trsptedSpecies):
            C_new = boundaries(C_new, sp, col)

    elif useNernstPlanckZeroCurrent:
        for sp, col in enumerate(trsptedSpecies):
            C_old = boundaries(C_old, sp, col)
        if npSolver == 'implicit' or hasElectrode or eoRequested:
            C_new = nernstPlanckCurrentTransport(C_old, C_new)
        else:
            C_new = nernstPlanckTransport(C_old, C_new)
        for sp, col in enumerate(trsptedSpecies):
            C_new = boundaries(C_new, sp, col)

    else:
        C_new = adeTransport(C_old)
    return C_new


def basicTransport(centralDict, commMtrx, trsptedSpecies):
    transportLog = []
    C_unchanged = commMtrx[trsptedSpecies].copy()
    saved = {k: centralDict.get(k) for k in ('boundaryConditions', 'dtStep', 'tTransportStart')}
    try:
        C = commMtrx
        for bc, dtSeg, tSeg in transportSegments(centralDict):
            if list(bc) != list(centralDict.get('bcActive') or bc):
                print(f"\n[boundary conditions from t = {tSeg}{centralDict['timeUnit']} : {bc[0]!r} (x = 0), "
                      f"{bc[1]!r} (x = L)]", end=" ", flush=True)
            centralDict['bcActive'] = list(bc)
            centralDict.update({'boundaryConditions': list(bc), 'dtStep': dtSeg, 'tTransportStart': tSeg})
            C_seg = transportCore(centralDict, C, trsptedSpecies, transportLog)
            C = C.copy()
            C[trsptedSpecies] = C_seg[trsptedSpecies].to_numpy()
        return C[trsptedSpecies].copy(), False, transportLog
    except TransportError as err:
        where, message = err.args
    except Exception as err:
        where = 'nativeTransport'
        message = f"unexpected {type(err).__name__} : {err}\n" + traceback.format_exc()
    finally:
        centralDict.update(saved)
    print(f"\nERROR ({where}) : {message}")
    transportLog.append({'level': 'ERROR', 'where': where, 'message': message, 'count': 1})
    return C_unchanged, True, transportLog


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
    centralDict['warningNbr'] = centralDict.get('warningNbr', 0) + len(transportLog)
    nErr = sum(entry['level'] == 'ERROR' for entry in transportLog)
    with open("warning.log", "a") as warningLog:
        warningLog.write(f"nativeTransport, time = {centralDict['tStep']}{centralDict['timeUnit']}, "
                         f"time step n°{centralDict['lStep']+1} : {nErr} error(s), "
                         f"{len(transportLog) - nErr} warning(s)\n")
        for entry in transportLog:
            count = f" [x{entry['count']}]" if entry['count'] > 1 else ""
            lines = entry['message'].rstrip('\n').split('\n')
            warningLog.write(f"    {entry['level']} ({entry['where']}){count} : {lines[0]}\n")
            for line in lines[1:]:
                warningLog.write(f"        {line}\n")
        warningLog.write("\n")



def trspt(centralDict):
    startTransport = time.time()
    print("Native transport", end=" ", flush=True)
    if (centralDict.get('NernstPlanck') or centralDict.get('donnan') is not None) and centralDict['PIDnbr'] > 1:
        warningManager.warn("nativeTransport (trspt) : PIDnbr forced to 1 (Nernst-Planck / Donnan couple all species)")
        centralDict['PIDnbr'] = 1
    waterMass = centralDict.get('cellWaterMass') if centralDict.get('molesStorage') else None
    if waterMass is not None:
        waterMass = np.asarray(waterMass, dtype=float)
        savedPorosity = {k: centralDict.get(k) for k in ('porosity', 'especePorosity')}
        centralDict['porosity'] = np.asarray(centralDict['porosity'], dtype=float) * waterMass
        if centralDict.get('especePorosity'):
            centralDict['especePorosity'] = {s: np.asarray(v, dtype=float) * waterMass
                                             for s, v in centralDict['especePorosity'].items()}
        moles = centralDict['commMtrx']
        waterCols = [s for s in centralDict['transportedSpecies'] if s in moles.columns]
        centralDict['commMtrx'] = pd.concat([moles[waterCols].div(waterMass, axis=0), moles.drop(columns=waterCols)],
                                            axis=1)[moles.columns]
    xSave = centralDict["commMtrx"]['x'].copy()
    initialColumns = list(centralDict['commMtrx'].columns)
    commMtrxOther = centralDict["commMtrx"][([s for s in initialColumns if s not in centralDict['transportedSpecies']])].copy()
    if centralDict['PIDnbr'] > 1:
        chunk_size = int(np.ceil(len(centralDict['transportedSpecies']) / centralDict['PIDnbr']))
        taille, reste = divmod(len(centralDict['transportedSpecies']), chunk_size)
        lst_split = [
            centralDict['transportedSpecies'][i:i + chunk_size]
            for i in range(0, len(centralDict['transportedSpecies']), chunk_size)
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=centralDict['PIDnbr']) as executor:
            futures = []
            for i,chunk in enumerate(lst_split):
                futures.append(warningManager.submit(executor, basicTransport, centralDict, centralDict['commMtrx'], chunk))
        results = []
        results = [f.result() for f in futures]
        df1, Bool, logs = zip(*results)
        abort = any(Bool)
        transportLog = [entry for log in logs for entry in log]
        commMtrxTrspt = pd.concat(df1,axis=1)
        commMtrxTrspt = pd.concat([commMtrxTrspt,commMtrxOther], axis=1)
        commMtrxTrspt = commMtrxTrspt[initialColumns]
    else:
        C_new, abort, transportLog = basicTransport(centralDict, centralDict['commMtrx'], centralDict['transportedSpecies'])
        commMtrxTrspt = pd.concat([C_new,commMtrxOther], axis=1)
        commMtrxTrspt = commMtrxTrspt[initialColumns]


    writeWarningLog(centralDict, transportLog)
    if abort:
        print('\nFatal nativeTransport error (see warning.log). Aborting run.')
        with open("warning.log", "a") as warningLog:
            warningLog.write("pycte : run aborted by nativeTransport\n")
        sys.exit()
    
    commMtrxTrspt['x'] = xSave.copy()
    if waterMass is not None:
        commMtrxTrspt = pd.concat([commMtrxTrspt[waterCols].mul(waterMass, axis=0),
                                   commMtrxTrspt.drop(columns=waterCols)], axis=1)[initialColumns]
        centralDict.update(savedPorosity)
    t = centralDict['nativeTransportClockTime'] + time.time() - startTransport
    
    if outputManager.wanted(centralDict, 'transport'):
        outDf = commMtrxTrspt.copy()
        if 'currentTotal' in centralDict:
            outDf['psi'] = centralDict['electricPotential']
            outDf['currentTotal_A'] = centralDict['currentTotal']
            if centralDict.get('electroOsmosis'):
                outDf['p_Pa'] = centralDict['electroOsmosis']['pressure_Pa']
                outDf['waterFlow_m3s'] = centralDict['electroOsmosis']['waterFlow_m3s']
        for col, val in (centralDict.get('donnanOutput') or {}).items():
            outDf[col] = val
        outDf.to_csv(outputManager.filePath(centralDict, 'transport', 'nativeTransport'),
                     index=False, header=True, sep='\t')
    
    
    # print(commMtrxTrspt)
    # sys.exit()
    
    centralDict.update({
        "commMtrx": commMtrxTrspt,
        "nativeTransportCalcTime_WallClock": centralDict["nativeTransportCalcTime_WallClock"] + t,
        "nativeTransportCalcTime_ProcessorTime": centralDict["nativeTransportCalcTime_ProcessorTime"]+ t,
        "nativeTransportInterfTime_WallClock": centralDict["nativeTransportInterfTime_WallClock"]+ t,
        "nativeTransportTotalTime": centralDict["nativeTransportTotalTime"] + t,
        })
    print(f"({writeTime((time.time() - startTransport))})")
    return centralDict