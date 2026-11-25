"""Operator-splitting schemes: one transport / speciation coupling step from t - dt to t.

Schemes: SNIA, Strang, Alternative, Additive, Symmetrical, Iterative (SIA, fixed point on the reacted amount) and
OSdefined (user list of (fraction of dt, solver) calls). A new scheme is registered in SCHEMES."""
import copy
import difflib
from fractions import Fraction

import numpy as np
import pandas as pd

try:
    from . import display
    from . import frames
except ImportError:
    import display
    import frames

_ACCUMULATOR_SUFFIXES = ('CalcTime_WallClock', 'CalcTime_ProcessorTime', 'InterfTime_WallClock', 'InitTime',
                         'TotalTime', 'ClockTime', 'WaitingTime')
_ACCUMULATOR_KEYS = {'warningNbr', 'extractDBTime'}


def _call(centralDict, launcher, dtStep, tTransportStart=None):
    centralDict['dtStep'] = dtStep
    if tTransportStart is not None:
        centralDict['tTransportStart'] = tTransportStart
    result = launcher(centralDict)
    if result is not centralDict:               # the solvers give centralDict back: updating it with itself is no-op
        centralDict.update(result)


def _isAccumulator(key):
    return key in _ACCUMULATOR_KEYS or (isinstance(key, str) and key.endswith(_ACCUMULATOR_SUFFIXES))


# keys only read during a run (the time steps, the output steps): kept in a snapshot as they are instead of a deep
# copy (22 ms per copy of dtpycte at 100 000 steps, several times per step with Additive, Symmetrical and Iterative)
_READ_ONLY = frozenset({'dtpycte', 'output'})


def _copied(value):
    """copy.deepcopy(value); a DataFrame keeps the columns Index object of value (the same labels: the positions kept
    for that Index by frames.indexer stay valid)."""
    copied = copy.deepcopy(value)
    if isinstance(value, pd.DataFrame) and isinstance(copied, pd.DataFrame):
        copied.columns = value.columns
    return copied


def _snapshot(centralDict, skip=()):
    """Deep copy of centralDict: accumulators and values that cannot be copied left out, read-only keys as they are.
    skip: keys left out (commMtrx of a state that is only blended: _combineStates never reads it)."""
    state = {}
    for key, value in centralDict.items():
        if _isAccumulator(key) or key in skip:
            continue
        if key in _READ_ONLY:
            state[key] = value
            continue
        try:
            state[key] = _copied(value)
        except Exception:
            pass
    return state


def _restore(centralDict, state, handOver=False):
    """centralDict back to `state` (a copy of each value). handOver: the values of state themselves, when state is not
    used again (True), or not again for these values (a function of the value): the same objects as their copy would be
    (a copy of a copy has the same values and layout), without copying them again."""
    for key in [k for k in centralDict if k not in state and not _isAccumulator(k)]:
        try:
            copy.deepcopy(centralDict[key])
        except Exception:
            continue
        del centralDict[key]
    for key, value in state.items():
        if key in _READ_ONLY or handOver is True or (callable(handOver) and handOver(value)):
            centralDict[key] = value
        else:
            centralDict[key] = _copied(value)


def _blendRead(value):
    """Whether _blend reads the content of a value of the first state (dicts and float arrays), and not only its type:
    such a value of a state restored and blended later is copied, the others handed over (_restore)."""
    return isinstance(value, (dict, np.ndarray))


def _blend(values, weights):
    last = values[-1]
    if isinstance(last, dict) and all(isinstance(v, dict) for v in values):
        return {k: (_blend([v[k] for v in values], weights) if all(k in v for v in values) else last[k]) for k in last}
    if isinstance(last, float) and all(isinstance(v, float) for v in values):
        return sum(w * v for w, v in zip(weights, values))
    if (isinstance(last, np.ndarray) and last.dtype.kind == 'f'
            and all(isinstance(v, np.ndarray) and v.shape == last.shape and v.dtype.kind == 'f' for v in values)):
        return sum(w * v for w, v in zip(weights, values))
    return last


def _combineStates(centralDict, states, weights):
    for key in states[-1]:
        if key == 'commMtrx' or _isAccumulator(key) or not all(key in s for s in states):
            continue
        centralDict[key] = _blend([s[key] for s in states], weights)


def _floatPositions(df, columns):
    if not (df.columns.is_unique and frames.allFloat64(df)):
        return None
    positions = frames.indexer(df.columns, columns)
    return positions if np.all(positions >= 0) else None


def _speciesValues(df, columns):
    positions = _floatPositions(df, columns)
    if positions is None:
        return df[columns].to_numpy(dtype=float)
    return df.to_numpy()[:, positions]


def _withSpecies(df, columns, values):
    positions = _floatPositions(df, columns)
    if positions is None or values.dtype != np.float64:
        df = df.copy()
        df[columns] = values
        return df
    full = df.to_numpy(dtype=float, copy=True)
    full[:, positions] = values
    return pd.DataFrame(full, index=df.index, columns=df.columns)


def _combine(centralDict, speciesValues):
    comm = centralDict['commMtrx']
    final = pd.concat([comm[centralDict['anythingButSpecies']], speciesValues], axis=1)
    centralDict['commMtrx'] = final[comm.columns].copy()


_COORDINATES = ('x', 'y', 'z', 'r')


def _coordinates(centralDict, *frames):
    return [c for c in centralDict.get('anythingButSpecies', [])
            if str(c).lower() in _COORDINATES and all(c in df.columns for df in frames)]


def _sameGrid(source, target, coords):
    if len(source) != len(target):
        return False
    if not coords:
        return True
    a, b = source[coords].to_numpy(dtype=float), target[coords].to_numpy(dtype=float)
    return np.allclose(a, b, rtol=1e-9, atol=1e-12 * (np.max(np.abs(b), initial=0.0) + 1e-300))


def _onGrid(centralDict, source, target, values):
    coords = _coordinates(centralDict, source, target)
    data = values.to_numpy(dtype=float)
    if _sameGrid(source, target, coords):
        return pd.DataFrame(data, index=target.index, columns=values.columns)
    if not coords:
        raise ValueError(f"pycte : Additive : the transport returned {len(target)} nodes instead of {len(source)} and "
                         "no coordinate column (x, y, z) allows to map the speciation branch onto the transport grid")
    a, b = source[coords].to_numpy(dtype=float), target[coords].to_numpy(dtype=float)
    if len(coords) == 1:
        order = np.argsort(a[:, 0], kind='stable')
        mapped = np.column_stack([np.interp(b[:, 0], a[order, 0], data[order, j]) for j in range(data.shape[1])])
    else:
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        mapped = LinearNDInterpolator(a, data)(b)
        outside = ~np.isfinite(mapped)
        if np.any(outside):
            mapped[outside] = NearestNDInterpolator(a, data)(b)[outside]
    return pd.DataFrame(mapped, index=target.index, columns=values.columns)


def snia(centralDict, transport, speciation, t, dt):
    _call(centralDict, transport, dt, t - dt)
    _call(centralDict, speciation, dt)


def strang(centralDict, transport, speciation, t, dt):
    _call(centralDict, transport, dt / 2, t - dt)
    _call(centralDict, speciation, dt)
    _call(centralDict, transport, dt / 2, t - dt / 2)
    centralDict['dtStep'] = dt


def alternative(centralDict, transport, speciation, t, dt):
    if centralDict['lStep'] % 2 == 0:
        _call(centralDict, transport, dt, t - dt)
        _call(centralDict, speciation, dt)
    else:
        _call(centralDict, speciation, dt)
        _call(centralDict, transport, dt, t - dt)


def additive(centralDict, transport, speciation, t, dt):
    species = centralDict['coupledVariables']
    initial = _snapshot(centralDict)
    grid = centralDict['commMtrx']
    # frames.columnsOf: commMtrx[species] built from its array (same frame), copied as before
    ic = frames.columnsOf(grid, species).copy()
    _call(centralDict, speciation, dt)
    spc = frames.columnsOf(centralDict['commMtrx'], species).copy()
    # states only blended (_combineStates): commMtrx left out, never read there
    stateS = _snapshot(centralDict, skip=('commMtrx',))
    # initial is blended below: what _blend reads of it is copied, the rest handed over
    _restore(centralDict, initial, handOver=lambda value: not _blendRead(value))
    _call(centralDict, transport, dt, t - dt)
    _combineStates(centralDict, [initial, stateS, _snapshot(centralDict, skip=('commMtrx',))], [-1.0, 1.0, 1.0])
    comm = centralDict['commMtrx']
    if comm.index.equals(grid.index) and _sameGrid(grid, comm, _coordinates(centralDict, grid, comm)):
        _combine(centralDict, spc + frames.columnsOf(comm, species) - ic)
        return
    reaction = pd.DataFrame(spc.to_numpy(dtype=float) - ic.to_numpy(dtype=float), columns=species)
    _combine(centralDict, comm[species] + _onGrid(centralDict, grid, comm, reaction))


def symmetrical(centralDict, transport, speciation, t, dt):
    species = centralDict['coupledVariables']
    initial = _snapshot(centralDict)
    _call(centralDict, transport, dt, t - dt)
    _call(centralDict, speciation, dt)
    branch1 = frames.columnsOf(centralDict['commMtrx'], species).copy()
    state1 = _snapshot(centralDict, skip=('commMtrx',))        # only blended: commMtrx never read there
    display.stepPrint('\\')
    _restore(centralDict, initial, handOver=True)               # initial is not used again
    _call(centralDict, speciation, dt)
    _call(centralDict, transport, dt, t - dt)
    _combineStates(centralDict, [state1, _snapshot(centralDict, skip=('commMtrx',))], [0.5, 0.5])
    _combine(centralDict, (branch1 + frames.columnsOf(centralDict['commMtrx'], species)) / 2)


def defined(centralDict, transport, speciation, t, dt, launchers=None):
    """OSdefined: the calls of centralDict['OSdefined'], [(role, fraction of dt, instance)]; launchers: {instance:
    launcher} (solverInstances), otherwise the transport / speciation launcher of the role. Each transport instance
    has its own time: a call starts where its previous call of the step ended."""
    timeColumn = 'time' in centralDict['crossDependencies']['names']
    transportDone = {}
    for item in centralDict['OSdefined']:
        role, fraction = item[0], item[1]
        label = item[2] if len(item) > 2 else None
        launcher = (launchers or {}).get(label) or (transport if role == 'transport' else speciation)
        if role == 'transport':
            done = transportDone.get(label, 0.0)
            _call(centralDict, launcher, fraction * dt, t - dt + done * dt)
            transportDone[label] = done + fraction
        else:
            if timeColumn:
                centralDict['commMtrx']['time'] = fraction * dt
            _call(centralDict, launcher, fraction * dt)
    centralDict['dtStep'] = dt
    if timeColumn:
        centralDict['commMtrx']['time'] = dt


def _nativeTransport():
    try:
        from . import nativeTransport
    except ImportError:
        import nativeTransport
    return nativeTransport


def _warningManager():
    try:
        from . import warningManager
    except ImportError:
        import warningManager
    return warningManager


def _warn(message):
    _warningManager().warn(message)


def _anderson(history, S, f, relax, depth):
    if depth <= 0 or len(history) < 2:
        return S + relax * f
    m = min(depth, len(history) - 1)
    Ss = np.array([h[0] for h in history[-m - 1:]])
    Fw = np.array([h[1] for h in history[-m - 1:]])
    Fr = np.array([h[2] for h in history[-m - 1:]])
    dS, dF, dFw = np.diff(Ss, axis=0).T, np.diff(Fr, axis=0).T, np.diff(Fw, axis=0).T
    if not (np.all(np.isfinite(dFw)) and np.all(np.isfinite(Fw[-1]))):
        return S + relax * f
    try:
        gamma = np.linalg.lstsq(dFw, Fw[-1], rcond=1e-10)[0]
    except np.linalg.LinAlgError:
        return S + relax * f
    new = S + relax * f - (dS + relax * dF) @ gamma
    return new if np.all(np.isfinite(new)) else S + relax * f


def _nonFinite(store, values, species):
    bad = np.argwhere(~np.isfinite(values))
    for node, j in bad:
        store.setdefault(f"{species[j]}@{node}", True)
    return len(bad) > 0


def iterative(centralDict, transport, speciation, t, dt):
    rtol = float(centralDict.get('iterTol') or 1e-4)
    atol = float(centralDict.get('iterAtol') or 1e-12)
    maxIter = int(centralDict.get('iterMaxIter') or 50)
    relax = float(centralDict.get('iterRelax') or 1.0)
    depth = int(centralDict.get('iterAnderson', 3) or 0)
    sinkMax = float(centralDict.get('iterSinkMax') or 1e8)
    mode = str(centralDict.get('iterSinkMode') or 'linear').lower()
    if mode not in SINK_MODES:
        raise ValueError(f"iterSinkMode must be one of {', '.join(repr(m) for m in SINK_MODES)} (got {mode!r})")
    comm = centralDict['commMtrx']
    species = [s for s in centralDict['transportedSpecies'] if s in comm.columns]
    index = comm.index
    shape = (len(index), len(species))
    last = centralDict.get('iterSourceLast')
    if isinstance(last, pd.DataFrame) and last.index.equals(index):
        U = dt * last.reindex(columns=species).fillna(0.0).to_numpy(dtype=float)
    else:
        U = np.zeros(shape)
    I0 = dt * np.abs(_speciesValues(comm, species))
    fem = centralDict['couplingInfo'][2] == 'COMSOL' and not centralDict.get('iterReactionMeasured')
    initial = _snapshot(centralDict)
    history, errors, weight = [], [], None
    nonFinite = {'transport': {}, 'chemistry': {}}
    for k in range(1, maxIter + 1):
        if k > 1:
            _restore(centralDict, initial, handOver=k == maxIter)    # last iteration: initial not used again
        source = (U if mode == 'constant' else np.maximum(U, 0.0)) / dt
        with np.errstate(divide='ignore', invalid='ignore'):
            sink = np.where((U < 0) & (mode != 'constant'), np.minimum(-U / I0, sinkMax / dt), 0.0)
        if fem:
            sink = np.where(I0 > atol * dt, sink, 0.0)
        source = np.where(np.isfinite(source), source, 0.0)
        sink = np.where(np.isfinite(sink), sink, 0.0)
        centralDict['transportSource'] = pd.DataFrame(source, index=index, columns=species)
        centralDict['transportSink'] = pd.DataFrame(sink, index=index, columns=species)
        try:
            _call(centralDict, transport, dt, t - dt)
        finally:
            centralDict.pop('transportSource', None)
            centralDict.pop('transportSink', None)
        I = _speciesValues(centralDict.pop('transportReactionIntegral'), species)
        amount = centralDict.pop('transportReactionAmount', None)
        fixedTransport = centralDict.pop('transportFixed', None)
        if amount is not None:
            centralDict['iterReactionMeasured'] = True
            fem = False
        if fixedTransport is not None:
            fixed = fixedTransport.reindex(columns=species).fillna(False).to_numpy(dtype=bool)
        elif centralDict.get('boundaryConditions') is not None:
            fixed = _nativeTransport().dirichletMask(centralDict, species, shape[0])
        else:
            fixed = np.zeros(shape, dtype=bool)
        I = np.where(np.isfinite(I), I, 0.0)
        ct = _speciesValues(centralDict['commMtrx'], species)
        _nonFinite(nonFinite['transport'], ct, species)
        ct = np.where(np.isfinite(ct), ct, 0.0)
        X = ct - (dt * source - sink * I)
        X[fixed] = ct[fixed]
        deficit = np.maximum(-X, 0.0)
        clipped = int(np.count_nonzero(deficit))
        centralDict['commMtrx'] = _withSpecies(centralDict['commMtrx'], species, X + deficit)
        _call(centralDict, speciation, dt)
        cc = _speciesValues(centralDict['commMtrx'], species) - deficit
        if _nonFinite(nonFinite['chemistry'], cc, species):
            cc = np.where(np.isfinite(cc), cc, X)
            clipped = clipped or 1
        if clipped:
            centralDict['commMtrx'] = _withSpecies(centralDict['commMtrx'], species, cc)
        Uchem = np.where(fixed, 0.0, cc - X)
        scale = rtol * (np.abs(cc) + np.max(np.abs(cc), initial=0.0)) + atol
        gap = np.where(fixed, 0.0, np.abs(cc - ct)) / scale
        errors.append(float(np.max(gap)) if gap.size else 0.0)
        if errors[-1] <= 1.0:
            break
        if k == 1:
            U = Uchem
            I0 = np.where(I > 0, I, I0)
            continue
        if weight is None:
            weight = 1.0 / (rtol * np.max(np.abs(cc), axis=0, keepdims=True) + atol)
        free = ~fixed
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            arrived = (I0 <= 0) & (I > 0)
            I0 = np.where(arrived, I, I0)
            ratio = np.where(I > 0, I0 / I, 1.0)
            target = np.where(Uchem > 0, Uchem, Uchem * ratio)
            consumed = (Uchem < 0) & (X > 0)
            if mode == 'constant':
                target = Uchem
            elif mode == 'log':
                removed = sink * I
                factor = (np.log1p(np.maximum(Uchem / X, -1.0 + 1e-15))
                          / np.log1p(np.maximum(-removed / X, -1.0 + 1e-15)))
                valid = consumed & (U < 0) & ~arrived & (removed > 0.5 * X) & np.isfinite(factor) & (factor > 0)
                target = np.where(valid, U * np.clip(factor, 0.1, 10.0), target)
            elif mode == 'exponential':
                rate = np.minimum(np.log(X / np.maximum(X + Uchem, 0.0)) / dt, sinkMax / dt)
                target = np.where(consumed & (I0 > 0) & np.isfinite(rate), -rate * I0, target)
            if fem:
                target = np.where(Uchem > 0, target, np.maximum(target, -sinkMax * I0 / dt))
            f = np.where(fixed, 0.0, target - U)
        f = np.where(np.isfinite(f), f, 0.0)
        history.append((U[free], (f * weight)[free], f[free]))
        Unew = np.zeros(shape)
        Unew[free] = _anderson(history, U[free], f[free], relax, depth)
        U = Unew
    converged = errors[-1] <= 1.0
    centralDict['iterSourceLast'] = pd.DataFrame(Uchem / dt, index=index, columns=species)
    centralDict['iterStats'] = {'iterations': len(errors), 'errors': errors, 'converged': converged,
                                'clipped': clipped}
    centralDict['dtStep'] = dt
    _warningManager().info(
        f"pycte : Iterative, time = {t}{centralDict['timeUnit']}, time step n°{centralDict['lStep'] + 1} : "
        f"{len(errors)} iteration(s), {'converged' if converged else 'not converged'} "
        "(|C_chem - C_transport| / (iterTol.(|C| + max|C|) + iterAtol), converged when <= 1)"
        + "".join(f"\niteration {i} : {e:.3g}" for i, e in enumerate(errors, 1)))
    for module, where in nonFinite.items():
        if where:
            _warn(f"pycte : Iterative : the {module} returned non-finite values (nan / inf) at t = {t}"
                  f"{centralDict['timeUnit']} for {len(where)} species / node(s), e.g. "
                  + ', '.join(list(where)[:8])
                  + (" : no reaction kept there (C_chem = C_transport-only state)" if module == 'chemistry'
                     else " : set to 0 (check the transport model and the reaction source terms)"))
    if not converged:
        _warn(f"pycte : Iterative : not converged at t = {t}{centralDict['timeUnit']} after {maxIter} iterations "
              f"(|C_chem - C_transport| = {errors[-1]:.3g} x (iterTol.|C| + iterAtol)) : the last iterate is kept ; "
              "increase iterMaxIter, use iterAnderson or reduce the time step")
    negative = int(np.count_nonzero(cc < (-atol if fem else 0.0)))
    if fem and np.any(cc < 0):
        comm = centralDict['commMtrx'].copy()
        comm[species] = np.maximum(cc, 0.0)
        centralDict['commMtrx'] = comm
    if negative:
        comm = centralDict['commMtrx'].copy()
        comm[species] = np.maximum(cc, 0.0)
        centralDict['commMtrx'] = comm
        _warn(f"pycte : Iterative : {negative} negative concentration(s) set to 0 at t = {t}{centralDict['timeUnit']} "
              "(the chemistry consumed more than the transport-only state) : mass is not conserved there ; reduce the "
              "time step")
    centralDict['iterStats']['negative'] = negative


SINK_MODES = ('linear', 'log', 'exponential', 'constant')

SCHEMES = {
    'SNIA':        (snia,        [1, '1', 'snia']),
    'Strang':      (strang,      [2, '2', 'strang']),
    'Alternative': (alternative, [3, '3', 'alternative']),
    'Additive':    (additive,    [4, '4', 'additive']),
    'Symmetrical': (symmetrical, [5, '5', 'symmetrical']),
    'Iterative':   (iterative,   [6, '6', 'iterative', 'sia']),
    'OSdefined':   (defined,     []),
}
ALIASES = {name: aliases for name, (_, aliases) in SCHEMES.items()}

ROLES = {'transport': 'transport', 'trspt': 'transport',
         'speciation': 'speciation', 'chemistry': 'speciation', 'chem': 'speciation'}
_SYNTAX = "OSdefined = [(0.5, 'comsol'), (1, 'phreeqc'), (0.5, 'comsol')]"


def _fraction(value):
    if isinstance(value, bool):
        raise ValueError(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        return float(Fraction(value.strip()))
    raise ValueError(value)


def _pair(item):
    if isinstance(item, dict):
        if len(item) != 1:
            raise ValueError(f"OSdefined : {item!r} : one call per dict ({{0.5: 'comsol'}}) ; write one dict per call")
        item = next(iter(item.items()))
    elif isinstance(item, str):
        item = item.split(':')
    if not isinstance(item, (tuple, list)) or len(item) != 2:
        raise ValueError(f"OSdefined : {item!r} is not a (fraction, module) call ; expected e.g. {_SYNTAX}")
    for fraction, name in (item, item[::-1]):
        if not isinstance(name, str):
            continue
        try:
            return _fraction(fraction), name.strip()
        except (ValueError, ZeroDivisionError):
            continue
    raise ValueError(f"OSdefined : {tuple(item)!r} : no (fraction, module) pair found ; expected e.g. {_SYNTAX}")


def _module(name, chemMap, trsptMap, labels=None, lenient=False):
    """Name of a call -> (role, name): an instance of solverInstances (labels {name: role}), a role ('transport',
    'speciation': name None) or a solver (canonical name). lenient: an unknown name is kept (role None), it may be
    an instance declared by solverInstances (checked at run start)."""
    key = name.lower()
    for label, role in (labels or {}).items():
        if label == name:
            return role, label
    for label, role in (labels or {}).items():
        if label.lower() == key:
            return role, label
    if key in ROLES:
        return ROLES[key], None
    names = list(ROLES) + list(labels or {})
    for role, mapping in (('speciation', chemMap), ('transport', trsptMap)):
        for canonical, aliases in mapping.items():
            words = [a for a in aliases if isinstance(a, str) and not a.isdigit()]
            names += words
            if key in (w.lower() for w in words):
                return role, canonical
    if lenient:
        return None, name
    match = difflib.get_close_matches(key, [n.lower() for n in names], n=1, cutoff=0.6)
    hint = [n for n in names if match and n.lower() == match[0]][:1]
    raise ValueError(f"OSdefined : unknown module {name!r}" + (f" (did you mean '{hint[0]}' ?)" if hint else "")
                     + " : use a solver name (chemistryModule / transportModule), 'transport' / 'speciation'"
                     + (", or an instance of solverInstances (" + ", ".join(labels) + ")" if labels else ""))


def parseDefined(value, chemMap, trsptMap, labels=None, lenient=False):
    """OSdefined -> [(role, fraction, name), ...]; role: 'transport' or 'speciation' (None: unknown name, lenient);
    name: instance of solverInstances (labels {name: role}), solver, or None for a role."""
    if isinstance(value, dict):
        raise ValueError("OSdefined must be a list, not a dict : a dict cannot hold the same key twice "
                         f"({{0.5: 'comsol', 1: 'phreeqc', 0.5: 'comsol'}} loses a call) ; write {_SYNTAX}")
    if isinstance(value, str) or not hasattr(value, '__iter__'):
        raise ValueError(f"OSdefined must be a list of (fraction, module) calls, e.g. {_SYNTAX} (got {value!r})")
    sequence = []
    for item in value:
        fraction, name = _pair(item)
        if not np.isfinite(fraction) or fraction <= 0:
            raise ValueError(f"OSdefined : {item!r} : the fraction of the time step must be > 0")
        role, solver = _module(name, chemMap, trsptMap, labels, lenient)
        sequence.append((role, fraction, solver))
    if not sequence:
        raise ValueError(f"OSdefined is empty ; expected e.g. {_SYNTAX}")
    return sequence


def bind(sequence, instances):
    """[(role, fraction, name)] -> [(role, fraction, instance label)]: a role or a solver designates its only
    instance; with several instances, each one is called by its name. instances: [(label, role, solver)]."""
    roleOf = {label: role for label, role, _ in instances}
    bound = []
    for role, fraction, name in sequence:
        if name in roleOf:
            bound.append((roleOf[name], fraction, name))
            continue
        candidates = [label for label, r, solver in instances if r == role and (name is None or solver == name)]
        if len(candidates) == 1:
            bound.append((role, fraction, candidates[0]))
            continue
        what = name or role
        if not candidates:
            raise ValueError(f"OSdefined calls {what} : no {role} instance runs {what} (instances : "
                             + ", ".join(f"{label} ({solver})" for label, r, solver in instances if r == role) + ")")
        raise ValueError(f"OSdefined calls {what!r}, which designates several instances ({', '.join(candidates)}) : "
                         "call each one by its name")
    return bound


def resolveModules(sequence, chem, trspt, declared=()):
    """(chemistryModule, transportModule) completed with the solvers named in OSdefined; declared: roles given by
    solverInstances (their solvers are its instances, not chemistryModule / transportModule)."""
    resolved = []
    for role, declaredModule, keyword in (('speciation', chem, 'chemistryModule'), ('transport', trspt, 'transportModule')):
        if role in declared:
            resolved.append(declaredModule)
            continue
        named = sorted({solver for r, _, solver in sequence if r == role and solver})
        if len(named) > 1:
            raise ValueError(f"OSdefined : several {role} solvers {named} : declare each one in solverInstances "
                             "(name, solver and settings) and call them by their name")
        if named and declaredModule is not None and named[0] != declaredModule:
            raise ValueError(f"OSdefined calls {named[0]} but {keyword} = {declaredModule}")
        resolved.append(declaredModule if declaredModule is not None else (named[0] if named else None))
    return tuple(resolved)


def describe(sequence, chem, trspt):
    """Readable OSdefined sequence, e.g. 'COMSOL(0.5) -> PhreeqC(1) -> COMSOL(0.5)' (instance names when bound)."""
    names = {'transport': trspt, 'speciation': chem}
    return ' -> '.join(f"{(item[2] if len(item) > 2 and item[2] else names[item[0]])}({item[1]:g})"
                       for item in sequence)


def checkFractions(sequence, chem, instances=None):
    """Warnings if the sum of the fractions of a module differs from 1. instances [(label, role, solver)]: run with
    several instances in a role, the sum of each instance."""
    messages = []
    if instances:
        for label, role, solver in instances:
            total = sum(item[1] for item in sequence if len(item) > 2 and item[2] == label)
            if abs(total - 1.0) > 1e-9 and total > 0:
                messages.append(f"pycte : OSdefined : the fractions of {label} sum to {total:g} instead of 1 : "
                                f"{label} ({solver}) covers {total:g} dt per coupling step")
        return messages
    for role in ('transport', 'speciation'):
        total = sum(fraction for r, fraction, *_ in sequence if r == role)
        if abs(total - 1.0) > 1e-9 and not (role == 'speciation' and chem == 'none' and total == 0):
            messages.append(f"pycte : OSdefined : the {role} fractions sum to {total:g} instead of 1 : the {role} "
                            f"module covers {total:g} dt per coupling step")
    return messages


def step(centralDict, transport, speciation, t, dt, launchers=None):
    """One coupling step with the scheme centralDict['couplingInfo'][0]; launchers {instance: launcher}: OSdefined
    with solverInstances."""
    scheme = SCHEMES[centralDict['couplingInfo'][0]][0]
    if scheme is defined:
        defined(centralDict, transport, speciation, t, dt, launchers)
    else:
        scheme(centralDict, transport, speciation, t, dt)
