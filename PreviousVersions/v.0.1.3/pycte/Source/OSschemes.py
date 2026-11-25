"""
Schemas de decoupage d'operateurs (operator splitting) de pycte : un pas de couplage transport / speciation.

Chaque schema est une fonction  schema(centralDict, transport, speciation, t, dt)  qui fait avancer
centralDict['commMtrx'] de t - dt a t (t, dt dans l'unite timeUnit). transport et speciation sont les lanceurs
generiques des modules (COMSOL, nativeTransport, PFLOTRAN / PhreeqC, xGEMS, ORCHESTRA, ...) : ils recoivent
centralDict et renvoient un dict de mises a jour, applique par centralDict.update, exactement comme avant dans
engine.py. Avant chaque appel, le schema fixe :
    dtStep           duree de l'appel (sous-pas du schema)
    tTransportStart  debut de l'appel de transport (conditions aux limites dependant du temps, rampe de tension)

Schemas (T : transport, S : speciation, sur [t - dt, t]) :
    SNIA         T(dt) puis S(dt)                                        ordre 1
    Strang       T(dt/2), S(dt), T(dt/2)                                 ordre 2
    Alternative  pas pairs : T puis S ; pas impairs : S puis T           ordre 2 sur deux pas (T S S T)
    Additive     C = S(IC) + T(IC) - IC  (branches paralleles)           ordre 1
    Symmetrical  C = (S(T(IC)) + T(S(IC))) / 2                            ordre 2
    OSdefined    sequence definie par l'utilisateur (mot-cle OSdefined, ou liste donnee a operatorSplitting)

OSdefined : liste ordonnee d'appels (fraction du pas de couplage dt, module), executes l'un apres l'autre, ex. Strang :
    OSdefined = [(0.5, 'comsol'), (1, 'phreeqc'), (0.5, 'comsol')]
Chaque appel peut aussi s'ecrire {0.5: 'comsol'}, '0.5:comsol' ou ('comsol', 0.5) ; la fraction peut etre un nombre
ou une chaine ('0.5', '1/3'). Le module est un nom de solveur (alias de chemModule / trsptModule : 'comsol',
'nativeTransport', 'phreeqc', 'orchestra', ...) ou un role generique ('transport', 'speciation', 'chemistry').
chemModule / trsptModule absents : deduits des solveurs nommes ; presents : ils doivent concorder. Chaque appel de
transport part du temps deja couvert par le transport dans le pas (t - dt + somme des fractions de transport
precedentes) ; une colonne 'time' (crossDependencies) recoit la duree de chaque appel de speciation. Une somme des
fractions differente de 1 pour un module donne un avertissement. Les operations en branches (Additive, Symmetrical)
et l'alternance (Alternative) ne s'ecrivent pas avec OSdefined.

Branches (Additive, Symmetrical) : les deux branches partent du meme etat. L'etat de centralDict est sauvegarde
avant la premiere et restaure avant la seconde (etats internes des modules : Donnan, bilans d'electrodes, pas
implicite, ...), sauf les compteurs de temps de calcul et le nombre d'avertissements, qui s'accumulent. L'etat
conserve apres le pas est celui de la derniere branche executee : Additive execute donc la branche speciation en
premier, pour garder l'etat du transport. Les etats internes aux codes externes (PhreeqC, COMSOL, ...) ne sont
pas concernes. Les branches sont recombinees colonne par colonne, par nom (especes : systemSpeciation ; autres
colonnes : celles de la derniere branche).

Ajouter un schema : ecrire la fonction et l'inscrire dans SCHEMES (nom canonique : (fonction, alias)).
"""
import copy
import difflib
from fractions import Fraction

import numpy as np
import pandas as pd

_ACCUMULATOR_SUFFIXES = ('CalcTime_WallClock', 'CalcTime_ProcessorTime', 'InterfTime_WallClock', 'InitTime',
                         'TotalTime', 'ClockTime', 'WaitingTime')
_ACCUMULATOR_KEYS = {'warningNbr', 'waitingTime', 'extractDBTime'}


def _call(centralDict, launcher, dtStep, tTransportStart=None):
    centralDict['dtStep'] = dtStep
    if tTransportStart is not None:
        centralDict['tTransportStart'] = tTransportStart
    centralDict.update(launcher(centralDict))


def _isAccumulator(key):
    return key in _ACCUMULATOR_KEYS or (isinstance(key, str) and key.endswith(_ACCUMULATOR_SUFFIXES))


def _snapshot(centralDict):
    state = {}
    for key, value in centralDict.items():
        if _isAccumulator(key):
            continue
        try:
            state[key] = copy.deepcopy(value)
        except Exception:
            pass
    return state


def _restore(centralDict, state):
    for key in [k for k in centralDict if k not in state and not _isAccumulator(k)]:
        try:
            copy.deepcopy(centralDict[key])
        except Exception:
            continue
        del centralDict[key]
    for key, value in state.items():
        centralDict[key] = copy.deepcopy(value)


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


def _combine(centralDict, speciesValues):
    comm = centralDict['commMtrx']
    final = pd.concat([comm[centralDict['anythingButSpecies']], speciesValues], axis=1)
    centralDict['commMtrx'] = final[comm.columns].copy()


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
    species = centralDict['systemSpeciation']
    initial = _snapshot(centralDict)
    ic = centralDict['commMtrx'][species].copy()
    _call(centralDict, speciation, dt)
    spc = centralDict['commMtrx'][species].copy()
    stateS = _snapshot(centralDict)
    _restore(centralDict, initial)
    _call(centralDict, transport, dt, t - dt)
    _combineStates(centralDict, [initial, stateS, _snapshot(centralDict)], [-1.0, 1.0, 1.0])
    _combine(centralDict, spc + centralDict['commMtrx'][species] - ic)


def symmetrical(centralDict, transport, speciation, t, dt):
    species = centralDict['systemSpeciation']
    initial = _snapshot(centralDict)
    _call(centralDict, transport, dt, t - dt)
    _call(centralDict, speciation, dt)
    branch1 = centralDict['commMtrx'][species].copy()
    state1 = _snapshot(centralDict)
    print('\\')
    _restore(centralDict, initial)
    _call(centralDict, speciation, dt)
    _call(centralDict, transport, dt, t - dt)
    _combineStates(centralDict, [state1, _snapshot(centralDict)], [0.5, 0.5])
    _combine(centralDict, (branch1 + centralDict['commMtrx'][species]) / 2)


def defined(centralDict, transport, speciation, t, dt):
    crossDep = centralDict.get('crossDependencies')
    timeColumn = bool(crossDep) and 'time' in (crossDep.get('totalCrossDep') or [])
    transportDone = 0.0
    for role, fraction in centralDict['OSdefined']:
        if role == 'transport':
            _call(centralDict, transport, fraction * dt, t - dt + transportDone * dt)
            transportDone += fraction
        else:
            if timeColumn:
                centralDict['commMtrx']['time'] = fraction * dt
            _call(centralDict, speciation, fraction * dt)
    centralDict['dtStep'] = dt
    if timeColumn:
        centralDict['commMtrx']['time'] = dt


SCHEMES = {
    'SNIA':        (snia,        [1, '1', 'snia']),
    'Strang':      (strang,      [2, '2', 'strang']),
    'Alternative': (alternative, [3, '3', 'alternative']),
    'Additive':    (additive,    [4, '4', 'additive']),
    'Symmetrical': (symmetrical, [5, '5', 'symmetrical']),
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


def _module(name, chemMap, trsptMap):
    key = name.lower()
    if key in ROLES:
        return ROLES[key], None
    names = list(ROLES)
    for role, mapping in (('speciation', chemMap), ('transport', trsptMap)):
        for canonical, aliases in mapping.items():
            words = [a for a in aliases if isinstance(a, str) and not a.isdigit()]
            names += words
            if key in (w.lower() for w in words):
                return role, canonical
    hint = difflib.get_close_matches(key, [n.lower() for n in names], n=1, cutoff=0.6)
    raise ValueError(f"OSdefined : unknown module {name!r}" + (f" (did you mean '{hint[0]}' ?)" if hint else "")
                     + " : use a solver name (chemModule / trsptModule) or 'transport' / 'speciation'")


def parseDefined(value, chemMap, trsptMap):
    """OSdefined -> [(role, fraction, solveur ou None), ...] ; role : 'transport' ou 'speciation'."""
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
        role, solver = _module(name, chemMap, trsptMap)
        sequence.append((role, fraction, solver))
    if not sequence:
        raise ValueError(f"OSdefined is empty ; expected e.g. {_SYNTAX}")
    return sequence


def resolveModules(sequence, chem, trspt):
    """(chemModule, trsptModule) declares (None si absents) -> completes par les solveurs nommes dans OSdefined."""
    resolved = []
    for role, declared, keyword in (('speciation', chem, 'chemModule'), ('transport', trspt, 'trsptModule')):
        named = sorted({solver for r, _, solver in sequence if r == role and solver})
        if len(named) > 1:
            raise ValueError(f"OSdefined : several {role} solvers {named} : pycte couples one speciation module "
                             "and one transport module")
        if named and declared is not None and named[0] != declared:
            raise ValueError(f"OSdefined calls {named[0]} but {keyword} = {declared}")
        resolved.append(declared if declared is not None else (named[0] if named else None))
    return tuple(resolved)


def describe(sequence, chem, trspt):
    """Sequence OSdefined lisible, ex. 'COMSOL(0.5) -> PhreeqC(1) -> COMSOL(0.5)'."""
    names = {'transport': trspt, 'speciation': chem}
    return ' -> '.join(f"{names[role]}({fraction:g})" for role, fraction, *_ in sequence)


def checkFractions(sequence, chem):
    """Avertissements si la somme des fractions d'un module differe de 1."""
    messages = []
    for role in ('transport', 'speciation'):
        total = sum(fraction for r, fraction, *_ in sequence if r == role)
        if abs(total - 1.0) > 1e-9 and not (role == 'speciation' and chem == 'none' and total == 0):
            messages.append(f"pycte : OSdefined : the {role} fractions sum to {total:g} instead of 1 : the {role} "
                            f"module covers {total:g} dt per coupling step")
    return messages


def step(centralDict, transport, speciation, t, dt):
    """Un pas de couplage avec le schema centralDict['couplingInfo'][0]."""
    SCHEMES[centralDict['couplingInfo'][0]][0](centralDict, transport, speciation, t, dt)
