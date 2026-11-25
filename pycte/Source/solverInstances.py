"""Named solver instances (solverInstances): the same solver several times in one run, each instance with its own
settings, its own transported species and its own state, called by its name in OSdefined.

    solverInstances = {
        'nativeTransport1': {'module': 'nativeTransport', 'diffCoeff': 1e-9, ...},
        'nativeTransport2': {'module': 'nativeTransport', 'transportedSpecies': ['TK'], 'diffCoeff': 1e-6,
                             'boundaryConditions': ['constant', 'closed'], 'firstBoundary': {'TK': 320}},
        'phreeqc':          {'module': 'phreeqc'},
    }
    OSdefined = [(1, 'nativeTransport2'), (1, 'nativeTransport1'), (1, 'phreeqc')]

Rules
- 'module' names the solver (chemistryModule / transportModule names); the other keys are pycte settings of this
  instance: they win over the setting of the whole run, which wins over the default;
- a role (speciation, transport) without an instance in solverInstances keeps chemistryModule / transportModule
  (one instance named after its solver);
- a role with several instances needs OSdefined, which calls each instance by its name (otherwise: error);
- transportedSpecies of an instance: a list of columns, as for a single transport (default: the species marked
  mobile by the database extraction); a column may be transported by several instances (each call transports it);
- isolation: an instance of a role with several instances keeps in its own store every key of centralDict that its
  set-up creates or changes and every key that its calls create; they are in centralDict only during its calls.
  The keys of SHARED (exchanges between solvers) stay common.
"""
import contextlib
import copy
import difflib
import re
from collections import Counter

try:
    from . import keywordValidation
except ImportError:
    import keywordValidation

# settings of the whole run, read before the solvers are set up (or by the coupling loop): not per instance
GLOBAL_ONLY = frozenset({
    'solverInstances', 'chemistryModule', 'transportModule', 'operatorSplitting', 'OSdefined',
    'maxTime', 'timeStep', 'dtpycte', 'timeUnit', 'output', 'outputFormat', 'intermediateOutput', 'comsolVTU',
    'outputComsol', 'consoleDisplay', 'description', 'system', 'geometry',
    'coupledVariables', 'initialConditions', 'crossDependencies', 'systemSpecies', 'speciesChargeGeometry',
    'firstStepEquilibrium', 'preliminaryEquilibrium', 'nonTrivialDecomposition', 'MultiCompoundTransport',
    'renewal', 'iterTol', 'iterAtol', 'iterMaxIter', 'iterRelax', 'iterAnderson', 'iterSinkMax', 'iterSinkMode',
})
# settings of the whole run that an instance may change for itself (read by the solver from centralDict)
CORE_OVERRIDES = frozenset({'speciationThreads', 'transportThreads'})
# solvers without state at module level (session, licence, process pool, work folder): may be used several times
MULTI_INSTANCE_SOLVERS = frozenset({'nativeTransport', 'nativeKinetics'})
# keys exchanged between solvers: common to every instance (never in the store of an instance)
SHARED = frozenset({'commMtrx', 'cellWaterMass', 'molesStorage', 'saturationColumn', 'gasVolumeColumn',
                    'warningNbr', 'extractDBTime'})

_LABEL = re.compile(r'^[A-Za-z][\w.-]*$')
_MISSING = object()


def _engine():
    try:
        from . import engine
    except ImportError:
        import engine
    return engine


def _roles():
    try:
        from . import OSschemes
    except ImportError:
        import OSschemes
    return OSschemes.ROLES


class Instance:
    """One solver of the coupling: label (name in OSdefined), solver (canonical name), role ('speciation' or
    'transport'), block (its own settings), declared (given in solverInstances)."""

    def __init__(self, label, solver, role, block=None, declared=False):
        self.label, self.solver, self.role = label, solver, role
        self.block = dict(block or {})
        self.declared = declared
        self.launcher = None
        self.own = None             # private keys (isolated instance); None: its keys stay in centralDict
        self.scoped = None          # ScopedSettings of the instance

    def __repr__(self):
        return f"Instance({self.label!r}, {self.solver}, {self.role})"


def solverOf(module):
    """Solver name (chemistryModule / transportModule) -> (role, canonical name)."""
    engine = _engine()
    if not isinstance(module, str):
        raise ValueError(f"'module' = {module!r} : expected the name of a solver (text), e.g. 'nativeTransport'")
    for role, mapping in (('speciation', engine.chem_map), ('transport', engine.trspt_map)):
        for canonical, aliases in mapping.items():
            if any(isinstance(a, str) and a.lower() == module.strip().lower() for a in aliases):
                return role, canonical
    names = [a for mapping in (engine.chem_map, engine.trspt_map) for aliases in mapping.values()
             for a in aliases if isinstance(a, str) and not a.isdigit()]
    hint = difflib.get_close_matches(module, names, n=1, cutoff=0.6)
    raise ValueError(f"'module' = {module!r} : not a solver" + (f" (did you mean {hint[0]!r} ?)" if hint else ""))


def _solverNamed(label):
    try:
        return solverOf(label)[1]
    except ValueError:
        return None


def parse(value):
    """solverInstances -> [Instance] (declared, in the given order); ValueError listing every problem."""
    if not isinstance(value, dict) or not value:
        raise ValueError("expected a dict {name: {'module': solver, setting: value, ...}}, e.g. "
                         "{'nativeTransport2': {'module': 'nativeTransport', 'transportedSpecies': ['TK']}}")
    problems, instances = [], []
    known = set(keywordValidation.RULES)
    roles = _roles()
    seen = {}
    for label, block in value.items():
        if not isinstance(label, str) or not _LABEL.match(label):
            problems.append(f"{label!r} : the name of an instance is a text starting with a letter (letters, digits, "
                            "_ . -)")
            continue
        if label.lower() in roles:
            problems.append(f"{label} : '{label}' designates a role in OSdefined : choose another name")
            continue
        if label.lower() in seen:
            problems.append(f"{label} and {seen[label.lower()]} : two instances with the same name (case ignored)")
            continue
        seen[label.lower()] = label
        if not isinstance(block, dict) or 'module' not in block:
            problems.append(f"{label} : expected a dict with the solver in 'module', e.g. "
                            "{'module': 'nativeTransport', 'diffCoeff': 1e-9}")
            continue
        try:
            role, solver = solverOf(block['module'])
        except ValueError as error:
            problems.append(f"{label} : {error}")
            continue
        other = _solverNamed(label)
        if other is not None and other != solver:
            problems.append(f"{label} : name of the solver {other}, given to an instance of {solver} : choose "
                            "another name")
        settings = {}
        for key, val in block.items():
            if key == 'module':
                continue
            name = keywordValidation.ALIASES.get(key, key)
            if name in GLOBAL_ONLY:
                problems.append(f"{label} : {key} is a setting of the whole run : give it outside solverInstances")
                continue
            if name not in known:
                hint = difflib.get_close_matches(str(key), sorted(known), n=1)
                problems.append(f"{label} : {key} : unknown setting" + (f" (did you mean '{hint[0]}' ?)" if hint else ""))
                continue
            problems += [f"{label} : {p}" for p in keywordValidation.errorsOf(name, val)]
            settings[name] = val
        instances.append(Instance(label, solver, role, settings, declared=True))
    for solver, n in Counter(i.solver for i in instances).items():
        if n > 1 and solver not in MULTI_INSTANCE_SOLVERS:
            problems.append(f"{solver} is given {n} times : only " + ", ".join(sorted(MULTI_INSTANCE_SOLVERS))
                            + f" can be used several times in a run ({solver} keeps a session, a process pool or a "
                            "work folder for the whole run)")
    if problems:
        raise ValueError(" ; ".join(problems))
    return instances


def check(value):
    """keywordValidation rule of solverInstances."""
    parse(value)


class ScopedSettings:
    """Settings seen by one instance: its block, then the settings of the run, then the default."""

    def __init__(self, settings, instance):
        self._settings, self._instance = settings, instance
        self.read = set()

    def given(self, name):
        return name in self._instance.block or self._settings.given(name)

    def peek(self, name):
        if name in self._instance.block:
            return self._instance.block[name]
        return self._settings.peek(name)

    def get(self, name, default=_MISSING):
        if name in self._instance.block:
            self.read.add(name)
            value = self._instance.block[name]
            try:
                return copy.deepcopy(value)
            except Exception:
                return value
        return self._settings.get(name) if default is _MISSING else self._settings.get(name, default)

    def __call__(self, name, default=_MISSING):
        return self.get(name, default)


def build(declared, chemistry, transport):
    """Every instance of the run: the declared ones, and one implicit instance (named after its solver) for a role
    without declared instance. chemistry / transport: solver of the role without declared instance."""
    instances = list(declared)
    for role, solver in (('speciation', chemistry), ('transport', transport)):
        if not any(i.role == role for i in instances):
            instances.append(Instance(solver, solver, role))
    # speciation first (set up first: the database extraction gives the mobile species)
    return [i for i in instances if i.role == 'speciation'] + [i for i in instances if i.role == 'transport']


def byRole(instances, role):
    return [i for i in instances if i.role == role]


def several(instances):
    """Roles with more than one instance."""
    return [role for role in ('speciation', 'transport') if len(byRole(instances, role)) > 1]


def setup(centralDict, instance, isolate, function):
    """Runs the set-up `function()` of `instance`; isolated instance: the keys it creates or changes (but SHARED)
    go to its own store and centralDict gets back its previous values."""
    before = dict(centralDict)
    if isolate:
        centralDict['solverInstance'] = instance.label
        info = list(centralDict['couplingInfo'])
        info[1 if instance.role == 'speciation' else 2] = instance.solver
        centralDict['couplingInfo'] = info
    try:
        instance.launcher = function()
    finally:
        if isolate:
            own = {k: v for k, v in centralDict.items()
                   if k not in SHARED and (k not in before or v is not before[k])}
            for k in own:
                if k in before:
                    centralDict[k] = before[k]
                else:
                    del centralDict[k]
            instance.own = own
    for key in CORE_OVERRIDES & set(instance.block):
        (instance.own if isolate else centralDict)[key] = instance.block[key]
        instance.scoped.read.add(key)
    return instance.launcher


@contextlib.contextmanager
def context(centralDict, instance):
    """centralDict as seen by `instance` during the block: its own keys in, back to its store at the end; the keys
    created meanwhile (but SHARED) belong to the instance."""
    own = instance.own
    if own is None:
        yield centralDict
        return
    hidden = {k: centralDict[k] for k in own if k in centralDict}
    start = set(centralDict)
    centralDict.update(own)
    try:
        yield centralDict
    finally:
        for k in list(own):
            if k in centralDict:
                own[k] = centralDict[k]
            else:
                del own[k]
        for k in [k for k in centralDict if k not in start and k not in SHARED]:
            own[k] = centralDict[k]
        for k in set(hidden) | set(own):
            if k in hidden:
                centralDict[k] = hidden[k]
            else:
                centralDict.pop(k, None)


def launcher(instance):
    """Launcher of `instance` for the coupling schemes: runs in its context, returns nothing to merge."""
    if instance.own is None:
        return instance.launcher

    def call(centralDict):
        with context(centralDict, instance):
            result = instance.launcher(centralDict)
            if result is not None and result is not centralDict:
                centralDict.update(result)
        return {}
    call.__name__ = f"{instance.label}_launcher"
    return call


def value(centralDict, instance, key, default=None):
    """Value of `key` for `instance` (its store, or centralDict)."""
    if instance.own is not None and key in instance.own:
        return instance.own[key]
    return centralDict.get(key, default)


def unread(instances):
    """Settings given in a block and never read by its solver: [(label, solver, [names])]."""
    out = []
    for i in instances:
        if i.scoped is None:
            continue
        names = [n for n in i.block if n not in i.scoped.read]
        if names:
            out.append((i.label, i.solver, names))
    return out
