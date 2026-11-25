"""Validation of the pycte settings: checkSetting when given, checkGiven and checkModules at run start.

RULES maps each setting to its rule; ALIASES maps former setting names to the current ones."""
import difflib
import importlib
import numbers
import os
import sys
import types

import numpy as np

try:
    from . import warningManager
except ImportError:
    import warningManager


def _module(name):
    """pycte module `name`, imported when a rule needs it (engine imports this module)."""
    return importlib.import_module(f"{__package__}.{name}" if __package__ else name)


class KeywordError(ValueError):
    """Invalid pycte setting(s): the message lists every problem found."""


ALIASES = {"systemSpeciation": "coupledVariables", "renouvellement": "renewal", "especeCharge": "speciesValence",
           "especeDiffCoeff": "speciesDiffCoeff", "especePorosity": "speciesPorosity",
           "preliminarEquilibrium": "preliminaryEquilibrium", "comsolIntFonction": "comsolIntFunction",
           "speciationEch": "primarySpeciesExch", "cutoffEch": "cutoffExch",
           "primarySpeciesEchSorbed": "primarySpeciesExchSorbed", "distAnodCathTotale": "electrodeDistance",
           "chemModule": "chemistryModule", "trsptModule": "transportModule"}


def _short(value, width=120):
    try:
        text = repr(value)
    except Exception:
        text = f"<{type(value).__name__}>"
    text = " ".join(text.split())
    return text if len(text) <= width else text[:width - 5] + " ...)"


def _isNumber(value):
    return isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))


def _isSequence(value):
    return (isinstance(value, (list, tuple, range)) or (isinstance(value, np.ndarray) and value.ndim == 1))


class Rule:
    """A check: errors(value, where) -> messages (empty: valid)."""
    text = "a value"

    def errors(self, value, where):
        return [] if self.accepts(value) else [f"{where} = {_short(value)} : expected {self.text}"]

    def accepts(self, value):
        return True


class Anything(Rule):
    def __init__(self, text="any value"):
        self.text = text


class Number(Rule):
    def __init__(self, lo=None, hi=None, loOpen=False, hiOpen=False):
        self.lo, self.hi, self.loOpen, self.hiOpen = lo, hi, loOpen, hiOpen
        if lo is not None and hi is not None:
            self.text = f"a number in {']' if loOpen else '['}{lo}, {hi}{'[' if hiOpen else ']'}"
        elif lo is not None:
            self.text = f"a number {'>' if loOpen else '>='} {lo}"
        elif hi is not None:
            self.text = f"a number {'<' if hiOpen else '<='} {hi}"
        else:
            self.text = "a number"

    def accepts(self, value):
        if not _isNumber(value) or not np.isfinite(value):
            return False
        if self.lo is not None and (value <= self.lo if self.loOpen else value < self.lo):
            return False
        if self.hi is not None and (value >= self.hi if self.hiOpen else value > self.hi):
            return False
        return True


class Integer(Number):
    def __init__(self, lo=None, hi=None, choices=None):
        super().__init__(lo, hi)
        self.choices = choices
        self.text = (f"one of the integers {', '.join(map(str, choices))}" if choices else
                     self.text.replace("a number", "an integer"))

    def accepts(self, value):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
            return False
        return value in self.choices if self.choices else super().accepts(value)


class Bool(Rule):
    text = "True or False"

    def accepts(self, value):
        return isinstance(value, (bool, np.bool_)) or (isinstance(value, numbers.Integral) and value in (0, 1))


class Text(Rule):
    def __init__(self, text="a text (str)", nonEmpty=False):
        self.text, self.nonEmpty = text, nonEmpty

    def accepts(self, value):
        return isinstance(value, str) and (bool(value.strip()) or not self.nonEmpty)


class Choice(Rule):
    """A text among `options`, compared after `normalize`."""

    def __init__(self, options, normalize=lambda s: s.strip().lower(), shown=None):
        self.normalize = normalize
        self.options = {normalize(o) for o in options}
        self.shown = list(shown or options)
        self.text = "one of " + ", ".join(repr(o) for o in self.shown)

    def errors(self, value, where):
        if isinstance(value, str) and self.normalize(value) in self.options:
            return []
        hint = difflib.get_close_matches(str(value), self.shown, n=1) if isinstance(value, str) else []
        return [f"{where} = {_short(value)} : expected {self.text}" + (f" (did you mean {hint[0]!r} ?)" if hint else "")]


class FilePath(Rule):
    text = "a file path (str or pathlib.Path)"

    def accepts(self, value):
        return isinstance(value, (str, os.PathLike)) and bool(str(os.fspath(value)).strip())


class Optional(Rule):
    """None, or `rule`."""

    def __init__(self, rule):
        self.rule = rule
        self.text = f"None or {rule.text}"

    def errors(self, value, where):
        if value is None:
            return []
        found = self.rule.errors(value, where)
        return [m.replace(f": expected {self.rule.text}", f": expected {self.text}") for m in found]


class AnyOf(Rule):
    """Accepted by one of `rules`."""

    def __init__(self, *rules, text=None, pick=None):
        self.rules, self.pick = rules, pick
        self.text = text or " or ".join(r.text for r in rules)

    def errors(self, value, where):
        found = [r.errors(value, where) for r in self.rules]
        if any(not f for f in found):
            return []
        index = self.pick(value) if self.pick else None
        if index is not None:
            return found[index]
        return [f"{where} = {_short(value)} : expected {self.text}"]


class ListOf(Rule):
    """A list, tuple or 1-D array of items checked by `item`."""

    def __init__(self, item, length=None, minLength=0, unique=False, text=None):
        self.item, self.length, self.minLength, self.unique = item, length, minLength, unique
        if length:
            self.text = text or f"a list of {length} items, each {item.text}"
        elif minLength:
            self.text = text or f"a non-empty list of {item.text}"
        else:
            self.text = text or f"a list of {item.text}"

    def errors(self, value, where):
        if not _isSequence(value):
            return [f"{where} = {_short(value)} : expected {self.text}"]
        items = list(value)
        if (self.length and len(items) != self.length) or len(items) < self.minLength:
            return [f"{where} = {_short(value)} : expected {self.text} ({len(items)} given)"]
        found = []
        for k, item in enumerate(items):
            found += self.item.errors(item, f"{where}[{k}]")
            if len(found) >= 3:
                break
        if not found and self.unique:
            seen = set()
            twice = [i for i in items if isinstance(i, str) and (i in seen or seen.add(i))]
            if twice:
                found.append(f"{where} : {sorted(set(twice))} given twice")
        return found


class Pair(ListOf):
    def __init__(self, item, text=None):
        super().__init__(item, length=2, text=text or f"a pair [value at x = 0, value at x = L], each {item.text}")


class DictOf(Rule):
    """A dict with keys checked by `key` and values by `value`."""

    def __init__(self, value=None, key=None, keys=None, text=None):
        self.value, self.key, self.keys = value or Anything(), key or Text(), keys
        self.text = text or (f"a dict {{{self.key.text.replace('a text (str)', 'name')}: {self.value.text}}}"
                             if value else "a dict")

    def errors(self, value, where):
        if not isinstance(value, dict):
            hint = " (a missing ':' inside the braces makes it a set)" if isinstance(value, set) else ""
            return [f"{where} = {_short(value)} : expected {self.text}{hint}"]
        found = []
        for k, v in value.items():
            if self.keys is not None and k not in self.keys:
                hint = difflib.get_close_matches(str(k), list(self.keys), n=1)
                found.append(f"{where} : unknown key {k!r}" + (f" (did you mean {hint[0]!r} ?)" if hint else "")
                             + f", allowed : {', '.join(map(repr, self.keys))}")
                continue
            found += self.key.errors(k, f"{where} key")
            found += self.value.errors(v, f"{where}[{k!r}]")
            if len(found) >= 3:
                break
        return found


class Nodes(Rule):
    """A number checked by `number`, or one per node."""

    def __init__(self, number):
        self.number = number
        self.text = f"{number.text}, or one such value per node (list or 1-D array)"

    def errors(self, value, where):
        if _isNumber(value):
            return [] if self.number.accepts(value) else [f"{where} = {_short(value)} : expected {self.text}"]
        if isinstance(value, np.ndarray) and value.ndim == 0:
            return self.errors(value.item(), where)
        if not _isSequence(value) or len(value) == 0:
            return [f"{where} = {_short(value)} : expected {self.text}"]
        for k, item in enumerate(value):
            if not self.number.accepts(item):
                return [f"{where}[{k}] = {_short(item)} : expected {self.number.text} ({where} : {self.text})"]
        return []


class Custom(Rule):
    """check(value) raises ValueError / TypeError when the value is refused."""

    def __init__(self, check, text):
        self.check, self.text = check, text

    def errors(self, value, where):
        try:
            self.check(value)
        except (ValueError, TypeError) as error:
            message = str(error).replace("pycte : ", "")
            return [f"{where} = {_short(value)} : {message}"]
        return []


def _chemistryModule(value):
    engine = _module('engine')
    engine.normalize_choice(value, engine.chem_map, 'chemistryModule')


def _transportModule(value):
    engine = _module('engine')
    engine.normalize_choice(value, engine.trspt_map, 'transportModule')


def _osDefined(value):
    engine = _module('engine')
    OSschemes = _module('OSschemes')
    # lenient: a name that is not a solver may be an instance of solverInstances (checked at run start)
    OSschemes.parseDefined(value, engine.chem_map, engine.trspt_map, lenient=True)


def _solverInstances(value):
    _module('solverInstances').check(value)


def _operatorSplitting(value):
    engine = _module('engine')
    OSschemes = _module('OSschemes')
    if isinstance(value, (list, tuple)):
        _osDefined(value)
    else:
        engine.normalize_choice(value, OSschemes.ALIASES, 'operatorSplitting')


def _output(value):
    outputManager = _module('outputManager')
    outputManager.normalize(value, sys.maxsize)


def _outputFormat(value):
    outputManager = _module('outputManager')
    outputManager.normalizeFormat(value)


# lowest temperature in K accepted: below, the value is most likely in °C (25 for 298.15 K)
MIN_KELVIN = 200.0


def _kelvin(value):
    if not _isNumber(value) or not value > 0:
        raise ValueError("expected a number > 0, in K")
    if value < MIN_KELVIN:
        raise ValueError(f"{value} K = {value - 273.15:.2f} °C : the temperature is in K, give {value + 273.15:g} "
                         f"for {value:g} °C")


def _timeUnit(value):
    timeUnits = _module('timeUnits')
    timeUnits.secondsPerUnit(value)


def _boundaryConditions(value):
    nativeTransport = _module('nativeTransport')
    nativeTransport.boundarySchedule(value)


def _fixedPhases(value):
    phreeqc = _module('phreeqc')
    if not isinstance(value, dict):
        raise ValueError("expected a dict {pseudo-phase: (SI, reactant, amount)}, e.g. {'Fix_ph': (-7.0, 'NaOH', 10)}")
    phreeqc.fixedPhaseSpecs(value)


def _dtpycte(value):
    if not _isSequence(value) or not all(_isNumber(t) and np.isfinite(t) for t in value):
        raise ValueError("expected a list of times (numbers, unit timeUnit), e.g. [10, 20, 50]")
    times = [float(t) for t in value]
    if times and times[0] < 0:
        raise ValueError("the times must be >= 0")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("the times must be strictly increasing")


def _meshing(value):
    if value in (0, None, False):
        return
    if not _isSequence(value) or not value:
        raise ValueError("expected 0 (uniform mesh) or a list of (number of cells, cell width), e.g. [(15, 0.02)]")
    for k, zone in enumerate(value):
        if (not _isSequence(zone) or len(zone) != 2 or isinstance(zone[0], bool)
                or not isinstance(zone[0], numbers.Integral) or zone[0] <= 0
                or not _isNumber(zone[1]) or not zone[1] > 0):
            raise ValueError(f"zone {k} = {_short(zone)} : expected (number of cells > 0, cell width > 0)")


def _tortuosity(value):
    if isinstance(value, str):
        if value.replace('_', '-').replace(' ', '-').lower() not in ('millington-quirk', 'millingtonquirk'):
            raise ValueError("expected a number in ]0, 1], one value per node, or 'millington-quirk'")
        return
    found = Nodes(Number(0, 1, loOpen=True)).errors(value, "tortuosity")
    if found:
        raise ValueError("expected a number in ]0, 1], one value per node, or 'millington-quirk'")


def _saturation(value):
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("expected a number in ]0, 1], one value per node, or the name of a column of the "
                             "initial conditions (crossDependencies of the speciation)")
        return
    if Nodes(FRACTION).errors(value, "saturation"):
        raise ValueError("expected a number in ]0, 1], one value per node, or the name of a column of the initial "
                         "conditions (crossDependencies of the speciation)")


def _areaSchedule(value):
    nativeTransport = _module('nativeTransport')
    nativeTransport.areaSchedule(value)


def _gasVolume(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expected the name of a column of the initial conditions (gas volume per node, m3)")


def _crossDependencies(value):
    if not isinstance(value, dict):
        raise ValueError("expected a dict {'speciation': [...], 'transport': [...]}")
    for key, names in value.items():
        if key not in ('speciation', 'transport'):
            raise ValueError(f"unknown key {key!r} : use 'speciation' and 'transport'")
        if names is None:
            continue
        if not _isSequence(names) or not all(isinstance(n, (str, dict)) for n in names):
            raise ValueError(f"crossDependencies[{key!r}] must be a list of names (str) or {{name: [...]}} groups")


# keys of the nativeSpeciation chemistry dict, as call.buildConfig reads them: a misspelled key was ignored
CHEMISTRY_KEYS = ('elements', 'chemPath', 'minerals', 'gases', 'sorbedSpecies', 'cec', 'exchanger', 'temperature', 'ph',
                  'phReagent', 'diagnostics', 'species', 'totalsOutput', 'redox', 'pe', 'tol', 'chargeBalance')


def _chemistry(value):
    if not isinstance(value, dict):
        raise ValueError("expected a dict {'elements': [...], ...}")
    for key in value:
        if key not in CHEMISTRY_KEYS:
            hint = difflib.get_close_matches(str(key), CHEMISTRY_KEYS, n=1)
            raise ValueError(f"unknown key {key!r}" + (f" (did you mean {hint[0]!r} ?)" if hint else "")
                             + f", allowed : {', '.join(CHEMISTRY_KEYS)}")
    if 'elements' not in value:
        raise ValueError("the key 'elements' (list of the elements of the chemical system) is required")
    # temperature: a number in K, or the name of a column (in K)
    if 'temperature' in value and not isinstance(value['temperature'], str):
        try:
            _kelvin(value['temperature'])
        except ValueError as error:
            raise ValueError(f"temperature : {error}") from None


def _renewal(value):
    groups = [value] if isinstance(value, dict) else value
    if not isinstance(groups, (list, tuple)) or not groups or not all(isinstance(g, dict) for g in groups):
        raise ValueError("expected a group or a list of groups {'cells': ..., 'times': [...], 'concentration': {...}}")
    for k, group in enumerate(groups):
        unknown = set(group) - {'cells', 'times', 'concentration'}
        if unknown or 'cells' not in group or 'times' not in group:
            raise ValueError(f"group {k} : keys 'cells', 'times' (required) and 'concentration'"
                             + (f", unknown {sorted(unknown)}" if unknown else ""))


def _electrodeReactions(value):
    if not isinstance(value, dict):
        raise ValueError("expected a dict {'anode': [reaction, ...], 'cathode': [reaction, ...]}")
    for side, reactions in value.items():
        if side not in ('anode', 'cathode'):
            raise ValueError(f"unknown electrode {side!r} : use 'anode' and 'cathode'")
        if not _isSequence(reactions) or not all(isinstance(r, dict) for r in reactions):
            raise ValueError(f"electrodeReactions[{side!r}] must be a list of reactions "
                             "{'name': ..., 'n': electrons, 'stoich': {species: coefficient}, 'efficiency': ...}")
        for r in reactions:
            if not _isNumber(r.get('n')) or not isinstance(r.get('stoich'), dict):
                raise ValueError(f"electrodeReactions[{side!r}] : each reaction needs 'n' (number of electrons) and "
                                 "'stoich' {species: coefficient}")


# keys of a reaction of kineticReactions, as nativeKinetics._prepare reads them: a misspelled 'products' was ignored
# (reactant consumed without product, mass lost) ; 'name' is a label
KINETIC_REACTION_KEYS = ('rate_law', 'consumes', 'products', 'k', 'name')


def _kineticReactions(value):
    if not _isSequence(value) or not value:
        raise ValueError("expected a non-empty list of reaction dicts {'rate_law': ..., 'products': ..., 'k': ...}")
    for i, rx in enumerate(value):
        if not isinstance(rx, dict):
            raise ValueError(f"reaction {i} : expected a dict {{'rate_law': ..., 'products': ..., 'k': ...}}")
        for key in rx:
            if key not in KINETIC_REACTION_KEYS:
                hint = difflib.get_close_matches(str(key), KINETIC_REACTION_KEYS, n=1)
                raise ValueError(f"reaction {i} : unknown key {key!r}" + (f" (did you mean {hint[0]!r} ?)" if hint else "")
                                 + f", allowed : {', '.join(KINETIC_REACTION_KEYS)}")
        if 'rate_law' not in rx or 'k' not in rx:
            raise ValueError(f"reaction {i} : the keys 'rate_law' [(species, order), ...] and 'k' (rate constant, "
                             "per second) are required")
        for key in ('rate_law', 'consumes', 'products'):
            pairs = rx.get(key, [])
            if (not _isSequence(pairs)
                    or not all(_isSequence(p) and len(p) == 2 and isinstance(p[0], str) and _isNumber(p[1]) for p in pairs)):
                raise ValueError(f"reaction {i} : {key!r} must be a list of (species, number) pairs, "
                                 f"e.g. [('Aldicarb', 1)]")
        if not _isNumber(rx['k']) or rx['k'] < 0:
            raise ValueError(f"reaction {i} : 'k' must be a number >= 0 (per second)")


def _kineticMethod(value):
    if isinstance(value, type):
        return
    methods = ('RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA')
    if value not in methods:
        lower = {m.lower(): m for m in methods}
        hint = [lower[h] for h in difflib.get_close_matches(str(value).lower(), list(lower), n=1)]
        raise ValueError(f"expected one of {', '.join(methods)} (scipy solve_ivp, case sensitive)"
                         + (f" : did you mean {hint[0]!r} ?" if hint else ""))


def _stoich(value):
    if not (isinstance(value, dict) or type(value).__name__ == 'DataFrame'):
        raise ValueError("expected a dict {species: [coefficients]} or a pandas DataFrame")


NUMBER = Number()
POSITIVE = Number(0, loOpen=True)
NON_NEGATIVE = Number(0)
FRACTION = Number(0, 1, loOpen=True)
NAMES = ListOf(Text(nonEmpty=True), text="a list of names (str)")
OPTIONAL_TEXT = Optional(Text())
BLOCK = Optional(AnyOf(Text(), Bool(), text="PhreeqC input text (str)"))
NODE_INDEX = Integer()

RULES = {
    "maxTime": POSITIVE,
    "timeStep": POSITIVE,
    "dtpycte": Custom(_dtpycte, "a list of increasing times"),
    "timeUnit": Custom(_timeUnit, "a time unit"),
    "output": Custom(_output, "{'stepEnd' | 'speciation' | 'transport': [steps]}"),
    "intermediateOutput": Anything("ignored (former keyword)"),
    "outputFormat": Custom(_outputFormat, "a list among 'txt', 'vtu'"),
    "consoleDisplay": Choice(('progressBar', 'steps')),
    "description": OPTIONAL_TEXT,
    "chemistryModule": Custom(_chemistryModule, "a chemistry module"),
    "transportModule": Custom(_transportModule, "a transport module"),
    "operatorSplitting": Optional(Custom(_operatorSplitting, "a coupling scheme or an OSdefined list")),
    "OSdefined": Optional(Custom(_osDefined, "a list of (solver, fraction of dt)")),
    "solverInstances": Optional(Custom(_solverInstances, "a dict {name: {'module': solver, setting: value, ...}}")),
    "iterTol": POSITIVE,
    "iterAtol": NON_NEGATIVE,
    "iterMaxIter": Integer(1),
    "iterRelax": POSITIVE,
    "iterAnderson": Integer(0),
    "iterSinkMax": POSITIVE,
    "iterSinkMode": Choice(('linear', 'log', 'exponential', 'constant')),
    "system": Anything(),
    "speciationThreads": Integer(1),
    "transportThreads": Optional(Integer(1)),
    "coupledVariables": Optional(ListOf(Text(nonEmpty=True), minLength=1, unique=True,
                                        text="a non-empty list of names (str), each name once")),
    "systemSpecies": Optional(NAMES),
    "geometry": Integer(choices=(1, 2, 3)),
    "firstStepEquilibrium": Bool(),
    "preliminaryEquilibrium": Bool(),
    "MultiCompoundTransport": Bool(),
    "nonTrivialDecomposition": Optional(DictOf(DictOf(NUMBER, text="a dict {primary species: coefficient}"),
                                               text="a dict {species: {primary species: coefficient}}")),
    "initialConditions": FilePath(),
    "crossDependencies": Optional(Custom(_crossDependencies, "{'speciation': [...], 'transport': [...]}")),
    "chemPath": Optional(FilePath()),
    "trsptPath": FilePath(),
    "phases": Optional(AnyOf(NAMES, Text(), text="a list of phase names (ORCHESTRA) or PhreeqC PHASES text",
                             pick=lambda v: 0 if _isSequence(v) else None)),

    "current": Anything(),
    "SI": Optional(DictOf(NUMBER, text="a dict {phase: saturation index}")),
    "mineralReversibility": Optional(DictOf(Text(), text="a dict {phase: '' | 'dissolve_only' | ...}")),
    "cutoffAq": NUMBER, "cutoffPha": NUMBER, "cutoffExch": NUMBER, "cutoffSurf": NUMBER,
    "masterSpecies": BLOCK, "solutionSpecies": BLOCK, "masterExchange": BLOCK, "exchangeSpecies": BLOCK,
    "masterSurface": BLOCK, "surfaceSpecies": BLOCK,
    "kineticRates": BLOCK,
    "kinetics": BLOCK,
    "kineticsSpecies": Optional(NAMES),
    "knobs": Optional(AnyOf(Text(), DictOf(), text="KNOBS text (str)")),
    "fixpH": Anything("pH fixed by EQUILIBRIUM_PHASES (False / None: not fixed)"),
    "fixedPhases": Optional(Custom(_fixedPhases, "{pseudo-phase: (SI, reactant, amount)}")),
    "userVarBool": Optional(DictOf(Bool(), text="a dict {variable: True | False}")),
    "userVarList": Optional(DictOf(NAMES, text="a dict {variable: [names]}")),
    "solMod": Bool(),
    "step_divide": Optional(POSITIVE),
    "speciesCharge": Optional(AnyOf(NAMES, Text(), text="a list of names (str)")),
    "speciesChargeGeometry": Optional(ListOf(Pair(Anything()), text="a list of (cell, species) pairs")),
    "speciationCharge": Bool(),
    "pHdefault": NUMBER,
    "tempDefault": Custom(_kelvin, "a temperature in K (298.15 for 25 °C)"),
    "pressureDefault": POSITIVE,
    "densityPhreeqC": Optional(NON_NEGATIVE),
    "electrodeDistance": Anything(), "catholyte": Anything(), "anolyte": Anything(), "solModCharge": Anything(),
    "water": Optional(AnyOf(Bool(), Nodes(POSITIVE), text="False or one water mass (kg) per cell")),
    "supplementarySolution": OPTIONAL_TEXT,
    "surfaceCounterIons": Bool(),
    "molesStorage": Bool(),
    "phasesAsPrimary": Bool(),

    "firstIC": Anything(),
    "primarySpeciesAq": NAMES, "primarySpeciesPha": NAMES, "primarySpeciesSurf": NAMES, "primarySpeciesExch": NAMES,
    "primarySpeciesPhantom": NAMES,
    "waterMass": Nodes(POSITIVE),
    "primarySpeciesExchSorbed": Optional(DictOf()),
    "couplingFormalism": Choice(('tot', 'literbulk', 'diss'), normalize=lambda s: s),
    "inputVariableOrchestra": Optional(ListOf(Text(nonEmpty=True), minLength=1, unique=True,
                                              text="a non-empty list of ORCHESTRA variables, each once")),
    "outputVariableOrchestra": Optional(ListOf(Text(nonEmpty=True), minLength=1,
                                               text="a non-empty list of ORCHESTRA variables")),
    "orchestraCutOff": Optional(NUMBER),
    "OrchestraMemory": Bool(),
    "stoich": Optional(Custom(_stoich, "a dict or a DataFrame")),
    "colAq": NAMES,
    "transportedSpecies": Optional(NAMES),

    "comsolComp": Text(nonEmpty=True), "comsolIntFunction": Text(nonEmpty=True), "comsolStudy": Text(nonEmpty=True),
    "comsolVTU": Anything("ignored (former keyword, see outputFormat)"),
    "outputComsol": Optional(NAMES),
    "comsolReactionTerms": Bool(),
    "pflotranReactionTerms": Optional(AnyOf(Bool(), Choice(('sandbox',)), text="True, False or 'sandbox'")),
    "pflotranScale": Optional(DictOf(NUMBER, text="a dict {column: factor} (column x factor in testpycte.h5)")),

    "chemistry": Optional(Custom(_chemistry, "a dict of nativeSpeciation settings {'elements': [...], ...}")),
    "kineticReactions": Optional(Custom(_kineticReactions, "a non-empty list of reaction dicts "
                                                           "{'rate_law': ..., 'products': ..., 'k': ...}")),
    "kineticMethod": Custom(_kineticMethod, "a scipy solve_ivp method"),
    "kineticRtol": POSITIVE, "kineticAtol": POSITIVE,
    "kineticChunkSize": Optional(Integer(1)),
    "kineticCheckMB": Bool(),
    "kineticTolMB": POSITIVE,

    "velocity": NUMBER,
    "advection": Optional(Bool()), "FickDiffusion": Optional(Bool()), "porousTransport": Optional(Bool()),
    "ADE": Optional(Bool()),
    "adeSolver": Choice(('explicit', 'implicit')),
    "adeSubSteps": Integer(1),
    "adeTimeScheme": Choice(('euler', 'implicit-euler', 'backward-euler', 'crank-nicolson', 'cn'),
                            normalize=lambda s: s.lower().replace('-', '').replace('_', '').replace(' ', ''),
                            shown=('euler', 'crank-nicolson')),
    "advectionScheme": Choice(('upwind', 'LUD', 'vanLeer')),
    "renewal": Optional(Custom(_renewal, "a list of groups")),
    "donnan": Optional(DictOf(text="a dict of Donnan settings {'porosityDL' | 'thickness', 'CEC' | 'chargeDL', ...}")),
    "dispersivity": NON_NEGATIVE,
    "firstBoundary": Optional(DictOf(NUMBER, text="a dict {species: boundary value at x = 0}")),
    "secondBoundary": Optional(DictOf(NUMBER, text="a dict {species: boundary value at x = L}")),
    "boundaryConditions": Custom(_boundaryConditions, "[condition at x = 0, condition at x = L]"),
    "boundaryTransferCoeff": AnyOf(NON_NEGATIVE, Pair(NON_NEGATIVE), text="a number >= 0 [m/s] or [h_0, h_L]"),
    "activityGradient": Bool(),
    "activityDatabase": Optional(FilePath()),
    "porosity": Nodes(FRACTION),
    "area": Nodes(POSITIVE),
    "areaSchedule": Optional(Custom(_areaSchedule, "a group or a list of groups {'cells': [...], 'times': [...], "
                                                   "'area': [...]}")),
    "diffCoeff": Nodes(NON_NEGATIVE),
    "meshing": Custom(_meshing, "0 or a list of (number of cells, cell width)"),
    "saturation": Custom(_saturation, "a number in ]0, 1], one per node, or a column name"),
    "gasVolume": Optional(Custom(_gasVolume, "the name of a column of the initial conditions (gas volume per node, m3)")),
    "tortuosity": Custom(_tortuosity, "a number in ]0, 1], one per node, or 'millington-quirk'"),
    "cflSafety": POSITIVE,
    "maxSubCycling": Integer(1),
    "coordinateSystem": Choice(('cartesian', 'cylindrical', 'spherical')),
    "cylinderHeight": Nodes(POSITIVE),
    "speciesValence": Optional(DictOf(NUMBER, text="a dict {species: charge}")),
    "speciesDiffCoeff": Optional(DictOf(Nodes(NON_NEGATIVE), text="a dict {species: diffusion coefficient [m2/s], "
                                                               "a number or one value per node}")),
    "speciesPorosity": Optional(DictOf(Nodes(FRACTION), text="a dict {species: porosity in ]0, 1], a number or one "
                                                           "value per node}")),
    "crossDiffusion": Optional(DictOf(DictOf(Nodes(NUMBER)),
                                      text="a dict {target species: {driving species: cross coefficient K "
                                           "[m2/s x unit of target / unit of driver], a number or one value per node}}")),
    "NernstPlanck": Bool(),
    "npSolver": Choice(('explicit', 'implicit')),
    "npImplicitAtol": POSITIVE, "npImplicitMaxChange": POSITIVE,
    "npImplicitMaxIter": Integer(1),
    "nahed": Optional(Bool()), "nahedEpsilon": NON_NEGATIVE,
    "imposedCurrent": Optional(NUMBER), "imposedCurrentDensity": Optional(NUMBER), "imposedVoltage": Optional(NUMBER),
    "rampPotential": Optional(ListOf(NUMBER, length=3, text="[V_start, V_end, ramp duration]")),
    "electrodeArea": Optional(Pair(Optional(POSITIVE), text="[area of electrode 0, area of electrode L] (m2, or None)")),
    "electrodeVolume": Optional(Pair(Optional(POSITIVE), text="[volume at x = 0, volume at x = L] (m3, or None)")),
    "electrodeNodes": Optional(Pair(NODE_INDEX, text="[i0, iL] (node indices)")),
    "electrodeReactions": Optional(Custom(_electrodeReactions, "{'anode': [...], 'cathode': [...]}")),
    "electrodeKinetics": Choice(('faraday', 'butler-volmer'), normalize=lambda s: s),
    "electrodeVoltageDrop": NUMBER,
    "voltageDrop": Bool(),
    "electrodeResistance": Optional(AnyOf(NON_NEGATIVE, Pair(NON_NEGATIVE), text="R or [R_0, R_L] (ohm, >= 0)")),
    "targetVoltage": Optional(POSITIVE),
    "electroOsmoticPermeability": Nodes(NUMBER),
    "permeability": Nodes(POSITIVE),
    "viscosity": POSITIVE,
    "hydraulicBoundary": Choice(('open', 'closed')),
    "boundaryPressure": Pair(NUMBER, text="[p_0, p_L] (Pa)"),
    "temperature": Custom(_kelvin, "a temperature in K (298.15 for 25 °C)"),
    "Kw": POSITIVE,
    "waterEquilibrium": Optional(Bool()),
}


def _ruleOf(name):
    return RULES.get(ALIASES.get(name, name))


def errorsOf(name, value):
    """Problems of setting `name` = `value` (empty: valid)."""
    rule = _ruleOf(name)
    return rule.errors(value, name) if rule is not None else []


def _report(problems, inRun):
    title = ("pycte : invalid setting(s), the run did not start :" if inRun
             else "pycte : invalid setting refused before the run :")
    message = title + "\n" + "\n".join(f"    {p}" for p in problems)
    if inRun:
        for problem in problems:
            warningManager.info(f"pycte : invalid setting : {problem}")
        warningManager.start(f"{title}\n")
    else:
        warningManager.appendOutsideRun(message + "\n")
    raise KeywordError(message)


def checkSetting(name, value):
    """Checks one setting when it is given (KeywordError, also in warning.log)."""
    problems = errorsOf(name, value)
    if problems:
        _report(problems, inRun=False)


def checkGiven(settings):
    """Run start: checks every given setting and refuses unknown names (KeywordError)."""
    known = set(settings.known or RULES) | set(ALIASES)
    problems = []
    for name in settings.givenNames():
        if name not in known:
            if not isinstance(settings.config, types.ModuleType):
                hint = difflib.get_close_matches(name, sorted(known), n=1)
                problems.append(f"{name} : unknown setting" + (f" (did you mean '{hint[0]}' ?)" if hint else ""))
            continue
        problems += errorsOf(name, settings.peek(name))
    if isinstance(settings.config, types.ModuleType):
        problems += _misspelledVariables(settings.config, known)
    if problems:
        _report(problems, inRun=True)


def _misspelledVariables(module, known):
    """Old-style script: its other variables are not refused, so a misspelled setting would be ignored and its default
    used. A variable named as a setting but for the case (timestep for timeStep) is an error, a variable close to a
    setting (outpt for output) a warning."""
    byLower = {k.lower(): k for k in known}
    problems = []
    for name, value in vars(module).items():
        if name.startswith('_') or name in known or isinstance(value, (types.ModuleType, types.FunctionType, type)):
            continue
        if name.lower() in byLower:
            problems.append(f"{name} : unknown setting (did you mean '{byLower[name.lower()]}' ? the names of the "
                            "settings are case-sensitive)")
            continue
        close = difflib.get_close_matches(name, sorted(known), n=1, cutoff=0.85)
        if close:
            _module('warningManager').warn(f"pycte : the variable {name} of the script is not a pycte setting and is "
                                           f"ignored : if it is meant as the setting '{close[0]}', rename it")
    return problems


def _existing(settings, key, default):
    value = settings.peek(key) if settings.given(key) else default
    return None if value is None else os.path.abspath(os.fspath(value))


def checkModules(settings, chemistryModule, transportModule, common=True):
    """Run start: rules depending on the chosen solvers (KeywordError). With solverInstances: once per instance, with
    the settings of the instance and None for the solver of the other role; common: rules of the run (initial
    conditions), checked once."""
    problems = []
    given = settings.given
    value = settings.peek

    if (transportModule is not None and given('transportThreads') and value('transportThreads') is None
            and transportModule != 'COMSOL'):
        problems.append(f"transportThreads = None (COMSOL takes the cores it needs) is only for transportModule = "
                        f"'comsol' : give a number of threads >= 1 for {transportModule}")

    if transportModule == 'nativeTransport' and given('geometry') and value('geometry') != 1:
        problems.append(f"geometry = {_short(value('geometry'))} : nativeTransport is a 1D code (one coordinate x, "
                        "geometry = 1 ; coordinateSystem = 'cylindrical' or 'spherical' for radial 1D)")

    files = [('initialConditions', 'ic.txt', 'initial conditions')] if common else []
    files += {'PhreeqC': [('chemPath', 'phreeqc.dat', 'PhreeqC database')],
              'PhreeqcRM': [('chemPath', 'phreeqc.dat', 'PhreeqC database')],
              'xGEMS': [('chemPath', 'dat.lst', 'xGEMS task file (.lst)')],
              'ORCHESTRA': [('chemPath', 'chemistry1.inp', 'ORCHESTRA input file')],
              'nativeSpeciation': [('chemPath', None, 'database')]}.get(chemistryModule, [])
    files += {'COMSOL': [('trsptPath', 'comsol.mph', 'COMSOL model (.mph)')],
              'PFLOTRAN': [('trsptPath', 'pflotran.in', 'PFLOTRAN input file')],
              'nativeTransport': [('activityDatabase', None, 'activity database')]}.get(transportModule, [])
    for key, default, what in files:
        path = _existing(settings, key, default)
        if path is not None and not os.path.isfile(path):
            problems.append(f"{key} = {_short(value(key) if given(key) else default)} : {what} not found "
                            f"({path})" + ("" if given(key) else f" ; {key} is not given, '{default}' of the run "
                                                                 "folder is used by default"))
    if transportModule == 'COMSOL' and given('trsptPath') and not str(value('trsptPath')).lower().endswith('.mph'):
        problems.append(f"trsptPath = {_short(value('trsptPath'))} : COMSOL needs a model file .mph")

    required = {'ORCHESTRA': ('inputVariableOrchestra', 'outputVariableOrchestra'),
                'nativeKinetics': ('kineticReactions',),
                'nativeSpeciation': ('chemistry',)}.get(chemistryModule, ())
    for key in required:
        if not given(key) or value(key) is None:
            problems.append(f"{key} is required with chemistryModule = '{chemistryModule}'")

    if chemistryModule == 'ORCHESTRA' and given('phases') and isinstance(value('phases'), str):
        problems.append("phases : ORCHESTRA expects a list of phase names, not PhreeqC PHASES text")

    if problems:
        _report(problems, inRun=True)
