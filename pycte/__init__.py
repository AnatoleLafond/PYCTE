"""
pycte - PYthon generiC inTerfacE : coupling of speciation and transport solvers.

Quick start
-----------
    import pycte

    pycte.load("CementClayInterface")    # a bundled example (pycte.examples() lists them)
    pycte.chemModule("orchestra")        # switch solver : the example's database and initial conditions follow
    pycte.maxTime(100)                   # any setting can be overridden

    if __name__ == "__main__":           # needed on Windows as soon as PIDnbr > 1 (multiprocessing)
        pycte.run()

Settings
--------
One function per setting read by the engine (see _RECOGNIZED_SETTINGS), e.g. pycte.maxTime(72000),
pycte.timeStep(720), pycte.dispersivity(0.002). Call the ones you need, in any order, before pycte.run() ; a setting
you never call keeps its default. Only these functions exist : a typo (pycte.dispersivty(...)) fails immediately with
AttributeError instead of being silently ignored.

Functions
---------
run()                 run with the settings declared so far, then clear them. If none were declared, reads the
                      calling script's own globals (old style : chemModule = ..., maxTime = ..., pycte.run()).
run(config)           run with an explicit config object or dict, bypassing the setting functions.
reset()               clear every setting declared so far (run() already does it after each run).
run_from_file(path)   load a standalone .py config script (old style) and run it.
examples()            names of the bundled examples.
load(name)            load a bundled example as the current config. Its file paths (chemPath,
                      initialConditions, ...) point inside the installed package, so it runs from any folder ;
                      the outputs are written in the current folder.
postProcess(species, folder=None, ...)
                      outputs of a run -> pandas.DataFrame : breakthrough curves (one cell, every step, default)
                      or columnProfile=... (every cell, one step t). `folder` : run folder, current one by default.
availableSpecies(folder=None)
                      species available in the outputs of a run.
getTimes(folder=None, short=True)
                      computation times read in warning.log -> one-row DataFrame (seconds, warnings_count,
                      errors_count, errors_detail, coupling_completed).

Examples and solver switching
-----------------------------
After load(name), the example settings are rebuilt from scratch at every chemModule(), trsptModule() or example
option call, in this order : SETTINGS, CHEM_VARIANTS[chemModule], TRSPT_VARIANTS[trsptModule],
COMBO_VARIANTS[(chemModule, trsptModule)], variant(chemModule, trsptModule, **options) if the example declares it,
then the settings given by the user after load() (they win, whatever the call order). Nothing is left over from a
previous solver.

Example options : an example may declare its own keywords, OPTIONS = {"MCT": True} (default value). They exist on
the pycte module only while this example is loaded, e.g.

    pycte.load("CementClayInterface")    # PhreeqC, multi-compound transport
    pycte.MCT(False)                     # multi-species transport
"""

import copy
import importlib
import multiprocessing
import pkgutil
import sys
import types

from .Source import engine
from .Source.postProcess import (
    postProcess, getTimes, availableSpecies,
)
from . import Examples as _examples_pkg

_RECOGNIZED_SETTINGS = (
    "maxTime", "timeStep", "stepReprise", "dtpycte", "speciesChargeGeometry",
    "fixpH", "intermediateOutput", "operatorSplitting", "OSdefined", "chemModule", "trsptModule",
    "system", "PIDnbr", "PIDextract", "systemSpeciation", "timeUnit", "geometry",
    "firstStepEquilibrium", "nonTrivialDecomposition", "preliminarEquilibrium",
    "PIDchem", "PIDtrspt", "MultiCompoundTransport", "initialConditions",
    "crossDependencies", "description","chemPath", "trsptPath", "phases","output",
    
    "current", "SI", "mineralReversibility",
    "cutoffAq", "cutoffPha", "cutoffEch", "cutoffSurf",
    "masterSpecies", "solutionSpecies", "masterExchange", "exchangeSpecies",
    "masterSurface", "surfaceSpecies", "kineticRates", "kinetics",
    "userVarBool", "userVarList", "solMod", "step_divide", "speciesCharge",
    "speciationCharge", "pHdefault", "tempDefault", "distAnodCathTotale",
    "catholyte", "anolyte", "solModCharge", "acidicTrspt", "water",
    "supplementarySolution", "surfaceCounterIons", "molesStorage",

    "attributes", "firstIC", "primarySpeciesAq", "primarySpeciesPha",
    "primarySpeciesSurf", "speciationEch", "primarySpeciesPhantom", "nrThreads",
    "waterMass", "primarySpeciesEchSorbed", "couplingFormalism",
    "transportedSpecies", "inputVariableOrchestra", "outputVariableOrchestra",
    
    "comsolComp", "comsolIntFonction", "comsolStudy", "comsolVTU",
    "comsolCore", "outputComsol",

    "velocity", "advection", "FickDiffusion", "porousTransport", "ADE", "advectionScheme", "renouvellement", "donnan",
    "dispersivity", "firstBoundary", "secondBoundary", "boundaryConditions", "boundaryTransferCoeff",
    "activityGradient", "activityDatabase",
    "porosity", "area", "diffCoeff", "meshing",

    'MultiCompoundTransport', 'chemistry',

    "systemSpecies", "phasesAsPrimary", "densityPhreeqC", "kineticsSpecies", "colAq",
    "kineticReactions",

    "adeSolver", "adeSubSteps", "cflSafety", "maxSubCycling", "tortuosity",
    "coordinateSystem", "cylinderHeight", "especeCharge", "especeDiffCoeff", "especePorosity",
    "NernstPlanck", "npSolver", "npImplicitAtol", "npImplicitMaxChange", "npImplicitMaxIter",
    "imposedCurrent", "imposedCurrentDensity", "imposedVoltage", "rampPotential",
    "electrodeArea", "electrodeVolume", "electrodeReactions", "electrodeKinetics", "electrodeVoltageDrop",
    "electroOsmoticPermeability", "permeability", "viscosity", "hydraulicBoundary", "boundaryPressure",
    "temperature", "Kw", "waterEquilibrium",

)

_settings = {}          # reglages effectifs pour le prochain run()
_user = {}              # reglages donnes apres load() : prioritaires sur ceux de l'exemple
_options = {}           # mots-cles propres a l'exemple charge (son OPTIONS), ex. {'MCT': True}
_current_example = None

_MISSING = object()


def _make_setter(_name):
    def _setter(value):
        _settings[_name] = value
        if _current_example is not None:
            _user[_name] = value
    _setter.__name__ = _name
    _setter.__qualname__ = f"pycte.{_name}"
    _setter.__doc__ = f"Set pycte's '{_name}' setting to `value` for the next pycte.run()."
    return _setter


for _name in _RECOGNIZED_SETTINGS:
    globals()[_name] = _make_setter(_name)
del _name, _make_setter


def _same_choice(a, b):
    """Compare deux valeurs de reglage ('COMSOL' == 'comsol')."""
    if isinstance(a, str) and isinstance(b, str):
        return a.lower() == b.lower()
    return a == b


def _lookup_variant(variants, value):
    """Cherche `value` dans un dict de variantes, sans tenir compte de la casse."""
    if not variants:
        return None
    if value in variants:
        return variants[value]
    for key, settings in variants.items():
        if _same_choice(key, value):
            return settings
    return None


def _rebuild():
    """
    Reglages de l'exemple charge, recalcules en entier a chaque changement de solveur ou d'option (rien ne reste
    d'un solveur ou d'une option precedents), dans cet ordre :
      SETTINGS
      CHEM_VARIANTS[chemModule], TRSPT_VARIANTS[trsptModule]
      COMBO_VARIANTS[(chemModule, trsptModule)]      reglages qui dependent du couple de solveurs
      variant(chemModule, trsptModule, **options)    si l'exemple la declare ; solveurs sous leur nom canonique
                                                     ('PhreeqC', 'ORCHESTRA', 'nativeTransport', ...), options
                                                     = mots-cles de son OPTIONS (ex. MCT)
      reglages donnes par l'utilisateur apres load()  prioritaires, quel que soit l'ordre des appels
    """
    example = _current_example
    settings = copy.deepcopy(example.SETTINGS)
    chem = _user.get("chemModule", settings.get("chemModule"))
    trspt = _user.get("trsptModule", settings.get("trsptModule"))
    for variants, value in ((getattr(example, "CHEM_VARIANTS", None), chem),
                            (getattr(example, "TRSPT_VARIANTS", None), trspt)):
        settings.update(copy.deepcopy(_lookup_variant(variants, value) or {}))
    for key, combo in (getattr(example, "COMBO_VARIANTS", None) or {}).items():
        if (isinstance(key, (tuple, list)) and len(key) == 2
                and _same_choice(key[0], chem) and _same_choice(key[1], trspt)):
            settings.update(copy.deepcopy(combo))
            break
    variant = getattr(example, "variant", None)
    if callable(variant):
        settings.update(copy.deepcopy(variant(
            chemModule=engine.normalize_choice(chem, engine.chem_map, "chemModule"),
            trsptModule=engine.normalize_choice(trspt, engine.trspt_map, "trsptModule"),
            **_options)))
    settings.update(_user)
    _settings.clear()
    _settings.update(settings)


def _select(name, value, target):
    """Solveur ou option d'exemple : sans exemple charge, simple reglage ; sinon recalcul des reglages de
    l'exemple (valeur refusee -> choix precedent conserve)."""
    if _current_example is None:
        _settings[name] = value
        return
    previous = target.get(name, _MISSING)
    target[name] = value
    try:
        _rebuild()
    except Exception:
        if previous is _MISSING:
            target.pop(name, None)
        else:
            target[name] = previous
        raise


def chemModule(value):
    """Set 'chemModule' ; avec un exemple charge, prend aussi ses reglages pour ce solveur (base de donnees,
    conditions initiales, especes, ...)."""
    _select("chemModule", value, _user)


def trsptModule(value):
    """Set 'trsptModule' ; avec un exemple charge, prend aussi ses reglages pour ce solveur."""
    _select("trsptModule", value, _user)


def _optionValue(name, value, default):
    """Valeur d'une option d'exemple ; une option booleenne accepte aussi 'True' / 'False' (texte) et 1 / 0."""
    if isinstance(default, bool):
        text = str(value).strip().lower()
        if text in ("true", "1"):
            return True
        if text in ("false", "0"):
            return False
        raise ValueError(f"{name} expects True or False, got {value!r}")
    return value


def __getattr__(name):
    """Mots-cles propres a l'exemple charge (son OPTIONS), ex. pycte.MCT(False) apres
    pycte.load("CementClayInterface") ; ils n'existent pas sans cet exemple."""
    options = getattr(_current_example, "OPTIONS", None) or {}
    if name not in options:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    example = _current_example

    def option(value):
        if _current_example is not example:
            raise RuntimeError(f"{name}() belongs to the example {example.__name__.rsplit('.', 1)[-1]!r}, "
                               f"which is no longer loaded")
        _select(name, _optionValue(name, value, options[name]), _options)
    option.__name__ = name
    option.__doc__ = f"Option {name!r} of the loaded example (default {options[name]!r})."
    return option


def reset():
    """Clear every setting declared so far via pycte.<name>(value)."""
    global _current_example
    _settings.clear()
    _user.clear()
    _options.clear()
    _current_example = None


def run(config=None):
    """
    Run a pycte coupling.

    - pycte.run() : uses whatever was declared via the pycte.<name>(value)
      functions (e.g. pycte.dispersivity(0.002)). Settings are cleared
      after the run. If none were declared at all, falls back to reading
      the calling script's own globals (older style).
    - pycte.run(config) : advanced use — pass an explicit config object
      or dict directly, bypassing the setting functions.
    """
    import sys
    if sys._getframe(1).f_globals.get('__name__') == '__mp_main__':
        return None
    if multiprocessing.current_process().name != "MainProcess":
        return None

    if config is not None:
        if isinstance(config, dict):
            config = types.SimpleNamespace(**config)
        return engine.main(config)

    if _settings:
        cfg = types.SimpleNamespace(**_settings)
        try:
            return engine.main(cfg)
        finally:
            reset()

    return engine.main(sys.modules['__main__'])


def examples():
    """List the names of bundled examples available to pycte.load()."""
    return sorted(m.name for m in pkgutil.iter_modules(_examples_pkg.__path__))


def load(name):
    """
    Load a bundled example as the current config for the next
    pycte.run(), e.g.:

        pycte.load("CationExchange")
        pycte.run()

    Clears any settings declared via pycte.<name>(value) first, then
    loads the example. You can still override individual settings
    afterwards, e.g. pycte.load("CationExchange"); pycte.maxTime(1000) ;
    these overrides are kept when the solvers or the example options change.
    """
    try:
        mod = importlib.import_module(f".{name}", package=_examples_pkg.__name__)
    except ModuleNotFoundError:
        raise ValueError(
            f"Unknown example {name!r}. Available: {', '.join(examples())}"
        ) from None
    global _current_example
    reset()
    _current_example = mod
    _options.update(getattr(mod, "OPTIONS", None) or {})
    _rebuild()


def run_from_file(path):
    """
    Load a config from a standalone .py file (old style: a script that
    just defines chemModule/trsptModule/maxTime/... as globals) and run
    it. Useful while migrating existing scripts to the pycte.<setting>(value) style.
    """
    import importlib.util
    import os

    module_name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(module_name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return run(loaded)
