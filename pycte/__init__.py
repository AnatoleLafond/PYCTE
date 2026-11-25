"""pycte - PYthon generiC inTerfacE: coupling of speciation and transport solvers.

    import pycte
    pycte.load("CementClayInterface")    # bundled example, see pycte.examples()
    pycte.maxTime(100)                   # one function per setting
    if __name__ == "__main__":
        pycte.run()"""

import copy
import importlib
import multiprocessing
import pkgutil
import sys
import types

from .Source import engine, keywordValidation
from .Source.keywordValidation import KeywordError
from .Source.postProcess import (
    postProcess, getTimes, availableSpecies,
)
from .Source.nahed import plotCouplingMatrix, readCouplingMatrix
from . import Examples as _examples_pkg

_RECOGNIZED_SETTINGS = (
    "maxTime", "timeStep", "dtpycte","speciesChargeGeometry",
    "fixpH", "fixedPhases", "knobs", "intermediateOutput", "operatorSplitting", "OSdefined", "chemistryModule", "transportModule",
    "solverInstances",
    "iterTol", "iterAtol", "iterMaxIter", "iterRelax", "iterAnderson", "iterSinkMax", "iterSinkMode",
    "system", "speciationThreads", "transportThreads", "coupledVariables", "timeUnit", "geometry",
    "firstStepEquilibrium", "nonTrivialDecomposition", "preliminaryEquilibrium",
    "MultiCompoundTransport", "initialConditions",
    "crossDependencies", "description","chemPath", "trsptPath", "phases","output", "outputFormat", "consoleDisplay",
    
    "current", "SI", "mineralReversibility",
    "cutoffAq", "cutoffPha", "cutoffExch", "cutoffSurf",
    "masterSpecies", "solutionSpecies", "masterExchange", "exchangeSpecies",
    "masterSurface", "surfaceSpecies", "kineticRates", "kinetics",
    "userVarBool", "userVarList", "solMod", "step_divide", "speciesCharge",
    "speciationCharge", "pHdefault", "tempDefault", "pressureDefault", "electrodeDistance",
    "catholyte", "anolyte", "solModCharge", "water",
    "supplementarySolution", "surfaceCounterIons", "molesStorage",

    "firstIC", "primarySpeciesAq", "primarySpeciesPha",
    "primarySpeciesSurf", "primarySpeciesExch", "primarySpeciesPhantom",
    "waterMass", "primarySpeciesExchSorbed", "couplingFormalism",
    "transportedSpecies", "inputVariableOrchestra", "outputVariableOrchestra",
    "orchestraCutOff", "OrchestraMemory", "stoich",
    
    "comsolComp", "comsolIntFunction", "comsolStudy", "comsolVTU",
    "outputComsol", "comsolReactionTerms", "pflotranReactionTerms", "pflotranScale",

    "velocity", "advection", "FickDiffusion", "porousTransport", "ADE", "advectionScheme", "renewal", "donnan",
    "dispersivity", "firstBoundary", "secondBoundary", "boundaryConditions", "boundaryTransferCoeff",
    "activityGradient", "activityDatabase",
    "porosity", "area", "areaSchedule", "diffCoeff", "meshing",

    'chemistry',

    "systemSpecies", "phasesAsPrimary", "densityPhreeqC", "kineticsSpecies", "colAq",
    "kineticReactions", "kineticMethod", "kineticRtol", "kineticAtol", "kineticChunkSize",
    "kineticCheckMB", "kineticTolMB",

    "adeSolver", "adeSubSteps", "adeTimeScheme", "cflSafety", "maxSubCycling", "tortuosity", "saturation", "gasVolume",
    "coordinateSystem", "cylinderHeight", "speciesValence", "speciesDiffCoeff", "speciesPorosity", "crossDiffusion",
    "NernstPlanck", "npSolver", "npImplicitAtol", "npImplicitMaxChange", "npImplicitMaxIter", "nahed", "nahedEpsilon",
    "imposedCurrent", "imposedCurrentDensity", "imposedVoltage", "rampPotential",
    "electrodeArea", "electrodeVolume", "electrodeNodes", "electrodeReactions", "electrodeKinetics", "electrodeVoltageDrop", "voltageDrop",
    "electrodeResistance", "targetVoltage",
    "electroOsmoticPermeability", "permeability", "viscosity", "hydraulicBoundary", "boundaryPressure",
    "temperature", "Kw", "waterEquilibrium",

)

# former names still accepted : setting -> current name
_ALIASES = keywordValidation.ALIASES

_settings = {}          # settings of the next run()
_user = {}              # settings given after load(), they win over the example
_options = {}           # options of the loaded example (its OPTIONS), e.g. {'MCT': True}
_current_example = None

_MISSING = object()


def _make_setter(_name, key):
    def _setter(value):
        keywordValidation.checkSetting(_name, value)
        _settings[key] = value
        if _current_example is not None:
            _user[key] = value
    _setter.__name__ = _name
    _setter.__qualname__ = f"pycte.{_name}"
    _setter.__doc__ = (f"Set pycte's '{key}' setting to `value` for the next pycte.run()."
                       + (f" Former name of {key}." if key != _name else "")
                       + " An invalid value raises keywordValidation.KeywordError (also written to warning.log).")
    return _setter


for _name in _RECOGNIZED_SETTINGS + tuple(_ALIASES):
    globals()[_name] = _make_setter(_name, _ALIASES.get(_name, _name))
del _name, _make_setter


def _canonical(settings):
    """Example settings with former setting names replaced by the current ones."""
    return {_ALIASES.get(key, key): value for key, value in (settings or {}).items()}


def _same_choice(a, b):
    """Compares two setting values, ignoring case for texts."""
    if isinstance(a, str) and isinstance(b, str):
        return a.lower() == b.lower()
    return a == b


def _lookup_variant(variants, value):
    """Looks `value` up in a dict of variants, ignoring case."""
    if not variants:
        return None
    if value in variants:
        return variants[value]
    for key, settings in variants.items():
        if _same_choice(key, value):
            return settings
    return None


def _rebuild():
    """Rebuilds the loaded example: SETTINGS, solver variants, variant(), then the user's settings."""
    example = _current_example
    settings = _canonical(copy.deepcopy(example.SETTINGS))
    chem = _user.get("chemistryModule", settings.get("chemistryModule"))
    trspt = _user.get("transportModule", settings.get("transportModule"))
    for variants, value in ((getattr(example, "CHEM_VARIANTS", None), chem),
                            (getattr(example, "TRSPT_VARIANTS", None), trspt)):
        settings.update(_canonical(copy.deepcopy(_lookup_variant(variants, value))))
    for key, combo in (getattr(example, "COMBO_VARIANTS", None) or {}).items():
        if (isinstance(key, (tuple, list)) and len(key) == 2
                and _same_choice(key[0], chem) and _same_choice(key[1], trspt)):
            settings.update(_canonical(copy.deepcopy(combo)))
            break
    variant = getattr(example, "variant", None)
    if callable(variant):
        settings.update(_canonical(copy.deepcopy(variant(
            chemistryModule=engine.normalize_choice(chem, engine.chem_map, "chemistryModule"),
            transportModule=engine.normalize_choice(trspt, engine.trspt_map, "transportModule"),
            **_options))))
    settings.update(_user)
    _settings.clear()
    _settings.update(settings)


def _select(name, value, target):
    """Solver or example option: plain setting, or example rebuilt (previous choice kept on error)."""
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


def chemistryModule(value):
    """Sets chemistryModule (and the loaded example's settings for this solver)."""
    keywordValidation.checkSetting("chemistryModule", value)
    _select("chemistryModule", value, _user)


def transportModule(value):
    """Sets transportModule (and the loaded example's settings for this solver)."""
    keywordValidation.checkSetting("transportModule", value)
    _select("transportModule", value, _user)


def _optionValue(name, value, default):
    """Value of an example option (a boolean also accepts 'True' / 'False' and 1 / 0)."""
    if isinstance(default, bool):
        text = str(value).strip().lower()
        if text in ("true", "1"):
            return True
        if text in ("false", "0"):
            return False
        raise ValueError(f"{name} expects True or False, got {value!r}")
    return value


def __getattr__(name):
    """Options of the loaded example (its OPTIONS), e.g. pycte.MCT(False)."""
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
    """Clears every declared setting."""
    global _current_example
    _settings.clear()
    _user.clear()
    _options.clear()
    _current_example = None


def _coupling(settings):
    """Settings of the run: operatorSplitting (named scheme) and OSdefined (sequence) exclude each other; the one
    given after load() replaces the example's, which is left out. Otherwise the example's OSdefined won over the
    user's operatorSplitting, and the user's OSdefined gave a warning naming the example's scheme. Both given by the
    user: kept, engine warns that operatorSplitting is ignored."""
    settings = dict(settings)
    if _current_example is not None:
        for mine, other in (("operatorSplitting", "OSdefined"), ("OSdefined", "operatorSplitting")):
            if mine in _user and other not in _user:
                settings.pop(other, None)
    return settings


def run(config=None):
    """Runs pycte with the declared settings, with `config`, or with the caller's globals. The declared settings are
    kept after the run (Jupyter: run again, or change one setting and run again); load() and reset() clear them."""
    import sys
    if sys._getframe(1).f_globals.get('__name__') == '__mp_main__':
        return None
    if multiprocessing.current_process().name != "MainProcess":
        return None

    if config is not None:
        if isinstance(config, dict):
            config = types.SimpleNamespace(**config)
        return engine.main(config, known=_RECOGNIZED_SETTINGS + tuple(_ALIASES))

    if _settings:
        config = types.SimpleNamespace(**_coupling(_settings))
        # output steps of the example (not given by the user): those beyond a shortened run are left out by engine
        config._exampleOutput = _current_example is not None and "output" not in _user
        return engine.main(config, known=_RECOGNIZED_SETTINGS + tuple(_ALIASES))

    return engine.main(sys.modules['__main__'], known=_RECOGNIZED_SETTINGS + tuple(_ALIASES))


def solvers(show=True):
    """Solvers this installation can run: {name: 'ok (...)' or 'missing : how to install'}; printed by default.
    COMSOL is found, not started (its licence is checked by the first run)."""
    import shutil
    status = {}

    def check(name, probe):
        try:
            detail = probe()
            status[name] = "ok" + (f" ({detail})" if detail else "")
        except Exception as error:
            status[name] = f"missing : {' '.join(str(error).split())[:200]}"

    def version(package):
        from importlib.metadata import version as installed
        return f"{package} {installed(package)}"

    def phreeqcProbe():
        from .Source import phreeqc
        phreeqc.newIPhreeqc().destroy_iphreeqc()
        notes = []
        if phreeqc._phreeqcRow is None:
            notes.append("selected output read with ctypes (pycte._phreeqcRow not compiled : same results, slightly "
                         "slower)")
        if phreeqc._floatText is None:
            notes.append("numbers of the scripts and text files written by Python (pycte._floatText not compiled : "
                         "same results, slower)")
        return ", ".join([version("phreeqpy")] + notes)

    def rmProbe():
        from .Source import rmphreeqc
        rmphreeqc._phreeqcrm()
        return version("phreeqcrm")

    def orchestraProbe():
        from .Source import orchestra
        return "bundled with pycte" if orchestra.PyORCHESTRA.__name__ == "pycte.PyORCHESTRA" else orchestra.PyORCHESTRA.__file__

    def gemsProbe():
        from .Source import gems
        return "bundled with pycte" if gems.xgems.__name__ == "pycte.PyxGEMS" else gems.xgems.__file__

    def comsolProbe():
        import mph.discovery
        backend = mph.discovery.backend()
        return f"COMSOL {backend['name']}"

    def pflotranProbe():
        from .Source import pflotran
        return pflotran.executable()

    for name, probe in (("PhreeqC", phreeqcProbe), ("PhreeqcRM", rmProbe), ("ORCHESTRA", orchestraProbe),
                        ("xGEMS", gemsProbe), ("nativeSpeciation / nativeKinetics / nativeTransport", lambda: ""),
                        ("COMSOL", comsolProbe), ("PFLOTRAN", pflotranProbe)):
        check(name, probe)
    if show:
        width = max(map(len, status))
        print("\n".join(f"{name:<{width}}  {text}" for name, text in status.items()))
    return status


def examples():
    """Names of the bundled examples."""
    return sorted(m.name for m in pkgutil.iter_modules(_examples_pkg.__path__))


def load(name):
    """Loads a bundled example as the current settings (later setting calls override it)."""
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
    """Runs an old-style config script (settings as globals)."""
    import importlib.util
    import os

    module_name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(module_name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return run(loaded)
