"""
piccts — Python Interface Coupling generiC Transport and Speciation.

Public API
----------
One dedicated function per setting piccts actually reads (generated
from every `getattr(config, 'name', default)` call in engine.py — see
_RECOGNIZED_SETTINGS below). Call the ones you need, in any order,
before piccts.run():

    import piccts

    piccts.chemModule("phreeqc")
    piccts.trsptModule("nativeTransport")
    piccts.maxTime(72000)
    piccts.timeStep(720)
    piccts.dispersivity(0.002)
    piccts.velocity(0.002 / 720)

    if __name__ == "__main__":
        piccts.run()

Only these functions exist on the `piccts` module — nothing else. A
typo (piccts.dispersivty(...)) fails immediately with AttributeError,
instead of silently being ignored. A setting you never call keeps its
documented default (same default as when it was an absent attribute on
an old-style input script).

run()                : run using whatever was declared via the setting
                       functions above. If none were called at all,
                       falls back to reading the calling script's own
                       globals (the older `import piccts; ...;
                       piccts.run()` style, kept for scripts that
                       define chemModule/maxTime/... directly).

run(config)           : advanced/escape hatch — pass an explicit config
                       object or dict directly, bypassing the setting
                       functions entirely.

reset()               : clear every setting declared so far, without
                       running (rarely needed — run() already clears
                       after each call).

run_from_file(path)   : legacy helper. Loads a standalone .py config
                       script (old copy/paste style) and runs it.

load(name)            : load a bundled example (see piccts.examples()
                       for the list of names) as the current config,
                       e.g. piccts.load("CationExchange"); piccts.run().
                       Every file path in the example (chemPath,
                       initialConditions, ...) is resolved to an
                       absolute path inside the installed package, so
                       it works regardless of your script's own
                       working directory.

examples()            : list the names of bundled examples available
                       to piccts.load().

chemModule(value)      : like every other setting function, but solver-aware
                       when an example is loaded: if the loaded example
                       declares a CHEM_VARIANTS[value] (e.g. "orchestra"),
                       those settings (chemPath, initialConditions, and any
                       solver-specific species settings) are merged in too,
                       e.g. piccts.load("CationExchange");
                       piccts.chemModule("orchestra") switches to
                       ORCHESTRA's database/initial conditions/species
                       instead of leaving PhreeqC's in place.
"""

import importlib
import multiprocessing
import pkgutil
import sys
import types

from .Source import engine
from . import Examples as _examples_pkg

# Every keyword piccts.engine.main() actually reads, i.e. every name
# that appears as getattr(config, 'name', default) in engine.py.
# Keep this in sync with engine.py if settings are added/removed there.
_RECOGNIZED_SETTINGS = (
    "maxTime", "timeStep", "stepReprise", "dtpycte", "speciesChargeGeometry",
    "fixpH", "intermediateOutput", "operatorSplitting", "chemModule", "trsptModule",
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
    "supplementarySolution", "surfaceCounterIons",
    
    "attributes", "firstIC", "primarySpeciesAq", "primarySpeciesPha",
    "primarySpeciesSurf", "speciationEch", "primarySpeciesPhantom", "nrThreads",
    "waterMass", "primarySpeciesEchSorbed", "couplingFormalism",
    "transportedSpecies", "inputVariableOrchestra", "outputVariableOrchestra",
    
    "comsolComp", "comsolIntFonction", "comsolStudy", "comsolVTU",
    "comsolCore", "outputComsol",

    "velocity", "advection", "FickDiffusion", "porousTransport", "ADE",
    "dispersivity", "firstBoundary", "secondBoundary", "boundaryConditions",
    "porosity", "area", "diffCoeff", "meshing",

)

_settings = {}
_current_example = None


def _make_setter(_name):
    def _setter(value):
        _settings[_name] = value
    _setter.__name__ = _name
    _setter.__qualname__ = f"piccts.{_name}"
    _setter.__doc__ = f"Set piccts' '{_name}' setting to `value` for the next piccts.run()."
    return _setter


for _name in _RECOGNIZED_SETTINGS:
    globals()[_name] = _make_setter(_name)
del _name, _make_setter


def _set_with_variant(setting_name, variants_attr_name, value):
    """
    Fixe _settings[setting_name] = value. Si un exemple est chargé via
    load() et déclare `variants_attr_name` (ex. "CHEM_VARIANTS",
    "TRSPT_VARIANTS") avec une variante pour `value`, fusionne aussi ces
    réglages par-dessus.
    """
    _settings[setting_name] = value
    variants = getattr(_current_example, variants_attr_name, None)
    if variants and value in variants:
        _settings.update(variants[value])


def chemModule(value):
    """Set 'chemModule', solver-aware si un exemple est chargé (CHEM_VARIANTS)."""
    _set_with_variant("chemModule", "CHEM_VARIANTS", value)


def trsptModule(value):
    """Set 'trsptModule', solver-aware si un exemple est chargé (TRSPT_VARIANTS)."""
    _set_with_variant("trsptModule", "TRSPT_VARIANTS", value)

# def chemModule(value):
#     """
#     Set piccts' 'chemModule' setting to `value`.

#     Overrides the generic auto-generated setter: if an example was loaded
#     via piccts.load(...) and it declares a CHEM_VARIANTS dict with a
#     variant for `value` (e.g. "orchestra"), those settings are merged on
#     top of the current ones too — so switching chemistry solver also
#     swaps whatever depends on it (chemPath, initialConditions, solver-
#     specific species settings, ...) instead of silently keeping the
#     previous solver's values. If no such variant exists (no example
#     loaded, or the example has nothing declared for `value`), this just
#     behaves like the plain setter.
#     """
#     _settings["chemModule"] = value
#     variants = getattr(_current_example, "CHEM_VARIANTS", None)
#     if variants and value in variants:
#         _settings.update(variants[value])


def reset():
    """Clear every setting declared so far via piccts.<name>(value)."""
    _settings.clear()
    global _current_example
    _current_example = None


def run(config=None):
    """
    Run a PICCTS coupling.

    - piccts.run() : uses whatever was declared via the piccts.<name>(value)
      functions (e.g. piccts.dispersivity(0.002)). Settings are cleared
      after the run. If none were declared at all, falls back to reading
      the calling script's own globals (older style).
    - piccts.run(config) : advanced use — pass an explicit config object
      or dict directly, bypassing the setting functions.
    """
    # Guard against multiprocessing's Windows 'spawn' start method (used by
    # extractDB's ProcessPoolExecutor for PhreeqC). Each worker boots a
    # fresh interpreter that re-executes the calling script's top-level
    # code (via runpy) to reconstruct __main__ — which re-triggers an
    # unguarded `pycte.run()` call at the bottom of the script, before
    # sys.modules['__main__'] actually holds the script's real settings.
    # By the time that re-execution happens, multiprocessing has already
    # renamed the current process away from "MainProcess", so we use that
    # to detect we're not the real entry point and no-op instead of
    # running (and crashing on missing settings). This makes the
    # `if __name__ == "__main__":` guard unnecessary for callers.
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
            _settings.clear()
            global _current_example
            _current_example = None

    return engine.main(sys.modules['__main__'])


def examples():
    """List the names of bundled examples available to piccts.load()."""
    return sorted(m.name for m in pkgutil.iter_modules(_examples_pkg.__path__))


def load(name):
    """
    Load a bundled example as the current config for the next
    piccts.run(), e.g.:

        piccts.load("CationExchange")
        piccts.run()

    Clears any settings declared via piccts.<name>(value) first, then
    loads the example. You can still override individual settings
    afterwards, e.g. piccts.load("CationExchange"); piccts.maxTime(1000).
    """
    try:
        mod = importlib.import_module(f".{name}", package=_examples_pkg.__name__)
    except ModuleNotFoundError:
        raise ValueError(
            f"Unknown example {name!r}. Available: {', '.join(examples())}"
        ) from None
    _settings.clear()
    _settings.update(mod.SETTINGS)
    global _current_example
    _current_example = mod


def run_from_file(path):
    """
    Load a config from a standalone .py file (old style: a script that
    just defines chemModule/trsptModule/maxTime/... as globals) and run
    it. Useful while migrating existing scripts to `import piccts`.
    """
    import importlib.util
    import os

    module_name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(module_name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return run(loaded)
