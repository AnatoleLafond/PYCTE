"""
pycte — Python Interface Coupling generiC Transport and Speciation.

Public API
----------
One dedicated function per setting pycte actually reads (generated
from every `getattr(config, 'name', default)` call in engine.py — see
_RECOGNIZED_SETTINGS below). Call the ones you need, in any order,
before pycte.run():

    import pycte

    pycte.chemModule("phreeqc")
    pycte.trsptModule("nativeTransport")
    pycte.maxTime(72000)
    pycte.timeStep(720)
    pycte.dispersivity(0.002)
    pycte.velocity(0.002 / 720)

    if __name__ == "__main__":
        pycte.run()

Only these functions exist on the `pycte` module — nothing else. A
typo (pycte.dispersivty(...)) fails immediately with AttributeError,
instead of silently being ignored. A setting you never call keeps its
documented default (same default as when it was an absent attribute on
an old-style input script).

run()                : run using whatever was declared via the setting
                       functions above. If none were called at all,
                       falls back to reading the calling script's own
                       globals (the older `import pycte; ...;
                       pycte.run()` style, kept for scripts that
                       define chemModule/maxTime/... directly).

run(config)           : advanced/escape hatch — pass an explicit config
                       object or dict directly, bypassing the setting
                       functions entirely.

reset()               : clear every setting declared so far, without
                       running (rarely needed — run() already clears
                       after each call).

run_from_file(path)   : legacy helper. Loads a standalone .py config
                       script (old copy/paste style) and runs it.
"""

import sys
import types

from . import engine

# Every keyword pycte.engine.main() actually reads, i.e. every name
# that appears as getattr(config, 'name', default) in engine.py.
# Keep this in sync with engine.py if settings are added/removed there.
_RECOGNIZED_SETTINGS = (
    "maxTime", "timeStep", "stepReprise", "dtpycte", "speciesChargeGeometry",
    "fixpH", "intermediateOutput", "operatorSplitting", "chemModule", "trsptModule",
    "system", "PIDnbr", "PIDextract", "systemSpeciation", "timeUnit", "geometry",
    "firstStepEquilibrium", "nonTrivialDecomposition", "preliminarEquilibrium",
    "PIDchem", "PIDtrspt", "MultiCompoundTransport", "initialConditions",
    "crossDependencies", "description",
    "chemPath", "trsptPath", "phases",
    
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


def _make_setter(_name):
    def _setter(value):
        _settings[_name] = value
    _setter.__name__ = _name
    _setter.__qualname__ = f"pycte.{_name}"
    _setter.__doc__ = f"Set pycte' '{_name}' setting to `value` for the next pycte.run()."
    return _setter


for _name in _RECOGNIZED_SETTINGS:
    globals()[_name] = _make_setter(_name)
del _name, _make_setter


def reset():
    _settings.clear()


def run(config=None):
    """
    - pycte.run() : pycte.<name>(value)
or
    - pycte.run(config) 
    """
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

    return engine.main(sys.modules['__main__'])


def run_from_file(path):
    """
    Load a config from a standalone .py file (old style: a script that
    just defines chemModule/trsptModule/maxTime/... as globals) and run
    it. Useful while migrating existing scripts to `import pycte`.
    """
    import importlib.util
    import os

    module_name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(module_name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return run(loaded)