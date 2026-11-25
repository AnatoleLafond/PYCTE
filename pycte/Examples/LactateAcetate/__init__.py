"""Multicomponent electromigration (VE3 of Lopez-Vizcaino et al. 2026), implicit Nernst-Planck.

    pycte.load("LactateAcetate")          # MnO4- injected
    pycte.injectedAnion("Lactate-")       # lactate injected"""
from pathlib import Path

_DIR = Path(__file__).parent

OPTIONS = {"injectedAnion": "MnO4-"}

_species = ["K+", "Na+", "MnO4-", "Lactate-", "HCO3-"]
_maxTime = 3
_timeStep = 0.25

SETTINGS = dict(
    description="Time-dependent multicomponent electromigration (VE3, Lopez-Vizcaino et al.)",
    chemistryModule="none",
    transportModule="nativeTransport",
    operatorSplitting="snia",
    timeUnit="d",
    maxTime=_maxTime,
    timeStep=_timeStep,
    porosity=0.4,
    tortuosity=0.4,
    coupledVariables=_species,
    transportedSpecies=_species,
    initialConditions=str(_DIR / "LactateAcetateIC.txt"),
    speciesValence={"K+": 1, "Na+": 1, "MnO4-": -1, "Lactate-": -1, "HCO3-": -1},
    speciesDiffCoeff={"K+": 1.96e-9, "Na+": 1.33e-9, "MnO4-": 1.63e-9, "Lactate-": 1.03e-9, "HCO3-": 1.18e-9},
    secondBoundary={"K+": 0, "Na+": 10e-3, "MnO4-": 0, "Lactate-": 0, "HCO3-": 10e-3},
    boundaryConditions=["constant", "constant"],
    temperature=298.15,
    NernstPlanck=True,
    npSolver="implicit",
    imposedVoltage=100,
    speciationThreads=1,
    transportThreads=1,
    output={"transport": list(range(1, round(_maxTime / _timeStep) + 1))},
)


def variant(chemistryModule, transportModule, injectedAnion):
    if chemistryModule != "none" or transportModule != "nativeTransport":
        raise ValueError(f"LactateAcetate is available with chemistryModule 'none' and transportModule 'nativeTransport' "
                         f"only, not {chemistryModule!r} / {transportModule!r}")
    if injectedAnion not in ("MnO4-", "Lactate-"):
        raise ValueError(f"injectedAnion is 'MnO4-' or 'Lactate-', not {injectedAnion!r}")
    firstBoundary = {s: 0 for s in _species}
    firstBoundary.update({"K+": 10e-3, injectedAnion: 10e-3})
    return dict(firstBoundary=firstBoundary,
                description=f"Time-dependent multicomponent electromigration (VE3, Lopez-Vizcaino et al.), "
                            f"{injectedAnion} injected")
