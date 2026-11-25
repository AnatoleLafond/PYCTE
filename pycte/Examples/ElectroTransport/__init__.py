"""NaCl electromigration with water electrolysis (current1.phr of Appelo, VE4 of Lopez-Vizcaino et al. 2026).

    pycte.load("ElectroTransport")"""
from pathlib import Path

import numpy as np

_DIR = Path(__file__).parent

_species = ["Na+", "Cl-", "H+", "OH-"]
_maxTime = 33000
_timeStep = 100

SETTINGS = dict(
    description="Electromigration of NaCl with water electrolysis at the electrodes (Appelo current1.phr)",
    chemistryModule="none",
    transportModule="nativeTransport",
    operatorSplitting="snia",
    timeUnit="min",
    maxTime=_maxTime,
    timeStep=_timeStep,
    porosity=0.1,
    tortuosity=0.1,
    area=np.pi * (0.1 / 2) ** 2,
    coupledVariables=_species,
    transportedSpecies=_species,
    initialConditions=str(_DIR / "ElectroTransportIC.txt"),
    speciesValence={"Na+": 1, "Cl-": -1, "H+": 1, "OH-": -1},
    speciesDiffCoeff={s: 1.0e-9 for s in _species},
    boundaryConditions=["electrode", "electrode"],
    firstBoundary={},
    secondBoundary={},
    electrodeVolume=[1.0e-3, 1.0e-3],
    electrodeKinetics="faraday",
    electrodeReactions={
        "anode": [{"name": "O2 evolution", "n": 4, "stoich": {"H+": 4}}],
        "cathode": [{"name": "H2 evolution", "n": 2, "stoich": {"OH-": 2}}],
    },
    waterEquilibrium=True,
    Kw=1.0e-14,
    temperature=298.15,
    NernstPlanck=True,
    imposedVoltage=3.0,
    npImplicitMaxChange=0.1,
    maxSubCycling=200000,
    speciationThreads=1,
    transportThreads=1,
    output={"transport": list(range(1, round(_maxTime / _timeStep) + 1))},
)


def variant(chemistryModule, transportModule):
    if chemistryModule != "none" or transportModule != "nativeTransport":
        raise ValueError(f"ElectroTransport is available with chemistryModule 'none' and transportModule 'nativeTransport' "
                         f"only, not {chemistryModule!r} / {transportModule!r}")
    return {}
