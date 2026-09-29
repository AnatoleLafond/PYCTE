from pathlib import Path

_DIR = Path(__file__).parent

_species = ["Aldicarb", "Oxime", "Sulfoxide", "Sulfoxide_ox", "Sulfone", "Sulfone_ox"]
_maxTime = 8 * 24
_timeStep = _maxTime / 100

SETTINGS = dict(
    description="Aldicarb infiltration in an unsaturated soil (COMSOL : Richards equation and transport, 2D "
                "axisymmetric) with a first-order kinetic degradation network (chemistry module). The reference is "
                "the full COMSOL model.",
    timeUnit="h", maxTime=_maxTime, timeStep=_timeStep,
    geometry=2,
    chemModule="nativeKinetics", trsptModule="comsol",
    operatorSplitting="snia", firstStepEquilibrium=True,
    systemSpeciation=_species + ["p", "dl.theta_l", "time"],
    initialConditions=str(_DIR / "AldicarbInfiltrationIC.txt"),
    trsptPath=str(_DIR / "AldicarbInfiltration.mph"),
    comsolStudy="std2",
    outputComsol=["data1"],
    output={"transport": [round(_maxTime / _timeStep)]},
)

_kineticsPhreeqC = """
Aldi_ox
    -formula Oxime 1 Aldicarb -1
    -m0 1
    -tol 1e-12
Aldi_sulfox
    -formula Sulfoxide 1 Aldicarb -1
    -m0 1
    -tol 1e-12
Sulfox_sulfox
    -formula Sulfoxide_ox 1 Sulfoxide -1
    -m0 1
    -tol 1e-12
Sulfox_sulfone
    -formula Sulfone 1 Sulfoxide -1
    -m0 1
    -tol 1e-12
Sulfone_sulfone
    -formula Sulfone_ox 1 Sulfone -1
    -m0 1
    -tol 1e-12
"""

CHEM_VARIANTS = {
    "phreeqc": dict(
        chemPath=str(_DIR / "AldicarbInfiltration.dat"),
        kinetics=_kineticsPhreeqC,
        kineticsSpecies=[],
        cutoffAq=1e-99,
        crossDependencies={"transport": ["p", "dl.theta_l", "time"]},
    ),
    "orchestra": dict(
        chemPath=str(_DIR / "chemistry1Kin.inp"),
        couplingFormalism="tot",
        primarySpeciesAq=_species,
        inputVariableOrchestra=[f"{s}.c0" for s in _species] + ["time"],
        outputVariableOrchestra=[f"{s}.ct" for s in _species],
        crossDependencies={"transport": ["p", "dl.theta_l", "time"], "speciation": ["time"]},
    ),
    "nativeKinetics": dict(
        kineticReactions=[
            {"rate_law": [("Aldicarb", 1)], "products": [("Sulfoxide", 1.0)], "k": 0.36 / 24},
            {"rate_law": [("Sulfoxide", 1)], "products": [("Sulfone", 1.0)], "k": 0.024 / 24},
            {"rate_law": [("Aldicarb", 1)], "products": [("Oxime", 1.0)], "k": 0.2 / 24},
            {"rate_law": [("Sulfoxide", 1)], "products": [("Sulfoxide_ox", 1.0)], "k": 0.01 / 24},
            {"rate_law": [("Sulfone", 1)], "products": [("Sulfone_ox", 1.0)], "k": 0.0524 / 24},
        ],
        crossDependencies={"transport": ["p", "dl.theta_l", "time"]},
    ),
}

SETTINGS.update(CHEM_VARIANTS["nativeKinetics"])
