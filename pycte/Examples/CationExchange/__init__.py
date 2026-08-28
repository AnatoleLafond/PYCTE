
from pathlib import Path

_DIR = Path(__file__).parent

systemSpeciationPhqc = [
    "CaX2", "KX", "NaX", "NH4X", "Ca+2", "CaOH+", "Cl-", "H+", "H2", "K+",
    "N2", "Na+", "NaOH", "NH3", "NH4+", "NO2-", "NO3-", "O2", "OH-"]

systemSpeciationOrch = ["Exch_X2-Ca", "Exch_X-K", "Exch_X-Na", "Exch_X-NH4", "Ca+2", "CaOH+", "Cl-", "H+", "H2", "K+",
"N2", "Na+", "NaOH", "NH3", "NH4+", "NO2-", "NO3-", "O2", "OH-"]

SETTINGS = dict(
    description="Cation exchange test case from Appelo et al. 2013",
    chemModule="phreeqc",
    trsptModule="nativeTransport",
    operatorSplitting="snia",
    maxTime=720 * 100,
    timeStep=720,
    firstStepEquilibrium=True,
    PIDnbr=1,
    PIDextract=1,
    coupling = {'coupling' : list(range(1, int(72000/720) + 1))}
)


CHEM_VARIANTS = {
    "phreeqc": dict(
        chemPath=str(_DIR / "phreeqc.dat"),
        initialConditions=str(_DIR / "phreeqcIC.txt"),
        systemSpeciation = systemSpeciationPhqc,
    ),
    "orchestra": dict(
        chemPath=str(_DIR / "chemistry1.inp"),
        initialConditions=str(_DIR / "orchestraIC.txt"),
        systemSpeciation = systemSpeciationOrch,
        primarySpeciesAq=["Ca", "Cl", "K", "N", "Na"],
        transportedSpecies=["Ca+2", "CaOH+", "Cl-", "H+", "H2", "K+", "N2", "Na+", "NaOH", "NH3", "NH4+", "NO2-", "NO3-", "O2", "OH-"],
        inputVariableOrchestra = [f'{spc}.tot' for spc in ["Ca", "Cl", "K", "N", "Na"]],
        outputVariableOrchestra = [f'{spc}.con' if spc not in ["Exch_X2-Ca", "Exch_X-K", "Exch_X-Na"] else f'{spc}.solid' for spc in systemSpeciationOrch]
    ),
}

TRSPT_VARIANTS = {
    "nativeTransport": dict(
        dispersivity=0.002,
        velocity=0.002 / 720,
        ADE=True,
        firstBoundary={"Ca+2": 0.6e-3, "Cl-": 1.2e-3},
        boundaryConditions = ['constant', 'closed'],
    ),
    "comsol": dict(
        trsptPath=str(_DIR / "comsol.mph"),

    ),
}

# default setting
SETTINGS.update(CHEM_VARIANTS["phreeqc"])
SETTINGS.update(TRSPT_VARIANTS["nativeTransport"])