"""Cation exchange (ex. 11 of PhreeqC, Appelo et al. 2013): CaCl2 displaces Na, K and NO3 from a 1D column.

    pycte.load("CationExchange")          # PhreeqC + nativeTransport
    pycte.chemistryModule("orchestra")    # or "nativeSpeciation", "phreeqcrm"
    pycte.transportModule("comsol")       # or "pflotran"
"""
from pathlib import Path

_DIR = Path(__file__).parent

_speciesPhreeqC = [
    "CaX2", "KX", "NaX", "NH4X", "Ca+2", "CaOH+", "Cl-", "H+", "H2", "K+", "KOH",
    "N2", "Na+", "NaOH", "NH3", "NH4+", "NO2-", "NO3-", "O2", "OH-"]
_exchangerPhreeqC = ["CaX2", "KX", "NaX", "NH4X"]

_speciesOrchestra = [
    "Exch_X2-Ca", "Exch_X-K", "Exch_X-Na", "Exch_X-NH4", "Ca+2", "CaOH+", "Cl-", "H+", "H2", "K+", "KOH",
    "N2", "Na+", "NaOH", "NH3", "NH4+", "NO2-", "NO3-", "O2", "OH-"]
# sorbed species -> ORCHESTRA output variable (sorbed amount of the element ; Exch_X-NH4 not in the database)
_exchangerOrchestra = {"Exch_X2-Ca": "Ca.solid", "Exch_X-K": "K.solid", "Exch_X-Na": "Na.solid",
                       "Exch_X-NH4": "Exch_X-NH4.con"}

_elements = ["Ca", "Cl", "K", "N", "Na"]

SETTINGS = dict(
    description="Cation exchange test case from Appelo et al. 2013",
    chemistryModule="phreeqc",
    transportModule="nativeTransport",
    operatorSplitting="snia",
    adeSolver = 'explicit',
    maxTime=720 * 100,
    timeStep=720,
    firstStepEquilibrium=True,
    speciationThreads=1,
    transportThreads=1,
    output={"stepEnd": list(range(1, 100 + 1))},
)


def _chemistry(chemistryModule):
    """Settings of the speciation solver and its aqueous species."""
    phreeqcFiles = dict(chemPath=str(_DIR / "phreeqc.dat"), initialConditions=str(_DIR / "phreeqcIC.txt"),
                        coupledVariables=_speciesPhreeqC)
    aqueousPhreeqC = [s for s in _speciesPhreeqC if s not in _exchangerPhreeqC]
    if chemistryModule in ("PhreeqC", "PhreeqcRM"):
        return phreeqcFiles, aqueousPhreeqC
    if chemistryModule == "nativeSpeciation":
        # initial conditions in species : MultiCompoundTransport=False (True would expect element totals)
        return dict(phreeqcFiles, MultiCompoundTransport=False,
                    chemistry={"elements": _elements, "exchanger": "X", "cec": 0,
                               "chemPath": str(_DIR / "phreeqc.dat")}), aqueousPhreeqC
    if chemistryModule == "ORCHESTRA":
        return dict(
            chemPath=str(_DIR / "chemistry1.inp"),
            initialConditions=str(_DIR / "orchestraIC.txt"),
            coupledVariables=_speciesOrchestra,
            primarySpeciesAq=_elements,
            inputVariableOrchestra=[f"{e}.tot" for e in _elements],
            outputVariableOrchestra=[_exchangerOrchestra.get(s, f"{s}.con") for s in _speciesOrchestra],
        ), [s for s in _speciesOrchestra if s not in _exchangerOrchestra]
    raise ValueError(f"CationExchange is available with chemistryModule 'phreeqc', 'phreeqcrm', 'orchestra' or 'nativeSpeciation', "
                     f"not {chemistryModule!r}")


def variant(chemistryModule, transportModule):
    """Settings of the solver pair (called by pycte at every solver change)."""
    settings, aqueous = _chemistry(chemistryModule)
    if transportModule == "nativeTransport":
        firstBoundary = {s: 0 for s in settings["coupledVariables"]}
        firstBoundary.update({"Ca+2": 0.6e-3, "Cl-": 1.2e-3})
        settings.update(dispersivity=0.002, velocity=0.002 / 720, ADE=True, transportedSpecies=aqueous,
                        firstBoundary=firstBoundary, boundaryConditions=["constant", "closed"],
                        adeSolver="explicit", advectionScheme="upwind", adeSubSteps=1)
    elif transportModule == "COMSOL":
        settings.update(trsptPath=str(_DIR / "comsol.mph"))
    elif transportModule == "PFLOTRAN":
        settings.update(trsptPath=str(_DIR / ("pflotranOrchestra.in" if chemistryModule == "ORCHESTRA" else "pflotran.in")))
    return settings
