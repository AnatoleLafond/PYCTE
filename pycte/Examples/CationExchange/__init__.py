"""
Cation exchange test case from Appelo et al. 2013 (ex. 11 of PhreeqC) : a CaCl2 solution displaces Na, K and NO3
from a 1D column with a cation exchanger X, 100 steps of 720 s.

    pycte.load("CationExchange")          # PhreeqC + nativeTransport (defaut)
    pycte.chemModule("orchestra")         # ou "nativeSpeciation"
    pycte.trsptModule("comsol")           # ou "pflotran"

Reglages propres au solveur de speciation :
    PhreeqC           19 especes (noms PhreeqC), base phreeqc.dat, conditions initiales phreeqcIC.txt
    ORCHESTRA         les memes 19 especes (noms de chemistry1.inp : Exch_X2-Ca, ...), base chemistry1.inp,
                      conditions initiales orchestraIC.txt (memes totaux par element que phreeqcIC.txt, tout en
                      solution : l'echangeur est equilibre par ORCHESTRA au premier pas). chemistry1.inp n'a pas
                      d'azote reduit : Exch_X-NH4, N2, NH3, NH4+ et NO2- y sont absents, ORCHESTRA les rend nuls
                      (warning "not in ORCHESTRA database") ; tout l'azote est en NO3-
    nativeSpeciation  especes PhreeqC, base phreeqc.dat, conditions initiales phreeqcIC.txt (en especes :
                      MultiCompoundTransport=False)
et au solveur de transport :
    nativeTransport   seules les especes aqueuses sont transportees ; x = 0 : CaCl2 (Ca+2 0.6 mM, Cl- 1.2 mM)
    COMSOL            comsol.mph
    PFLOTRAN          pflotran.in (PhreeqC) ou pflotranOrchestra.in (ORCHESTRA)
"""
from pathlib import Path

_DIR = Path(__file__).parent

_speciesPhreeqC = [
    "CaX2", "KX", "NaX", "NH4X", "Ca+2", "CaOH+", "Cl-", "H+", "H2", "K+",
    "N2", "Na+", "NaOH", "NH3", "NH4+", "NO2-", "NO3-", "O2", "OH-"]
_exchangerPhreeqC = ["CaX2", "KX", "NaX", "NH4X"]

_speciesOrchestra = [
    "Exch_X2-Ca", "Exch_X-K", "Exch_X-Na", "Exch_X-NH4", "Ca+2", "CaOH+", "Cl-", "H+", "H2", "K+",
    "N2", "Na+", "NaOH", "NH3", "NH4+", "NO2-", "NO3-", "O2", "OH-"]
# espece sorbee -> variable de sortie ORCHESTRA (quantite sorbee de l'element ; Exch_X-NH4 absent de la base)
_exchangerOrchestra = {"Exch_X2-Ca": "Ca.solid", "Exch_X-K": "K.solid", "Exch_X-Na": "Na.solid",
                       "Exch_X-NH4": "Exch_X-NH4.con"}

_elements = ["Ca", "Cl", "K", "N", "Na"]

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
    output={"coupling": list(range(1, 100 + 1))},
)


def _chemistry(chemModule):
    """Reglages du solveur de speciation, et ses especes aqueuses (transportees par nativeTransport)."""
    phreeqcFiles = dict(chemPath=str(_DIR / "phreeqc.dat"), initialConditions=str(_DIR / "phreeqcIC.txt"),
                        systemSpeciation=_speciesPhreeqC)
    aqueousPhreeqC = [s for s in _speciesPhreeqC if s not in _exchangerPhreeqC]
    if chemModule == "PhreeqC":
        return phreeqcFiles, aqueousPhreeqC
    if chemModule == "nativeSpeciation":
        # conditions initiales en especes : MultiCompoundTransport=False (True attendrait des totaux par element)
        return dict(phreeqcFiles, MultiCompoundTransport=False,
                    chemistry={"elements": _elements, "exchanger": "X", "cec": 0,
                               "chemPath": str(_DIR / "phreeqc.dat")}), aqueousPhreeqC
    if chemModule == "ORCHESTRA":
        return dict(
            chemPath=str(_DIR / "chemistry1.inp"),
            initialConditions=str(_DIR / "orchestraIC.txt"),
            systemSpeciation=_speciesOrchestra,
            primarySpeciesAq=_elements,
            inputVariableOrchestra=[f"{e}.tot" for e in _elements],
            outputVariableOrchestra=[_exchangerOrchestra.get(s, f"{s}.con") for s in _speciesOrchestra],
            nrThreads=1,
        ), [s for s in _speciesOrchestra if s not in _exchangerOrchestra]
    raise ValueError(f"CationExchange is available with chemModule 'phreeqc', 'orchestra' or 'nativeSpeciation', "
                     f"not {chemModule!r}")


def variant(chemModule, trsptModule):
    """Reglages propres au couple de solveurs (appelee par pycte a chaque changement de solveur)."""
    settings, aqueous = _chemistry(chemModule)
    if trsptModule == "nativeTransport":
        firstBoundary = {s: 0 for s in settings["systemSpeciation"]}
        firstBoundary.update({"Ca+2": 0.6e-3, "Cl-": 1.2e-3})
        settings.update(dispersivity=0.002, velocity=0.002 / 720, ADE=True, transportedSpecies=aqueous,
                        firstBoundary=firstBoundary, boundaryConditions=["constant", "closed"])
    elif trsptModule == "COMSOL":
        settings.update(trsptPath=str(_DIR / "comsol.mph"))
    elif trsptModule == "PFLOTRAN":
        settings.update(trsptPath=str(_DIR / ("pflotranOrchestra.in" if chemModule == "ORCHESTRA" else "pflotran.in")))
    return settings
