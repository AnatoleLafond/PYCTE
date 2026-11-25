"""Calcite dissolution / dolomite precipitation by a MgCl2 solution in a 1D column (Azad et al. 2016).

    pycte.load("CalciteDolomite")          # xGEMS + nativeTransport
    pycte.chemistryModule("orchestra")     # same GEMS database, converted for ORCHESTRA (orchestra/ folder)
    pycte.chemistryModule("phreeqc")       # same GEMS database, converted for PhreeqC (phreeqc/ folder)
    pycte.transportModule("comsol")        # or "pflotran" (xGEMS only)

The ORCHESTRA and PhreeqC files (orchestra/, phreeqc/) are generated from the GEMS database Resources/CalciteIC
by gems2orchestra.py (logK from the GEMS G0 at 25 C, Davies, all elements on mass balance: pH and pe are results,
as in GEMS). Every solver converges to the relative mass balance tolerance of GEMS (pa_DHB = 1e-14).
"""
import re
from pathlib import Path

_DIR = Path(__file__).parent
_ORCH = _DIR / "orchestra"

_speciesGems = [
    "Ca(CO3)@", "Ca(HCO3)+", "Ca+2", "CaOH+", "Mg(CO3)@", "Mg(HCO3)+",
    "Mg+2", "MgOH+", "CO2@", "CO3-2", "HCO3-", "CH4@", "ClO4-", "Cl-",
    "H2@", "O2@", "OH-", "H+", "H2O@", "CO2", "CH4", "H2", "O2", "Cal",
    "Dis-Dol", "Sn",
]

# GEMS -> ORCHESTRA names (orchestra/species.csv). Not coupled with ORCHESTRA: the gases (absent at equilibrium,
# no gas phase), the water (not a mass balance) and Sn (inert component)
_gemsToOrchestra = {
    "Ca(CO3)@": "Ca[CO3]", "Ca(HCO3)+": "Ca[HCO3]+", "Ca+2": "Ca+2", "CaOH+": "CaOH+",
    "Mg(CO3)@": "Mg[CO3]", "Mg(HCO3)+": "Mg[HCO3]+", "Mg+2": "Mg+2", "MgOH+": "MgOH+",
    "CO2@": "CO2", "CO3-2": "CO3-2", "HCO3-": "HCO3-", "CH4@": "CH4", "ClO4-": "ClO4-", "Cl-": "Cl-",
    "H2@": "H2", "O2@": "O2", "OH-": "OH-", "H+": "H+", "Cal": "Calcite[s]", "Dis-Dol": "Dolomite-dis[s]",
}
_speciesOrchestra = list(_gemsToOrchestra.values())

# GEMS -> PHREEQC names (phreeqc/calciteDolomite.dat): '@' removed, phases named after the GEMS phase. The water is
# coupled (H and O balances of SOLUTION_MODIFY), the gases and Sn are not
_gemsToPhreeqC = {s: s.replace("@", "") for s in _speciesGems[:19]}
_gemsToPhreeqC.update({"Cal": "Calcite", "Dis-Dol": "Dolomite-dis"})
_speciesPhreeqC = list(_gemsToPhreeqC.values())
_phasesPhreeqC = ["Calcite", "Dolomite-dis"]

# convergence tolerance of every solver = relative mass balance tolerance of GEMS (pa_DHB of CalciteIC-ipm.dat)
_TOLERANCE = 1e-14
_mineralsOrchestra = ["Calcite[s]", "Dolomite-dis[s]"]
_elementsOrchestra = ["C", "Ca", "Cl", "H", "Mg", "E"]     # H, E: proton and electron excess (may be < 0)

_timeStep = 5e-3 / 9.375e-6

SETTINGS = dict(
    description="Calcite dolomite test case from Azad and al. 2016",
    timeUnit="s", maxTime=21e3, timeStep=_timeStep,
    chemistryModule="gems", transportModule="nativeTransport",
    operatorSplitting="snia", firstStepEquilibrium=True,
    output={'stepEnd': list(range(1, int(21e3 / _timeStep) + 1))},
)

# injected MgCl2 solution (GEMS names)
_boundaryGems = {s: 0 for s in _speciesGems}
_boundaryGems.update({'Ca(CO3)@': 3.49122e-17, 'Ca(HCO3)+': 7.26344e-15, 'Ca+2': 1e-8, 'CaOH+': 1.09822e-14,
                      'Mg(CO3)@': 3.97585e-12, 'Mg(HCO3)+': 1.33202e-10, 'Mg+2': (0.00199995182445532 / 2),
                      'MgOH+': 4.80383669042909e-08, 'CO2@': 1.93e-9, 'CO3-2': 4.03182994984226e-12,
                      'HCO3-': 7.92277169973322e-09, 'Cl-': 2e-3, 'H+': 1.28190667315784e-07,
                      'H2O@': 55.5083731906404})


def _charge(species):
    """Charge from the name: 'Ca(HCO3)+' -> 1, 'CO3-2' -> -2, 'CO2@' -> 0."""
    m = re.search(r"([+-])(\d*)$", species)
    return 0 if m is None else (1 if m.group(1) == "+" else -1) * int(m.group(2) or 1)


# electroneutrality: the solution above carries +1.4e-7 eq/kgw, balanced with OH- (PhreeqC and GEMS would neutralise
# it, ORCHESTRA would keep it)
_boundaryGems['OH-'] = sum(_charge(s) * v for s, v in _boundaryGems.items() if s != 'OH-')


def _chemistry(chemistryModule):
    """Settings of the speciation solver: database, initial conditions, coupled species, injected solution."""
    if chemistryModule == "xGEMS":
        return dict(chemPath=str(_DIR / "Resources" / "CalciteIC" / "CalciteIC-dat.lst"),
                    initialConditions=str(_DIR / "ic_spc_1d.txt"),
                    coupledVariables=_speciesGems), dict(_boundaryGems)
    if chemistryModule == "ORCHESTRA":
        return dict(
            chemPath=str(_ORCH / "calciteDolomite.inp"),
            initialConditions=str(_ORCH / "ic_orchestra_1d.txt"),
            coupledVariables=_speciesOrchestra,
            couplingFormalism="tot",
            primarySpeciesAq=_elementsOrchestra,
            inputVariableOrchestra=[f"{e}.tot" for e in _elementsOrchestra],
            outputVariableOrchestra=[f"{s}.min" if s in _mineralsOrchestra else f"{s}.con" for s in _speciesOrchestra],
            transportedSpecies=[s for s in _speciesOrchestra if s not in _mineralsOrchestra],
        ), {_gemsToOrchestra[s]: v for s, v in _boundaryGems.items() if s in _gemsToOrchestra}
    if chemistryModule in ("PhreeqC", "PhreeqcRM"):
        return dict(
            chemPath=str(_DIR / "phreeqc" / "calciteDolomite.dat"),
            initialConditions=str(_DIR / "phreeqc" / "ic_phreeqc_1d.txt"),
            coupledVariables=_speciesPhreeqC,
            phases=_phasesPhreeqC,
            transportedSpecies=[s for s in _speciesPhreeqC if s not in _phasesPhreeqC],
            solMod=True,      # SOLUTION_MODIFY with the H, O totals and -cb 0 : pH and pe from the mass balances
            knobs={"convergence_tolerance": _TOLERANCE},
        ), {_gemsToPhreeqC[s]: v for s, v in _boundaryGems.items() if s in _gemsToPhreeqC}
    raise ValueError(f"CalciteDolomite is available with chemistryModule 'gems', 'orchestra' or 'phreeqc', "
                     f"not {chemistryModule!r}")


def variant(chemistryModule, transportModule):
    """Settings of the solver pair (called by pycte at every solver change)."""
    settings, boundary = _chemistry(chemistryModule)
    if transportModule == "nativeTransport":
        settings.update(geometry=1, dispersivity=0.0067, velocity=9.375e-6, ADE=True, firstBoundary=boundary)
    elif chemistryModule == "ORCHESTRA":
        # calciteDolomite.mph and pflotran.in are written with the GEMS species
        raise ValueError(f"CalciteDolomite with ORCHESTRA is available with transportModule 'nativeTransport' only, "
                         f"not {transportModule!r}")
    elif transportModule == "COMSOL":
        settings.update(trsptPath=str(_DIR / "calciteDolomite.mph"))
    elif transportModule == "PFLOTRAN":
        settings.update(MultiCompoundTransport=True, geometry=3, initialConditions=str(_DIR / "ic_spc.txt"),
                        trsptPath=str(_DIR / "pflotran.in"))
    return settings
