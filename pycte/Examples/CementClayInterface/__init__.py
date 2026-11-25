
"""Cement / clay interface benchmark: 15 cells of cement paste, 75 cells of clay, diffusion only.

    pycte.load("CementClayInterface")    # PhreeqC, multi-compound transport
    pycte.MCT(False)                     # multi-species transport
    pycte.chemistryModule("orchestra")   # or "phreeqcrm", "nativeSpeciation"
    pycte.diffusion("NernstPlanck")      # Nernst-Planck instead of Fick (species transported, MCT ignored)
    pycte.transportModule("pflotran")    # PFLOTRAN (Fick) : pflotran_<chemistry>_<MCT|MS>.in, tracers of pflotranDB.txt
    pycte.transportModule("comsol")      # COMSOL (Fick) : comsol_<chemistry>_<MCT|MS>.mph

The cement water of CI_phreeqc.txt carries a charge imbalance of +0.0695 eq/kgw. The Nernst-Planck initial
conditions (CI_*_NP.txt) add that amount of OH- to the cement water and re-equilibrate it with its phases
(PhreeqC, pH 13.61 -> 13.66); the clay is unchanged. At zero current the charge of a cell is conserved, an
imbalance would stay in the cement.

Nernst-Planck: same porosities, De = porosity * tortuosity * Dw(species), Dw from phreeqc.dat (PHREEQC 3,
phreeqc_rates.dat lineage, 25 °C; 1e-9 m2/s, the PHREEQC default, for the species without -dw), tortuosity
= Fick pore coefficient / Dw(H2O), so that water diffuses as in the Fick case.
"""
import re
from pathlib import Path

import pandas as pd

_DIR = Path(__file__).parent

OPTIONS = {"MCT": True, "diffusion": "Fick"}

_porosity = [0.13] * 15 + [0.18] * 75

SETTINGS = dict(
    description="cement clay interface benchmark",
    chemistryModule="phreeqc",
    transportModule="nativeTransport",
    operatorSplitting="snia",
    timeUnit="y",
    maxTime=1e5,
    timeStep=1,
    firstStepEquilibrium=True,
    meshing=[(15, 0.3 / 15), (35, 1.5 / 35), (10, 1.0 / 10), (30, 7.5 / 30)],
    ADE=True,
    porosity=_porosity,
    diffCoeff=[3.38e-11] * 15 + [1.44e-10] * 75,
    boundaryConditions=["closed", "constant"],
    output={"stepEnd": [1e3, 1e4, 1e5]},
    speciationThreads=1,
    transportThreads=1,
)


_aqueousPhreeqC = [
    'OH-', 'H+', 'H2O', 'Al(OH)4-', 'NaAl(OH)4', 'Al(OH)3', 'Al(OH)2+', 'Al(OH)+2', 'AlH3SiO4+2', 'Al+3',
    'Al(SO4)+', 'CH4', 'CH3-', 'Na(CO3)-', 'CO3-2', 'CaCO3', 'HCO3-', 'Na(HCO3)', 'Ca(HCO3)+', 'Mg(CO3)', 'CO2',
    'Mg(HCO3)+', 'Sr(CO3)', 'Fe(OH)CO3', 'FeCO3', 'Fe(CO3)2-2', 'Sr(HCO3)+', 'Fe(CO3)3-3', 'CO', 'Ca(OH)+', 'Ca+2',
    'Ca(SO4)', 'Ca(H3SiO4)+', 'CaCl+', 'CaCl2', 'Ca(S2O3)', 'Cl-', 'KCl', 'NaCl', 'MgCl+', 'HCl', 'SrCl+', 'FeCl+',
    'FeCl+2', 'FeCl2+', 'FeCl3', 'Cl2', 'FeCl4-', 'ClO4-', 'Fe(OH)4-2', 'Fe(OH)3-', 'Fe(OH)2', 'Fe(OH)+', 'Fe+2',
    'Fe(SO4)', 'Fe(HS)+', 'Fe(HS)2', 'Fe(OH)4-', 'Fe(OH)3', 'Fe(OH)2+', 'Fe(OH)+2', 'Fe(H3SiO4)+2', 'Fe+3',
    'Fe(SO4)+', 'Fe(SO4)2-', 'Fe2(OH)2+4', 'Fe3(OH)4+5', 'FeS2O3+', 'H2', 'K+', 'K(OH)', 'K(SO4)-', 'Mg(OH)+',
    'Mg+2', 'Mg(SO4)', 'Mg(H3SiO4)+', 'Mg4(OH)4+4', 'Mg(S2O3)', 'Na+', 'Na(OH)', 'Na(SO4)-', 'Na(S2O3)-', 'O2',
    'HS-', 'S-2', 'H2S', 'S2-2', 'S3-2', 'S4-2', 'S5-2', 'S2O3-2', 'H(S2O3)-', 'Sr(S2O3)', 'H2(S2O3)', 'S2O4-2',
    'HS2O4-', 'H2S2O4', 'SO3-2', 'H(SO3)-', 'SO2', 'H2(SO3)', 'S2O5-2', 'S3O6-2', 'S4O6-2', 'S5O6-2', 'SO4-2',
    'H(SO4)-', 'Sr(SO4)', 'HSO5-', 'S2O8-2', 'H2(SiO4)-2', 'H3(SiO4)-', 'H4(SiO4)', 'Si2O3(OH)4-2', 'Si2O2(OH)5-',
    'Si3O6(OH)3-3', 'Si3O5(OH)5-3', 'Si4O7(OH)6-4', 'Si4O8(OH)4-4', 'Si4O6(OH)6-2', 'Si6O15-6', 'Sr+2', 'Sr(OH)+',
]

_phasesPhreeqC = [
    'Brucite', 'C3AH6', 'C4FH13', 'CSH0.8', 'CSH1.2', 'CSH1.6', 'Calcite', 'Celestite', 'Dolomite', 'Ettringite',
    'Ettringite-Fe', 'Ferrihydrite(am)', 'Hydrotalcite', 'Hydrotalcite-CO3', 'Magnetite', 'Monosulfate-Fe',
    'Portlandite', 'Pyrite', 'Pyrrhotite', 'Saponite-FeCa', 'SiO2(am)', 'Siderite', 'Stratlingite',
]


_secondBoundaryPhreeqCMCT = {
    'Al': 8.519998902e-08, 'C(+4)': 0.0040527660588813574, 'C(-4)': 3.36048e-17, 'Ca': 0.007713414966080056,
    'Cl': 0.041200000313349, 'Fe(+2)': 6.11181766428239e-05, 'Fe(+3)': 1.853659474032439e-09,
    'K': 0.0005109997648652, 'Mg': 0.00514973160860011, 'Na': 0.04010000029724816, 'S(+6)': 0.0111081534532578,
    'S(-2)': 1.2519090021629906e-10, 'S(+2)': 5.8479432234234366e-15, 'S(+3)': 2.841260640805699e-34,
    'S(+4)': 7.548770853000002e-15, 'Si': 0.00017999956704318576, 'Sr': 0.0002521539815104087,
    'H': 111.0215997343147, 'O': 55.56547191826152, 'chargebalance': -2.21778e-08,
}



_crossOrchestra = [
    'chargebalance', 'totvolume', 'A', 'poreD', 'dx', 'porosity', 'saturation', 'pH', 'pe',
]

_elementsOrchestra = [
    'Al', 'C', 'Ca', 'Cl', 'E', 'Fe', 'H', 'K', 'Mg', 'Na', 'S', 'Si', 'Sr',
]

_aqueousOrchestra = [
    'OH-', 'H+', 'H2O', 'Al[OH]4-', 'NaAl[OH]4', 'Al[OH]3', 'Al[OH]2+', 'Al[OH]+2', 'AlH3SiO4+2', 'Al+3',
    'Al[SO4]+', 'CH4', 'CH3-', 'Na[CO3]-', 'CO3-2', 'CaCO3', 'HCO3-', 'Na[HCO3]', 'Ca[HCO3]+', 'Mg[CO3]', 'CO2',
    'Mg[HCO3]+', 'Sr[CO3]', 'Fe[OH]CO3', 'FeCO3', 'Fe[CO3]2-2', 'Sr[HCO3]+', 'Fe[CO3]3-3', 'CO', 'Ca[OH]+', 'Ca+2',
    'Ca[SO4]', 'Ca[H3SiO4]+', 'CaCl+', 'CaCl2', 'Ca[S2O3]', 'Cl-', 'KCl', 'NaCl', 'MgCl+', 'HCl', 'SrCl+', 'FeCl+',
    'FeCl+2', 'FeCl2+', 'FeCl3', 'Cl2', 'FeCl4-', 'ClO4-', 'Fe[OH]4-2', 'Fe[OH]3-', 'Fe[OH]2', 'Fe[OH]+', 'Fe+2',
    'Fe[SO4]', 'Fe[HS]+', 'Fe[HS]2', 'Fe[OH]4-', 'Fe[OH]3', 'Fe[OH]2+', 'Fe[OH]+2', 'Fe[H3SiO4]+2', 'Fe+3',
    'Fe[SO4]+', 'Fe[SO4]2-', 'Fe2[OH]2+4', 'Fe3[OH]4+5', 'FeS2O3+', 'H2', 'K+', 'K[OH]', 'K[SO4]-', 'Mg[OH]+',
    'Mg+2', 'Mg[SO4]', 'Mg[H3SiO4]+', 'Mg4[OH]4+4', 'Mg[S2O3]', 'Na+', 'Na[OH]', 'Na[SO4]-', 'Na[S2O3]-', 'O2',
    'HS-', 'S-2', 'H2S', 'S2-2', 'S3-2', 'S4-2', 'S5-2', 'S2O3-2', 'H[S2O3]-', 'Sr[S2O3]', 'H2[S2O3]', 'S2O4-2',
    'HS2O4-', 'H2S2O4', 'SO3-2', 'H[SO3]-', 'SO2', 'H2[SO3]', 'S2O5-2', 'S3O6-2', 'S4O6-2', 'S5O6-2', 'SO4-2',
    'H[SO4]-', 'Sr[SO4]', 'HSO5-', 'S2O8-2', 'H2[SiO4]-2', 'H3[SiO4]-', 'H4[SiO4]', 'Si2O3[OH]4-2', 'Si2O2[OH]5-',
    'Si3O6[OH]3-3', 'Si3O5[OH]5-3', 'Si4O7[OH]6-4', 'Si4O8[OH]4-4', 'Si4O6[OH]6-2', 'Si6O15-6', 'Sr+2', 'Sr[OH]+',
]

_phasesOrchestra = [
    'Brucite[s]', 'C3AH6[s]', 'C4FH13[s]', 'CSH0_8[s]', 'CSH1_2[s]', 'CSH1_6[s]', 'Calcite[s]', 'Celestite[s]',
    'Dolomite[s]', 'Ettringite[s]', 'Ettringite-Fe[s]', 'Ferrihydrite[am][s]', 'Hydrotalcite[s]',
    'Hydrotalcite-CO3[s]', 'Magnetite[s]', 'Monosulfate-Fe[s]', 'Portlandite[s]', 'Pyrite[s]', 'Pyrrhotite[s]',
    'Saponite-FeCa[s]', 'SiO2[am][s]', 'Siderite[s]', 'Stratlingite[s]',
]


_secondBoundaryOrchestraMCT = {
    'Al': 8.52e-08, 'C': 0.0040530702, 'Ca': 0.00771359589, 'Cl': 0.0412, 'E': -8.48323151e-10,
    'Fe': 6.11213814e-05, 'H': 0.00455772125, 'K': 0.000511, 'Mg': 0.00514985287, 'Na': 0.0401, 'S': 0.0111081565,
    'Si': 0.00018, 'Sr': 0.000252156576,
}


_secondBoundaryOrchestraMS = {
    'OH-': 1.398e-07, 'H+': 1.129e-07, 'H2O': 55.51, 'Al[OH]4-': 6.984e-08, 'NaAl[OH]4': 2.916e-10,
    'Al[OH]3': 1.383e-08, 'Al[OH]2+': 1.102e-09, 'Al[OH]+2': 8.409e-11, 'AlH3SiO4+2': 5.724e-12, 'Al+3': 2.121e-12,
    'Al[SO4]+': 1.533e-12, 'CH4': 3.371e-17, 'CH3-': 0.0, 'Na[CO3]-': 9.85e-07, 'CO3-2': 3.42e-06,
    'CaCO3': 5.397e-06, 'HCO3-': 0.0033, 'Na[HCO3]': 4.459e-05, 'Ca[HCO3]+': 0.0001006, 'Mg[CO3]': 2.1e-06,
    'CO2': 0.000519, 'Mg[HCO3]+': 5.926e-05, 'Sr[CO3]': 6.797e-08, 'Fe[OH]CO3': 1.846e-09, 'FeCO3': 3.821e-06,
    'Fe[CO3]2-2': 7.836e-10, 'Sr[HCO3]+': 3.917e-06, 'Fe[CO3]3-3': 4.963e-14, 'CO': 1.062e-17,
    'Ca[OH]+': 5.621e-09, 'Ca+2': 0.00606, 'Ca[SO4]': 0.001486, 'Ca[H3SiO4]+': 9.178e-09, 'CaCl+': 5.053e-05,
    'CaCl2': 5.7e-07, 'Ca[S2O3]': 5.483e-17, 'Cl-': 0.04067, 'KCl': 3.958e-06, 'NaCl': 0.000309,
    'MgCl+': 0.0001491, 'HCl': 5.561e-10, 'SrCl+': 5.416e-06, 'FeCl+': 6.216e-06, 'FeCl+2': 6.611e-21,
    'FeCl2+': 5.387e-22, 'FeCl3': 8.584e-25, 'Cl2': 0.0, 'FeCl4-': 3.555e-28, 'ClO4-': 0.0, 'Fe[OH]4-2': 2.796e-23,
    'Fe[OH]3-': 3.818e-18, 'Fe[OH]2': 2.682e-12, 'Fe[OH]+': 1.202e-07, 'Fe+2': 3.824e-05, 'Fe[SO4]': 1.265e-05,
    'Fe[HS]+': 2.074e-11, 'Fe[HS]2': 1.029e-19, 'Fe[OH]4-': 1.191e-14, 'Fe[OH]3': 9.172e-13, 'Fe[OH]2+': 6.079e-12,
    'Fe[OH]+2': 4.859e-16, 'Fe[H3SiO4]+2': 2.881e-17, 'Fe+3': 1.943e-20, 'Fe[SO4]+': 1.688e-19,
    'Fe[SO4]2-': 4.821e-20, 'Fe2[OH]2+4': 4.463e-29, 'Fe3[OH]4+5': 3.548e-38, 'FeS2O3+': 2.538e-32,
    'H2': 2.006e-12, 'K+': 0.0004955, 'K[OH]': 1.492e-11, 'K[SO4]-': 1.15e-05, 'Mg[OH]+': 4.784e-08,
    'Mg+2': 0.004097, 'Mg[SO4]': 0.0008356, 'Mg[H3SiO4]+': 1.103e-08, 'Mg4[OH]4+4': 7.655e-22,
    'Mg[S2O3]': 1.094e-16, 'Na+': 0.03869, 'Na[OH]': 5.974e-10, 'Na[SO4]-': 0.001031, 'Na[S2O3]-': 1.623e-16,
    'O2': 0.0, 'HS-': 6.203e-11, 'S-2': 1.092e-20, 'H2S': 4.25e-11, 'S2-2': 7.146e-23, 'S3-2': 1.381e-27,
    'S4-2': 2.118e-33, 'S5-2': 1.959e-39, 'S2O3-2': 2.576e-15, 'H[S2O3]-': 6.101e-21, 'Sr[S2O3]': 8.693e-18,
    'H2[S2O3]': 1.703e-27, 'S2O4-2': 1.412e-34, 'HS2O4-': 2.015e-39, 'H2S2O4': 0.0, 'SO3-2': 4.518e-15,
    'H[SO3]-': 3.016e-15, 'SO2': 1.535e-20, 'H2[SO3]': 1.532e-20, 'S2O5-2': 4.665e-31, 'S3O6-2': 0.0,
    'S4O6-2': 1.317e-36, 'S5O6-2': 0.0, 'SO4-2': 0.007654, 'H[SO4]-': 3.299e-08, 'Sr[SO4]': 4.701e-05,
    'HSO5-': 0.0, 'S2O8-2': 0.0, 'H2[SiO4]-2': 4.112e-13, 'H3[SiO4]-': 3.703e-07, 'H4[SiO4]': 0.0001796,
    'Si2O3[OH]4-2': 4.14e-13, 'Si2O2[OH]5-': 1.484e-09, 'Si3O6[OH]3-3': 3.352e-19, 'Si3O5[OH]5-3': 2.658e-19,
    'Si4O7[OH]6-4': 5.386e-25, 'Si4O8[OH]4-4': 4.286e-25, 'Si4O6[OH]6-2': 8.783e-17, 'Si6O15-6': 4.419e-39,
    'Sr+2': 0.0001962, 'Sr[OH]+': 5.623e-11,
}


def _phreeqc(MCT):
    coupledVariables = ["chargebalance", "pH", "pe"] + _aqueousPhreeqC + _phasesPhreeqC
    transported = [s for s in coupledVariables if s not in _phasesPhreeqC + ["pH", "pe"]]
    initialConditions = _DIR / "CI_phreeqc.txt"
    if MCT:
        secondBoundary = _secondBoundaryPhreeqCMCT
    else:
        ci = pd.read_csv(initialConditions, sep=r"\s+", comment="%", header=None)
        ci.columns = ["x"] + coupledVariables
        secondBoundary = {s: float(ci[s].iloc[-1]) for s in transported}
    return dict(
        chemPath=str(_DIR / "PHREEQC_Davies_e-_ThermoChimie_v12a.dat"),
        initialConditions=str(initialConditions),
        coupledVariables=coupledVariables,
        phases=_phasesPhreeqC,
        transportedSpecies=transported,
        crossDependencies={"speciation": ["chargebalance", "pH", "pe"]},
        solMod=True,
        MultiCompoundTransport=MCT,
        secondBoundary=secondBoundary,
    )


def _orchestra(MCT):
    aqueous = _elementsOrchestra if MCT else _aqueousOrchestra
    settings = dict(
        chemPath=str(_DIR / "chemistry1.inp"),
        couplingFormalism="literbulk",
        inputVariableOrchestra=_crossOrchestra + [f"{e}.literbulk" for e in _elementsOrchestra],
        outputVariableOrchestra=(_crossOrchestra + [f"{s}.diss" if MCT else f"{s}.con" for s in aqueous]
                                 + [f"{p}.con" for p in _phasesOrchestra]),
        coupledVariables=_crossOrchestra + aqueous + _phasesOrchestra,
        colAq=aqueous,
        transportedSpecies=aqueous,
        crossDependencies={"speciation": _crossOrchestra},
        waterMass=_porosity,                # saturation = 1
    )
    if MCT:
        settings.update(initialConditions=str(_DIR / "CI_orchestra_multiCompound.txt"),
                        secondBoundary=_secondBoundaryOrchestraMCT)
    else:
        settings.update(initialConditions=str(_DIR / "CI_orchestra_multiSpecies.txt"),
                        secondBoundary=_secondBoundaryOrchestraMS)
    return settings


_elementsNative = ['Al', 'C', 'Ca', 'Cl', 'E', 'Fe', 'K', 'Mg', 'Na', 'S', 'Si', 'Sr']


def _native(MCT):
    settings = _phreeqc(False)
    transported = [s for s in _aqueousPhreeqC if s != 'H2O']
    settings.update(
        transportedSpecies=transported,
        secondBoundary={s: settings['secondBoundary'][s] for s in transported},
        chemistry={'elements': _elementsNative, 'minerals': _phasesPhreeqC, 'exchanger': None,
                   'diagnostics': {'pH': 'pH', 'pe': 'pe'}},
        MultiCompoundTransport=MCT,
    )
    for key in ('phases', 'crossDependencies', 'solMod') + (('transportedSpecies',) if MCT else ()):
        settings.pop(key)
    return settings


_DW_H2O = 2.299e-9
_DW_DEFAULT = 1e-9
_DW = {
    'OH-': 5.27e-09, 'H+': 9.31e-09, 'H2O': 2.299e-09, 'Al(OH)4-': 1.04e-09, 'Al+3': 5.59e-10, 'CH4': 1.85e-09,
    'Na(CO3)-': 5.85e-10, 'CO3-2': 9.55e-10, 'CaCO3': 4.46e-10, 'HCO3-': 1.18e-09, 'Na(HCO3)': 6.73e-10,
    'Ca(HCO3)+': 5.06e-10, 'Mg(CO3)': 4.21e-10, 'CO2': 1.92e-09, 'Mg(HCO3)+': 4.78e-10, 'Ca+2': 7.92e-10,
    'Ca(SO4)': 4.71e-10, 'Cl-': 2.033e-09, 'Fe+2': 7.19e-10, 'H2': 5.13e-09, 'K+': 1.96e-09, 'K(SO4)-': 9e-10,
    'Mg+2': 7.05e-10, 'Mg(SO4)': 4.45e-10, 'Na+': 1.33e-09, 'Na(SO4)-': 1.13e-09, 'O2': 2.35e-09, 'HS-': 1.73e-09,
    'S-2': 7.31e-10, 'H2S': 2.1e-09, 'SO4-2': 1.07e-09, 'H(SO4)-': 1.22e-09, 'H4(SiO4)': 1.1e-09, 'Sr+2': 7.94e-10,
}


def _valence(species):
    """Charge of a species from its name (trailing sign and number)."""
    sign = re.search(r'([+-])(\d*)$', species)
    return 0 if sign is None else int(sign.group(2) or 1) * (1 if sign.group(1) == '+' else -1)


def _nernstPlanck(settings, orchestra):
    """Nernst-Planck settings: Dw per species, tortuosity giving the Fick pore coefficient to water."""
    transported = [s for s in settings['transportedSpecies'] if s != 'chargebalance']
    name = (lambda s: s.replace('(', '[').replace(')', ']')) if orchestra else (lambda s: s)
    dw = {name(s): d for s, d in _DW.items()}
    return dict(
        initialConditions=str(_DIR / ("CI_orchestra_multiSpecies_NP.txt" if orchestra else "CI_phreeqc_NP.txt")),
        ADE=False,
        NernstPlanck=True,
        diffCoeff=_DW_DEFAULT,
        tortuosity=[d / _DW_H2O for d in SETTINGS['diffCoeff']],
        transportedSpecies=transported,
        speciesDiffCoeff={s: dw[s] for s in transported if s in dw},
        speciesValence={s: _valence(s) for s in transported},
        secondBoundary={s: v for s, v in settings['secondBoundary'].items() if s in transported},
    )


_TRANSPORT_INPUT = {"PhreeqC": "phreeqc", "PhreeqcRM": "phreeqc", "ORCHESTRA": "orchestra",
                   "nativeSpeciation": "nativeSpeciation"}


def variant(chemistryModule, transportModule, MCT, diffusion):
    nernstPlanck = str(diffusion).replace('-', '').replace(' ', '').lower() == "nernstplanck"
    if not nernstPlanck and str(diffusion).lower() != "fick":
        raise ValueError(f"diffusion is 'Fick' or 'NernstPlanck', not {diffusion!r}")
    if nernstPlanck and transportModule != "nativeTransport":
        raise ValueError(f"diffusion = 'NernstPlanck' is available with transportModule 'nativeTransport' only, "
                         f"not {transportModule!r}")
    MCT = MCT and not nernstPlanck
    mode = "multi-compound" if MCT else "multi-species"
    if chemistryModule in ("PhreeqC", "PhreeqcRM"):
        settings = _phreeqc(MCT)
    elif chemistryModule == "ORCHESTRA":
        settings = _orchestra(MCT)
    elif chemistryModule == "nativeSpeciation":
        settings = _native(MCT)
    else:
        raise ValueError(f"CementClayInterface is available with chemistryModule 'phreeqc', 'phreeqcrm', 'orchestra' or "
                         f"'nativeSpeciation', "
                         f"not {chemistryModule!r}")
    if nernstPlanck:
        settings.update(_nernstPlanck(settings, chemistryModule == "ORCHESTRA"))
        mode += " Nernst-Planck"
    if transportModule in ("PFLOTRAN", "COMSOL"):
        stem = f"{transportModule.lower()}_{_TRANSPORT_INPUT[chemistryModule]}_{'MCT' if MCT else 'MS'}"
        settings["trsptPath"] = str(_DIR / (stem + (".in" if transportModule == "PFLOTRAN" else ".mph")))
    settings["description"] = f"cement clay interface benchmark, {chemistryModule}, {mode} transport"
    return settings
