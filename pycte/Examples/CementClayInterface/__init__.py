
from pathlib import Path

import pandas as pd

_DIR = Path(__file__).parent

# mots-cles propres a cet exemple, avec leur valeur par defaut : pycte.MCT(True | False)
OPTIONS = {"MCT": True}

_porosity = [0.13] * 15 + [0.18] * 75       # beton (0.3 m) puis argilite

SETTINGS = dict(
    description="cement clay interface benchmark",
    chemModule="phreeqc",
    trsptModule="nativeTransport",
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
    output={"coupling": [1e3, 1e4, 1e5]},
    PIDnbr=1,
    PIDextract=1,
)

# ---- PhreeqC (noms de la base ThermoChimie PhreeqC)

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

# multi-composes : totaux par etat redox, H, O et chargebalance au bord x = L

_secondBoundaryPhreeqCMCT = {
    'Al': 8.519998902e-08, 'C(+4)': 0.0040527660588813574, 'C(-4)': 3.36048e-17, 'Ca': 0.007713414966080056,
    'Cl': 0.041200000313349, 'Fe(+2)': 6.11181766428239e-05, 'Fe(+3)': 1.853659474032439e-09,
    'K': 0.0005109997648652, 'Mg': 0.00514973160860011, 'Na': 0.04010000029724816, 'S(+6)': 0.0111081534532578,
    'S(-2)': 1.2519090021629906e-10, 'S(+2)': 5.8479432234234366e-15, 'S(+3)': 2.841260640805699e-34,
    'S(+4)': 7.548770853000002e-15, 'Si': 0.00017999956704318576, 'Sr': 0.0002521539815104087,
    'H': 111.0215997343147, 'O': 55.56547191826152, 'chargebalance': -2.21778e-08,
}


# ---- ORCHESTRA (noms de chemistry1.inp : [] au lieu de (), phases en [s])

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
    systemSpeciation = ["chargebalance", "pH", "pe"] + _aqueousPhreeqC + _phasesPhreeqC
    transported = [s for s in systemSpeciation if s not in _phasesPhreeqC + ["pH", "pe"]]
    initialConditions = _DIR / "CI_phreeqc.txt"
    if MCT:
        secondBoundary = _secondBoundaryPhreeqCMCT
    else:
        # multi-especes : au bord x = L, les especes de la derniere maille des conditions initiales
        ci = pd.read_csv(initialConditions, sep=r"\s+", comment="%", header=None)
        ci.columns = ["x"] + systemSpeciation
        secondBoundary = {s: float(ci[s].iloc[-1]) for s in transported}
    return dict(
        chemPath=str(_DIR / "PHREEQC_Davies_e-_ThermoChimie_v12a.dat"),
        initialConditions=str(initialConditions),
        systemSpeciation=systemSpeciation,
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
        systemSpeciation=_crossOrchestra + aqueous + _phasesOrchestra,
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
                        systemSpecies=aqueous + _phasesOrchestra,
                        secondBoundary=_secondBoundaryOrchestraMS)
    return settings


def variant(chemModule, trsptModule, MCT):
    mode = "multi-compound" if MCT else "multi-species"
    if chemModule == "PhreeqC":
        settings = _phreeqc(MCT)
    elif chemModule == "ORCHESTRA":
        settings = _orchestra(MCT)
    else:
        raise ValueError(f"CementClayInterface is available with chemModule 'phreeqc' or 'orchestra', "
                         f"not {chemModule!r}")
    settings["description"] = f"cement clay interface benchmark, {chemModule}, {mode} transport"
    return settings
