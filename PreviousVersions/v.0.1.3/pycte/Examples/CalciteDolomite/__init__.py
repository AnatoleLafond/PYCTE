from pathlib import Path

_DIR = Path(__file__).parent

_systemSpeciation = [
    "Ca(CO3)@", "Ca(HCO3)+", "Ca+2", "CaOH+", "Mg(CO3)@", "Mg(HCO3)+",
    "Mg+2", "MgOH+", "CO2@", "CO3-2", "HCO3-", "CH4@", "ClO4-", "Cl-",
    "H2@", "O2@", "OH-", "H+", "H2O@", "CO2", "CH4", "H2", "O2", "Cal",
    "Dis-Dol", "Sn",
]

SETTINGS = dict(
    description="Calcite dolomite test case from Azad and al. 2016",
    timeUnit="s", maxTime=21e3, timeStep=5e-3 / 9.375e-6,
     chemModule="gems", trsptModule="nativeTransport",
    operatorSplitting="snia", firstStepEquilibrium=True,
    systemSpeciation=_systemSpeciation,
    output = {'coupling' : list(range(1, int(21e3/(5e-3 / 9.375e-6)) + 1))},
    initialConditions=str(_DIR / "ic_spc.txt"), chemPath= str(_DIR / "Resources" / "CalciteIC" / "CalciteIC-dat.lst"),
)

firstBoundary = {s:0 for s in _systemSpeciation}
firstBoundary.update({'Ca(CO3)@': 3.49122e-17, 'CaHCO3+': 7.26344e-15,'Ca+2':1e-8,'CaOH+':1.09822e-14,'Mg(CO3)@':3.97585e-12,'Mg(HCO3)+':1.33202e-10,'Mg+2':(0.00199995182445532/2),
                      'MgOH+':4.80383669042909e-08,'CO2@':1.93e-9,'CO3-2':4.03182994984226e-12,'HCO3-':7.92277169973322e-09,'Cl-':2e-3,'H+':1.28190667315784e-07,'H2O@':55.5083731906404})

TRSPT_VARIANTS = {
    "nativeTransport": dict(
        initialConditions=str(_DIR / "ic_spc_1d.txt"),
        geometry=1, dispersivity=0.0067, velocity=9.375e-6, ADE=True,
        firstBoundary=firstBoundary,
    ),
    "comsol": dict(trsptPath=str(_DIR / "calciteDolomite.mph")),
    "pflotran": dict(MultiCompoundTransport=True, geometry=3, trsptPath=str(_DIR / "pflotran.in")),
}

SETTINGS.update(TRSPT_VARIANTS["nativeTransport"])