import importlib.util
import os

chemModule = 'phreeqc' # phreeqc or orchestra
trsptModule = 'nativeTransport' # nativeTransport or comsol
operatorSplitting = 'strang' # choose between 'additive', 'alternative', 'strang', 'symmetrical' and 'snia'
# pblm additive ..

maxTime = 720*100
timeStep = 720

firstStepEquilibrium = True # this keyword will equilibrate first your initial conditions.

systemSpeciation = ["CaX2", "KX", "NaX", "NH4X", "Ca+2", "CaOH+","Cl-", "H+", "H2", "K+", "N2", "Na+", "NaOH", "NH3", "NH4+", "NO2-", "NO3-", "O2", "OH-"]


if chemModule == 'orchestra' :
    transportedSpecies = ["Ca+2", "CaOH+","Cl-", "H+", "H2", "K+", "N2", "Na+", "NaOH", "NH3", "NH4+", "NO2-", "NO3-", "O2", "OH-"]
    primarySpeciesAq = ["Ca","Cl", "K", "N", "Na"]
    chemPath = 'chemistry1.inp'
    speciesAttributes = { # this shall be retrieved from the .inp file ..
        "con" : [spc for spc in systemSpeciation if spc not in ["CaX2", "KX", "NaX", "NH4X"]],
        "solid" : {'CaX2':'Ca','KX':'K','NaX':'Na','NH4X':'NH4'},
        }
    
    initialConditions = "orchestraIC.txt"
elif chemModule == 'phreeqc' :
    initialConditions = "phreeqcIC.txt"
    chemPath = "phreeqc.dat" 
    

if trsptModule == 'comsol' :
    trsptPath = "comsol.mph"
elif trsptModule == 'nativeTransport':
    dispersivity = 0.002
    velocity = 0.002/720
    ADE = True
    firstBoundary = {s:0 for s in systemSpeciation}
    firstBoundary.update({'Ca+2': 0.6e-3, 'Cl-': 1.2e-3})


PIDnbr = 1
PIDextract = 1

enginePath = r"D:\Source" # change your path
if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location(rf"{enginePath}\engine",rf"{enginePath}\engine.py")
    with open("store.txt", "w", encoding="utf-8") as fichier:
        fichier.write(os.path.dirname(os.path.abspath(__file__))+'\n')
        fichier.write(os.path.basename(__file__))
    code = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(code)
    code.main()