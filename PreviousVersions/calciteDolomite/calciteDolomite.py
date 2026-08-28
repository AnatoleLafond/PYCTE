import importlib.util
import os


description = "Calcite dolomite test case, using several transport solvers"
# after running this file, please run the 'profileColumn1D.py' to see concentration = f(x) of each species at each time step

timeUnit = 's'
maxTime = 21e3
timeStep = 5e-3/9.375e-6 #  CFL=1

geometry = 3

chemModule = 'gems' # only gems
trsptModule = 'nativeTransport' # comsol, pflotran or nativeTransport
operatorSplitting = 'snia' # snia, strang, symmetrical, 

firstStepEquilibrium = True

systemSpeciation = ["Ca(CO3)@","Ca(HCO3)+","Ca+2","CaOH+","Mg(CO3)@","Mg(HCO3)+","Mg+2","MgOH+","CO2@","CO3-2","HCO3-","CH4@","ClO4-","Cl-","H2@","O2@","OH-","H+","H2O@","CO2","CH4","H2","O2","Cal","Dis-Dol","Sn"]
initialConditions =  "ic_spc.txt"



if trsptModule == 'pflotran':
    MultiCompoundTransport = True
    trsptPath = 'pflotran.in' # please open this file to update your database path (line 95)
elif trsptModule == 'nativeTransport': 
    # native transport is a little 1D transport code i coded.
    initialConditions =  "ic_spc_1d.txt" 
    geometry = 1
    dispersivity = 0.0067
    velocity = 9.375e-6
    firstBoundary = {s:0 for s in systemSpeciation}
    firstBoundary.update({'Ca(CO3)@': 3.49122e-17, 'CaHCO3+': 7.26344e-15,'Ca+2':1e-8,'CaOH+':1.09822e-14,'Mg(CO3)@':3.97585e-12,'Mg(HCO3)+':1.33202e-10,'Mg+2':(0.00199995182445532/2),
                          'MgOH+':4.80383669042909e-08,'CO2@':1.93e-9,'CO3-2':4.03182994984226e-12,'HCO3-':7.92277169973322e-09,'Cl-':2e-3,'H+':1.28190667315784e-07,'H2O@':55.5083731906404})
    ADE = True # stands for advection-dispersion equation. There is advection and Fick alone as well.

elif trsptModule == 'comsol':
    trsptPath = "calciteDolomite.mph"

chemPath =r'.../Resources/CalciteIC/CalciteIC-dat.lst'

    
enginePath = ".../Source"

if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("engine", os.path.join(enginePath, "engine.py"))
    with open("store.txt", "w", encoding="utf-8") as fichier:
        fichier.write(os.path.dirname(os.path.abspath(__file__))+'\n')
        fichier.write(os.path.basename(__file__))
    code = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(code)
    code.main()
