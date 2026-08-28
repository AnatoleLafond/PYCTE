import importlib.util
import os

description = "Aldicarb infiltration under kinetic equilibrium. The reference (true values) is the attached comsol file 'reference.mph'"
timeUnit = 'h'
maxTime = 8*24
timeStep = 2
geometry = 2

PIDnbr = 1 # number of PID carrying PhreeqC calculation
operatorSplitting = 'snia' # change the operator splitting using 'additive', 'alternative', 'strang', 'symmetrical' and 'snia'

systemSpeciation = ['p','dl.theta_l',"Aldicarb","Oxime","Sulfoxide","Sulfoxide_ox","Sulfone","Sulfone_ox"]
primarySpeciesAq = systemSpeciation

partialCrossDependencies = {'chem' : ['p','dl.theta_l']}

initialConditions = 'AldicarbInfiltrationIC.txt'
chemPath = "AldicarbInfiltration.dat"
trsptPath = "AldicarbInfiltration.mph" # version 6.4

cutoffAq = 1e-99

comsolStudy = 'std2'

kinetics ="""
Aldi_ox
    -formula Oxime 1 Aldicarb -1 
Aldi_sulfox
    -formula Sulfoxide 1 Aldicarb -1
Sulfox_sulfox
    -formula Sulfoxide_ox 1 Sulfoxide -1
Sulfox_sulfone
    -formula Sulfone 1 Sulfoxide -1
Sulfone_sulfone
    -formula Sulfone_ox 1 Sulfone -1
"""

enginePath = r"...\Source"
if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("engine", os.path.join(enginePath, "engine.py"))
    with open("store.txt", "w", encoding="utf-8") as fichier:
        fichier.write(os.path.dirname(os.path.abspath(__file__))+'\n')
        fichier.write(os.path.basename(__file__))
    code = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(code)
    code.main()
