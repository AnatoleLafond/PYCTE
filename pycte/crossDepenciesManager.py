from itertools import product
import sys

def crossDep(centralDict, crossDependencies):
    'link cross dependencies (CD) with respect to coupled modules'
    'we need to differentiate input CD and output CD'

    def fillCrossDepDic(coupledParam, inputOrOuput, crossDepDict, CD):
        for cpld in coupledParam:
            if isinstance(cpld, dict):
                for key in cpld.keys():
                    for cells in CD[inputOrOuput]:
                        if key in cells:
                            val = cpld[key]
                            if isinstance(val, str):
                                crossDepDict[inputOrOuput] += [f"{cells[0]}('{val}')"]
                            elif any(isinstance(v, list) for v in val):
                                val_normalized = [v if isinstance(v, list) else [v] for v in val]
                                crossDepDict[inputOrOuput] += [
                                    f'{cells[0]}(' + ','.join(f"'{item}'" for item in combo) + ')'
                                    for combo in product(*val_normalized) ]
                            else:
                                crossDepDict[inputOrOuput] += [f"{cells[0]}('{v}')" for v in val]
                            break
            else:
                for cells in CD[inputOrOuput]:
                    if cpld in cells:
                        crossDepDict[inputOrOuput] += [cells[0]]
                        break
        return crossDepDict


    transportCD = {}
    speciationCD = {}
    
    if centralDict["couplingInfo"][1] == 'PhreeqC':
        'first cell of each CD is the phreeqpy command'
        'second cell is the name of the CD variable'
        'CD shall carry the same name across solvers ...'
        
        speciationCD = {
            'input':  [
            ['tk','temperature(K)','t','temp','tempK','tK','temperatureK','temperature'],
            ['tc','temperature(C)','tC','tempC','temperatureC'],
            ['DebyeLength', 'DebyeLength(m)', 'debyeLength'],
            ['-la("H+")','ph','pH'],
            ['-la("e-")','pe','Eh','eh']
            ],
            
            "output" : [
            ['tk','temperature(K)','t','temp','tempK','tK','temperatureK','temperature'],
            ['tc','temperature(C)','tC','tempC','temperatureC'],
            ['viscos','viscosity(mPa·s)','v','visc'],
            ['mu','ionicStrength(mol/kgw)','IS','is'],
            ['EDL','EDL_','edl'],
            ['TOT','TotAq_','tot_aq'],
            ['-la("H+")','ph','pH'],
            ['-la("e-")','pe','Eh','eh']
                ]},
            
    elif centralDict["couplingInfo"][1] == 'xGEMS':
        speciationCD = {
            'input':  [
            ['temperature(K)','tk','t','temp','tempK','tK','temperatureK','temperature'],
            ['temperature(C)','tc','tC','tempC','temperatureC'],
            ['pressure','p','P'],
            ],
            
            "output" : [
            ['temperature(K)','tk','t','temp','tempK','tK','temperatureK','temperature'],
            ['temperature(C)','tc','tC','tempC','temperatureC'],
            ['pressure','p','P'],
                ]}

    if centralDict["couplingInfo"][2] == 'COMSOL':
        transportCD = {'input' : [['dV']], 
                    'output' : [['dV']]}
    
    elif centralDict["couplingInfo"][2] == 'PFLOTRAN':
        transportCD = {
                    'input':  [
                    ['temperature(K)','tk','t','temp','tempK','tK','temperatureK','temperature'],
                    ['temperature(C)','tc','tC','tempC','temperatureC'],
                    ['pressure','p','P'],
                    ['porosity'],
                    ],
                    
                    "output" : [
                    ['temperature(K)','tk','t','temp','tempK','tK','temperatureK','temperature'],
                    ['temperature(C)','tc','tC','tempC','temperatureC'],
                    ['pressure','p','P'],
                    ['porosity'],
                        ]}

    crossDepSpc = {'input' : [], 'output' : []}
    crossDepTrspt = {'input' : [], 'output' : []}

    crossDepSpc = fillCrossDepDic(crossDependencies.get('speciation', []), 'input',  crossDepSpc, speciationCD)
    crossDepSpc = fillCrossDepDic(crossDependencies.get('speciation', []), 'output', crossDepSpc, speciationCD)
    
    crossDepTrspt = fillCrossDepDic(crossDependencies.get('transport', []), 'input', crossDepTrspt, transportCD)
    crossDepTrspt = fillCrossDepDic(crossDependencies.get('transport', []), 'output', crossDepTrspt, transportCD)

    test = {'speciation' : {
                            'input' : list(dict.fromkeys(crossDepSpc['input'])), 
                            'output': list(dict.fromkeys(crossDepSpc['output'])),
                            'total' : list(dict.fromkeys(crossDepSpc['input'] + crossDepSpc['output']))},
            'transport': {
                            'input': list(dict.fromkeys(crossDepTrspt['input'])),
                            'output': list(dict.fromkeys(crossDepTrspt['output'])),
                            'total': list(dict.fromkeys(crossDepTrspt['output']+crossDepTrspt['input'])) },
            'totalCrossDep' : list(dict.fromkeys(crossDepTrspt['output']+crossDepTrspt['input']+crossDepSpc['input'] + crossDepSpc['output']))
            }

    return test
