from itertools import product
import sys

def crossDep(crossDict, crossDependencies):
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
    
    if crossDict["couplingInfo"][1] == 'PhreeqC':
        'first cell of each CD is the phreeqpy command'
        'second cell is the name of the CD variable'
        
        availableCD = {
            'PhreeqC' : {
            'input':  [
            ['tk','temperature(K)','t','temp','tempK','tK','temperatureK','temperature'],
            ['tc','temperature(°C)','tC','tempC','temperatureC'],
            ['DebyeLength', 'DebyeLength(m)', 'debyeLength'],
            ['-la("H+")','ph','pH'],
            ['-la("e-")','pe','Eh','eh']
            ],
            
            "output" : [
            ['tk','temperature(K)','t','temp','tempK','tK','temperatureK','temperature'],
            ['tc','temperature(°C)','tC','tempC','temperatureC'],
            ['viscos','viscosity(mPa·s)','v','visc'],
            ['mu','ionicStrength(mol/kgw)','IS','is'],
            ['EDL','EDL_','edl'],
            ['TOT','TotAq_','tot_aq'],
            ['-la("H+")','ph','pH'],
            ['-la("e-")','pe','Eh','eh']
                ]},
            'COMSOL' : {'input' : [['dV']], 
                        'output' : [['dV']]}}
        
        crossDepSpc = {'input' : [], 'output' : []}
        crossDepTrspt = {'input' : [], 'output' : []}
        
        crossDepSpc = fillCrossDepDic(crossDependencies['speciation'],'input', crossDepSpc, availableCD['PhreeqC'])
        crossDepSpc = fillCrossDepDic(crossDependencies['speciation'], 'output', crossDepSpc, availableCD['PhreeqC'])
        
        crossDepTrspt = fillCrossDepDic(crossDependencies['transport'], 'output', crossDepTrspt, availableCD['COMSOL'])

        crossDict.update({'crossDependencies' : {'speciation' : { 
                                                'input' : list(dict.fromkeys(crossDepSpc['input'])), 
                                                'output': list(dict.fromkeys(crossDepSpc['output'])),
                                                'total' : list(dict.fromkeys(crossDepSpc['input'] + crossDepSpc['output']))},
                                                'transport': {
                                                'input': [],
                                                'output': list(dict.fromkeys(crossDepTrspt['output'])),
                                                'total': list(dict.fromkeys(crossDepTrspt['output']))
                                            }
                                                }})

    return crossDict
