import time
PICCTSstart = time.time()
import pandas as pd
import sys 
import importlib.util
import os
import shutil
import difflib
import numpy as np
import re
from pathlib import Path

current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

with open("store.txt", 'r', encoding='utf-8') as fichier:
    pathPicctsInput =  Path(fichier.readline().strip())
    nameInput =  fichier.readline().strip()


module_name = os.path.splitext(nameInput)[0]
file_path = os.path.join(pathPicctsInput, nameInput)
spec = importlib.util.spec_from_file_location(module_name, file_path)

PICCTS_input = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PICCTS_input)

maxTime = getattr(PICCTS_input, 'maxTime', 1)
timeStep = getattr(PICCTS_input, 'timeStep', None)
stepReprise = getattr(PICCTS_input, 'stepReprise', 0)
dtPICCTS = getattr(PICCTS_input, 'dtPICCTS', [])
if dtPICCTS and dtPICCTS[0]==0: del dtPICCTS[0]

if not dtPICCTS:
    dtPICCTS = [timeStep]
    while dtPICCTS[-1] < maxTime:
        dtPICCTS += [dtPICCTS[-1] + timeStep]

if stepReprise:
    # stepReprise was the last completed step. The Xeme step is the (X-1)eme step for piccts
    # So the Xeme step is the step to be completed
    startTimeStep = start = stepReprise
    timeStepReprise = dtPICCTS[stepReprise-1]
else:
    timeStepReprise = 0
    startTimeStep = start = 0





def normalize_choice(value, mapping, field_name):
    if isinstance(value, str):
        value_norm = value.lower()
    else:
        value_norm = value
    for canonical, aliases in mapping.items():
        for alias in aliases:
            if isinstance(alias, str):
                if value_norm == alias.lower():
                    return canonical
            else:
                if value_norm == alias:
                    return canonical

    str_aliases = [
        alias.lower()
        for aliases in mapping.values()
        for alias in aliases
        if isinstance(alias, str)
    ]

    suggestion = None
    if isinstance(value, str):
        matches = difflib.get_close_matches(value_norm, str_aliases, n=1, cutoff=0.6)
        if matches:
            suggestion = matches[0]

    if suggestion:
        raise ValueError(
            f"Invalid value for '{field_name}': {value!r}. "
            f"Do you mean '{suggestion}' ? "
        )
    else:
        raise ValueError(
            f"Invalid value for '{field_name}': {value!r}. "
        )

operator_map = {
    'SNIA': [1, '1', 'snia'],
    'Strang': [2, '2', 'strang'],
    'Alternative': [3, '3', 'alternative'],
    'Additive': [4, '4', 'additive'],
    'Symmetrical': [5, '5', 'symmetrical'],
}

chem_map = {
    'PhreeqC': [1, '1', 'phreeqc','PhreeqC','Phreeqc'],
    'xGEMS': [2, '2', 'xgems','xGEMS','gems'],
    'ORCHESTRA': [3, '3', 'orchestra','ORCHESTRA', 'Orchestra'],
}

trspt_map = {
    'COMSOL': [1, '1', 'comsol','Comsol','COMSOL'],
    'nativeTransport': [2, '2', 'nativeTransport','native','trspt'],
    'PFLOTRAN': [3, '3', 'Pflotran','pflotran','PFLOTRAN'],
}

def main():
    
    
    
    def writeTime(tps, arr=10):
        if tps >= 3600 * 24:
            return f"{tps / (3600 * 24):.{arr}f} d"
        elif tps >= 3600:
            return f"{tps / 3600:.{arr}f} h"
        elif tps >= 60:
            return f"{tps / 60:.{arr}f} min"
        else:
            return f"{tps:.{arr}f} sec"
    
    def readInputFile(txtPath, coord,  inputHeaders = None) :
        dataframe = pd.read_csv(txtPath,sep=r"\s+",comment="%",dtype=float)
        if len(dataframe.columns) != len(coord + inputHeaders):
            print(f"input file : {len(dataframe.columns)} columns")
            print(f"coupled variables : {len(coord + inputHeaders)}")
            print(coord +inputHeaders)
            sys.exit()
        if (coord + inputHeaders) != dataframe.columns.tolist():
            dataframe = pd.read_csv(txtPath,sep=r"\s+",comment="%",dtype=float,names=coord + inputHeaders)
        return dataframe
    
    # 
        
    # Decompose secundary species into primary species (e.g. Na2S2O3 into 2Na, 2S and 3O)
    def decomposingIntoPrimSpecies(formula, primarySpecies,):
        def multiply_dict(d, factor): 
            return {k: v * factor for k, v in d.items()}
        def merge_dicts(a, b): 
            for k, v in b.items():
                a[k] = a.get(k, 0) + v
            return a
    
        charge_match = re.search(r'([+-]\d*)$', formula)
        if charge_match:
            charge = charge_match.group(1)
            formula = formula[:charge_match.start()]
        else:
            charge = None
    
        formula = re.sub(r'__([0-9]+)', r')\1', formula) 
    
        primaryspecies_trié = sorted(primarySpecies, key=len, reverse=True)
        pattern = re.compile('|'.join(re.escape(ps) for ps in primaryspecies_trié)) 
    
        index = 0
        tokens = []
    
        while index < len(formula): 
            m = pattern.match(formula, index) 
            if m:
                name = m.group(0)
                j = index + len(name) 
                coef_match = re.match(r'\d+', formula[j:]) 
                if coef_match:
                    count = int(coef_match.group())
                    j += len(coef_match.group()) 
                else:
                    count = 1 
                tokens.append(('group', name, count)) 
                index = j 
                continue
    
            if formula[index] == '(':
                tokens.append(('(',))
                index += 1
                continue
            elif formula[index] == ')': 
                j = index + 1
                while j < len(formula) and formula[j].isdigit():
                    j += 1
                multiplicateur = int(formula[index+1:j]) if j > index+1 else 1 
            
                tokens.append((')', multiplicateur))
                index = j
                continue
    
            index += 1
        stack = []
        current = {}
    
        for token in tokens:
            if token[0] == 'group': 
                name, count = token[1], token[2]
                current[name] = current.get(name, 0) + count 
            elif token[0] == 'element':
                elem, count = token[1], token[2]
                current[elem] = current.get(elem, 0) + count
            elif token[0] == '(':
                stack.append(current)
                current = {}
            elif token[0] == ')':
                multiplicateur = token[1]
                current = multiply_dict(current, multiplicateur)
                prev = stack.pop()
                current = merge_dicts(prev, current)
                
        return current, charge
   
    
    maillesChargeGeom = []
    specieChargeGeom = []
    if getattr(PICCTS_input, 'speciesChargeGeometry', None):
        for (maille, spc) in getattr(PICCTS_input, 'speciesChargeGeometry', None): 
            maillesChargeGeom += [maille]
            specieChargeGeom += [spc]
    
    phases = getattr(PICCTS_input, "phases", "")
    if getattr(PICCTS_input, "fixpH", None):
        phases += "Fix_ph\nH+=H+; log_k 0"
    
    
    
    
    outputFolder= ["PrimarySpecies","Speciation","CouplingHistory"]
    
    paths = {} # variables are keys, paths are values
    
    if getattr(PICCTS_input, 'intermediateOutput', True) :
        for i,doss in enumerate(outputFolder):
                paths[doss] = os.path.join(pathPicctsInput, outputFolder[i])
                # if os.path.exists(paths[doss]) and not stepReprise:
                #     shutil.rmtree(paths[doss]) # Quite dangerous but I like risk (will permanently delete your previous files when running a new PICCTS run)
                #     os.makedirs(paths[doss])
                # elif not stepReprise:
                #     os.makedirs(paths[doss])
    
    
    
    # outputName = ["trsptPath","chemPath","initialConditions"]
    # outputDefault = ["comsol.mph","phreeqc.dat","ic.txt"]
    # for i, txt in enumerate(outputName):
    #     if not getattr(PICCTS_input, txt, None):
    #         paths[txt] = Path(getattr(PICCTS_input, txt, outputDefault[i]))
    #     else:
    #         paths[txt] = Path(getattr(PICCTS_input, txt))

    
    
    
    centralDict = { # gather keywords which do not depend on components
        "warningRun": 0,
        "inputPath" : pathPicctsInput,
        "AcidicEcho": pd.DataFrame(),
        "waitingTime" : 0,
        "couplingInfo" :[normalize_choice(getattr(PICCTS_input, 'operatorSplitting', 1), operator_map, 'operatorSplitting'),
                         normalize_choice(getattr(PICCTS_input, 'chemModule', 1), chem_map, 'chemModule'),
                         normalize_choice(getattr(PICCTS_input, 'trsptModule', 1), trspt_map, 'trsptModule')],
        
        # "couplingInfo" : [ ['SNIA','Strang','Alternative','Additive','Symmetrical'][getattr(PICCTS_input, 'operatorSplitting', 1)-1],
        #                   ['PhreeqC','xGEMS','ORCHESTRA'][getattr(PICCTS_input, 'chemModule', 1)-1],
        #                   ['COMSOL'][getattr(PICCTS_input, 'trsptModule', 1)-1],
        #     ],
        
        "paths" : paths,
        "system" : getattr(PICCTS_input, 'system', 1),
        "stepReprise" : getattr(PICCTS_input, 'stepReprise', False),
        "PIDnbr": getattr(PICCTS_input, 'PIDnbr', 1),
        "PIDextract": getattr(PICCTS_input, 'PIDextract', 3),
        "systemSpeciation" : getattr(PICCTS_input, 'systemSpeciation', None),
        "timeUnit" : getattr(PICCTS_input, 'timeUnit', 's'),
        "geometry" : getattr(PICCTS_input, 'geometry', 1),

        "coord" : ['x', 'y', 'z'][:getattr(PICCTS_input, 'geometry', 1)],
        "chemModule" : getattr(PICCTS_input, 'chemModule', 1),
        "trsptModule" : getattr(PICCTS_input, 'trsptModule', 1),
        "firstStepEquilibrium" : getattr(PICCTS_input, 'firstStepEquilibrium', False),
        'nonTrivialDecomposition' : getattr(PICCTS_input, 'nonTrivialDecomposition', None),
        "preliminarEquilibrium" : getattr(PICCTS_input, 'preliminarEquilibrium', False),
        'extractDBTime' : 0,
        'PIDchem' : getattr(PICCTS_input, 'PIDchem', 1),
        'PIDtrspt' : getattr(PICCTS_input, 'PIDtrspt', 1),
        'MultiCompoundTransport' : getattr(PICCTS_input, 'MultiCompoundTransport', False),
        "initialConditions" : Path(getattr(PICCTS_input, 'initialConditions', os.path.join(pathPicctsInput, 'ic.txt'))),
        
        }
    

    
    crossDep = getattr(PICCTS_input, 'crossDependencies', None)
    if crossDep :
        if not crossDep.get('speciation'):
            crossDep['speciation'] = []
        if not crossDep.get('transport'):
            crossDep['transport'] = []
        
        import crossDepenciesManager
        centralDict.update(crossDepenciesManager.crossDep(centralDict, crossDep))

    else: 
        centralDict.update({'crossDependencies' : None})

    
    centralDict.update({"commMtrx": readInputFile(centralDict['initialConditions'],['x', 'y', 'z'][:getattr(PICCTS_input, 'geometry', 1)],
    (centralDict['systemSpeciation'] + (centralDict['crossDependencies']['speciation']['total'] if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation') else [])+
     (centralDict['crossDependencies']['transport']['total'] if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('transport') else [])))})
    

    centralDict.update({"anythingButSpecies" : [c for c in list(centralDict["commMtrx"].columns) if c not in centralDict['systemSpeciation']]})

    print(f"""
####   #  ####  ####  #####  ####     {centralDict['couplingInfo'][0]} splitting :
#  #   #  #     #       #    #        {centralDict['couplingInfo'][1]}--->
####   #  #     #       #    ####            <---{centralDict['couplingInfo'][2]}
#      #  #     #       #       #     {maxTime-timeStepReprise}{centralDict['timeUnit']} in {len(dtPICCTS)-stepReprise} steps        
#      #  ####  ####    #    ####     
       """, flush=True)
    
    if centralDict["couplingInfo"][1] == 'PhreeqC':
        centralDict['chemPath'] = Path(getattr(PICCTS_input, 'chemPath', os.path.join(pathPicctsInput, 'phreeqc.dat')))
        import phreeqc
        import extractDB
        centralDict.update(extractDB.extract(centralDict))

        speciationLauncher = {'PhreeqC': phreeqc.spct}                
        
        centralDict.update({
        # "primToSecSpecies" : getattr(PICCTS_input, 'primToSecSpecies', {}),
        "current" : getattr(PICCTS_input, 'current', None),
        "SI" : getattr(PICCTS_input, "SI", {ph : 0 for ph in centralDict['primarySpecies']['phases']}  ),
        "mineralReversibility" : getattr(PICCTS_input, "mineralReversibility", {ph : "" for ph in centralDict['primarySpecies']['phases']}),
        
        "cutoffs" : {'solution' : getattr(PICCTS_input, 'cutoffAq', 1e-20),
                    'phases' : getattr(PICCTS_input, 'cutoffPha', -1e-99),
                    'exchange' : getattr(PICCTS_input, 'cutoffEch', 1e-20),
                    'surface' : getattr(PICCTS_input, 'cutoffSurf', 1e-20)},

        "database": ["",getattr(PICCTS_input, 'masterSpecies', None),
                     getattr(PICCTS_input, 'solutionSpecies', None),
                     phases,
                     getattr(PICCTS_input, 'masterExchange', None),
                     getattr(PICCTS_input, 'exchangeSpecies', None),
                     getattr(PICCTS_input, 'masterSurface', None),
                     getattr(PICCTS_input, 'surfaceSpecies', None),
                     getattr(PICCTS_input, 'kineticRates', None),
                     getattr(PICCTS_input, 'kinetics', None),],
        "fixpH": getattr(PICCTS_input, "fixpH", None),
        "userVarBool": getattr(PICCTS_input, "userVarBool", {}),
        "userVarList": getattr(PICCTS_input, "userVarList", {}),
        "maillesChargeGeom" : maillesChargeGeom,
        "solMod" : getattr(PICCTS_input, 'solMod', False),
        "step_divide" : getattr(PICCTS_input, 'step_divide', None),
        "speciesCharge" : getattr(PICCTS_input, "speciesCharge", ['pH']),
        "speciesChargeGeometry" : getattr(PICCTS_input, 'speciesChargeGeometry', None),
        "speciationCharge" : getattr(PICCTS_input, "speciationCharge", False),
        "pHdefault" : getattr(PICCTS_input, "pHdefault", 7),
        "tempDefault" : getattr(PICCTS_input, "tempDefault", 25),
        "distAnodCathTotale" : getattr(PICCTS_input, "distAnodCathTotale", None),
        "catholyte" : getattr(PICCTS_input, "catholyte", None),
        "anolyte" : getattr(PICCTS_input, "anolyte", None),
        "solModCharge" : getattr(PICCTS_input, "solModCharge", 0),
        "acidicTrspt": getattr(PICCTS_input, 'acidicTrspt', False),
        "water" : getattr(PICCTS_input, 'water', False),
        "kinetics" : getattr(PICCTS_input, 'kinetics', False),
        'supplementarySolution' : getattr(PICCTS_input, 'supplementarySolution', None),
        "surfaceCounterIons" : getattr(PICCTS_input, 'surfaceCounterIons', False),
        "beforeTrsptMtrx" : pd.DataFrame(),
        "PhreeqCCalcTime_WallClock": 0,
        "PhreeqCCalcTime_ProcessorTime": 0,
        "PhreeqCInterfTime_WallClock": 0,
        "PhreeqCInitTime" : 0,
        "PhreeqCTotalTime" : 0,

        })
        


        

    elif centralDict["couplingInfo"][1] == 'xGEMS':

        
        centralDict.update({
        "chemPath" : Path(getattr(PICCTS_input, 'chemPath', os.path.join(pathPicctsInput, 'dat.lst'))),
        # "independentComponents" : getattr(PICCTS_input, "independentComponents", None),
        "xGEMSClockTime" : 0,
        # "constantSpecies" : getattr(PICCTS_input, 'constantSpecies', {}),
        "xGEMSPrcsTime" : 0,
        # "transportedSpecies" :  getattr(PICCTS_input, "transportedSpecies"),
        "xGEMSCalcTime_WallClock": 0,
        "xGEMSCalcTime_ProcessorTime": 0,
        "xGEMSInterfTime_WallClock": 0,
        "xGEMSInitTime" : 0,
        "xGEMSTotalTime" : 0,
        })
        import extractDB
        
        centralDict.update(extractDB.extract(centralDict))

        import gems
        speciationLauncher = {
            'xGEMS': gems.spct,}
        
        os.chdir(pathPicctsInput)

        
    elif centralDict["couplingInfo"][1] == 'ORCHESTRA':
        import orchestra

        
        
        speciationLauncher = {
            'ORCHESTRA': orchestra.spct}
        
        speciesAttributes = {}
        dico = getattr(PICCTS_input, 'speciesAttributes', { })
        for ky in dico.keys():
            if isinstance(dico[ky], dict) : 
                for subkey in dico[ky]:
                    speciesAttributes[subkey] = f"{dico[ky][subkey]}.{ky}"
            else:
                for spc in centralDict['systemSpeciation']:
                    if spc in dico[ky]:
                        speciesAttributes[spc] = f"{spc}.{ky}"
                        continue
        

        centralDict.update({
        "chemPath" : Path(getattr(PICCTS_input, 'chemPath', os.path.join(pathPicctsInput, 'chemistry1.inp'))),
        "primarySpecies" : [getattr(PICCTS_input, "primarySpeciesAq", []),
                           getattr(PICCTS_input, "primarySpeciesPha", []),
                           getattr(PICCTS_input, "primarySpeciesSurf", []),
                           getattr(PICCTS_input, "speciationEch", []),
                           getattr(PICCTS_input, "primarySpeciesPhantom", ['OH','H']),
                           (getattr(PICCTS_input, "primarySpeciesAq", []) + getattr(PICCTS_input, "primarySpeciesPha", []) +
                            getattr(PICCTS_input, "primarySpeciesSurf", [])+getattr(PICCTS_input, "speciationEch", []))
                           ],
        
        "primarySpeciesEchSorbed" :  getattr(PICCTS_input, "primarySpeciesEchSorbed", {}),
        "ORCHESTRACalcTime_WallClock": 0,
        "ORCHESTRACalcTime_ProcessorTime": 0,
        "ORCHESTRAInterfTime_WallClock": 0,
        "ORCHESTRAInitTime" : 0,
        "ORCHESTRATotalTime" : 0,
        "transportedSpecies" :  getattr(PICCTS_input, "transportedSpecies"),
        })
        
        import extractDB
        centralDict.update(extractDB.extract(centralDict))
        
        centralDict.update({
        "inputVariableOrchestra" : [f"{spc}.tot" for spc in centralDict['primarySpecies'][-1]],
        "outputVariableOrchestra" : [f"{speciesAttributes[spc]}" for spc in centralDict['systemSpeciation']],
            })

    else: 
        print(f'No module is associated with chemModule = {getattr(PICCTS_input, "chemModule", 1)}')
        sys.exit()


    if centralDict["couplingInfo"][2] == 'COMSOL':
        import comsol
        
        transportLauncher = {
            "COMSOL": comsol.trspt, 
        }

        
        centralDict.update({
        "trsptPath" : Path(getattr(PICCTS_input, 'trsptPath', os.path.join(pathPicctsInput, 'comsol.mph'))),
        "comsolTags" : [getattr(PICCTS_input, 'comsolComp', 'comp1'),
                       getattr(PICCTS_input, 'comsolIntFonction', 'int1'),
                       getattr(PICCTS_input, "comsolStudy", 'std1'), 
                    ],
        # "comsolCoupling": getattr(PICCTS_input, 'comsolCoupling', 'data1'),
        "outputComsol": [],
        "comsolVTU":  getattr(PICCTS_input, 'comsolVTU', True),
        "comsolCore" : getattr(PICCTS_input, 'comsolCore', None),
        "COMSOLCalcTime_WallClock": 0,
        "COMSOLCalcTime_ProcessorTime": 0, 
        "COMSOLInterfTime_WallClock": 0,
        "COMSOLInitTime" : 0,
        "COMSOLTotalTime" : 0,
        "COMSOLWaitingTime" : 0
        })

        if getattr(PICCTS_input, 'outputComsol', None):
            for i,out in enumerate(getattr(PICCTS_input, 'outputComsol', None)):
                centralDict['outputComsol'] += [out]
                centralDict['paths'][f'Transport{out}'] =  os.path.join(pathPicctsInput, f'outputTransport{out}')
                centralDict['paths'][f'TransportVTU{out}'] =  os.path.join(pathPicctsInput, f'outputVTU{out}')

    elif centralDict["couplingInfo"][2] == 'nativeTransport':
        import nativeTransport

        transportLauncher = {
            "nativeTransport": nativeTransport.trspt, 
        }

        centralDict.update({
            "velocity" : getattr(PICCTS_input, 'velocity', 0),
            "advection" : getattr(PICCTS_input, 'advection', None),
            "FickDiffusion" : getattr(PICCTS_input, 'FickDiffusion', None),
            "porousTransport" : getattr(PICCTS_input, 'porousTransport', None),
            "ADE" : getattr(PICCTS_input, 'ADE', None),
            "diffCoeff" : getattr(PICCTS_input, 'diffCoeff', 0),
            "dispersivity" : getattr(PICCTS_input, 'dispersivity', 0),
            "firstBoundary" : getattr(PICCTS_input, 'firstBoundary', {cle: 0 for cle in centralDict['systemSpeciation']}),
            "secondBoundary" : getattr(PICCTS_input, 'secondBoundary', {cle: 0 for cle in centralDict['systemSpeciation']}),
            "boundaryConditions" : getattr(PICCTS_input, 'boundaryConditions', ['flux', 'flux']),
            "nativeTransportClockTime" : 0,
            "nodeSize" : getattr(PICCTS_input, 'nodeSize', 1),
            "advectionScheme" : 'upwind',
            "diffusionScheme" : 'fick',
            "porosity" : (
                    np.array([getattr(PICCTS_input, "porosity",1)] * len(centralDict["commMtrx"]))
                    if isinstance(getattr(PICCTS_input, "porosity", 1), (int, float))
                    else np.array(getattr(PICCTS_input, "porosity", [1] * len(centralDict["commMtrx"])))
                ),
            "area" : np.array(getattr(PICCTS_input, 'area', [1]*len(centralDict['commMtrx']))),
            "nativeTransportCalcTime_WallClock": 0,
            "nativeTransportCalcTime_ProcessorTime": 0,
            "nativeTransportInterfTime_WallClock": 0,
            "nativeTransportTotalTime" : 0,
        })

        x = centralDict['commMtrx'][centralDict['coord']].copy().to_numpy().reshape(-1)

        f_left = np.zeros(len(centralDict['commMtrx']))
        f_right = np.zeros(len(centralDict['commMtrx']))
        
        f_left[1:]   = x[1:] - x[:-1]
        f_right[:-1] = x[1:] - x[:-1]

        centralDict.update({'f_right' : f_right, 'f_left' : f_left})
        dx = np.diff(x)
        dx_min = np.min(dx)
        

        nodeSize = np.empty(len(x))
        nodeSize[1:-1] = (x[2:] - x[:-2]) / 2     
        nodeSize[0]    = (x[1]  - x[0])  #/ 2       
        nodeSize[-1]   = (x[-1] - x[-2]) #/ 2    
        centralDict.update({"x": x,
                            "dx" : dx,
                            'nodeSize': nodeSize,
                            'commMtrxCoord' : centralDict['commMtrx'][centralDict['coord']].copy()})
    

        if centralDict['timeUnit'] == 'd':
            dt = timeStep*3600*24
        elif centralDict['timeUnit'] == 'h':
            dt = timeStep*3600
        elif centralDict['timeUnit'] == 'm' or centralDict['timeUnit'] == 'min':
            dt = timeStep*60
        else: dt = timeStep

        n_adv = 1
        n_diff = 1
        if centralDict['velocity'] != 0 and dt > round(dx_min / abs(centralDict['velocity']),4):
            # centralDict['warningRun'] += 1
            n_adv = int(np.ceil(dt / (dx_min / abs(centralDict['velocity']))))
            print(f"Sub-cycling advection within {n_adv} sub-steps, dt_max={round(dx_min / abs(centralDict['velocity']),4)} sec")

        if (centralDict['diffCoeff'] +centralDict['velocity']*centralDict['dispersivity']):
            dt_diff = dx_min**2 / (2 * (centralDict['diffCoeff'] +centralDict['velocity']*centralDict['dispersivity']))
            if dt > round(dt_diff,4):
                n_diff = int(round(dt / dt_diff))
                print(f"Sub-cycling diffusion/dispersion within {n_diff} sub-steps, dt_max={round(dt_diff,4)} sec")
        
        centralDict.update({'subCyclingDiff' : n_diff,
                            'subCyclingAdv' : n_adv})
    
    
    elif centralDict["couplingInfo"][2] == 'PFLOTRAN':
        import pflotran

        transportLauncher = {
            "PFLOTRAN": pflotran.trspt, 
        }
    
        centralDict.update({
            "trsptPath" : Path(getattr(PICCTS_input, 'trsptPath', os.path.join(pathPicctsInput, 'pflotran.in'))),
            "PFLOTRANCalcTime_WallClock": 0,
            "PFLOTRANCalcTime_ProcessorTime": 0,
            "PFLOTRANInterfTime_WallClock": 0,
            "PFLOTRANInitTime" : 0,
            "PFLOTRANTotalTime" : 0,})
        
    
    
    
    else:
        print(f'No module is associated with trsptModule = {getattr(PICCTS_input, "trsptModule", 1)}')
        sys.exit()
    
    # print(centralDict['paths'])
    for p, path in centralDict['paths'].items():
        path = Path(path)
        if path.suffix:
            continue
        if path.exists():
            shutil.rmtree(path)
    
        path.mkdir(parents=True, exist_ok=True)
        
    if not stepReprise:
        with open("warning.log", "w") as warningLog:
            warningLog.write(f"PICCTS : {centralDict['couplingInfo'][0]} {centralDict['couplingInfo'][1]}-{centralDict['couplingInfo'][2]}.\n")
            if getattr(PICCTS_input, "description", None): warningLog.write(f"PICCTS : Description :\n{PICCTS_input.description}\n")
            warningLog.write(f"PICCTS : Time steps ({centralDict['timeUnit']}) :\n")    
            for dtt in dtPICCTS:
                if dtt == dtPICCTS[-1]: warningLog.write(f"{dtt}.\n")
                else : warningLog.write(f"{dtt}, ")
    else:
        with open("warning.log", "a") as warningLog:
            warningLog.write(f"PICCTS, time = {dtPICCTS[stepReprise]}{centralDict['timeUnit']} : run reprise ...\n")


    for l,t in enumerate(dtPICCTS[start:], start = startTimeStep):
        
        if l==startTimeStep and not stepReprise: dt = t
        else: dt = t-dtPICCTS[l-1]

        centralDict.update({
            "dtStep":dt,
            "tStep":t,
            "lStep":l,
            })
        
        print(f"#######  step n°{l+1}/{len(dtPICCTS)}, dt = {dt}{centralDict['timeUnit']}  #######")

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', 10)

        if centralDict['preliminarEquilibrium']:
            centralDict['beforeTrsptMtrx'] = centralDict['commMtrx'].copy() # not OS dependent ..

        if centralDict["firstStepEquilibrium"] and l==startTimeStep:
            centralDict["dtStep"]=0
            centralDict.update(speciationLauncher.get(centralDict['couplingInfo'][1])(centralDict))
            if centralDict['preliminarEquilibrium']:
                centralDict['beforeTrsptMtrx'] = centralDict['commMtrx'].copy()
            centralDict["firstStepEquilibrium"]=False
            centralDict["dtStep"]=dt
            # print(centralDict['commMtrx'])
            # sys.exit()

        if centralDict['couplingInfo'][0]=='SNIA':
            centralDict.update(transportLauncher.get(centralDict['couplingInfo'][2])(centralDict))

            centralDict.update(speciationLauncher.get(centralDict['couplingInfo'][1])(centralDict))    

        elif centralDict['couplingInfo'][0]=='Strang':
            centralDict['dtStep'] = dt/2
            centralDict.update(transportLauncher.get(centralDict['couplingInfo'][2])(centralDict))
            centralDict['dtStep'] = dt
            centralDict.update(speciationLauncher.get(centralDict['couplingInfo'][1])(centralDict))
            centralDict['dtStep'] = dt/2
            centralDict.update(transportLauncher.get(centralDict['couplingInfo'][2])(centralDict))

        elif centralDict['couplingInfo'][0]=='Alternative':
            if l%2==0:
                centralDict.update(transportLauncher.get(centralDict['couplingInfo'][2])(centralDict))
                centralDict.update(speciationLauncher.get(centralDict['couplingInfo'][1])(centralDict))
            else:
                centralDict.update(speciationLauncher.get(centralDict['couplingInfo'][1])(centralDict))
                centralDict.update(transportLauncher.get(centralDict['couplingInfo'][2])(centralDict))

        elif centralDict['couplingInfo'][0]=='Additive':
            commMtrxIC = centralDict['commMtrx'].copy()
            centralDict.update(transportLauncher.get(centralDict['couplingInfo'][2])(centralDict))

            commMtrxTrspt = centralDict['commMtrx'].copy()
            centralDict['commMtrx'] = commMtrxIC.copy()
            centralDict.update(speciationLauncher.get(centralDict['couplingInfo'][1])(centralDict))

            commMtrx_Final = centralDict['commMtrx'][centralDict['systemSpeciation']] + commMtrxTrspt[centralDict['systemSpeciation']] - commMtrxIC[centralDict['systemSpeciation']]
            commMtrx_Final = pd.concat([centralDict['commMtrx'][centralDict["anythingButSpecies"]],commMtrx_Final], axis=1)
            commMtrx_Final.columns = centralDict['commMtrx'].columns

            centralDict['commMtrx'] = commMtrx_Final.copy()
            
        elif centralDict['couplingInfo'][0]=='Symmetrical': # Output solely for last symmetrical coupling ..
            commMtrxIC = centralDict['commMtrx'].copy()
            centralDict.update(transportLauncher.get(centralDict['couplingInfo'][2])(centralDict))
            centralDict.update(speciationLauncher.get(centralDict['couplingInfo'][1])(centralDict))
            commMtrxSym1 = centralDict['commMtrx'].copy()
            print('\\')
            centralDict['commMtrx'] = commMtrxIC.copy() 
            centralDict.update(speciationLauncher.get(centralDict['couplingInfo'][1])(centralDict))
            centralDict.update(transportLauncher.get(centralDict['couplingInfo'][2])(centralDict))
            
            commMtrx_Final = (commMtrxSym1 + centralDict['commMtrx'])/2
            centralDict['commMtrx'] = commMtrx_Final.copy()
        
        stepTime = time.time() - PICCTSstart
        
        centralDict['commMtrx'].to_csv(os.path.join(paths['CouplingHistory'], f"CommMtrx_{l+1}.txt"), index=False, header=True, sep='\t')

        if t!=max(dtPICCTS):
            if 'extractFromDbTime' in centralDict:
                print(f"Remaining calculation time ~ {writeTime((max(dtPICCTS) * stepTime - timeStepReprise - centralDict['extractFromDbTime'])/ (t-timeStepReprise) - stepTime,2)}")
            else:
                print(f"Remaining calculation time ~ {writeTime((max(dtPICCTS) * stepTime - timeStepReprise)/ (t-timeStepReprise) - stepTime,2)}")

    with open("warning.log", "a") as f:
        f.write("PICCTS : Coupling completed\n")
        f.write(f"PICCTS : {centralDict['warningRun']} warning(s) have occured during the coupling\n\n")
    
    if centralDict['warningRun']:
        print(f"Oh no, {centralDict['warningRun']} warnings occured :( ! See warning.log file.")
    
    with open("warning.log", "a") as warningLog:
        chem = centralDict["couplingInfo"][1]
        trspt = centralDict["couplingInfo"][2]
        warningLog.write(f"PICCTS : Total coupling calculation time : {writeTime(time.time() - PICCTSstart - centralDict['waitingTime'])}\n")
        if centralDict["extractDBTime"]:
            warningLog.write(f"PICCTS : Extracting {chem} database time : {writeTime(centralDict['extractDBTime'])}\n")
        warningLog.write(f"{chem} : Total {chem} calculation time : {writeTime(centralDict[f'{chem}TotalTime'])}\n")
        warningLog.write(f"{trspt} : Total {trspt} calculation time : {writeTime(centralDict[f'{trspt}TotalTime'])}\n\n")
        warningLog.write(f"{chem} : Wall clock time of total speciation calculation : {writeTime(centralDict[f'{chem}CalcTime_WallClock'])}\n")
        warningLog.write(f"{chem} : Processor time of total speciation calculation : {writeTime(centralDict[f'{chem}CalcTime_ProcessorTime'])}\n")
        warningLog.write(f"{chem} : Wall clock time of interfacing data : {writeTime(centralDict[f'{chem}InterfTime_WallClock'])}\n")

        warningLog.write(f"{chem} : Initialisation time : {writeTime(centralDict[f'{chem}InitTime'])}\n\n")
        
        warningLog.write(f"{trspt} : Wall clock time of total transport calculation : {writeTime(centralDict[f'{trspt}CalcTime_WallClock'])}\n")
        warningLog.write(f"{trspt} : Processor time of total transport calculation : {writeTime(centralDict[f'{trspt}CalcTime_ProcessorTime'])}\n")
        warningLog.write(f"{trspt} : Wall clock time of interfacing data : {writeTime(centralDict[f'{trspt}InterfTime_WallClock'])}\n")
        
        if trspt == 'COMSOL': 
            warningLog.write(f"{trspt} : Licence waiting time: {writeTime(centralDict['COMSOLWaitingTime'])} (this may only apply for shared licences between users)\n")

        warningLog.write(f"\nPICCTS : Interfacing time from PICCTS coupler : {writeTime(time.time()-PICCTSstart-centralDict[f'{chem}TotalTime']-centralDict[f'{trspt}TotalTime'])}\n")



    if getattr(PICCTS_input, "description", None): print(f"Description = {PICCTS_input.description}")
    print(f"Total coupling time : {writeTime(time.time() - PICCTSstart - centralDict['waitingTime'],2)}")
    