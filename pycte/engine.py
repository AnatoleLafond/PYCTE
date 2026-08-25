import time
pyctestart = time.time()
import pandas as pd
import sys 
import importlib.util
import os
import shutil
import difflib
import numpy as np
import re
from pathlib import Path

# current_dir = Path(__file__).parent
# if str(current_dir) not in sys.path:
#     sys.path.insert(0, str(current_dir))

# with open("store.txt", 'r', encoding='utf-8') as fichier:
#     pathpycteInput =  Path(fichier.readline().strip())
#     nameInput =  fichier.readline().strip()


# module_name = os.path.splitext(nameInput)[0]
# file_path = os.path.join(pathpycteInput, nameInput)
# spec = importlib.util.spec_from_file_location(module_name, file_path)

# pycte_input = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(pycte_input)






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

def main(pycte_input):
    
    pathpycteInput = Path(os.getcwd())
    maxTime = getattr(pycte_input, 'maxTime', 1)
    timeStep = getattr(pycte_input, 'timeStep', None)
    stepReprise = getattr(pycte_input, 'stepReprise', 0)
    dtpycte = getattr(pycte_input, 'dtpycte', [])
    if dtpycte and dtpycte[0]==0: del dtpycte[0]
    
    if not dtpycte:
        dtpycte = [timeStep]
        while dtpycte[-1] < maxTime:
            dtpycte += [dtpycte[-1] + timeStep]
    
    if stepReprise:
        # stepReprise was the last completed step. The Xeme step is the (X-1)eme step for pycte
        # So the Xeme step is the step to be completed
        startTimeStep = start = stepReprise
        timeStepReprise = dtpycte[stepReprise-1]
    else:
        timeStepReprise = 0
        startTimeStep = start = 0

    
    
    def writeTime(tps, arr=10):
        if tps >= 3600 * 24:
            return f"{tps / (3600 * 24):.{arr}f} d"
        elif tps >= 3600:
            return f"{tps / 3600:.{arr}f} h"
        elif tps >= 60:
            return f"{tps / 60:.{arr}f} min"
        elif tps < 1:
            return f"{tps *1000 :.{arr}f} msec"
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
            dataframe.columns = coord + inputHeaders
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
    if getattr(pycte_input, 'speciesChargeGeometry', None):
        for (maille, spc) in getattr(pycte_input, 'speciesChargeGeometry', None): 
            maillesChargeGeom += [maille]
            specieChargeGeom += [spc]
    
    phases = getattr(pycte_input, "phases", "")
    if getattr(pycte_input, "fixpH", None):
        phases += "Fix_ph\nH+=H+; log_k 0"
    
    
    
    
    outputFolder= ["PrimarySpecies","Speciation","CouplingHistory"]
    
    paths = {} # variables are keys, paths are values
    
    if getattr(pycte_input, 'intermediateOutput', True) :
        for i,doss in enumerate(outputFolder):
                paths[doss] = os.path.join(pathpycteInput, outputFolder[i])
                # if os.path.exists(paths[doss]) and not stepReprise:
                #     shutil.rmtree(paths[doss]) # Quite dangerous but I like risk (will permanently delete your previous files when running a new pycte run)
                #     os.makedirs(paths[doss])
                # elif not stepReprise:
                #     os.makedirs(paths[doss])
    
    
    
    # outputName = ["trsptPath","chemPath","initialConditions"]
    # outputDefault = ["comsol.mph","phreeqc.dat","ic.txt"]
    # for i, txt in enumerate(outputName):
    #     if not getattr(pycte_input, txt, None):
    #         paths[txt] = Path(getattr(pycte_input, txt, outputDefault[i]))
    #     else:
    #         paths[txt] = Path(getattr(pycte_input, txt))

    
    
    
    centralDict = { # gather keywords which do not depend on components
        "warningRun": 0,
        "inputPath" : pathpycteInput,
        "AcidicEcho": pd.DataFrame(),
        "waitingTime" : 0,
        "couplingInfo" :[normalize_choice(getattr(pycte_input, 'operatorSplitting', 1), operator_map, 'operatorSplitting'),
                         normalize_choice(getattr(pycte_input, 'chemModule', 1), chem_map, 'chemModule'),
                         normalize_choice(getattr(pycte_input, 'trsptModule', 1), trspt_map, 'trsptModule')],
        
        # "couplingInfo" : [ ['SNIA','Strang','Alternative','Additive','Symmetrical'][getattr(pycte_input, 'operatorSplitting', 1)-1],
        #                   ['PhreeqC','xGEMS','ORCHESTRA'][getattr(pycte_input, 'chemModule', 1)-1],
        #                   ['COMSOL'][getattr(pycte_input, 'trsptModule', 1)-1],
        #     ],
        
        "paths" : paths,
        "system" : getattr(pycte_input, 'system', 1),
        "stepReprise" : getattr(pycte_input, 'stepReprise', False),
        "PIDnbr": getattr(pycte_input, 'PIDnbr', 1),
        "PIDextract": getattr(pycte_input, 'PIDextract', 3),
        "systemSpeciation" : getattr(pycte_input, 'systemSpeciation', None),
        "timeUnit" : getattr(pycte_input, 'timeUnit', 's'),
        "geometry" : getattr(pycte_input, 'geometry', 1),

        "coord" : ['x', 'y', 'z'][:getattr(pycte_input, 'geometry', 1)],
        "chemModule" : getattr(pycte_input, 'chemModule', 1),
        "trsptModule" : getattr(pycte_input, 'trsptModule', 1),
        "firstStepEquilibrium" : getattr(pycte_input, 'firstStepEquilibrium', False),
        'nonTrivialDecomposition' : getattr(pycte_input, 'nonTrivialDecomposition', None),
        "preliminarEquilibrium" : getattr(pycte_input, 'preliminarEquilibrium', False),
        'extractDBTime' : 0,
        'PIDchem' : getattr(pycte_input, 'PIDchem', 1),
        'PIDtrspt' : getattr(pycte_input, 'PIDtrspt', 1),
        'MultiCompoundTransport' : getattr(pycte_input, 'MultiCompoundTransport', False),
        "initialConditions" : Path(getattr(pycte_input, 'initialConditions', os.path.join(pathpycteInput, 'ic.txt'))),
        "dtpycte" : dtpycte,
        }
    

    
    crossDep = getattr(pycte_input, 'crossDependencies', None)
    if crossDep :
        if not crossDep.get('speciation'):
            crossDep['speciation'] = []
        if not crossDep.get('transport'):
            crossDep['transport'] = []
        from . import crossDepenciesManager

        centralDict['crossDependencies'] = crossDepenciesManager.crossDep(centralDict, crossDep)

    else: 
        centralDict['crossDependencies'] = None


    centralDict.update({"commMtrx": readInputFile(centralDict['initialConditions'],['x', 'y', 'z'][:getattr(pycte_input, 'geometry', 1)],
    (centralDict['systemSpeciation'] + (centralDict['crossDependencies']['totalCrossDep'] if centralDict['crossDependencies'] else [])))})

    centralDict.update({"anythingButSpecies" : [c for c in list(centralDict["commMtrx"].columns) if c not in centralDict['systemSpeciation']]})

    print(f"""
####   #  ####  ####  #####  ####     {centralDict['couplingInfo'][0]} splitting :
#  #   #  #     #       #    #        {centralDict['couplingInfo'][1]}--->
####   #  #     #       #    ####            <---{centralDict['couplingInfo'][2]}
#      #  #     #       #       #     {maxTime-timeStepReprise}{centralDict['timeUnit']} in {len(dtpycte)-stepReprise} steps        
#      #  ####  ####    #    ####     
       """, flush=True)
    
    if centralDict["couplingInfo"][1] == 'PhreeqC':
        centralDict['chemPath'] = Path(getattr(pycte_input, 'chemPath', os.path.join(pathpycteInput, 'phreeqc.dat')))
        from . import phreeqc
        from . import extractDB
        centralDict.update(extractDB.extract(centralDict))

        speciationLauncher = {'PhreeqC': phreeqc.spct}                
        
        centralDict.update({
        # "primToSecSpecies" : getattr(pycte_input, 'primToSecSpecies', {}),
        "current" : getattr(pycte_input, 'current', None),
        "SI" : getattr(pycte_input, "SI", {ph : 0 for ph in centralDict['primarySpecies']['phases']}  ),
        "mineralReversibility" : getattr(pycte_input, "mineralReversibility", {ph : "" for ph in centralDict['primarySpecies']['phases']}),
        
        "cutoffs" : {'solution' : getattr(pycte_input, 'cutoffAq', 1e-20),
                    'phases' : getattr(pycte_input, 'cutoffPha', -1e-99),
                    'exchange' : getattr(pycte_input, 'cutoffEch', 1e-20),
                    'surface' : getattr(pycte_input, 'cutoffSurf', 1e-20)},

        "database": ["",getattr(pycte_input, 'masterSpecies', None),
                     getattr(pycte_input, 'solutionSpecies', None),
                     phases,
                     getattr(pycte_input, 'masterExchange', None),
                     getattr(pycte_input, 'exchangeSpecies', None),
                     getattr(pycte_input, 'masterSurface', None),
                     getattr(pycte_input, 'surfaceSpecies', None),
                     getattr(pycte_input, 'kineticRates', None),
                     getattr(pycte_input, 'kinetics', None),],
        "fixpH": getattr(pycte_input, "fixpH", None),
        "userVarBool": getattr(pycte_input, "userVarBool", {}),
        "userVarList": getattr(pycte_input, "userVarList", {}),
        "maillesChargeGeom" : maillesChargeGeom,
        "solMod" : getattr(pycte_input, 'solMod', False),
        "step_divide" : getattr(pycte_input, 'step_divide', None),
        "speciesCharge" : getattr(pycte_input, "speciesCharge", ['pH']),
        "speciesChargeGeometry" : getattr(pycte_input, 'speciesChargeGeometry', None),
        "speciationCharge" : getattr(pycte_input, "speciationCharge", False),
        "pHdefault" : getattr(pycte_input, "pHdefault", 7),
        "tempDefault" : getattr(pycte_input, "tempDefault", 25),
        "distAnodCathTotale" : getattr(pycte_input, "distAnodCathTotale", None),
        "catholyte" : getattr(pycte_input, "catholyte", None),
        "anolyte" : getattr(pycte_input, "anolyte", None),
        "solModCharge" : getattr(pycte_input, "solModCharge", 0),
        "acidicTrspt": getattr(pycte_input, 'acidicTrspt', False),
        "water" : getattr(pycte_input, 'water', False),
        "kinetics" : getattr(pycte_input, 'kinetics', False),
        'supplementarySolution' : getattr(pycte_input, 'supplementarySolution', None),
        "surfaceCounterIons" : getattr(pycte_input, 'surfaceCounterIons', False),
        "beforeTrsptMtrx" : pd.DataFrame(),
        "PhreeqCCalcTime_WallClock": 0,
        "PhreeqCCalcTime_ProcessorTime": 0,
        "PhreeqCInterfTime_WallClock": 0,
        "PhreeqCInitTime" : 0,
        "PhreeqCTotalTime" : 0,

        })
        
    elif centralDict["couplingInfo"][1] == 'xGEMS':

        
        centralDict.update({
        "chemPath" : Path(getattr(pycte_input, 'chemPath', os.path.join(pathpycteInput, 'dat.lst'))),
        # "independentComponents" : getattr(pycte_input, "independentComponents", None),
        "xGEMSClockTime" : 0,
        # "constantSpecies" : getattr(pycte_input, 'constantSpecies', {}),
        "xGEMSPrcsTime" : 0,
        # "transportedSpecies" :  getattr(pycte_input, "transportedSpecies"),
        "xGEMSCalcTime_WallClock": 0,
        "xGEMSCalcTime_ProcessorTime": 0,
        "xGEMSInterfTime_WallClock": 0,
        "xGEMSInitTime" : 0,
        "xGEMSTotalTime" : 0,})

        from . import extractDB
        
        centralDict.update(extractDB.extract(centralDict))

        centralDict.update({
        "transportedIC" : [s for s in centralDict['independentComponents'] if s not in ( ['Zz'] + centralDict['crossDependencies']['speciation']['total'] if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation') else [])  ]
        })

        from . import gems
        
        speciationLauncher = {'xGEMS': gems.spct}


        os.chdir(pathpycteInput)
        
        
    elif centralDict["couplingInfo"][1] == 'ORCHESTRA':
        from . import orchestra
        
        # centralDict['firstIC'] = pd.read_csv('literbulkIC.txt',sep=r"\s+",comment="%",dtype=float) # working IC from HM

        speciationLauncher = {
            'ORCHESTRA': orchestra.spct}
        
        centralDict.update({
        "attributes":getattr(pycte_input, "attributes", None),
        "firstIC" : getattr(pycte_input, "firstIC", None),
        "chemPath" : Path(getattr(pycte_input, 'chemPath', os.path.join(pathpycteInput, 'chemistry1.inp'))),
        "primarySpecies" : [getattr(pycte_input, "primarySpeciesAq", []),
                           getattr(pycte_input, "primarySpeciesPha", []),
                           getattr(pycte_input, "primarySpeciesSurf", []),
                           getattr(pycte_input, "speciationEch", []),
                           getattr(pycte_input, "primarySpeciesPhantom", ['OH','H']),
                           (getattr(pycte_input, "primarySpeciesAq", []) + getattr(pycte_input, "primarySpeciesPha", []) +
                            getattr(pycte_input, "primarySpeciesSurf", [])+getattr(pycte_input, "speciationEch", []))
                           ],
        "nrThreads" : getattr(pycte_input, "nrThreads",1),
        "phases" : getattr(pycte_input, "phases", []),
        "waterMass" : getattr(pycte_input, "waterMass", [1]*len(centralDict['commMtrx'])),
        "primarySpeciesEchSorbed" :  getattr(pycte_input, "primarySpeciesEchSorbed", {}),
        "couplingFormalism" : getattr(pycte_input, "couplingFormalism", 'tot'),
        "ORCHESTRACalcTime_WallClock": 0,
        "ORCHESTRACalcTime_ProcessorTime": 0,
        "ORCHESTRAInterfTime_WallClock": 0,
        "ORCHESTRAInitTime" : 0,
        "ORCHESTRATotalTime" : 0,
        "transportedSpecies" :  getattr(pycte_input, "transportedSpecies"),
        })
        
        from . import extractDB
        centralDict.update(extractDB.extract(centralDict))
        
        centralDict.update({
        "inputVariableOrchestra": getattr(pycte_input, "inputVariableOrchestra", None),
        'outputVariableOrchestra': getattr(pycte_input, "outputVariableOrchestra", None)
            })
        
        from .parse_reactions import build_stoich_matrix
        stoich = build_stoich_matrix(centralDict['reactions'], centralDict['primary_entities'], centralDict['systemSpeciation'])
        cols = [s for s in centralDict['systemSpeciation'] if s not in centralDict['phases']]
        centralDict.update({
        'stoich': stoich,
        "colAq": cols,
        })
        
    else: 
        print(f'No module is associated with chemModule = {getattr(pycte_input, "chemModule", 1)}')
        sys.exit()


    if centralDict["couplingInfo"][2] == 'COMSOL':
        from . import comsol
        
        transportLauncher = {
            "COMSOL": comsol.trspt, 
        }

        
        centralDict.update({
        "trsptPath" : Path(getattr(pycte_input, 'trsptPath', os.path.join(pathpycteInput, 'comsol.mph'))),
        "comsolTags" : [getattr(pycte_input, 'comsolComp', 'comp1'),
                       getattr(pycte_input, 'comsolIntFonction', 'int1'),
                       getattr(pycte_input, "comsolStudy", 'std1'), 
                    ],
        # "comsolCoupling": getattr(pycte_input, 'comsolCoupling', 'data1'),
        "outputComsol": [],
        "comsolVTU":  getattr(pycte_input, 'comsolVTU', True),
        "comsolCore" : getattr(pycte_input, 'comsolCore', None),
        "COMSOLCalcTime_WallClock": 0,
        "COMSOLCalcTime_ProcessorTime": 0, 
        "COMSOLInterfTime_WallClock": 0,
        "COMSOLInitTime" : 0,
        "COMSOLTotalTime" : 0,
        "COMSOLWaitingTime" : 0
        })

        if getattr(pycte_input, 'outputComsol', None):
            for i,out in enumerate(getattr(pycte_input, 'outputComsol', None)):
                centralDict['outputComsol'] += [out]
                centralDict['paths'][f'Transport{out}'] =  os.path.join(pathpycteInput, f'outputTransport{out}')
                centralDict['paths'][f'TransportVTU{out}'] =  os.path.join(pathpycteInput, f'outputVTU{out}')

    elif centralDict["couplingInfo"][2] == 'nativeTransport':
        from . import nativeTransport

        transportLauncher = {
            "nativeTransport": nativeTransport.trspt, 
        }

        centralDict.update({
            "velocity" : getattr(pycte_input, 'velocity', 0),
            "advection" : getattr(pycte_input, 'advection', None),
            "FickDiffusion" : getattr(pycte_input, 'FickDiffusion', None),
            "porousTransport" : getattr(pycte_input, 'porousTransport', None),
            "ADE" : getattr(pycte_input, 'ADE', None),
            # "diffCoeff" : 
            "dispersivity" : getattr(pycte_input, 'dispersivity', 0),
            "firstBoundary" : getattr(pycte_input, 'firstBoundary', {cle: 0 for cle in centralDict['systemSpeciation']}),
            "secondBoundary" : getattr(pycte_input, 'secondBoundary', {cle: 0 for cle in centralDict['systemSpeciation']}),
            "boundaryConditions" : getattr(pycte_input, 'boundaryConditions', ['flux', 'flux']),
            "nativeTransportClockTime" : 0,
            # "nodeSize" : getattr(pycte_input, 'nodeSize', 1),
            "advectionScheme" : 'upwind',
            "diffusionScheme" : 'fick',
            "porosity" : (
                    np.array([getattr(pycte_input, "porosity",1)] * len(centralDict["commMtrx"]))
                    if isinstance(getattr(pycte_input, "porosity", 1), (int, float))
                    else np.array(getattr(pycte_input, "porosity", [1] * len(centralDict["commMtrx"])))
                ),
            "area" : np.array(getattr(pycte_input, 'area', [1]*len(centralDict["commMtrx"]))),
            "nativeTransportCalcTime_WallClock": 0,
            "nativeTransportCalcTime_ProcessorTime": 0,
            "nativeTransportInterfTime_WallClock": 0,
            "nativeTransportTotalTime" : 0,
            'nativeTransportInitTime' : 0,
        })

        tempDiffCoeff = getattr(pycte_input, 'diffCoeff', 0)
        x = centralDict["commMtrx"][centralDict['coord']].copy().to_numpy().reshape(-1)



        f_left = np.zeros(len(centralDict["commMtrx"]))
        f_right = np.zeros(len(centralDict["commMtrx"]))
        
        f_left[1:]   = x[1:] - x[:-1]
        f_right[:-1] = x[1:] - x[:-1]

        centralDict.update({'f_right' : f_right, 'f_left' : f_left})
        dx = np.diff(x)
        dx_min = np.min(dx)
        
        dx_flux = x[1:] - x[:-1]
        nodeSize = np.empty(len(x))
        nodeSize[1:-1] = (x[2:] - x[:-2]) / 2     
        nodeSize[0]    = (x[1]  - x[0])  #/ 2       
        nodeSize[-1]   = (x[-1] - x[-2]) #/ 2    
        centralDict.update({"x": x,
                            "dx" : dx,
                            "dx_flux" : dx_flux,
                            'nodeSize': nodeSize,
                            'centralDictCoord' : centralDict["commMtrx"][centralDict['coord']].copy()})
    
        
        if centralDict['timeUnit'] == 'y':
            dt = timeStep*3600*24*365.25
        elif centralDict['timeUnit'] == 'd':
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
            print(f"Sub-cycling advection within {n_adv} sub-steps, dt_max={writeTime(round(dx_min / abs(centralDict['velocity']),4),5)}")
        
        if isinstance(tempDiffCoeff, (int,float)):
            centralDict['diffCoeff'] = np.full(len(centralDict["commMtrx"]),tempDiffCoeff)
            # diffcoeff = centralDict['diffCoeff']
        else:
            centralDict['diffCoeff'] = tempDiffCoeff
        
        if centralDict['velocity'] != 0  and centralDict['dispersivity']:
            centralDict['diffCoeff']  = centralDict['diffCoeff'] + centralDict['velocity'] * centralDict['dispersivity']
                    
        if (np.mean(centralDict['diffCoeff']) +centralDict['velocity']*centralDict['dispersivity']):
            dt_diff = dx_min**2 / (2 * (np.mean(centralDict['diffCoeff']) + centralDict['velocity']*centralDict['dispersivity']))
            if dt > round(dt_diff,4):
                n_diff = int(round(dt / dt_diff))
                print(f"Sub-cycling diffusion/dispersion within {n_diff} sub-steps, dt_max={writeTime(round(dt_diff,4),4)}")
        
        centralDict.update({'subCyclingDiff' : n_diff,
                            'subCyclingAdv' : n_adv})
        
          # (nb mailles, largeur maille)
        meshing = getattr(pycte_input, 'meshing', 0)
        if not meshing :
            x = np.asarray(x)
            largeur = x[1] - x[0]
            zones = [(len(x), largeur)]
            dxHalfCell = np.concatenate([np.full(nb, largeur / 2) for nb, largeur in zones])
        else:    
            dxHalfCell = np.concatenate([np.full(nb, largeur/2) for nb, largeur in meshing ])
        
        
        centralDict.update({'dxHalfCell': dxHalfCell})
 
 
             
    elif centralDict["couplingInfo"][2] == 'PFLOTRAN':
        import pflotran

        transportLauncher = {
            "PFLOTRAN": pflotran.trspt, 
        }
    
        centralDict.update({
            "trsptPath" : Path(getattr(pycte_input, 'trsptPath', os.path.join(pathpycteInput, 'pflotran.in'))),
            "PFLOTRANCalcTime_WallClock": 0,
            "PFLOTRANCalcTime_ProcessorTime": 0,
            "PFLOTRANInterfTime_WallClock": 0,
            "PFLOTRANInitTime" : 0,
            "PFLOTRANTotalTime" : 0,})

    else:
        print(f'No module is associated with trsptModule = {getattr(pycte_input, "trsptModule", 1)}')
        sys.exit()
    
    for p, path in centralDict['paths'].items():
        path = Path(path)
        if path.suffix:
            continue
        if path.exists():
            shutil.rmtree(path)
    
        path.mkdir(parents=True, exist_ok=True)
        
    if not stepReprise:
        with open("warning.log", "w") as warningLog:
            warningLog.write(f"pycte : {centralDict['couplingInfo'][0]} {centralDict['couplingInfo'][1]}-{centralDict['couplingInfo'][2]}.\n")
            if getattr(pycte_input, "description", None): warningLog.write(f"pycte : Description :\n{pycte_input.description}\n")
            warningLog.write(f"pycte : Time steps ({centralDict['timeUnit']}) :\n")    
            for dtt in dtpycte:
                if dtt == dtpycte[-1]: warningLog.write(f"{dtt}.\n")
                else : warningLog.write(f"{dtt}, ")
    else:
        with open("warning.log", "a") as warningLog:
            warningLog.write(f"pycte, time = {dtpycte[stepReprise]}{centralDict['timeUnit']} : run reprise ...\n")

    for l,t in enumerate(dtpycte[start:], start = startTimeStep):
        
        if l==startTimeStep and not stepReprise: dt = t
        else: dt = t-dtpycte[l-1]

        centralDict.update({
            "dtStep":dt,
            "tStep":t,
            "lStep":l,
            })
        
        print(f"#######  step n°{l+1}/{len(dtpycte)}, dt = {dt}{centralDict['timeUnit']}  #######")
        
        if centralDict['preliminarEquilibrium']:
            centralDict['beforeTrsptMtrx'] = centralDict["commMtrx"].copy() # not OS dependent ..

        if centralDict["firstStepEquilibrium"] and l==startTimeStep:
            centralDict["dtStep"]=0

            centralDict.update(speciationLauncher.get(centralDict['couplingInfo'][1])(centralDict))
            if centralDict['preliminarEquilibrium']:
                centralDict['beforeTrsptMtrx'] = centralDict["commMtrx"].copy()
            centralDict["firstStepEquilibrium"]=False
            centralDict["dtStep"]=dt

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
            centralDictIC = centralDict["commMtrx"].copy()
            centralDict.update(transportLauncher.get(centralDict['couplingInfo'][2])(centralDict))

            centralDictTrspt = centralDict["commMtrx"].copy()
            centralDict["commMtrx"] = centralDictIC.copy()
            centralDict.update(speciationLauncher.get(centralDict['couplingInfo'][1])(centralDict))

            centralDict_Final = centralDict["commMtrx"][centralDict['systemSpeciation']] + centralDictTrspt[centralDict['systemSpeciation']] - centralDictIC[centralDict['systemSpeciation']]
            centralDict_Final = pd.concat([centralDict["commMtrx"][centralDict["anythingButSpecies"]],centralDict_Final], axis=1)
            centralDict_Final.columns = centralDict["commMtrx"].columns

            centralDict["commMtrx"] = centralDict_Final.copy()
            
        elif centralDict['couplingInfo'][0]=='Symmetrical': # Output solely for last symmetrical coupling ..
            centralDictIC = centralDict["commMtrx"].copy()
            centralDict.update(transportLauncher.get(centralDict['couplingInfo'][2])(centralDict))
            centralDict.update(speciationLauncher.get(centralDict['couplingInfo'][1])(centralDict))
            centralDictSym1 = centralDict["commMtrx"].copy()
            print('\\')
            centralDict["commMtrx"] = centralDictIC.copy() 
            centralDict.update(speciationLauncher.get(centralDict['couplingInfo'][1])(centralDict))
            centralDict.update(transportLauncher.get(centralDict['couplingInfo'][2])(centralDict))
            
            centralDict_Final = (centralDictSym1 + centralDict["commMtrx"])/2
            centralDict["commMtrx"] = centralDict_Final.copy()
        
        stepTime = time.time() - pyctestart
        
        if True : # (centralDict['lStep']+1) % 500 == 0
           centralDict["commMtrx"].to_csv(os.path.join(centralDict['paths']['CouplingHistory'], f"CouplingHistory_{centralDict['lStep']}.txt"), index=False, header=True, sep='\t')
        
        if t != maxTime:
            if 'extractFromDbTime' in centralDict:
                print(f"Remaining calculation time ~ {writeTime((maxTime * stepTime - timeStepReprise - centralDict['extractFromDbTime'])/ (t-timeStepReprise) - stepTime,2)}")
            else:
                print(f"Remaining calculation time ~ {writeTime((maxTime * stepTime - timeStepReprise)/ (t-timeStepReprise) - stepTime,2)}")


    centralDict["commMtrx"].to_csv(os.path.join(centralDict['paths']['CouplingHistory'], f"CouplingHistory_{centralDict['lStep']}.txt"), index=False, header=True, sep='\t')
            

    with open("warning.log", "a") as f:
        f.write("pycte : Coupling completed\n")
        f.write(f"pycte : {centralDict['warningRun']} warning(s) have occured during the coupling\n\n")
    
    if centralDict['warningRun']:
        print(f"Oh no, {centralDict['warningRun']} warnings occured :( ! See warning.log file.")
    
    with open("warning.log", "a") as warningLog:
        chem = centralDict["couplingInfo"][1]
        trspt = centralDict["couplingInfo"][2]
        warningLog.write(f"pycte : Total coupling calculation time : {writeTime(time.time() - pyctestart - centralDict['waitingTime'])}\n")
        warningLog.write(f"{chem} : Total {chem} time : {writeTime(centralDict[f'{chem}TotalTime'])}\n")
        warningLog.write(f"{trspt} : Total {trspt} time : {writeTime(centralDict[f'{trspt}TotalTime'])}\n")
        warningLog.write(f"pycte : Communication time between {chem} and {trspt} : {writeTime(time.time()-pyctestart-centralDict[f'{chem}TotalTime']-centralDict[f'{trspt}TotalTime'])}\n\n")

        warningLog.write(f"{chem} : Wall clock time of total speciation calculation : {writeTime(centralDict[f'{chem}CalcTime_WallClock'])}\n")
        warningLog.write(f"{chem} : Processor time of total speciation calculation : {writeTime(centralDict[f'{chem}CalcTime_ProcessorTime'])}\n")
        warningLog.write(f"{chem} : Wall clock time of interfacing data : {writeTime(centralDict[f'{chem}InterfTime_WallClock'])}\n")
        if centralDict["extractDBTime"]:
            warningLog.write(f"{chem} : Database extraction : {writeTime(centralDict['extractDBTime'])}\n")
        warningLog.write(f"{chem} : Initialisation time : {writeTime(centralDict[f'{chem}InitTime'])}\n\n")
        
        warningLog.write(f"{trspt} : Wall clock time of total transport calculation : {writeTime(centralDict[f'{trspt}CalcTime_WallClock'])}\n")
        warningLog.write(f"{trspt} : Processor time of total transport calculation : {writeTime(centralDict[f'{trspt}CalcTime_ProcessorTime'])}\n")
        warningLog.write(f"{trspt} : Wall clock time of interfacing data : {writeTime(centralDict[f'{trspt}InterfTime_WallClock'])}\n")
        warningLog.write(f"{trspt} : Initialisation time : {writeTime(centralDict[f'{trspt}InitTime'])}\n\n")
        
        if trspt == 'COMSOL': 
            warningLog.write(f"{trspt} : Licence waiting time: {writeTime(centralDict['COMSOLWaitingTime'])} (this may only apply for shared licences between users)\n")




    if getattr(pycte_input, "description", None): print(f"Description = {pycte_input.description}")
    print(f"Total coupling time : {writeTime(time.time() - pyctestart - centralDict['waitingTime'],2)}")
    