import time
import pandas as pd
import math
import sys 
import importlib.util
import os
import shutil
import difflib
import numpy as np
import re
from pathlib import Path

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

from . import OSschemes                      # schemas de couplage (operator splitting) : un pas transport / speciation
from . import warningManager                 # warnings du run -> warning.log (tous, a chaque occurrence)

operator_map = OSschemes.ALIASES             # noms canoniques et alias de operatorSplitting

chem_map = {
    'none': [0, '0', 'none', None, 'transportOnly'],
    'PhreeqC': [1, '1', 'phreeqc','PhreeqC','Phreeqc'],
    'xGEMS': [2, '2', 'xgems','xGEMS','gems'],
    'ORCHESTRA': [3, '3', 'orchestra','ORCHESTRA', 'Orchestra'],
    'nativeKinetics': [4, '4', 'nativeKinetics','kinetics', 'nativekinetics'],
    'nativeSpeciation': [5, '5', 'nativeSpeciation','speciation', 'nativespeciation'],
}

trspt_map = {
    'COMSOL': [1, '1', 'comsol','Comsol','COMSOL'],
    'nativeTransport': [2, '2', 'nativeTransport','native','trspt'],
    'PFLOTRAN': [3, '3', 'Pflotran','pflotran','PFLOTRAN'],
}

def main(pycte_input):
    # les warnings emis pendant le run (y compris avant sys.exit ou une erreur) finissent dans warning.log
    warningManager.begin()
    try:
        return _main(pycte_input)
    finally:
        warningManager.end()

def _main(pycte_input):
    pyctestart = time.time()
    pathpycteInput = Path(os.getcwd())
    maxTime = getattr(pycte_input, 'maxTime', 1)
    timeStep = getattr(pycte_input, 'timeStep', 1)
    dtpycte = getattr(pycte_input, 'dtpycte', [])
    
    if dtpycte and dtpycte[0]==0: del dtpycte[0]
    if not dtpycte:
        # print(maxTime, timeStep)
        # sys.exit()
        n = math.ceil(maxTime / timeStep)
        dtpycte = (np.arange(1, n + 1) * timeStep).tolist()
       
    
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
    
    def readInputFile(txtPath, coord, inputHeaders) :

        dataframe = pd.read_csv(txtPath, sep=r"\s+", comment="%", dtype=float, header=None)
        
        if not inputHeaders:
            print('No headers !')
            print(inputHeaders)
            sys.exit()
        if len(dataframe.columns) != len(coord + inputHeaders):
            print(f"input file : {len(dataframe.columns)} columns")
            print(f"coupled variables : {len(coord + inputHeaders)}")
            print(coord + inputHeaders)
            sys.exit()
        dataframe.columns = coord + inputHeaders
        return dataframe

    maillesChargeGeom = []
    specieChargeGeom = []
    if getattr(pycte_input, 'speciesChargeGeometry', None):
        for (maille, spc) in getattr(pycte_input, 'speciesChargeGeometry', None): 
            maillesChargeGeom += [maille]
            specieChargeGeom += [spc]
    
    phases = getattr(pycte_input, "phases", "")
    if getattr(pycte_input, "fixpH", None):
        phases += "Fix_ph\nH+=H+; log_k 0"
    
    # sorties intermediaires : mot-cle output (voir outputManager.py) ; absent -> {'coupling': [dernier pas]}
    from . import outputManager
    output = outputManager.normalize(getattr(pycte_input, 'output', None), len(dtpycte))
    if getattr(pycte_input, 'intermediateOutput', None) is not None:
        warningManager.warn("pycte : intermediateOutput is no longer used and is ignored : the outputs are set by "
                            "output = {'coupling' | 'speciation' | 'transport': [steps]}")
    comsolTags = list(getattr(pycte_input, 'outputComsol', None) or [])
    paths = outputManager.folders(pathpycteInput, output, comsolTags)   # cle de centralDict['paths'] -> dossier

    # couplage : schema nomme (operatorSplitting) ou sequence definie par l'utilisateur (OSdefined, ou liste donnee
    # a operatorSplitting), voir OSschemes.py. Avec OSdefined, chemModule / trsptModule absents sont deduits des
    # solveurs nommes dans la sequence ; presents, ils doivent concorder avec elle.
    osDefined = getattr(pycte_input, 'OSdefined', None)
    osChoice = getattr(pycte_input, 'operatorSplitting', None)
    if osDefined is None and isinstance(osChoice, (list, tuple)):
        osDefined, osChoice = osChoice, None
    chemChoice = (normalize_choice(pycte_input.chemModule, chem_map, 'chemModule')
                  if hasattr(pycte_input, 'chemModule') else None)
    trsptChoice = (normalize_choice(pycte_input.trsptModule, trspt_map, 'trsptModule')
                   if hasattr(pycte_input, 'trsptModule') else None)
    osSequence = None
    if osDefined is not None:
        osSequence = OSschemes.parseDefined(osDefined, chem_map, trspt_map)
        chemChoice, trsptChoice = OSschemes.resolveModules(osSequence, chemChoice, trsptChoice)
        if osChoice is not None:
            warningManager.warn(f"pycte : operatorSplitting = {osChoice!r} is ignored : OSdefined sets the coupling")
        osName = 'OSdefined'
    else:
        osName = normalize_choice(1 if osChoice is None else osChoice, operator_map, 'operatorSplitting')
    couplingInfo = [osName,
                    chemChoice if chemChoice is not None else normalize_choice(1, chem_map, 'chemModule'),
                    trsptChoice if trsptChoice is not None else normalize_choice(1, trspt_map, 'trsptModule')]
    splittingLabel = osName
    if osSequence is not None:
        splittingLabel = f"OSdefined [{OSschemes.describe(osSequence, couplingInfo[1], couplingInfo[2])}]"
        for message in OSschemes.checkFractions(osSequence, couplingInfo[1]):
            warningManager.warn(message)


    centralDict = { # gather keywords which do not depend on components
        "warningNbr": 0,
        "inputPath" : pathpycteInput,
        "AcidicEcho": pd.DataFrame(),
        "waitingTime" : 0,
        "couplingInfo" : couplingInfo,
        "OSdefined" : [(role, fraction) for role, fraction, _ in osSequence] if osSequence else None,


        "paths" : paths,
        "system" : getattr(pycte_input, 'system', 1),
        "PIDnbr": getattr(pycte_input, 'PIDnbr', 1),
        "PIDextract": getattr(pycte_input, 'PIDextract', 1),
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
        "output" : output,
        }
    
    # print(centralDict['systemSpeciation'])
    # sys.exit()
    
    
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
    

    centralDict["commMtrx"] = readInputFile(centralDict['initialConditions'],['x', 'y', 'z'][:getattr(pycte_input, 'geometry', 1)],centralDict['systemSpeciation'])

    if not getattr(pycte_input, 'systemSpecies', None):
        if (centralDict['crossDependencies']['totalCrossDep'] if centralDict['crossDependencies'] else []):
            # PhreeqC : {'totalName': [...], 'totalPhreeqCCmd': [...]} ; autres modules (ORCHESTRA, nativeKinetics, ...) : liste
            spcTotal = centralDict['crossDependencies']['speciation']['total']
            spcTotal = spcTotal['totalName'] if isinstance(spcTotal, dict) else spcTotal
            centralDict['systemSpecies'] = [s for s in centralDict['systemSpeciation']  if s not in (spcTotal + centralDict['crossDependencies']['transport']['total']) ]
        else:
            centralDict['systemSpecies'] = centralDict['systemSpeciation'].copy()
    else:
        centralDict['systemSpecies'] =  getattr(pycte_input, 'systemSpecies')

    
    centralDict.update({"anythingButSpecies" : [c for c in list(centralDict["commMtrx"].columns) if c not in centralDict['systemSpeciation']]})

    print(f"""
####  #  #  ####  #####  ####     {splittingLabel} splitting :
#  #  #  #  #       #    #        {centralDict['couplingInfo'][1]}--->
####  ####  #       #    ####            <---{centralDict['couplingInfo'][2]}
#        #  #       #    #        {maxTime}{centralDict['timeUnit']} in {len(dtpycte)} steps        
#     ####  ####    #    ####     
       """, flush=True)
    
    # renouvellement periodique de mailles : liste de groupes {'cells', 'times', 'concentration'} (voir renouvellement.py)
    rnvBool = False
    if getattr(pycte_input, 'renouvellement', None):
        from . import renouvellement as rnv
        centralDict.update(rnv.setup(centralDict, getattr(pycte_input, 'renouvellement')))
        rnvBool = True
        
    
    
    
    
    if centralDict["couplingInfo"][1] == 'PhreeqC':

        phreeqcHeadersNames = centralDict['systemSpeciation'].copy()
        i = 0

        if centralDict['crossDependencies']:
            assert len(centralDict['systemSpecies'] + centralDict['crossDependencies']['totalCrossDep']) == len(centralDict['systemSpeciation'])
        
        for j,s in enumerate(centralDict['systemSpeciation']):
            if s not in (centralDict['systemSpecies'] + centralDict['coord']):
                phreeqcHeadersNames[j] = centralDict['crossDependencies']['totalCrossDep'][i]
                i += 1

        centralDict['phreeqcHeadersNames'] = phreeqcHeadersNames
        centralDict['phasesAsPrimary'] = getattr(pycte_input, 'phasesAsPrimary', True)
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
        # "phreeqcCommMtrxHeadersName" : [s for s in centralDict['commMtrx'].comlumns if s in centralDict['systemSpecies'] else ]
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
        "densityPhreeqC" : getattr(pycte_input, 'densityPhreeqC', 1),
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
        "kineticsSpecies" : getattr(pycte_input, 'kineticsSpecies', None),
        "PhreeqCCalcTime_WallClock": 0,
        "PhreeqCCalcTime_ProcessorTime": 0,
        "PhreeqCInterfTime_WallClock": 0,
        "PhreeqCInitTime" : 0,
        "PhreeqCTotalTime" : 0,
        })

        # molesStorage (defaut True, chemin SOLUTION_MODIFY : solMod ou MultiCompoundTransport) : commMtrx porte des
        # moles par maille (rapportees a 1 kg d'eau initial) au lieu de molalites ; la masse d'eau de chaque maille
        # (cellWaterMass, kg) suit le bilan de O au lieu d'etre ramenee a 1 kg a chaque pas. nativeTransport fait
        # diffuser les concentrations (moles / cellWaterMass) avec la teneur en eau porosity.cellWaterMass.
        centralDict['molesStorage'] = (bool(getattr(pycte_input, 'molesStorage', True))
                                       and (centralDict['solMod'] or centralDict['MultiCompoundTransport'])
                                       and not centralDict['acidicTrspt'])
        if centralDict['molesStorage']:
            # masse d'eau initiale : colonne H2O de la CI (55.50868 mol = 1 kg), sinon 1 kg
            centralDict['cellWaterMass'] = (centralDict['commMtrx']['H2O'].to_numpy(dtype=float) / 55.50868
                                            if 'H2O' in centralDict['commMtrx'] else np.ones(len(centralDict['commMtrx'])))



    elif centralDict["couplingInfo"][1] == 'xGEMS':

        
        centralDict.update({
        "chemPath" : Path(getattr(pycte_input, 'chemPath', os.path.join(pathpycteInput, 'dat.lst'))),
        # "independentComponents" : getattr(pycte_input, "independentComponents", None),
        "xGEMSClockTime" : 0,
        # "constantSpecies" : getattr(pycte_input, 'constantSpecies', {}),
        "xGEMSPrcsTime" : 0,
        # "transportedSpecies" :  getattr(pycte_input, "transportedSpecies",),
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
        # "attributes":getattr(pycte_input, "attributes", None),
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
        "inputVariableOrchestra": getattr(pycte_input, "inputVariableOrchestra", None),
        "inputVariableOrchestraSpecies": [
            s
            for s in (getattr(pycte_input, "inputVariableOrchestra", None) or [])
            if (
                not (centralDict['crossDependencies'] and centralDict['crossDependencies'].get("speciation"))
                or s not in centralDict["crossDependencies"]["speciation"]["total"]
            )
        ],        
        'outputVariableOrchestra': getattr(pycte_input, "outputVariableOrchestra", None)
        })
        
        from . import extractDB
        centralDict.update(extractDB.extract(centralDict))

        
        from .parse_reactions import build_stoich_matrix
        
        stoich = build_stoich_matrix(centralDict['reactions'], centralDict['primary_entities'], centralDict['systemSpeciation'])
        # print(stoich,centralDict['reactions'], centralDict['primary_entities'], centralDict['systemSpeciation'])
        
        if centralDict['crossDependencies'] and centralDict['crossDependencies']['speciation']:
            stoich = stoich.drop(index=centralDict['crossDependencies']['speciation']['total']) 
        # sys.exit()
        cols = [s for s in centralDict['systemSpeciation'] if s not in centralDict['phases']]
        
        centralDict.update({
        'stoich': stoich,
        "colAq": getattr(pycte_input, "colAq", cols),
        })
        
        
        
    elif centralDict["couplingInfo"][1] == 'nativeKinetics':
        from . import nativeKinetics 
        speciationLauncher = {
            'nativeKinetics': nativeKinetics.spct}
        centralDict.update({"kineticReactions" : getattr(pycte_input, "kineticReactions", None),
                            "nativeKineticsInterfTime_WallClock": 0,
                            "nativeKineticsCalcTime_WallClock": 0,
                            "nativeKineticsCalcTime_ProcessorTime": 0,
                            "nativeKineticsTotalTime": 0,
                            "nativeKineticsInitTime": 0,
                            })
    
    elif centralDict["couplingInfo"][1] == 'nativeSpeciation':
        from . import call
        speciationLauncher = {
            'nativeSpeciation': call.spct}
        centralDict.update({
            'MultiCompoundTransport' : getattr(pycte_input, "MultiCompoundTransport", False),
            "chemistry" :  getattr(pycte_input, "chemistry", None),
                            # "exchanger" : getattr(pycte_input, "exchanger", None),
                            "chemPath" : getattr(pycte_input, "chemPath", None),})
        
        # print(centralDict['elements'])
        # sys.exit()
    elif centralDict["couplingInfo"][1] == 'none':
        speciationLauncher = {'none': lambda cd: {}}
        centralDict.update({f"none{k}": 0 for k in ("CalcTime_WallClock", "CalcTime_ProcessorTime",
                                                    "InterfTime_WallClock", "InitTime", "TotalTime")})

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

        # sorties COMSOL utilisateur : transport/<tag>/ et transport/<tag>_VTU/ (si output['transport'])
        centralDict['outputComsol'] = list(comsolTags)

        # print(centralDict['paths'])
        # sys.exit()
        
    elif centralDict["couplingInfo"][2] == 'nativeTransport':
        from . import nativeTransport

        transportLauncher = {
            "nativeTransport": nativeTransport.trspt,
        }
        bcSchedule = nativeTransport.boundarySchedule(getattr(pycte_input, 'boundaryConditions', ['flux', 'flux']))

        centralDict.update({
            "velocity" : getattr(pycte_input, 'velocity', 0),
            "advection" : getattr(pycte_input, 'advection', None),
            "transportedSpecies" : getattr(pycte_input, 'transportedSpecies', centralDict['systemSpecies']),
            "FickDiffusion" : getattr(pycte_input, 'FickDiffusion', None),
            "porousTransport" : getattr(pycte_input, 'porousTransport', None),
            "ADE" : getattr(pycte_input, 'ADE', None),
            'adeSolver' : getattr(pycte_input, 'adeSolver', 'explicit'),        # 'explicit' | 'implicit'
            'adeSubSteps' : getattr(pycte_input, 'adeSubSteps', 1),            # pas implicites par pas de transport
            'dispersionInTransport' : True,     # la dispersion (theta.D_meca = alphaL.|q|) est calculee par nativeTransport
            "dispersivity" : getattr(pycte_input, 'dispersivity', 0),
            "firstBoundary" : getattr(pycte_input, 'firstBoundary', {cle: 0 for cle in centralDict['systemSpeciation']}),
            "secondBoundary" : getattr(pycte_input, 'secondBoundary', {cle: 0 for cle in centralDict['systemSpeciation']}),
            # [x = 0, x = L] : 'constant' : Dirichlet | 'closed' : Neumann (flux nul) | 'flux' : Cauchy
            # bord electrode : 'electrode+constant' | 'electrode+closed' | 'electrode+flux' ('electrode' = 'electrode+closed')
            # dependant du temps : {condition: temps de debut} ou {temps de debut: condition}, condition au temps 0 requise
            # ex. [{'closed': 0, 'constant': 100}, {'flux': 0, 'closed': 50}] ; boundaryConditions = conditions au temps 0
            "boundaryConditions" : [entries[0][1] for entries in bcSchedule],
            "boundarySchedule" : bcSchedule if any(len(entries) > 1 for entries in bcSchedule) else None,
            # Cauchy : J_in = Q_in.c_ext + h.A.(c_ext - c_bord), h [m/s] : nombre ou [h_0, h_L]
            "boundaryTransferCoeff" : getattr(pycte_input, 'boundaryTransferCoeff', 0.0),
            # diffusion sur le gradient d'activite (ADE et Nernst-Planck, explicite et implicite) ;
            # activityDatabase : base PHREEQC ('-gamma' : Debye-Huckel etendu), sinon Davies / Setchenow
            "activityGradient" : getattr(pycte_input, 'activityGradient', False),
            "activityDatabase" : getattr(pycte_input, 'activityDatabase', None),
            "nativeTransportClockTime" : 0,
            # "nodeSize" : getattr(pycte_input, 'nodeSize', 1),
            # ADE : 'upwind' (premier ordre) | 'LUD' (linear upwind differencing, ordre 2) | 'vanLeer' (TVD, ordre 2) ; LUD et vanLeer : ADE explicite seulement
            "advectionScheme" : getattr(pycte_input, 'advectionScheme', 'upwind'),
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
            'coordinateSystem' : getattr(pycte_input, 'coordinateSystem', 'cartesian'), # coordinateSystem = cylindrical, spherical
            'cylinderHeight' : getattr(pycte_input, 'cylinderHeight', 1),

            'imposedCurrent' : getattr(pycte_input, 'imposedCurrent', None),
            'imposedCurrentDensity' : getattr(pycte_input, 'imposedCurrentDensity', None),
            'imposedVoltage' : getattr(pycte_input, 'imposedVoltage', None),
            'rampPotential' : getattr(pycte_input, 'rampPotential', None),
            'temperature' : getattr(pycte_input, 'temperature', 298.15),
            
            'electroOsmoticPermeability' : getattr(pycte_input, 'electroOsmoticPermeability', 0), # k_eo [m2/(V.s)], scalar or list
            'permeability' : getattr(pycte_input, 'permeability', 1e-19), # k [m2], scalar or list
            'viscosity' : getattr(pycte_input, 'viscosity', 8.9e-4 ),  # mu [Pa.s],
            'hydraulicBoundary' : getattr(pycte_input, 'hydraulicBoundary', 'open'),  # 'open' ou 'closed'
            'boundaryPressure' : getattr(pycte_input, 'boundaryPressure', [0.0, 0.0] ), # [p_0, p_L] en Pa, en mode 'open'
            
            
            'NernstPlanck' : getattr(pycte_input, 'NernstPlanck', False),
            'especeCharge' : getattr(pycte_input, 'especeCharge', {}),
            'especeDiffCoeff' : getattr(pycte_input, 'especeDiffCoeff', {}),
            'especePorosity' : getattr(pycte_input, 'especePorosity', {}),
            'cflSafety' : getattr(pycte_input, 'cflSafety', 0.9),
            'maxSubCycling' : getattr(pycte_input, 'maxSubCycling', 100000),
            'electrodeReactions' : getattr(pycte_input, 'electrodeReactions', None),
            'electrodeVolume' : getattr(pycte_input, 'electrodeVolume', None),
            'electrodeVoltageDrop' : getattr(pycte_input, 'electrodeVoltageDrop', 0.0),
            'waterEquilibrium' : getattr(pycte_input, 'waterEquilibrium', None),
            'Kw' : getattr(pycte_input, 'Kw', 1.0e-14),
            'electrodeKinetics' : getattr(pycte_input, 'electrodeKinetics', 'faraday'),
            'electrodeArea' : getattr(pycte_input, 'electrodeArea', None),
            'npSolver' : getattr(pycte_input, 'npSolver', 'explicit'),
            'npImplicitMaxChange' : getattr(pycte_input, 'npImplicitMaxChange', 0.1),
            'npImplicitAtol' : getattr(pycte_input, 'npImplicitAtol', 1e-6),
            'npImplicitMaxIter' : getattr(pycte_input, 'npImplicitMaxIter', 30),
            # double porosite de Donnan (ADE explicite et implicite, Nernst-Planck implicite), voir nativeTransport.py :
            # {'porosityDL' | 'thickness', 'CEC' + 'bulkDensity' | 'chargeDL', 'stern', 'tortuosityDL', 'activityDL', ...}
            'donnan' : getattr(pycte_input, 'donnan', None),
            })

        nNodes = len(centralDict["commMtrx"])
        for key in ('especeDiffCoeff', 'especePorosity'):
            perSpecies = centralDict[key] or {}
            if not isinstance(perSpecies, dict):
                print(f"{key} must be a dict " + "{species: value}, got a "
                      + type(perSpecies).__name__ + " -> a missing ':' inside the braces makes it a set")
                sys.exit()
            resolved = {}
            for spc, val in perSpecies.items():
                val = np.asarray(val, dtype=float)
                if val.ndim == 0:
                    val = np.full(nNodes, float(val))
                elif len(val) != nNodes:
                    print(f"{key}['{spc}'] has {len(val)} values, expected {nNodes} (one per node)")
                    sys.exit()
                resolved[spc] = val
            centralDict[key] = resolved

        tempDiffCoeff = getattr(pycte_input, 'diffCoeff', 0)
        x = centralDict["commMtrx"][centralDict['coord']].copy().to_numpy().reshape(-1)



        f_left = np.zeros(len(centralDict["commMtrx"]))
        f_right = np.zeros(len(centralDict["commMtrx"]))
        
        f_left[1:]   = x[1:] - x[:-1]
        f_right[:-1] = x[1:] - x[:-1]

        centralDict.update({'f_right' : f_right, 'f_left' : f_left})
        dx = np.diff(x)
        
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

        if isinstance(tempDiffCoeff, (int,float)):
            centralDict['diffCoeff'] = np.full(len(centralDict["commMtrx"]),tempDiffCoeff)
        else:
            centralDict['diffCoeff'] = tempDiffCoeff

        # Tortuosite, convention PFLOTRAN (0 < tau <= 1) : diffCoeff et especeDiffCoeff sont donnes dans l'eau (Dw).
        # Coefficient de pore Dp = tau * Dw ; nativeTransport applique ensuite la porosite : De = porosity * tau * Dw.
        # Appliquee avant la dispersion (v . alpha n'est pas reduit par la tortuosite). Defaut 1 : comportement inchange.
        tortuosity = np.asarray(getattr(pycte_input, 'tortuosity', 1), dtype=float)
        if tortuosity.ndim == 0:
            tortuosity = np.full(len(centralDict["commMtrx"]), float(tortuosity))
        elif len(tortuosity) != len(centralDict["commMtrx"]):
            print(f"tortuosity has {len(tortuosity)} values, expected {len(centralDict['commMtrx'])} (one per node)")
            sys.exit()
        if np.any(tortuosity <= 0) or np.any(tortuosity > 1):
            print("tortuosity must be in ]0, 1] (PFLOTRAN convention : De = porosity * tortuosity * Dw)")
            sys.exit()
        centralDict['tortuosity'] = tortuosity
        centralDict['diffCoeff'] = tortuosity * np.asarray(centralDict['diffCoeff'], dtype=float)
        centralDict['especeDiffCoeff'] = {spc: tortuosity * val for spc, val in centralDict['especeDiffCoeff'].items()}

        # dispersion : n'est plus ajoutee a diffCoeff ; nativeTransport l'applique (theta.D_meca = alphaL.|q|,
        # q local en geometrie radiale). diffCoeff reste le coefficient de pore Dp = tortuosity * Dw.


        meshing = getattr(pycte_input, 'meshing', 0)
        if not meshing : # uniform meshing
            x = np.asarray(x)
            # sans meshing, dxHalfCell = (x[1] - x[0]) / 2 partout : les x doivent etre equidistants, sinon les
            # distances entre noeuds (dxHalfCell) ne correspondent plus aux volumes des mailles (nodeSize, lu sur x)
            if np.any(dx <= 0) or not np.allclose(dx, dx[0], rtol=1e-3, atol=0):
                k = int(np.argmax(np.abs(dx - dx[0])))
                print(f"without meshing, x must be strictly increasing and uniform (x[1] - x[0] = {dx[0]:.6g}, "
                      f"x[{k+1}] - x[{k}] = {dx[k]:.6g}) : give meshing = [(nb1, width1), (nb2, width2), ...] "
                      "for a non-uniform mesh")
                sys.exit()
            largeur = x[1] - x[0]
            zones = [(len(x), largeur)]
            dxHalfCell = np.concatenate([np.full(nb, largeur / 2) for nb, largeur in zones])
        else:
            # meshing = [(nb1, largeur1), (nb2, largeur2), ...] : noeuds au centre de leur maille, la face k+1/2 est a
            # dxHalfCell_k du noeud k -> les x du fichier doivent verifier x[k+1] - x[k] = (largeur_k + largeur_k+1) / 2
            if any(int(nb) != nb or nb <= 0 or not largeur > 0 for nb, largeur in meshing):
                print(f"meshing must be a list of (number of cells > 0, cell width > 0) (got {meshing})")
                sys.exit()
            dxHalfCell = np.concatenate([np.full(int(nb), largeur/2) for nb, largeur in meshing])
            if len(dxHalfCell) != len(x):
                print(f"meshing has {len(dxHalfCell)} cells, expected {len(x)} (one per node)")
                sys.exit()
            gap = dxHalfCell[:-1] + dxHalfCell[1:]
            if not np.allclose(dx, gap, rtol=1e-3, atol=0):
                k = int(np.argmax(np.abs(dx - gap) / gap))
                print(f"x does not match meshing : x[{k+1}] - x[{k}] = {dx[k]:.6g}, expected {gap[k]:.6g} "
                      "= (width_k + width_k+1) / 2 (nodes at the centre of their cell)")
                sys.exit()
            centralDict['nodeSize'] = 2 * dxHalfCell

        centralDict.update({'dxHalfCell': dxHalfCell})

        # ADE : nativeTransport calcule lui-meme le nombre de sous-pas (advection + diffusion + dispersion) en
        # explicite, et n'en a pas besoin en implicite (adeSolver = 'implicit')

        useNernstPlanck = bool(centralDict['NernstPlanck']) and not (
            centralDict['FickDiffusion'] or centralDict['advection'] or centralDict['ADE'])

        if useNernstPlanck and str(centralDict.get('npSolver') or 'explicit').lower() == 'explicit':

            especeDiffCoeff = centralDict['especeDiffCoeff'] or {}
            especePorosity = centralDict['especePorosity'] or {}
            nNodes = len(centralDict["commMtrx"])
            dt_diff = np.inf
            limitingSpecies = None

            for spc in centralDict['transportedSpecies']:
                D_spc = np.asarray(especeDiffCoeff.get(spc, centralDict['diffCoeff']), dtype=float)
                th_spc = np.asarray(especePorosity.get(spc, centralDict['porosity']), dtype=float)
                if D_spc.ndim == 0:
                    D_spc = np.full(nNodes, float(D_spc))
                if th_spc.ndim == 0:
                    th_spc = np.full(nNodes, float(th_spc))

                cond = th_spc * centralDict['area'] * D_spc
                with np.errstate(divide='ignore', invalid='ignore'):
                    R = dxHalfCell / np.where(cond > 0, cond, 1.0)
                R[cond <= 0] = np.inf
                R_sum = R[:-1] + R[1:]
                T_interface = np.zeros(len(cond) - 1)
                finite_R = np.isfinite(R_sum) & (R_sum > 0)
                T_interface[finite_R] = 1000.0 / R_sum[finite_R]

                T_sum = np.zeros(len(cond))
                T_sum[:-1] += T_interface
                T_sum[1:] += T_interface

                storage = 1000.0 * np.asarray(centralDict['nodeSize'], dtype=float) \
                          * np.asarray(centralDict['area'], dtype=float) * th_spc
                with np.errstate(divide='ignore', invalid='ignore'):
                    dt_node = np.where(T_sum > 0, storage / np.where(T_sum > 0, T_sum, 1.0), np.inf)

                dt_spc = float(np.min(dt_node))
                if dt_spc < dt_diff:
                    dt_diff = dt_spc
                    limitingSpecies = spc

            dt_diff *= getattr(pycte_input, 'cflSafety', 1.0)

            if np.isfinite(dt_diff) and dt_diff > 0 and dt > dt_diff:
                n_diff = int(np.ceil(dt / dt_diff))
                print(f"Sub-cycling Nernst-Planck within {n_diff} sub-steps, "
                      f"dt_max={writeTime(dt_diff,4)} (limiting species : {limitingSpecies})")

        centralDict.update({'subCyclingDiff' : n_diff,
                            'subCyclingAdv' : n_adv})
 
             
    elif centralDict["couplingInfo"][2] == 'PFLOTRAN':
        from . import pflotran

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

    # MultiCompoundTransport (PhreeqC) : commMtrx ne porte plus que les totaux ; conversion unique des conditions
    # initiales et aux limites (especes ou totaux) avant le premier pas, voir phreeqc.multiCompoundSetup
    if centralDict['MultiCompoundTransport'] and centralDict["couplingInfo"][1] == 'PhreeqC':
        phreeqc.multiCompoundSetup(centralDict)

    # dossiers de sortie : suppression des dossiers connus de pycte, recreation (vides) des dossiers demandes
    outputManager.prepareFolders(pathpycteInput, centralDict['paths'])

    # warning.log : en-tete du run puis les warnings emis pendant l'initialisation
    header = f"pycte : {centralDict['couplingInfo'][0]} {centralDict['couplingInfo'][1]}-{centralDict['couplingInfo'][2]}.\n"
    if osSequence is not None:
        header += f"pycte : OSdefined (fractions of dt) : {OSschemes.describe(osSequence, centralDict['couplingInfo'][1], centralDict['couplingInfo'][2])}\n"
    if getattr(pycte_input, "description", None): header += f"pycte : Description :\n{pycte_input.description}\n"
    header += f"pycte : Time steps ({centralDict['timeUnit']}) :\n{', '.join(str(dtt) for dtt in dtpycte)}.\n"
    warningManager.start(header, centralDict)

    def writeCoupling(step=None):
        # coupling/Coupling_n.txt, n = numero de pas (0 : etat initial)
        if outputManager.wanted(centralDict, 'coupling', step):
            centralDict["commMtrx"].to_csv(outputManager.filePath(centralDict, 'coupling', 'Coupling', step),
                                           index=False, header=True, sep='\t')

    for l,t in enumerate(dtpycte):
        
        if l==0: dt = t
        else: dt = t-dtpycte[l-1]

        timeCrossDep = bool(centralDict['crossDependencies']) and 'time' in centralDict['crossDependencies'].get('totalCrossDep')
        if timeCrossDep:
            centralDict['commMtrx']['time'] = dt

        centralDict.update({
            "dtStep":dt,
            "tStep":t,
            "lStep":l,
            })
        
        print(f"#######  step n°{l+1}/{len(dtpycte)}, dt = {dt}{centralDict['timeUnit']}  #######")
        # print(centralDict["systemSpeciation"])
        
        # sys.exit()
        if centralDict['preliminarEquilibrium']:
            # print('icic')
            centralDict['beforeTrsptMtrx'] = centralDict["commMtrx"].copy() # not OS dependent ..
            # sys.exit()
        
        if centralDict["firstStepEquilibrium"] and l==0:
            centralDict["dtStep"] = 0
            # duree 'time' transmise a la speciation (ex. cinetique ORCHESTRA) : nulle pour l'equilibre initial
            if timeCrossDep:
                centralDict['commMtrx']['time'] = 0
            # print(centralDict['beforeTrsptMtrx'],centralDict["commMtrx"])
            # sys.exit()
            centralDict.update(speciationLauncher.get(centralDict['couplingInfo'][1])(centralDict))
            if centralDict['preliminarEquilibrium']:
                centralDict['beforeTrsptMtrx'] = centralDict["commMtrx"].copy()
            centralDict["firstStepEquilibrium"]=False
            centralDict["dtStep"]=dt
            if timeCrossDep:
                centralDict['commMtrx']['time'] = dt
        if l == 0:
            writeCoupling(0)                       # etat initial (apres firstStepEquilibrium s'il est demande)
        pd.set_option('display.max_columns', None)
        
        # print(centralDict["commMtrx"].head(8))
        # sys.exit()

        # un pas de couplage transport / speciation avec le schema choisi (OSschemes.py) ; le schema fixe dtStep et
        # tTransportStart (debut du transport : conditions aux limites dependant du temps, rampe) avant chaque appel
        OSschemes.step(centralDict,
                       transportLauncher.get(centralDict['couplingInfo'][2]),
                       speciationLauncher.get(centralDict['couplingInfo'][1]),
                       t, dt)

        stepTime = time.time() - pyctestart
        writeCoupling()

        # renouvellement en fin de pas (apres transport et chimie) : le pas suivant part de la solution renouvelee ;
        # Coupling_*.txt ci-dessus garde l'etat calcule avant renouvellement
        if rnvBool:
            centralDict.update(rnv.apply(centralDict))

        warningManager.flush(centralDict)           # nouveaux warnings du pas, en un seul bloc

        if t != maxTime:
            if 'extractFromDbTime' in centralDict:
                print(f"Remaining calculation time ~ {writeTime((maxTime * stepTime - centralDict['extractFromDbTime'])/ (t) - stepTime,2)}")
            else:
                print(f"Remaining calculation time ~ {writeTime((maxTime * stepTime )/ (t) - stepTime,2)}")

    warningManager.flush(centralDict)
    with open("warning.log", "a") as f:
        f.write("pycte : Coupling completed\n")
        f.write(f"pycte : {centralDict['warningNbr']} warning(s) have occured during the coupling\n\n")
    
    if centralDict['warningNbr']:
        print(f"Oh no, {centralDict['warningNbr']} warnings occured :( ! See warning.log file.")
    
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