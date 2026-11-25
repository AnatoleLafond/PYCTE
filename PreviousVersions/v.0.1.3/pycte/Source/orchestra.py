import numpy as np
import time 
import pandas as pd
import sys 
import os
from pathlib import Path
import concurrent.futures
import contextlib
try:
    import PyORCHESTRA
except ImportError as err:
    raise ImportError("chemModule = 'ORCHESTRA' needs PyORCHESTRA, the Python interface of ORCHESTRA (H. Meeussen), "
                      "compiled from its sources : pip install <PyORCHESTRA source folder> (C++ compiler needed). "
                      "Do not 'pip install PyORCHESTRA' from PyPI : that name is an unrelated project.") from err
import faulthandler; faulthandler.enable() 
try:
    from . import outputManager
except ImportError:
    import outputManager


solver = None
initWorker = 0
firstCalc = True

@contextlib.contextmanager
def redirect_stdout_fd(filepath, t,dt, lStep,timeUnit):
    stdout_fd = 1
    stderr_fd = 2
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    with open(filepath, 'a') as f:
        print(f'\nPICCTS: time-step n°{lStep}, t={t} {timeUnit}, dt={dt} {timeUnit}: \n', file=f)
        os.dup2(f.fileno(), stdout_fd)
        os.dup2(f.fileno(), stderr_fd)
        try:
            yield
        finally:
            os.dup2(saved_stdout, stdout_fd)
            os.dup2(saved_stderr, stderr_fd)
            os.close(saved_stdout)
            os.close(saved_stderr)

def writeTime(tps, arr=2):
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

def resetOrchestra():
    global solver, initWorker, firstCalc
    solver = None
    initWorker = 0
    firstCalc = True
    print("ORCHESTRA solver reset.")


def _init_worker(chemPath, inVars, outVars,t,dt,l,unit):
    global solver, initWorker
    with redirect_stdout_fd("orchestra_output.log",t,dt,l,unit):
        ref = time.perf_counter()
        solver = PyORCHESTRA.ORCHESTRA()
        solver.initialise(str(chemPath), 1, inVars, outVars)
        initWorker = time.perf_counter() - ref

def speciationOrchestra(centralDict,commMtrxPart):
    global solver, initWorker, firstCalc
    pd.set_option('display.max_columns', None)
    """
    Site quantity is natively defined in the .inp file. This file could change depending on the node number ...
    side note:
        Sorbed attribute is .solid
        Speciation is Na+.con, Na.diss en solution, Na.tot total.
        Species attributes are defined by speciesAttributes keyword. It may refer to the .inp file.
    """
    
    commMtrxPart = commMtrxPart.copy()


    if centralDict['couplingFormalism'] == 'literbulk' :

        commMtrxPart.loc[:, centralDict['colAq']] = (commMtrxPart.loc[:, centralDict['colAq']].mul(centralDict['waterMass'], axis=0))
        
        if centralDict['crossDependencies'] and centralDict['crossDependencies']['speciation']:
            commMtrxPart.drop(columns=centralDict['crossDependencies']['speciation']['total'], inplace=True, )

        if 'O' in centralDict['stoich']:
            del centralDict['stoich']['O']
        

        commMtrx_primSpecies = pd.DataFrame(
            commMtrxPart.to_numpy() @ centralDict['stoich'].to_numpy(),
            columns=centralDict["inputVariableOrchestraSpecies"]
            )

        if centralDict['crossDependencies'] and centralDict['crossDependencies']['speciation']:
                commMtrx_primSpecies = pd.concat([centralDict['commMtrx'][centralDict['crossDependencies']['speciation']['total']],commMtrx_primSpecies], axis=1, )



        refdf = commMtrx_primSpecies.copy()

        rows = commMtrx_primSpecies.to_numpy()

        
    elif centralDict['couplingFormalism'] == 'tot':
        # print(centralDict['stoichReduced'],commMtrxPart[centralDict['systemSpecies']])
        # sys.exit()
        # especes (mailles x systemSpecies) @ stoichReduced (systemSpecies x composants) -> totaux des composants
        components = centralDict['primarySpecies'][-1]       # ex. ['Ca', 'Cl', 'K', 'N', 'Na']
        commMtrx_primSpecies = pd.DataFrame(
            commMtrxPart[centralDict['systemSpecies']].to_numpy() @ centralDict['stoichReduced'].to_numpy(),
            columns=components, index=commMtrxPart.index
        ).clip(lower=0)

        if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation'):
            cross = centralDict['crossDependencies']['speciation']['total']
            commMtrx_primSpecies[cross] = commMtrxPart[cross]    # pH, pe, ... : pas de clip (pe < 0 possible)

        # entrees ORCHESTRA par nom : 'Ca.tot' <- composant 'Ca', variable croisee telle quelle ; l'ordre de
        # inputVariableOrchestra est libre (un renommage par position echangerait Na et N si l'ordre differe)
        source = [v if v in commMtrx_primSpecies.columns else v.rsplit('.', 1)[0]
                  for v in centralDict["inputVariableOrchestra"]]
        commMtrx_primSpecies = commMtrx_primSpecies[source]
        commMtrx_primSpecies.columns = centralDict["inputVariableOrchestra"]
        rows = commMtrx_primSpecies.to_numpy()

    

        
        
        
    
    else:
        print(f'ORCHESTRA coupling formalism not recognized : {centralDict["couplingFormalism"]} (tot, literbulk, diss available)')
        sys.exit()
    
    

    rows = np.empty((len(commMtrx_primSpecies), len(centralDict["inputVariableOrchestra"])), dtype=np.float64, order="C")
    for j, col in enumerate(centralDict["inputVariableOrchestra"]):
        rows[:, j] = commMtrx_primSpecies[col].to_numpy(dtype=np.float64)
        

    nrCells = rows.shape[0]
    calcTime = 0

    if solver is None: 
        _cwd = os.getcwd()
        os.chdir(Path(centralDict['chemPath']).parent)

        try:
            # journal dans le dossier du run, pas dans celui de la base (package installe : dossier en lecture seule)
            with redirect_stdout_fd(os.path.join(_cwd, "orchestra_output.log"),centralDict['tStep'],centralDict['dtStep'],centralDict['lStep'],centralDict['timeUnit']):
                ref = time.perf_counter()
                solver = PyORCHESTRA.ORCHESTRA()
                solver.initialise(Path(centralDict['chemPath']).name, nrCells, centralDict['inputVariableOrchestra'], centralDict['outputVariableOrchestra'])
                init = time.perf_counter() - ref
        finally:
            os.chdir(_cwd)
    else:
        init = 0

    
    with redirect_stdout_fd("orchestra_output.log",centralDict['tStep'],centralDict['dtStep'],centralDict['lStep'],centralDict['timeUnit']):
        memory_option = 1 if firstCalc else 0
        ref = time.perf_counter()
        outputarray = solver.set_and_calculate_multi(rows, centralDict['nrThreads'], memory_option)

        calcTime = time.perf_counter() - ref
        firstCalc = False

    if False :
        output = pd.DataFrame(outputarray, columns=(centralDict['systemSpeciation']))      
        output.loc[:, centralDict['colAq']] = (output.loc[:, centralDict['colAq']].mul(centralDict['waterMass'], axis=0))
    
        output = pd.DataFrame(
            output.to_numpy() @ centralDict['stoich'].to_numpy(),
            columns=centralDict['stoich'].columns)
        output.columns = refdf.columns
        tol = 1e-11
        diff = (output - refdf).abs()
        
        if (diff > tol).any().any():
            print(diff)
            with open("warning.log", "a") as warningLog:
                warningLog.write('probleme MB')
                warningLog.write('\noutput orchestra:\n')
                warningLog.write(output.to_string())
                warningLog.write("\n")
                warningLog.write('\nrefdf:\n')
                warningLog.write(refdf.to_string())
                warningLog.write('\ndiff: \n')
                warningLog.write(diff.to_string())
            print('pblm MB, tol = ', tol)
            sys.exit()
        
    else:
        kept = (centralDict['crossDependencies']['transport']['total']
                if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('transport') else [])
        outputOrchestra = pd.DataFrame(outputarray, columns=[s for s in centralDict['systemSpeciation'] if s not in kept])


    return outputOrchestra, commMtrx_primSpecies, calcTime, init

def spct(centralDict):
    print("ORCHESTRA", end=" ", flush=True)
    startOrchestra = time.time()
    
    if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation'):
        col = [c for c in centralDict['systemSpeciation'] if c in (centralDict['systemSpecies'] + centralDict['crossDependencies']['speciation']['total']) ]
    else:
        col = centralDict['systemSpecies']
    
    commMtrxSpct, commMtrx_primSpecies, calcWallClock, init = speciationOrchestra(centralDict, centralDict['commMtrx'][col])

    
    if 'failed' in commMtrxSpct.columns:
        if (commMtrxSpct["failed"] == 1).any():
            print('Failed simulation')
            commMtrxSpct.to_csv("commMtrx_failed.txt", index=False, header=True, sep='\t')
            sys.exit()
        commMtrxSpct = commMtrxSpct.drop(columns="failed")
    
    
    if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('transport'):
        col = centralDict["anythingButSpecies"] + centralDict['crossDependencies']['transport']['total']
    else:
        col = centralDict["anythingButSpecies"]
    
    commMtrxSpct = pd.concat([centralDict['commMtrx'][col],commMtrxSpct], axis=1)
    commMtrxSpct = commMtrxSpct[centralDict['commMtrx'].columns]
    if outputManager.wanted(centralDict, 'speciation'):
       commMtrx_primSpecies.to_csv(outputManager.filePath(centralDict, 'primarySpecies', 'PrimarySpecies'), index=False, header=True, sep='\t')
       commMtrxSpct.to_csv(outputManager.filePath(centralDict, 'speciation', 'ORCHESTRA'), index=False, header=True, sep='\t')

    
    
    if centralDict['lStep'] == len(centralDict['dtpycte'])-1:
        resetOrchestra()
    
    # print(commMtrxSpct)
    # sys.exit()
    
    
    centralDict.update({
        "commMtrx": commMtrxSpct,
        "ORCHESTRAInterfTime_WallClock": centralDict['ORCHESTRAInterfTime_WallClock'] + time.time() - startOrchestra - calcWallClock - init,
        "ORCHESTRACalcTime_WallClock": centralDict['ORCHESTRACalcTime_WallClock'] + calcWallClock,
        "ORCHESTRACalcTime_ProcessorTime": centralDict['ORCHESTRACalcTime_ProcessorTime'] + calcWallClock,
        "ORCHESTRAInitTime" : centralDict['ORCHESTRAInitTime'] + init,
        "ORCHESTRATotalTime" : centralDict['ORCHESTRATotalTime'] + time.time() - startOrchestra
        })

    print(f"({writeTime((time.time() - startOrchestra))})") 

    return centralDict