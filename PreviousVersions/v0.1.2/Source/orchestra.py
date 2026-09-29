import numpy as np
import time 
import pandas as pd
import sys 
import importlib.util
import os
import concurrent.futures

import contextlib

import PyORCHESTRA


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
    else:
        return f"{tps:.{arr}f} sec"


def _init_worker(chemPath, inVars, outVars,t,dt,l,unit):
    global solver, initWorker
    with redirect_stdout_fd("orchestra_output.log",t,dt,l,unit):
        ref = time.perf_counter()
        solver = PyORCHESTRA.ORCHESTRA()
        solver.initialise(str(chemPath), 1, inVars, outVars)
        initWorker = time.perf_counter() - ref

def speciationOrchestra(centralDict,commMtrxPart):
    global solver, initWorker
    """
    Site quantity is natively defined in the .inp file. This file could change depending on the node number ...
    side note:
        Sorbed attribute is .solid
        Speciation is Na+.con, Na.diss en solution, Na.tot total.
        Species attributes are defined by speciesAttributes keyword. It may refer to the .inp file.
    """

    species_cols = centralDict['systemSpeciation']
    prim_species = centralDict['primarySpecies'][-1]

    stoich = pd.DataFrame(0.0, index=species_cols, columns=prim_species)
    for comp in species_cols:
        for prim, coef in centralDict['primToSecSpecies'][comp].items():
            stoich.loc[comp, prim] = coef
    
    commMtrx_primSpecies = pd.DataFrame(
        commMtrxPart.to_numpy() @ stoich.to_numpy(),
        columns=prim_species
    )
    
    if centralDict['PIDnbr'] == 1:
        with redirect_stdout_fd("orchestra_output.log",centralDict['tStep'],centralDict['dtStep'],centralDict['lStep'],centralDict['timeUnit']):
            ref = time.perf_counter()
            solver = PyORCHESTRA.ORCHESTRA()
            solver.initialise(str(centralDict['chemPath']), 1, centralDict['inputVariableOrchestra'], centralDict['outputVariableOrchestra'])
            init = time.perf_counter() - ref
    else:
        init = initWorker
        initWorker = 0
    rows = commMtrx_primSpecies.to_numpy()
    
    results = []
    calcTime = 0
    
    
    with redirect_stdout_fd("orchestra_output.log",centralDict['tStep'],centralDict['dtStep'],centralDict['lStep'],centralDict['timeUnit']):
        for row in rows:
            ref = time.perf_counter()
            results.append(solver.set_and_calculate_single(row)[0])
            calcTime += time.perf_counter() - ref

    outputOrchestra = pd.DataFrame(results, columns=centralDict['outputVariableOrchestra'])
    outputOrchestra.columns = centralDict['systemSpeciation']

    return outputOrchestra, commMtrx_primSpecies, calcTime, init



def spct(centralDict):
    print("ORCHESTRA", end=" ", flush=True)
    startOrchestra = time.time()

    if centralDict["PIDnbr"] > 1:
        chunk_size = int(np.ceil(len(centralDict["commMtrx"]) / centralDict["PIDnbr"]))
        commMtrxSplit = [centralDict["commMtrx"][centralDict['systemSpeciation']].iloc[i:i + chunk_size] for i in range(0, len(centralDict["commMtrx"]), chunk_size)]
        
        ### one initializer for all workers
        with concurrent.futures.ProcessPoolExecutor(max_workers=centralDict["PIDnbr"],initializer=_init_worker, initargs=(centralDict['chemPath'], centralDict['inputVariableOrchestra'],
                  centralDict['outputVariableOrchestra'],centralDict['tStep'],centralDict['dtStep'],centralDict['lStep'],centralDict['timeUnit'])) as executor:
            
            futures = [executor.submit(speciationOrchestra, centralDict, chunk) for chunk in commMtrxSplit]

        results = []
        results = [f.result() for f in futures]
        df1, df2, calc, init = zip(*results)
    
        commMtrxSpct = pd.concat(df1, ignore_index=True)
        commMtrx_primSpecies = pd.concat(df2, ignore_index=True)
        calcPrcsTime = sum(calc)
        calcWallClock = max(calc)
        
    else:
        commMtrxSpct, commMtrx_primSpecies, calcWallClock, init = speciationOrchestra(centralDict, centralDict['commMtrx'][centralDict['systemSpeciation']])
        calcPrcsTime = calcWallClock
        
        
    commMtrxSpct = pd.concat([centralDict['commMtrx'][centralDict["anythingButSpecies"]],commMtrxSpct], axis=1)
    commMtrxSpct.columns = centralDict['commMtrx'].columns
    commMtrx_primSpecies = pd.concat([centralDict['commMtrx'][['x', 'y', 'z'][:centralDict['geometry']]],commMtrx_primSpecies], axis=1)
    

    if centralDict["firstStepEquilibrium"]==True:
        commMtrx_primSpecies.to_csv(os.path.join(centralDict['paths']['PrimarySpecies'], f"PrimarySpecies_{centralDict['lStep']}.txt"), index=False, header=True, sep='\t')
        commMtrxSpct.to_csv(os.path.join(centralDict['paths']['Speciation'], f"ORCHESTRA_{centralDict['lStep']}.txt"), index=False, header=True, sep='\t')
    else:
        commMtrx_primSpecies.to_csv(os.path.join(centralDict['paths']['PrimarySpecies'], f"PrimarySpecies_{centralDict['lStep']+1}.txt"), index=False, header=True, sep='\t')
        commMtrxSpct.to_csv(os.path.join(centralDict['paths']['Speciation'], f"ORCHESTRA_{centralDict['lStep']+1}.txt"), index=False, header=True, sep='\t')
   
    centralDict.update({
        "commMtrx": commMtrxSpct,
        "ORCHESTRAInterfTime_WallClock": centralDict['ORCHESTRAInterfTime_WallClock'] + time.time() - startOrchestra - calcWallClock - init,
        "ORCHESTRACalcTime_WallClock": centralDict['ORCHESTRACalcTime_WallClock'] + calcWallClock,
        "ORCHESTRACalcTime_ProcessorTime": centralDict['ORCHESTRACalcTime_ProcessorTime'] + calcPrcsTime,
        "ORCHESTRAInitTime" : centralDict['ORCHESTRAInitTime'] + init,
        "ORCHESTRATotalTime" : centralDict['ORCHESTRATotalTime'] + time.time() - startOrchestra
        })
     
   
    print(f"({writeTime((time.time() - startOrchestra))})") 

    return centralDict