try:
    import xgems
except ImportError as err:
    raise ImportError("chemModule = 'xGEMS' needs xGEMS, which is not on PyPI : install it with conda "
                      "(conda install -c conda-forge xgems)") from err
import pandas as pd
import sys
import numpy as np
import time 
import importlib.util
import os
import concurrent.futures
from pathlib import Path
try:
    from . import outputManager
    from . import warningManager
except ImportError:
    import outputManager
    import warningManager

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


gemsStatus= {
0: "No GEM re-calculation needed",
1: "Need GEM calculation with LPP (automatic) initial approximation (AIA)",
2: "OK after GEM calculation with LPP AIA",
3: "Bad (not fully trustful) result after GEM calculation with LPP AIA",
4: "Failure (no result) in GEM calculation with LPP AIA",
5: "Need GEM calculation with no-LPP (smart) IA, SIA using the previous speciation",
6: "OK after GEM calculation with SIA",
7: "Bad (not fully trustful) result after GEM calculation with SIA",
8: "Failure (no result) in GEM calculation with SIA",
9: "Terminal error in GEMS3K (e.g., memory corruption). Restart required.",
    }

def speciation_xGEMS(centralDict,commMtrx):
    chem_dir = Path(centralDict['chemPath']).parent
    _cwd = os.getcwd()
    try:
        os.chdir(chem_dir)
        ref = time.perf_counter()
        engine = xgems.ChemicalEngine(str(centralDict['chemPath']))
        init = time.perf_counter() - ref
    finally:
        os.chdir(_cwd)
    

    outputGems = pd.DataFrame(columns=commMtrx.columns)
    
    if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation'):
        commMtrxCrossDep = commMtrx[centralDict['crossDependencies']['speciation']['total']].copy()
        commMtrx.drop(columns=centralDict['crossDependencies']['speciation']['total'], inplace=True)
    
    else:
        commMtrxCrossDep = pd.DataFrame(np.array([[1e5, 298.15]] * len(commMtrx)), columns=["pressure", "temperature(K)"])
    
    MultiCompoundTransport = centralDict['MultiCompoundTransport'] 
    if centralDict['lStep'] ==0 :
    
        notThere = (centralDict['coord']+centralDict['speciesByClass']['T']+centralDict['speciesByClass']['W']+centralDict['speciesByClass']['O']+centralDict['speciesByClass']['G'] + centralDict['crossDependencies']['speciation']['total'] if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation') else []) 
    
        l = [s for s in list(centralDict['commMtrx'].columns) if s not in notThere]
        if set(l).issubset(centralDict['speciesByClass']['S']):
            if centralDict['MultiCompoundTransport'] :
                MultiCompoundTransport = False
            

    if MultiCompoundTransport :
        commMtrx_primSpecies = commMtrx.copy()
        
        ic_list = centralDict['independentComponents']
        species_list = commMtrx.columns
        
        already_primary = [s for s in species_list if s in ic_list]
        
        to_decompose = [s for s in species_list if s not in ic_list]
        
        stoich_matrix = pd.DataFrame(0.0, index=to_decompose, columns=ic_list)
        for comp in to_decompose:
            for prim, coeff in centralDict['primToSecSpecies'][comp].items():
                stoich_matrix.at[comp, prim] = coeff
        
        decomposed = commMtrx[to_decompose].values @ stoich_matrix.values
        decomposed_df = pd.DataFrame(decomposed, columns=ic_list, index=commMtrx.index)
        
        result_df = pd.DataFrame(0.0, columns=ic_list, index=commMtrx.index)
        
        for s in already_primary:
            result_df[s] += commMtrx[s]
        
        result_df += decomposed_df
        
        commMtrx_primSpecies = result_df
        

        
    else:
        ic_list = centralDict['independentComponents']
        species_list = commMtrx.columns
        
        stoich_matrix = pd.DataFrame(
            0.0,
            index=centralDict['systemSpeciation'],
            columns=centralDict['independentComponents']
        )

        for comp in centralDict['systemSpeciation']:
            for prim, coeff in centralDict['primToSecSpecies'][comp].items():
                stoich_matrix.at[comp, prim] = coeff
        

        result = commMtrx.values @ stoich_matrix.values

        commMtrx_primSpecies = pd.DataFrame(result, columns=ic_list, index=commMtrx.index)

    gemsStatusList = []
    gemsIterations = []

    calcTime = 0
    
    
    for index in commMtrx_primSpecies.index:
        ref = time.perf_counter()
        status = engine.equilibrate(
            commMtrxCrossDep.at[index, 'temperature(K)'],commMtrxCrossDep.at[index, 'pressure'], 
            [
                commMtrx_primSpecies.at[index, spc]
                for spc in centralDict['independentComponents']
            ])

        calcTime += time.perf_counter() - ref

        outputGems.loc[len(outputGems)] = (list(engine.speciesAmounts())+ [engine.pressure()]+ [engine.temperature()])
        
        gemsStatusList += [status]
        gemsIterations += [engine.numIterations()]
        
    outputGems.index = commMtrx.index
        
    outputGems[centralDict['systemSpeciation']] = outputGems[centralDict['systemSpeciation']].clip(lower=0)

    if centralDict['MultiCompoundTransport']:
        
        if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation'):
            commMtrxCrossDep = outputGems[centralDict['crossDependencies']['speciation']['total']].copy()
            outputGems.drop(columns=centralDict['crossDependencies']['speciation']['total'], inplace=True)
        else:
            commMtrxCrossDep = pd.DataFrame()
        
        outputSpecies = outputGems.copy()
        ic_list = centralDict['independentComponents']
        species_list = outputGems.columns
        
        stoich_matrix = pd.DataFrame(0.0,index=species_list,columns=ic_list)


        for comp in (centralDict['transportedSpecies']) :
            for prim, coeff in centralDict['primToSecSpecies'][comp].items():
                stoich_matrix.at[comp, prim] = coeff
        result = outputGems.values @ stoich_matrix.values
        
        outputGems = pd.DataFrame(result, columns=ic_list, index=commMtrx.index)
        outputGems[centralDict['fixedSpecies']] = outputSpecies[centralDict['fixedSpecies']].copy()
        outputGems = pd.concat([outputGems,commMtrxCrossDep], axis=1)

        return outputGems,outputSpecies,gemsStatusList,gemsIterations, calcTime, init
    else:
        
        return outputGems,commMtrx_primSpecies,gemsStatusList,gemsIterations, calcTime, init
	
def spct(centralDict): 
    print("xGEMS", end=" ", flush=True)
    startGems = time.time()

    if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation'):
        gemsInput = [s for s in centralDict['commMtrx'].columns if s in (centralDict['systemSpeciation'] + centralDict['crossDependencies']['speciation']['input'])] 
    else: 
        gemsInput = [s for s in centralDict['commMtrx'].columns if s not in centralDict['coord']]


    if centralDict['PIDnbr'] > 1:
    
        chunk_size = int(np.ceil(len(centralDict['commMtrx']) / centralDict['PIDnbr']))
    
        commMtrxSplit = [centralDict['commMtrx'][gemsInput].iloc[i:i + chunk_size] for i in range(0, len(centralDict['commMtrx']), chunk_size)]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=centralDict['PIDnbr']) as executor:
            futures = []
            for chunk in commMtrxSplit:
                futures.append(
                    warningManager.submit(
                        executor,
                        speciation_xGEMS,
                        centralDict,
                        chunk,
                    )
                )
        
       
        results = []   
        
        results = [f.result() for f in futures]
    
        output,primSpc, status, iteration, calc, initWorker = zip(*results)
    
        commMtrxSpct = pd.concat(output, ignore_index=True)
        commMtrx_primSpecies = pd.concat(primSpc, ignore_index=True)
        calcPrcsTime = sum(calc)
        calcWallClock = max(calc)
        init = max(initWorker)
        
        gemsStatusList = [s for block in status for s in block]         # tous les workers, dans l'ordre des mailles
        gemsIterations = [n for block in iteration for n in block]
        
    else:
        commMtrxSpct, commMtrx_primSpecies, gemsStatusList, gemsIterations, calcWallClock, init = speciation_xGEMS(centralDict,centralDict['commMtrx'][gemsInput])
        calcPrcsTime = calcWallClock

    commMtrxSpct = pd.concat([centralDict['commMtrx'][['x', 'y', 'z'][:centralDict['geometry']]],commMtrxSpct], axis=1)
    commMtrx_primSpecies = pd.concat([centralDict['commMtrx'][['x', 'y', 'z'][:centralDict['geometry']]],commMtrx_primSpecies], axis=1)
    
    

    if centralDict['crossDependencies']:
        if centralDict['crossDependencies'].get('transport'):
            a = list(set(centralDict['crossDependencies']['transport']['total']) - set(centralDict['crossDependencies']['speciation']['total']))
            if a :
                commMtrxSpct = pd.concat([commMtrxSpct,centralDict['commMtrx'][a]], axis=1)
                commMtrx_primSpecies = pd.concat([commMtrx_primSpecies,centralDict['commMtrx'][a]], axis=1)
        
        
        if centralDict['crossDependencies'].get('speciation') and list(set(centralDict['crossDependencies']['speciation']['input']) - set(centralDict['crossDependencies']['speciation']['output'])):
            a = list(set(centralDict['crossDependencies']['speciation']['input']) - set(centralDict['crossDependencies']['speciation']['output']))
            commMtrxSpct = pd.concat([commMtrxSpct,centralDict['commMtrx'][a]], axis=1)
            commMtrx_primSpecies = pd.concat([commMtrx_primSpecies,centralDict['commMtrx'][a]], axis=1)
        
    
    with open("warning.log", "a") as warningLog:
        for i,status in enumerate(gemsStatusList):
            if status!=2:
                warningLog.write(f"GEMS, node n°{i}, time step n°{centralDict['lStep']+1}, t={centralDict['tStep']}{centralDict['timeUnit']}, PID={os.getpid()} : with {gemsIterations[i]} :\n")
                warningLog.write(f"GEMS, node n°{i}, time step n°{centralDict['lStep']+1}, t={centralDict['tStep']}{centralDict['timeUnit']}, PID={os.getpid()} : {gemsStatus[status]}\n")


    if outputManager.wanted(centralDict, 'speciation'):
        commMtrx_primSpecies.to_csv(outputManager.filePath(centralDict, 'primarySpecies', 'PrimarySpecies'), index=False, header=True, sep='\t')
        commMtrxSpct.to_csv(outputManager.filePath(centralDict, 'speciation', 'xGEMS'), index=False, header=True, sep='\t')
    print()
    print(commMtrxSpct)
    
    centralDict.update({
        "commMtrx": commMtrxSpct,
        "xGEMSCalcTime_WallClock": centralDict["xGEMSCalcTime_WallClock"] + calcWallClock,
        "xGEMSCalcTime_ProcessorTime": centralDict["xGEMSCalcTime_ProcessorTime"] + calcPrcsTime,
        "xGEMSInterfTime_WallClock": centralDict["xGEMSInterfTime_WallClock"] + time.time() - startGems - calcWallClock - init,
        "xGEMSInitTime" : centralDict["xGEMSInitTime"] + init,
        "xGEMSTotalTime" : centralDict["xGEMSTotalTime"] + time.time() - startGems,})
 
    
    print(f"({writeTime((time.time() - startGems))})") 


    return centralDict