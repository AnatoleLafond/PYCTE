import xgems
import pandas as pd
import sys
import numpy as np
import time 
import importlib.util
import os
import concurrent.futures

# from pathlib import Path
# current_dir = Path(__file__).parent
# if str(current_dir) not in sys.path:
#     sys.path.insert(0, str(current_dir))

# with open("store.txt", 'r', encoding='utf-8') as fichier:
#     inputPath =  fichier.readline().strip() 
#     nameInput =  fichier.readline().strip()

# module_name = os.path.splitext(nameInput)[0]
# file_path = os.path.join(inputPath, nameInput)
# spec = importlib.util.spec_from_file_location(module_name, file_path)

# PICCTS_input = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(PICCTS_input)

def writeTime(tps, arr=2):
    if tps >= 3600 * 24:
        return f"{tps / (3600 * 24):.{arr}f} d"
    elif tps >= 3600:
        return f"{tps / 3600:.{arr}f} h"
    elif tps >= 60:
        return f"{tps / 60:.{arr}f} min"
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
    ref = time.perf_counter() 
    engine = xgems.ChemicalEngine(str(centralDict['chemPath']))
    init = time.perf_counter() - ref

    outputGems = pd.DataFrame(columns=centralDict['systemSpeciation'])
    MultiCompoundTransport = centralDict['MultiCompoundTransport'] 
    
    if centralDict['lStep'] ==0 : # initial conditions are species or IC
        l = [s for s in list(centralDict['commMtrx'].columns) if s not in (centralDict['coord']+centralDict['speciesByClass']['T']+centralDict['speciesByClass']['W']+centralDict['speciesByClass']['O']+centralDict['speciesByClass']['G'])]
        if set(l).issubset(centralDict['speciesByClass']['S']):
            if centralDict['MultiCompoundTransport'] :
                MultiCompoundTransport = False
            # else:
            #     MultiCompoundTransport = True
        # elif not centralDict['MultiCompoundTransport'] :
            # MultiCompoundTransport = True
            # if centralDict['MultiCompoundTransport'] : MultiCompoundTransport = True
            # else: MultiCompoundTransport = False
            

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
        species_list = commMtrx.columns  # ordre des espèces tel qu'il apparaît dans outputGems
        
        stoich_matrix = pd.DataFrame(
            0.0,
            index=species_list,
            columns=ic_list
        )
        
        for comp in species_list:
            for prim, coeff in centralDict['primToSecSpecies'][comp].items():
                stoich_matrix.at[comp, prim] = coeff
        result = commMtrx.values @ stoich_matrix.values
        
        commMtrx_primSpecies = pd.DataFrame(result, columns=ic_list, index=commMtrx.index)
        

    gemsStatusList = []
    gemsIterations = []
    
    # print(commMtrx_primSpecies)
    # sys.exit()
    calcTime = 0
    for index in commMtrx_primSpecies.index:
        ref = time.perf_counter()
        status = engine.equilibrate(
            298,1e5, # to continue ..
            [
                commMtrx_primSpecies.at[index, spc]
                for spc in centralDict['independentComponents']
            ])
        calcTime += time.perf_counter() - ref
            
        outputGems.loc[len(outputGems)] = engine.speciesAmounts()
        gemsStatusList += [status]
        gemsIterations += [engine.numIterations()]
        
    outputGems.index = commMtrx.index
    # print(outputGems)

    # phase species shall not be decomposed onto IC
    if centralDict['MultiCompoundTransport']:
        # multi-component transport
        
        outputSpecies = outputGems.copy()
        ic_list = centralDict['independentComponents']
        species_list = outputGems.columns
        
        stoich_matrix = pd.DataFrame(0.0,index=species_list,columns=ic_list)
        # print(centralDict['transportedSpecies'])
        # sys.exit()
        for comp in (centralDict['transportedSpecies']) : #species_list
            for prim, coeff in centralDict['primToSecSpecies'][comp].items():
                stoich_matrix.at[comp, prim] = coeff
        result = outputGems.values @ stoich_matrix.values
        
        outputGems = pd.DataFrame(result, columns=ic_list, index=commMtrx.index)
        # print(outputGems)
        outputGems = outputGems.drop(columns=centralDict['fixedSpecies'],errors="ignore")
        outputGems = pd.concat([outputGems, outputSpecies[centralDict['fixedSpecies'] ]], axis=1)
        
        
        # print(outputGems,'\n',outputSpecies)
        # sys.exit()
        
        
        
        # dico = {}
        # dico = {spc: [0]*len(outputGems) for spc in centralDict['independentComponents']}
        # ligne = 0
        # for _, row in outputGems.iterrows():
        #     for comp, conc in row.items():
        #         for prim in centralDict['primToSecSpecies'][comp] :
        #             dico[prim][ligne] += centralDict['primToSecSpecies'][comp][prim] * conc
        #     ligne += 1

        # outputGems = pd.DataFrame(dico)
        # outputGems.index = commMtrx.index
        
        return outputGems,outputSpecies,gemsStatusList,gemsIterations, calcTime, init
    else:
        
        return outputGems,commMtrx_primSpecies,gemsStatusList,gemsIterations, calcTime, init
	
def spct(centralDict): 
    print("xGEMS", end=" ", flush=True)
    startGems = time.time()
    
    
    
    if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation'):
        gemsInput = [s for s in centralDict['commMtrx'].columns if s not in centralDict['coord']] + list(centralDict['crossDependencies']['speciation']['input'])
    else: 
        gemsInput = [s for s in centralDict['commMtrx'].columns if s not in centralDict['coord']]
    
    # print(gemsInput,centralDict['commMtrx'])
    # sys.exit()
    if centralDict['PIDnbr'] > 1:
    
        chunk_size = int(np.ceil(len(centralDict['commMtrx']) / centralDict['PIDnbr']))
    
        commMtrxSplit = [centralDict['commMtrx'][gemsInput].iloc[i:i + chunk_size] for i in range(0, len(centralDict['commMtrx']), chunk_size)]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=centralDict['PIDnbr']) as executor:
            futures = []
            for chunk in commMtrxSplit:
                futures.append(
                    executor.submit(
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
        
        gemsStatusList = status[0]
        gemsIterations = iteration[0]
        
    else:
        commMtrxSpct, commMtrx_primSpecies, gemsStatusList, gemsIterations, calcWallClock, init = speciation_xGEMS(centralDict,centralDict['commMtrx'][gemsInput])
        calcPrcsTime = calcWallClock

    commMtrxSpct = pd.concat([centralDict['commMtrx'][centralDict["anythingButSpecies"]],commMtrxSpct], axis=1)
    commMtrx_primSpecies = pd.concat([centralDict['commMtrx'][['x', 'y', 'z'][:centralDict['geometry']]],commMtrx_primSpecies], axis=1)

    # commMtrxSpct.columns = centralDict['commMtrx'].columns
        
    # print(commMtrxSpct)
    # sys.exit()

    with open("warning.log", "a") as warningLog:
        for i,status in enumerate(gemsStatusList):
            if status!=2:
                warningLog.write(f"GEMS, node n°{i}, time step n°{centralDict['lStep']+1}, t={centralDict['tStep']}{centralDict['timeUnit']}, PID={os.getpid()} : with {gemsIterations[i]} :\n")
                warningLog.write(f"GEMS, node n°{i}, time step n°{centralDict['lStep']+1}, t={centralDict['tStep']}{centralDict['timeUnit']}, PID={os.getpid()} : {gemsStatus[status]}\n")

    
    if centralDict["firstStepEquilibrium"]==True:
        commMtrx_primSpecies.to_csv(os.path.join(centralDict['paths']['PrimarySpecies'], f"PrimarySpecies_{centralDict['lStep']}.txt"), index=False, header=True, sep='\t')
        commMtrxSpct.to_csv(os.path.join(centralDict['paths']['Speciation'], f"xGEMS_{centralDict['lStep']}.txt"), index=False, header=True, sep='\t')
    else:
        commMtrx_primSpecies.to_csv(os.path.join(centralDict['paths']['PrimarySpecies'], f"PrimarySpecies_{centralDict['lStep']+1}.txt"), index=False, header=True, sep='\t')
        commMtrxSpct.to_csv(os.path.join(centralDict['paths']['Speciation'], f"xGEMS_{centralDict['lStep']+1}.txt"), index=False, header=True, sep='\t')
       
    
    centralDict.update({
        "commMtrx": commMtrxSpct,
        "xGEMSCalcTime_WallClock": centralDict["xGEMSCalcTime_WallClock"] + calcWallClock,
        "xGEMSCalcTime_ProcessorTime": centralDict["xGEMSCalcTime_ProcessorTime"] + calcPrcsTime,
        "xGEMSInterfTime_WallClock": centralDict["xGEMSInterfTime_WallClock"] + time.time() - startGems - calcWallClock - init,
        "xGEMSInitTime" : centralDict["xGEMSInitTime"] + init,
        "xGEMSTotalTime" : centralDict["xGEMSTotalTime"] + time.time() - startGems,})
 
    
    print(f"({writeTime((time.time() - startGems))})") 


    return centralDict