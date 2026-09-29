import numpy as np
import time 
import pandas as pd
import sys 
import os
from pathlib import Path
import concurrent.futures
import contextlib
import PyORCHESTRA

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
    """
    Site quantity is natively defined in the .inp file. This file could change depending on the node number ...
    side note:
        Sorbed attribute is .solid
        Speciation is Na+.con, Na.diss en solution, Na.tot total.
        Species attributes are defined by speciesAttributes keyword. It may refer to the .inp file.
    """
    
    commMtrxPart = commMtrxPart.copy()
    
    if centralDict['couplingFormalism'] == 'literbulk' and centralDict['lStep'] == 0 and centralDict['firstIC']:

        commMtrx_primSpecies = pd.read_csv('literbulkIC.txt',sep=r"\s+",comment="%",dtype=float)

        rows = commMtrx_primSpecies.to_numpy()
    
        refdf = commMtrx_primSpecies.copy()
        pd.set_option('display.float_format', lambda x: f'{x:.15f}')


    elif centralDict['couplingFormalism'] == 'literbulk' :
        
        commMtrxPart.loc[:, centralDict['colAq']] = (commMtrxPart.loc[:, centralDict['colAq']].mul(centralDict['waterMass'], axis=0))
    
        commMtrx_primSpecies = pd.DataFrame(
            commMtrxPart.to_numpy() @ centralDict['stoich'].to_numpy(),
            columns=centralDict['stoich'].columns)

        commMtrx_primSpecies = commMtrx_primSpecies.drop(columns="O")
        t = ["chargebalance","totvolume","A","poreD","dx","porosity","saturation","pH","pe",]
        commMtrx_primSpecies = pd.concat([centralDict['commMtrx'][t],commMtrx_primSpecies], axis=1)
        commMtrx_primSpecies.columns =  centralDict["inputVariableOrchestra"]
        
        refdf =  commMtrx_primSpecies.copy()
        rows = commMtrx_primSpecies.to_numpy()
    
    elif centralDict['couplingFormalism'] == 'tot':
        commMtrx_primSpecies = pd.DataFrame(
            commMtrxPart.to_numpy() @ centralDict['stoichReduced'].to_numpy(),
            columns=centralDict['primarySpecies'][-1]
        )
        rows = commMtrx_primSpecies.to_numpy()

    else:
        print(f'ORCHESTRA coupling formalism not recognized : {centralDict["couplingFormalism"]} (tot and literbulk available)')
        sys.exit()
    
    if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation'):
        commMtrx_primSpecies[centralDict['crossDependencies']['speciation']['total']] = centralDict['commMtrx'][centralDict['crossDependencies']['speciation']['total']].copy()
        commMtrx_primSpecies.columns = centralDict["inputVariableOrchestra"] 

    
    # df = pd.read_csv(r"C:\Users\AL274877\Desktop\usbSave0608\Aldicarb\test.txt", names = centralDict["inputVariableOrchestra"] , sep=r"\s+")
    # df.columns = centralDict["inputVariableOrchestra"] 
    # rows = df.to_numpy()
    # print()
    # print(df)
    # print(commMtrx_primSpecies)
    # a = df.to_numpy()                                # celui qui marche
    # b = commMtrx_primSpecies.to_numpy()              # celui qui ne marche pas
    
    # for nom, x in (("test.txt", a), ("PyCTE", b)):
    #     print(nom, x.shape, x.dtype, "C-contig:", x.flags['C_CONTIGUOUS'],
    #           "finite:", np.isfinite(x).all())
    # print(commMtrx_primSpecies.dtypes)               # dtype COLONNE PAR COLONNE
    nrCells = rows.shape[0]
    # print(nrCells)
    calcTime = 0
    # sys.exit()
    # print(centralDict['inputVariableOrchestra'], centralDict['outputVariableOrchestra'],commMtrx_primSpecies)
    # sys.exit()
    
    # from pathlib import Path
    # p = Path(centralDict['chemPath']).resolve()
    # print(p, p.stat().st_mtime)
    # print((p.parent / "objects2026Kin.txt").resolve(),
    #       (p.parent / "objects2026Kin.txt").stat().st_mtime)
    # print("kin_version présent :", "kin_version" in (p.parent / "objects2026Kin.txt").read_text(errors="replace"))


    if solver is None: 
        chem_dir = Path(centralDict['chemPath']).parent
        _cwd = os.getcwd()
        os.chdir(chem_dir)
        try:
            with redirect_stdout_fd("orchestra_output.log",centralDict['tStep'],centralDict['dtStep'],centralDict['lStep'],centralDict['timeUnit']):
                ref = time.perf_counter()
                solver = PyORCHESTRA.ORCHESTRA()
                solver.initialise(str(centralDict['chemPath']), nrCells, centralDict['inputVariableOrchestra'], centralDict['outputVariableOrchestra'])
                init= time.perf_counter() - ref
        finally:
            os.chdir(_cwd)
    else:
        init = 0
    # print('ici')

    # print(df)
    # rows = df.to_numpy()
    # print(df)
    with redirect_stdout_fd("orchestra_output.log",centralDict['tStep'],centralDict['dtStep'],centralDict['lStep'],centralDict['timeUnit']):
        memory_option = 1 if firstCalc else 0
        ref = time.perf_counter()
        outputarray = solver.set_and_calculate_multi(rows, centralDict['nrThreads'], memory_option)
        # outputarray = solver.set_and_calculate_multi(rows, 1, 1)

        calcTime = time.perf_counter() - ref
        firstCalc = False



    if centralDict['couplingFormalism'] == 'literbulk' : #centralDict['lStep'] :
        outputOrchestra = pd.DataFrame(outputarray, columns=(["chargebalance","totvolume","A","poreD","dx","porosity","saturation","pH","pe",'failed']+ centralDict['systemSpeciation']))      
    else:
        # outputOrchestra = pd.DataFrame(outputarray, columns=(centralDict['outputVariableOrchestra']))   
        # print(outputOrchestra)
        # sys.exit()
        outputOrchestra = pd.DataFrame(outputarray, columns=(centralDict['systemSpeciation']))      


    
    if centralDict['couplingFormalism'] == 'literbulk' :
        ### no MB after equilibrium, decomp to check
        output = outputOrchestra[centralDict['systemSpeciation']].copy()
        
        


        #cols = [s for s in centralDict['systemSpeciation'] if s not in centralDict['phases']]    
        
        output.loc[:, centralDict['colAq']] = (output.loc[:, centralDict['colAq']].mul(centralDict['waterMass'], axis=0))
    
        output = pd.DataFrame(
            output.to_numpy() @ centralDict['stoich'].to_numpy(),
            columns=centralDict['stoich'].columns)
        
        output = output.drop(columns="O")
        # print(output)
        # output = output.map(lambda x: float(f"{x:.10g}") if isinstance(x, (int, float)) else x)
        # refdf = refdf.map(lambda x: float(f"{x:.10g}") if isinstance(x, (int, float)) else x)

        # print(refdf)
        refdf = refdf.drop(columns = ["chargebalance","totvolume","A","poreD","dx","porosity","saturation","pH","pe"])
    
        output.columns = refdf.columns
        # print(output)
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
        pd.set_option('display.max_columns', None)
        pd.set_option('display.float_format', '{:.3e}'.format)
        
        print(diff)
    # if centralDict['lStep'] == 301:
        sys.exit()
    
    return outputOrchestra, commMtrx_primSpecies, calcTime, init

def spct(centralDict):
    print("ORCHESTRA", end=" ", flush=True)
    startOrchestra = time.time()

    colSave = centralDict['commMtrx'].columns
    # if centralDict['lStep'] > 0:
    #     print(centralDict['commMtrx'][['pH','Na+']])
    # sys.exit()
    commMtrxSpct, commMtrx_primSpecies, calcWallClock, init = speciationOrchestra(centralDict, centralDict['commMtrx'][centralDict['systemSpeciation']])
    calcPrcsTime = calcWallClock
        
    commMtrxSpct = pd.concat([centralDict['commMtrx'][centralDict["anythingButSpecies"]],commMtrxSpct], axis=1)

    #commMtrxSpct.columns = list(centralDict['commMtrx'].columns) + ['failed']


    
    if 'failed' in commMtrxSpct.columns:
        if (commMtrxSpct["failed"] == 1).any():
            print('Failed simulation')
            commMtrxSpct.to_csv("commMtrx_failed.txt", index=False, header=True, sep='\t')
            sys.exit()
        commMtrxSpct = commMtrxSpct.drop(columns="failed")
    commMtrx_primSpecies = pd.concat([centralDict['commMtrx'][['x', 'y', 'z'][:centralDict['geometry']]],commMtrx_primSpecies], axis=1)


    if centralDict['output'] and centralDict['output'].get('speciation') and (centralDict['lStep']+1) in centralDict['output']['speciation'] :
       commMtrx_primSpecies.to_csv(os.path.join(centralDict['paths']['PrimarySpecies'], f"PrimarySpecies_{centralDict['lStep']+1}.txt"), index=False, header=True, sep='\t')
       commMtrxSpct.to_csv(os.path.join(centralDict['paths']['Speciation'], f"ORCHESTRA_{centralDict['lStep']+1}.txt"), index=False, header=True, sep='\t')

    
    # print(commMtrxSpct)
    # sys.exit()
    
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