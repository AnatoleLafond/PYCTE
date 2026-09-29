import time
import numpy as np
import os
import concurrent.futures
import subprocess
import re
from pathlib import Path
import pandas as pd
import h5py
import sys


def writeTime(tps, arr=2):
    if tps >= 3600 * 24:
        return f"{tps / (3600 * 24):.{arr}f} d"
    elif tps >= 3600:
        return f"{tps / 3600:.{arr}f} h"
    elif tps >= 60:
        return f"{tps / 60:.{arr}f} min"
    else:
        return f"{tps:.{arr}f} sec"


def transport(centralDict):
    'write datasets, launch pflotran, read output file'
    

    trsptSpc = [s for s in centralDict['commMtrx'].columns if s not in (centralDict['fixedSpecies']+["Zz"]+centralDict['coord'])]

    spcArray = {col: centralDict['commMtrx'][col].to_numpy() for col in centralDict['commMtrx'].columns if col in trsptSpc}


    with h5py.File(os.path.join(centralDict['trsptPath'].parent, 'testPiccts.h5'), 'w') as f:
        f.create_dataset('Cell Ids', data = np.array(list(range(1,len(centralDict['commMtrx'])+1)), dtype = 'i8'))
        for s in trsptSpc:
            f.create_dataset(s+'i', data = spcArray[s])

    pflotran = os.path.join(
        os.environ["PFLOTRAN_DIR"],
        "src",
        "pflotran",
        "pflotran"
    )
    
    with open("pflotran_output.txt", "w") as f:
        ref = time.perf_counter() 
        subprocess.run(
            ["mpiexec", "-n", "1", pflotran, f"{os.path.join(centralDict['trsptPath'].parent, 'pflotran.in')}"],
            stdout=f,
            cwd=Path(centralDict['trsptPath'].parent),
            stderr=f,
            text=True,
            check=True        
        )
    calcTime = time.perf_counter() - ref
    
    pattern = re.compile(r"pflotran-(\d+)\.tec$")
        
    files = [f for f in Path(centralDict['trsptPath'].parent).iterdir() if pattern.match(f.name)] # will catch the highest one .. may be a probleme in the future
    
    if not files:
        raise FileNotFoundError("Aucun fichier 'pflotran-*.tec' trouvé.")
    
    last_file = max(files, key=lambda f: int(pattern.match(f.name).group(1)))
    
    variables_line = None
    with last_file.open() as f:
        for line in f:
            if line.startswith("VARIABLES="):
                variables_line = line
                break
    
    if variables_line is None:
        raise ValueError(f"Aucune ligne VARIABLES= trouvée dans {last_file}")
    
    variables = [
        re.sub(r"\s*\[.*?\]", "", v).strip()
        for v in re.findall(r'"([^"]+)"', variables_line)
    ]
    
    df = pd.read_csv(
        last_file,
        sep=r"\s+",
        skiprows=3,
        names=variables,
        engine="python"
    )

    df = df.drop(columns="Material ID")
    df.columns = centralDict['coord'] + trsptSpc
    df = pd.concat([df,centralDict['commMtrx'][centralDict['fixedSpecies']]], axis = 1)

    return df, calcTime


def trspt(centralDict): 
    print("PFLOTRAN", end=" ", flush=True)
    startPflotran = time.time()
    
    
    if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation'):
        gemsInput = centralDict['transportedSpecies'] + list(centralDict['crossDependencies']['speciation']['input'])
    else: 
        gemsInput = centralDict['transportedSpecies'] 
    

    if centralDict['PIDnbr'] > 1:
    
        chunk_size = int(np.ceil(len(centralDict['commMtrx']) / centralDict['PIDnbr']))
    
        commMtrxSplit = [centralDict['commMtrx'][gemsInput].iloc[i:i + chunk_size] for i in range(0, len(centralDict['commMtrx']), chunk_size)]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=centralDict['PIDnbr']) as executor:
            futures = []
            for chunk in commMtrxSplit:
                futures.append(
                    executor.submit(
                        transport,
                        centralDict,
                        # chunk,
                    )
                )

    else:
        comm, calcWallClock = transport(centralDict)
        calcPrcsTime = calcWallClock
    
    
    centralDict.update({
        "commMtrx": comm,
        "PFLOTRANCalcTime_WallClock": centralDict["PFLOTRANCalcTime_WallClock"] + calcWallClock,
        "PFLOTRANCalcTime_ProcessorTime": centralDict["PFLOTRANCalcTime_ProcessorTime"] + calcPrcsTime,
        # "PFLOTRANInterfTime_WallClock": centralDict["PFLOTRANInterfTime_WallClock"] + time.time() - startPflotran - calcWallClock - init,
        # "PFLOTRANInitTime" : centralDict["PFLOTRANInitTime"] + init,
        "PFLOTRANTotalTime" : centralDict["PFLOTRANTotalTime"] + time.time() - startPflotran,})
 
    
    print(f"({writeTime((time.time() - startPflotran))})") 


    return centralDict
