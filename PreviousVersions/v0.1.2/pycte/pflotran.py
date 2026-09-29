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
    elif tps < 1:
        return f"{tps *1000 :.{arr}f} msec"
    else:
        return f"{tps:.{arr}f} sec"


def update_pflotran_time(input_path, new_time, unit):
    """
    Met a jour le FINAL_TIME (bloc TIME) et le TIMES (bloc OUTPUT) d'un
    fichier d'entree Pflotran, pour n'avoir qu'un seul temps de sortie,
    egal au nouveau temps final.
    """
    with open(input_path, "r") as f:
        text = f.read()

    pattern_final_time = re.compile(
        r'^(\s*FINAL_TIME\s+)([^\s#]+)(\s+)([^\s#]+)',
        re.MULTILINE,
    )
    text, n1 = pattern_final_time.subn(
        lambda m: f"{m.group(1)}{new_time}{m.group(3)}{unit}", text, count=1
    )
    if n1 == 0:
        raise ValueError("Ligne FINAL_TIME introuvable dans le fichier.")

    pattern_times = re.compile(
        r'^(\s*TIMES\s+)(\S+)(\s+).*$',
        re.MULTILINE,
    )
    text, n2 = pattern_times.subn(
        lambda m: f"{m.group(1)}{unit}{m.group(3)}{new_time}", text, count=1
    )
    if n2 == 0:
        raise ValueError("Line TIMES not found in OUTPUT block.")

    with open(input_path, "w") as f:
        f.write(text)

    # print(f"OK - FINAL_TIME et TIMES mis a jour a {new_time} {unit}")
    # print(f"Fichier ecrit : {output_path}")

def normalizeVarName(name):

    s = name.lower()
    s = s.replace("total ", "")
    s = s.replace("minus", "-")
    s = s.replace("plus", "+")
    s = re.sub(r"dc\b", "", s) # thats my pflotran formalism, but can be changed
    s = s.rstrip("]")            
    s = re.sub(r"\s+", " ", s).strip()
    return s


def matchVariables(source_names, target_names):

    norm_source = {name: normalizeVarName(name) for name in source_names}
    norm_target = {name: normalizeVarName(name) for name in target_names}

    def priority(name):
        return 1 if "rate" in name.lower() else 0

    mapping = {}
    used_source = set()
    remaining = list(target_names)

    def try_pass(candidate_fn):
        nonlocal remaining
        still_remaining = []
        for tgt in remaining:
            tgt_norm = norm_target[tgt]
            candidates = [s for s, sn in norm_source.items()
                          if s not in used_source and candidate_fn(tgt_norm, sn)]
            if len(candidates) >= 1:
                best = sorted(candidates, key=lambda s: priority(s))[0]
                mapping[tgt] = best
                used_source.add(best)
            else:
                still_remaining.append(tgt)
        remaining = still_remaining

    try_pass(lambda tn, sn: sn == tn)
    try_pass(lambda tn, sn: bool(tn) and tn in sn.split())
    try_pass(lambda tn, sn: bool(tn) and tn in sn)

    if remaining:
        print(
            f"No correspondance for : {remaining} "
            f"among files : {source_names}"
        )

    return mapping

_UNIT_SUFFIX_RE = re.compile(r"^(.*)\(([A-Za-z]+)\)$")
_KNOWN_UNITS = {"K", "C"}


def extractTargetUnit(name):
    """Si `name` se termine par une unite reconnue entre parentheses
    (ex: 'temperature(K)'), retourne (nom_de_base, unite) = ('temperature', 'K').
    Sinon (ex: 'H2(aq)', 'MX-80'), retourne (name, None) inchange -- les
    parentheses de phase chimique ('aq', 'g', 'l', ...) ne sont jamais
    interpretees comme une unite car elles ne sont pas dans _KNOWN_UNITS.
    """
    m = _UNIT_SUFFIX_RE.match(name)
    if m and m.group(2) in _KNOWN_UNITS:
        return m.group(1).strip(), m.group(2)
    return name, None


def convertUnit(value, from_unit, to_unit):
    if from_unit is None or to_unit is None or from_unit.upper() == to_unit.upper():
        return value
    fu, tu = from_unit.upper(), to_unit.upper()
    if fu == "C" and tu == "K":
        return value + 273.15
    if fu == "K" and tu == "C":
        return value - 273.15
    raise ValueError(f"Cannot convert units: '{from_unit}' -> '{to_unit}'")


def transport(centralDict):
    'write datasets, launch pflotran, read output file'

    spcArray = {
        col: np.where((arr := centralDict['commMtrx'][col].to_numpy()) == 0, 1e-99, arr)
        for col in centralDict['commMtrx'].columns
        if col not in (["Zz"] + centralDict['coord'])
    }

    with h5py.File(os.path.join(centralDict['trsptPath'].parent, 'testPiccts.h5'), 'w') as f:
        f.create_dataset('Cell Ids', data = np.array(list(range(1,len(centralDict['commMtrx'])+1)), dtype = 'i8'))
        for s in spcArray:
            if s == 'MX-80':
                f.create_dataset(s+'i', data = 1000*spcArray[s]*0.000137469997406006)
            elif s == 'hMX-80':
                f.create_dataset(s+'i', data = 1000*spcArray[s]*0.000215)
            #elif s == 'tk':
                #f.create_dataset(s+'i', data = (spcArray[s]-273.15))
            elif s == 'p':
                f.create_dataset(s+'i', data = (spcArray[s]))

            else:
                f.create_dataset(s+'i', data = spcArray[s])

    if centralDict['lStep'] == 0 or centralDict['dtStep'] != (centralDict['dtPICCTS'][centralDict['lStep']]-centralDict['dtPICCTS'][centralDict['lStep']-1]):
        update_pflotran_time(centralDict['trsptPath'],centralDict['dtStep'], centralDict['timeUnit'])

    pflotran = os.path.join(
        os.environ["PFLOTRAN_DIR"],
        "src",
        "pflotran",
        "pflotran"
    )

    with open("pflotran_output.txt", "w") as f:
        ref = time.perf_counter()
        subprocess.run(
            [
                "mpirun",
                "-n",
                "1",
                pflotran,
                "-pflotranin",
                centralDict["trsptPath"].name,
            ],
            stdout=f,
            cwd=centralDict["trsptPath"].parent,
            stderr=f,
            text=True,
            check=True,
        )

    calcTime = time.perf_counter() - ref

    pattern = re.compile(rf"{Path(centralDict['trsptPath']).stem}-(\d+)\.tec$")

    files = [f for f in Path(centralDict['trsptPath'].parent).iterdir() if pattern.match(f.name)] # will catch the highest one .. may be a probleme in the future

    if not files:
        raise FileNotFoundError("No '{Path(centralDict['trsptPath']).stem}-*.tec' file found.")

    last_file = max(files, key=lambda f: int(pattern.match(f.name).group(1)))

    variables_line = None
    with last_file.open() as f:
        for line in f:
            if line.startswith("VARIABLES="):
                variables_line = line
                break

    if variables_line is None:
        raise ValueError(f"No line VARIABLES= found in {last_file}")

    variables = []
    variableUnits = {}  
    for raw in re.findall(r'"([^"]+)"', variables_line):
        m = re.match(r"(.*?)\s*\[(.*?)\]\s*$", raw)
        if m:
            name, unit = m.group(1).strip(), m.group(2).strip()
        else:
            name, unit = raw.strip(), None
        variables.append(name)
        variableUnits[name] = unit

    df = pd.read_csv(
        last_file,
        sep=r"\s+",
        skiprows=3,
        names=variables,
        engine="python"
    )

    df = df.drop(columns="Material ID")
    
    targetNames = centralDict['coord'] + list(spcArray.keys())

    targetBaseAndUnit = [extractTargetUnit(t) for t in targetNames]
    matchTargets = [base for base, _ in targetBaseAndUnit]

    varMapping = matchVariables(list(df.columns), matchTargets)   

    df = df[[varMapping[base] for base in matchTargets]]          
    df.columns = targetNames                                      

    # convert units
    for target, (base, desiredUnit) in zip(targetNames, targetBaseAndUnit):
        if desiredUnit is None:
            continue
        sourceName = varMapping[base]
        sourceUnit = variableUnits.get(sourceName)
        df[target] = convertUnit(df[target], sourceUnit, desiredUnit)
    
    # to change ..
    df['hMX-80'] = df['hMX-80']/(1000*0.000215)
    df['MX-80'] = df['MX-80']/(1000*0.000137469997406006)
    
    return df, calcTime


def trspt(centralDict):
    print("PFLOTRAN", end=" ", flush=True)
    startPflotran = time.time()

    toAdd = False
    pflotranInput =  centralDict['commMtrx'].copy()
    if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('transport'):
        pflotranHeaders = centralDict['transportedSpecies'] + list(centralDict['crossDependencies']['transport']['total'])
    else:
        pflotranHeaders = centralDict['transportedSpecies']
    if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('transport'):
        a = list(set(centralDict['crossDependencies']['speciation']['total']) - set(centralDict['crossDependencies']['transport']['total']))
        if a:
            toAdd = a
            

    pflotranInput = pflotranInput[pflotranHeaders].copy()
    
    if centralDict['PIDnbr'] > 1:

        chunk_size = int(np.ceil(len(pflotranInput) / centralDict['PIDnbr']))

        commMtrxSplit = [pflotranInput.iloc[i:i + chunk_size] for i in range(0, len(pflotranInput), chunk_size)]

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

    
    if toAdd:
        comm = pd.concat([comm,centralDict['commMtrx'][toAdd]], axis=1)

    # if centralDict['crossDependencies']:
    #     if centralDict['crossDependencies'].get('speciation'):
    #         a=list(set(centralDict['crossDependencies']['speciation']['total']) - set(centralDict['crossDependencies']['transport']['total']))
    #         if a:
    #             comm = pd.concat([comm,centralDict['commMtrx'][a]], axis=1)

        # if centralDict['crossDependencies'].get('transport') and list(set(centralDict['crossDependencies']['transport']['input']) - set(centralDict['crossDependencies']['transport']['total'])):
            # a = list(set(centralDict['crossDependencies']['transport']['input']) - set(centralDict['crossDependencies']['transport']['total']))
            # comm = pd.concat([comm,centralDict['commMtrx'][a]], axis=1)

    centralDict.update({
        "commMtrx": comm,
        "PFLOTRANCalcTime_WallClock": centralDict["PFLOTRANCalcTime_WallClock"] + calcWallClock,
        "PFLOTRANCalcTime_ProcessorTime": centralDict["PFLOTRANCalcTime_ProcessorTime"] + calcPrcsTime,
        "PFLOTRANTotalTime" : centralDict["PFLOTRANTotalTime"] + time.time() - startPflotran,})


    print(f"({writeTime((time.time() - startPflotran))})")


    return centralDict