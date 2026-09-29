import time
import pandas as pd
import sys
import importlib.util
import os
import numpy as np
try:
    from . import outputManager
    from . import warningManager
except ImportError:
    import outputManager
    import warningManager

_comsol = {}

def writeTime(tps, arr=2):
    if tps >= 3600 * 24:
        return f"{tps / (3600 * 24):.{arr}f} d"
    elif tps >= 3600:
        return f"{tps / 3600:.{arr}f} h"
    elif tps >= 60:
        return f"{tps / 60:.{arr}f} min"
    elif tps < 1:
        return f"{tps*1000 :.{arr}f} msec"
    else:
        return f"{tps:.{arr}f} sec"

def Convert_Species_PhreeqC_to_COMSOL(entry_list):
    exit_list = []
    for name in entry_list:
        if name.startswith("("):
            name = name.replace("(","")

        name = name.replace("(", "_").replace(")", "_").replace(",", "_").replace(":", "_").replace('.','_')
        name = name.replace("-7", "minus7")
        name = name.replace("-6", "minus6")
        name = name.replace("-5", "minus5")
        name = name.replace("-4", "minus4")
        name = name.replace("-3", "minus3")
        name = name.replace("-2", "minus2")
        name = name.replace("-", "minus")

        name = name.replace("+7", "plus7")
        name = name.replace("+6", "plus6")
        name = name.replace("+5", "plus5")
        name = name.replace("+4", "plus4")
        name = name.replace("+3", "plus3")
        name = name.replace("+2", "plus2")
        name = name.replace("+", "plus")
        exit_list.append(name)
    return exit_list

def initComsol(centralDict):
    import mph
    scriptWarning = ""
    warning = 0
    waitingLicence = 0
    init = 0
    inputPath = os.path.join(centralDict['inputPath'], 'inputTransport.txt')
    licenceTaken = False
    licence = False
    startWaiting = time.perf_counter()         
    while not licence:
        ref = time.perf_counter()
        try:
            client = mph.start()
            if centralDict['comsolCore']: client = mph.Client(cores=centralDict['comsolCore'])
            model = client.load(f"{centralDict['trsptPath']}")
            javamodel = model.java
            javamodel.component(f"{centralDict['comsolTags'][0]}").func(f"{centralDict['comsolTags'][1]}").set("filename", f"{inputPath}")
            init += time.perf_counter() - ref
            licence = True
            if licenceTaken:
                waitingLicence = time.perf_counter() - startWaiting  
                print("Licence released :)", end=" ", flush=True)
        except Exception as r:
            if not licenceTaken:
                print(r)
                print("Waiting for a licence ? ...", flush=True)
                licenceTaken = True
            time.sleep(3)
    if licenceTaken:
        warning += 1
        print(f"({writeTime(waitingLicence)} of waiting)", flush=True)
        scriptWarning += f"COMSOL init : {writeTime(waitingLicence)} of waiting\n"  
    _comsol['comsolClient'] = client
    _comsol['comsolModel'] = model
    _comsol['comsolJavaModel'] = javamodel
    return centralDict, warning, scriptWarning, waitingLicence, init


def closeComsol(centralDict):
    client = _comsol.get('comsolClient')
    if client is not None:
        try:
            client.clear()
        except Exception as r:
            warningManager.warn(f"COMSOL : the COMSOL client could not be closed : {r}")
    _comsol.clear()

def transportComsol(centralDict):

    scriptWarning = ""
    warning = 0
    abort = False
    calcTime = 0
    init = 0
    waitingLicence = 0

    inputPath = os.path.join(centralDict['inputPath'], 'inputTransport.txt')

    centralDict['commMtrx'].to_csv(inputPath, index=False, header=False, sep='\t')

    if centralDict.get('comsolClient') is None:
        centralDict, initWarning, initScriptWarning, waitingLicence, initTime = initComsol(centralDict)
        warning += initWarning
        scriptWarning += initScriptWarning
        init += initTime

    model = _comsol['comsolModel']
    javamodel = _comsol['comsolJavaModel']

    try:
        ref = time.perf_counter()
        javamodel.component(f"{centralDict['comsolTags'][0]}").func(f"{centralDict['comsolTags'][1]}").set("filename", f"{inputPath}")
        javamodel.component(f"{centralDict['comsolTags'][0]}").func(f"{centralDict['comsolTags'][1]}").refresh()
        init += time.perf_counter() - ref

        if centralDict['timeUnit'] == 'y': timeUnit = 'a'
        else: timeUnit = centralDict['timeUnit']

        ref = time.perf_counter()
        javamodel.study(f"{centralDict['comsolTags'][2]}").feature("time").set("tunit", f"{timeUnit}")
        javamodel.study(f"{centralDict['comsolTags'][2]}").feature("time").set("tlist", f"range(0,{centralDict['dtStep']},{centralDict['dtStep']})")
        init += time.perf_counter() - ref
    except Exception as r:
        print(r)
        warning += 1
        scriptWarning += f"COMSOL, time = {centralDict['tStep']}{centralDict['timeUnit']}, time-step n° {centralDict['lStep']+1} : Fatal error occured : \n{r}\n"
        abort = True

    if not abort:
        try:
            ref = time.perf_counter()
            model.solve()
            calcTime += time.perf_counter() - ref
        except Exception as r:
            # modele en echec sauve dans le dossier du run : le modele d'origine (ex. celui d'un exemple installe)
            # reste intact
            model.save(os.path.join(centralDict['inputPath'],
                                    os.path.splitext(os.path.basename(centralDict['trsptPath']))[0] + "_failed.mph"))
            print(r)
            with open("warning.log", "a") as warningLog: warningLog.write(f"COMSOL, t={centralDict['tStep']}{centralDict['timeUnit']}, time-step n° {centralDict['lStep']} : {r}\n")
            abort = True

    if not abort:




        outputPath = os.path.join(centralDict['inputPath'], 'outputTransport.txt')
        ref = time.perf_counter()
        javamodel.result().export("data1").setIndex("looplevelinput", "last", 0);
        javamodel.result().export("data1").set("exporttype", "text");
        javamodel.result().export("data1").set("filename", f"{outputPath}");
        javamodel.result().export("data1").run();
        init += time.perf_counter() - ref

        exports = [(centralDict['lStep'] + 1, "last")] + ([(0, "first")] if centralDict['lStep'] == 0 else [])
        for step, level in exports:
            if not outputManager.wanted(centralDict, 'transport', step):
                continue
            for tag in centralDict['outputComsol']:
                formats = [("text", f'Transport{tag}', "txt")] + ([("vtu", f'TransportVTU{tag}', "vtu")] if centralDict['comsolVTU'] else [])
                for exportType, pathKey, ext in formats:
                    ref = time.perf_counter()
                    path = os.path.join(centralDict['paths'][pathKey], f'COMSOL_{step}.{ext}')
                    javamodel.result().export(f"{tag}").setIndex("looplevelinput", level, 0);
                    javamodel.result().export(f"{tag}").set("exporttype", exportType);
                    javamodel.result().export(f"{tag}").set("filename", f"{path}");
                    javamodel.result().export(f"{tag}").run();
                    init += time.perf_counter() - ref

        commMtrx_Comsol = pd.read_csv(outputPath,sep=r"\s+",comment="%",header=None, names=list(centralDict['commMtrx'].columns))
        
        
        closeComsol(centralDict)
        
        return commMtrx_Comsol, warning, scriptWarning,waitingLicence, abort,calcTime, init

    else:
        return pd.DataFrame(), warning, scriptWarning, waitingLicence, abort, 0, init

def trspt(centralDict):
    print("COMSOL", end=" ", flush=True)
    startComsol = time.time()

    comm, warning, scriptWarning,waitingLicence, abort, calcTime, init = transportComsol(centralDict)

    if warning:
        with open("warning.log", "a") as warningLog:
            warningLog.write(f"COMSOL, time = {centralDict['tStep']}{centralDict['timeUnit']}, time step n°{centralDict['lStep']+1} : the {warning} following warnings occured ...\n {scriptWarning}")
    if abort:
        with open("warning.log", "a") as warningLog:
            warningLog.write(f"COMSOL, time = {centralDict['tStep']}{centralDict['timeUnit']}, time step n°{centralDict['lStep']+1} : Fatal COMSOL error. Aborting run.")
        print('Fatal COMSOL error. Aborting run.')
        closeComsol(centralDict)            # libere le modele (sinon fichier verrouille pour un run suivant)
        sys.exit()

    if centralDict["preliminarEquilibrium"]:
    
        try:
            assert len(comm) == len(centralDict['commMtrx'])
        except:
            print('Mtrx post transport length different from input: post trspt: ', len(comm), ' input: ', len(centralDict['commMtrx']))
            print('2nd try')
            comm, warning, scriptWarning,waitingLicence, abort, calcTime, init = transportComsol(centralDict)

            if warning:
                with open("warning.log", "a") as warningLog:
                    warningLog.write(f"COMSOL, time = {centralDict['tStep']}{centralDict['timeUnit']}, time step n°{centralDict['lStep']+1} : the {warning} following warnings occured ...\n {scriptWarning}")
            if abort:
                with open("warning.log", "a") as warningLog:
                    warningLog.write(f"COMSOL, time = {centralDict['tStep']}{centralDict['timeUnit']}, time step n°{centralDict['lStep']+1} : Fatal COMSOL error. Aborting run.")
                print('Fatal COMSOL error. Aborting run.')
                closeComsol(centralDict)
                sys.exit()
            
            try:
                assert len(comm) == len(centralDict['commMtrx'])
            except:
    
                print('Mtrx post transport length different from input: post trspt: ', len(comm), ' input: ', len(centralDict['commMtrx']))
                print('aborting')
                sys.exit()


    if centralDict['lStep'] == len(centralDict['dtpycte'])-1:
        closeComsol(centralDict)
        centralDict.pop('comsolClient', None)
        centralDict.pop('comsolModel', None)
        centralDict.pop('comsolJavaModel', None)

    centralDict.update({
            "commMtrx": comm,
            "waitingTime" : centralDict['waitingTime'] + waitingLicence,
            "COMSOLCalcTime_WallClock": centralDict["COMSOLCalcTime_WallClock"] + calcTime,
            "COMSOLCalcTime_ProcessorTime": centralDict["COMSOLCalcTime_ProcessorTime"] + calcTime,
            "COMSOLInterfTime_WallClock": centralDict['COMSOLInterfTime_WallClock'] + time.time() - startComsol - calcTime - init,
            "COMSOLInitTime" : centralDict["COMSOLInitTime"] + init,
            "COMSOLTotalTime" : centralDict["COMSOLTotalTime"] + time.time() - startComsol,
            })


    print(f"({writeTime((time.time()-startComsol))})")

    return centralDict