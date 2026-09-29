import time
import pandas as pd
import sys
import importlib.util
import os
import numpy as np

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
    """
    Démarre le client COMSOL et charge le modèle UNE SEULE FOIS.

    Le client / modèle / javamodel sont stockés dans centralDict
    ('comsolClient', 'comsolModel', 'comsolJavaModel') afin que
    transportComsol() les réutilise à chaque pas de temps sans jamais
    relancer COMSOL ni recharger le modèle. Cette fonction contient donc
    toute la logique d'attente de licence, qui ne s'exécute plus qu'une
    seule fois pour l'ensemble de la simulation.
    """
    import mph

    scriptWarning = ""
    warning = 0
    waitingLicence = 0
    init = 0

    inputPath = os.path.join(centralDict['inputPath'], 'inputTransport.txt')

    licenceTaken = False
    licence = False
    startWaiting = 0
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
                startWaiting = time.time()
                print(r)
                print("Waiting for a licence ? ...", flush=True)
                licenceTaken = True
                scriptWarning += f"COMSOL waiting licence : {writeTime(waitingLicence)}\n"
            if licenceTaken: time.sleep(3)

    if licenceTaken:
        warning += 1
        print(f"({writeTime(waitingLicence)} of waiting)", flush=True)
        scriptWarning += f"COMSOL init : {writeTime(waitingLicence)} of waiting \n"

    centralDict['comsolClient'] = client
    centralDict['comsolModel'] = model
    centralDict['comsolJavaModel'] = javamodel

    return centralDict, warning, scriptWarning, waitingLicence, init


def closeComsol(centralDict):
    """
    A appeler UNE SEULE FOIS, à la toute fin de la simulation (après le
    dernier pas de temps), pour libérer le modèle et rendre la licence
    COMSOL. Ne pas appeler entre deux pas de temps : ça obligerait à tout
    recharger au pas suivant.
    """
    client = centralDict.get('comsolClient')
    if client is not None:
        try:
            client.clear()
        except Exception as r:
            print(f"Warning while closing COMSOL client: {r}")

    centralDict.pop('comsolClient', None)
    centralDict.pop('comsolModel', None)
    centralDict.pop('comsolJavaModel', None)

    return centralDict


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

    client = centralDict['comsolClient']
    model = centralDict['comsolModel']
    javamodel = centralDict['comsolJavaModel']

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
            print(r)
            with open("warning.log", "a") as warningLog: warningLog.write(f"COMSOL, t={centralDict['tStep']}{centralDict['timeUnit']}, time-step n° {centralDict['lStep']} : {r}\n")
            abort = True

    if not abort:

        #### appears to not work all the times ...
        # spc = [s if s in centralDict['transportedSpecies'] else s+'i' for s in centralDict['systemSpeciation']]
        # coord_values = model.evaluate(['x', 'y', 'z'][:centralDict['geometry']],inner='last')
        # coord = pd.DataFrame(
        #     {name: values for name, values in zip(['x', 'y', 'z'], coord_values)})

        # values = model.evaluate(Convert_Species_PhreeqC_to_COMSOL(spc), inner="last")
        # commMtrx_Comsol = pd.DataFrame(
        #     np.array(values).T/1000, # if you want to correct with the density, here it is
        #     columns=centralDict['systemSpeciation']
        # )
        # commMtrx_Comsol = pd.concat([coord,commMtrx_Comsol], axis=1)


        outputPath = os.path.join(centralDict['inputPath'], 'outputTransport.txt')
        ref = time.perf_counter()
        javamodel.result().export("data1").setIndex("looplevelinput", "last", 0);
        javamodel.result().export("data1").set("exporttype", "text");
        javamodel.result().export("data1").set("filename", f"{outputPath}");
        javamodel.result().export("data1").run();
        init += time.perf_counter() - ref

        if centralDict['output'] and centralDict['output'].get('transport') and (centralDict['lStep']+1) in centralDict['output']['transport'] :
            if centralDict['outputComsol']:
                for i,tag in enumerate(centralDict['outputComsol']): # user defined variables ..
                    ref = time.perf_counter()
                    path = os.path.join(centralDict['paths'][f'Transport{tag}'], f'COMSOL_{centralDict["lStep"]+1}.txt')
                    javamodel.result().export(f"{tag}").setIndex("looplevelinput", "last", 0);
                    javamodel.result().export(f"{tag}").set("exporttype", "text");
                    javamodel.result().export(f"{tag}").set("filename", f"{path}");
                    javamodel.result().export(f"{tag}").run();
                    init += time.perf_counter() - ref
                    if centralDict['lStep'] == 0:
                        path = os.path.join(centralDict['paths'][f'Transport{tag}'],'COMSOL_0.txt')
                        ref = time.perf_counter()
                        javamodel.result().export(f"{tag}").setIndex("looplevelinput", "first", 0);
                        javamodel.result().export(f"{tag}").set("exporttype", "text");
                        javamodel.result().export(f"{tag}").set("filename", f"{path}");
                        javamodel.result().export(f"{tag}").run();
                        init += time.perf_counter() - ref
                    if centralDict['comsolVTU']:
                        path = os.path.join(centralDict['paths'][f'TransportVTU{tag}'],f'COMSOL_{centralDict["lStep"]+1}.vtu')
                        ref = time.perf_counter()
                        javamodel.result().export(f"{tag}").setIndex("looplevelinput", "last", 0);
                        javamodel.result().export(f"{tag}").set("exporttype", "vtu");
                        javamodel.result().export(f"{tag}").set("filename", f"{path}");
                        javamodel.result().export(f"{tag}").run();
                        init += time.perf_counter() - ref
                        if centralDict['lStep'] == 0:
                            path = os.path.join(centralDict['paths'][f'TransportVTU{tag}'],'COMSOL_0.vtu')
                            ref = time.perf_counter()
                            javamodel.result().export(f"{tag}").setIndex("looplevelinput", "first", 0);
                            javamodel.result().export(f"{tag}").set("exporttype", "vtu");
                            javamodel.result().export(f"{tag}").set("filename", f"{path}");
                            javamodel.result().export(f"{tag}").run();
                            init += time.perf_counter() - ref



        # col =  centralDict["coord"] + centralDict['systemSpeciation'] + (centralDict['crossDependencies']['speciation']['total'] if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation') else [])
        commMtrx_Comsol = pd.read_csv(outputPath,sep=r"\s+",comment="%",header=None, names=list(centralDict['commMtrx'].columns))
        # results = [commMtrx_Comsol, warning, scriptWarning,waitingLicence, abort,calcTime]




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
        sys.exit()

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
            sys.exit()

        try:
            assert len(comm) == len(centralDict['commMtrx'])
        except:

            print('Mtrx post transport length different from input: post trspt: ', len(comm), ' input: ', len(centralDict['commMtrx']))
            print('aborting')
            sys.exit()



    centralDict.update({
            "commMtrx": comm,
            "waitingTime" : centralDict['waitingTime'] + waitingLicence,
            "COMSOLCalcTime_WallClock": centralDict["COMSOLCalcTime_WallClock"] + calcTime,
            "COMSOLCalcTime_ProcessorTime": centralDict["COMSOLCalcTime_ProcessorTime"] + calcTime, # not correct but i dont have access to that info (to my knowledge)
            "COMSOLInterfTime_WallClock": centralDict['COMSOLInterfTime_WallClock'] + time.time() - startComsol - calcTime - init,
            "COMSOLInitTime" : centralDict["COMSOLInitTime"] + init,
            "COMSOLTotalTime" : centralDict["COMSOLTotalTime"] + time.time() - startComsol,
            })


    print(f"({writeTime((time.time()-startComsol))})")

    return centralDict