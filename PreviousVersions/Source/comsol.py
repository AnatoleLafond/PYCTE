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


def transportComsol(comsolDict):

    import mph
    
    scriptWarning = ""
    warning = 0
    abort = False
    calcTime = 0
    init = 0
    inputPath = os.path.join(comsolDict['inputPath'], 'inputTransport.txt')
    
    comsolDict['commMtrx'].to_csv(inputPath, index=False, header=False, sep='\t') 
    ref = time.perf_counter()
    client = mph.start()
    init += time.perf_counter() - ref
    if comsolDict['comsolCore']: client = mph.Client(cores=comsolDict['comsolCore'])
    
    # if comsolDict['lStep']==0:
    licenceTaken = False
    licence = False
    while not licence :
        try: 
            ref = time.perf_counter()
            model = client.load(f"{comsolDict['trsptPath']}")
            javamodel = model.java
            javamodel.component(f"{comsolDict['comsolTags'][0]}").func(f"{comsolDict['comsolTags'][1]}").set("filename", f"{inputPath}")
            init += time.perf_counter() - ref
            licence = True
            if licenceTaken:
                endWaiting = time.time()
                print("Licence released :)", end=" ", flush=True)

        except Exception as r :
            if not licenceTaken:
                startWaiting = time.time()
                print(r)
                print("Waiting for a licence ? ...", flush=True)
                licenceTaken = True
                scriptWarning+= f"COMSOL, time = {comsolDict['tStep']}{comsolDict['timeUnit']}, time-step n° {comsolDict['lStep']+1} : {r}\n"
                time.sleep(3)
            if licenceTaken: time.sleep(3)

    if licenceTaken:
        warning += 1
        waitingLicence = endWaiting - startWaiting
        print(f"({writeTime(waitingLicence)} of waiting)", flush = True)
        scriptWarning+=f"COMSOL, t={comsolDict['tStep']}{comsolDict['timeUnit']}, time-step n° {comsolDict['lStep']+1} : {writeTime(waitingLicence)} of waiting \n"
        waitingLicence = endWaiting - startWaiting
    else: waitingLicence=0 
    # else:waitingLicence=0 
        
    # some Java to dynamically couple COMSOL ..
    # COMSOL output should fit the expected communication matrix (cf user guide)
    try:
        ref = time.perf_counter()
        javamodel.component(f"{comsolDict['comsolTags'][0]}").func(f"{comsolDict['comsolTags'][1]}").refresh();
        init += time.perf_counter() - ref
        if comsolDict['timeUnit'] =='y': timeUnit = 'a'
        else: timeUnit = comsolDict['timeUnit']
        ref = time.perf_counter()
        javamodel.study(f"{comsolDict['comsolTags'][2]}").feature("time").set("tunit", f"{timeUnit}");

        javamodel.study(f"{comsolDict['comsolTags'][2]}").feature("time").set("tlist", f"range(0,{comsolDict['dtStep']},{comsolDict['dtStep']})");
        init += time.perf_counter() - ref
    except Exception as r:
        print(r)
        warning += 1
        scriptWarning+= f"COMSOL, time = {comsolDict['tStep']}{comsolDict['timeUnit']}, time-step n° {comsolDict['lStep']+1} : Fatal error occured : \n{r}\n"
        abort = True
    
    if not abort:
        try:
            ref = time.perf_counter()
            model.solve()
            calcTime += time.perf_counter() - ref
        except Exception as r:
            print(r)
            with open("warning.log", "a") as warningLog: warningLog.write(f"COMSOL, t={comsolDict['tStep']}{comsolDict['timeUnit']}, time-step n° {comsolDict['lStep']} : {r}\n")
            abort = True
    if not abort:
        
        #### appears to not work all the times ... 
        # spc = [s if s in comsolDict['transportedSpecies'] else s+'i' for s in comsolDict['systemSpeciation']]
        # coord_values = model.evaluate(['x', 'y', 'z'][:comsolDict['geometry']],inner='last')
        # coord = pd.DataFrame(
        #     {name: values for name, values in zip(['x', 'y', 'z'], coord_values)})

        # values = model.evaluate(Convert_Species_PhreeqC_to_COMSOL(spc), inner="last")
        # commMtrx_Comsol = pd.DataFrame(
        #     np.array(values).T/1000, # if you want to correct with the density, here it is
        #     columns=comsolDict['systemSpeciation']
        # )
        # commMtrx_Comsol = pd.concat([coord,commMtrx_Comsol], axis=1)
        
        
        outputPath = os.path.join(comsolDict['inputPath'], 'outputTransport.txt')
        ref = time.perf_counter()
        javamodel.result().export("data1").setIndex("looplevelinput", "last", 0);
        javamodel.result().export("data1").set("exporttype", "text");
        javamodel.result().export("data1").set("filename", f"{outputPath}");
        javamodel.result().export("data1").run();
        init += time.perf_counter() - ref
        
        
        if comsolDict['outputComsol']:
            for i,tag in enumerate(comsolDict['outputComsol']): # user defined variables ..
                ref = time.perf_counter()
                path = os.path.join(comsolDict['paths'][f'Transport{tag}'], f'COMSOL_{comsolDict["lStep"]+1}.txt')
                javamodel.result().export(f"{tag}").setIndex("looplevelinput", "last", 0);
                javamodel.result().export(f"{tag}").set("exporttype", "text");
                javamodel.result().export(f"{tag}").set("filename", f"{path}");
                javamodel.result().export(f"{tag}").run();
                init += time.perf_counter() - ref
                if comsolDict['lStep'] == 0:
                    path = os.path.join(comsolDict['paths'][f'Transport{tag}'],'COMSOL_0.txt')
                    ref = time.perf_counter()
                    javamodel.result().export(f"{tag}").setIndex("looplevelinput", "first", 0);
                    javamodel.result().export(f"{tag}").set("exporttype", "text");
                    javamodel.result().export(f"{tag}").set("filename", f"{path}");
                    javamodel.result().export(f"{tag}").run();
                    init += time.perf_counter() - ref
                if comsolDict['comsolVTU']:
                    path = os.path.join(comsolDict['paths'][f'TransportVTU{tag}'],f'COMSOL_{comsolDict["lStep"]+1}.vtu')
                    ref = time.perf_counter()
                    javamodel.result().export(f"{tag}").setIndex("looplevelinput", "last", 0);
                    javamodel.result().export(f"{tag}").set("exporttype", "vtu");
                    javamodel.result().export(f"{tag}").set("filename", f"{path}");
                    javamodel.result().export(f"{tag}").run();
                    init += time.perf_counter() - ref
                    if comsolDict['lStep'] == 0:
                        path = os.path.join(comsolDict['paths'][f'TransportVTU{tag}'],'COMSOL_0.vtu')
                        ref = time.perf_counter()
                        javamodel.result().export(f"{tag}").setIndex("looplevelinput", "first", 0);
                        javamodel.result().export(f"{tag}").set("exporttype", "vtu");
                        javamodel.result().export(f"{tag}").set("filename", f"{path}");
                        javamodel.result().export(f"{tag}").run();
                        init += time.perf_counter() - ref
    
            
        ref = time.perf_counter()
        # javamodel.sol("sol1").clearSolutionData();
        client.clear()
        init += time.perf_counter() - ref

        # col =  comsolDict["coord"] + comsolDict['systemSpeciation'] + (comsolDict['crossDependencies']['speciation']['total'] if comsolDict['crossDependencies'] and comsolDict['crossDependencies'].get('speciation') else [])
        commMtrx_Comsol = pd.read_csv(outputPath,sep=r"\s+",comment="%",header=None, names=list(comsolDict['commMtrx'].columns))
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