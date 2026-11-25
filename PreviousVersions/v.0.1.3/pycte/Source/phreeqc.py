import numpy as np
import time 
import pandas as pd
import sys 
import importlib.util
import os
import concurrent.futures
from concurrent.futures import as_completed
try:
    from . import outputManager
    from . import warningManager
except ImportError:
    import outputManager
    import warningManager

phreeqc = None

def resetPhreeqC(verbose=False):

    global phreeqc
    if phreeqc is None:
        return False
    for meth in ('destroy_iphreeqc', 'destroy', 'DestroyIPhreeqc'):
        f = getattr(phreeqc, meth, None)
        if callable(f):
            try:
                f()
                if verbose: print(f"IPhreeqc released via {meth}() (PID={os.getpid()})")
            except Exception as r:
                warningManager.warn(f"PhreeqC : IPhreeqc not released by {meth}() : {r}")
            break
    phreeqc = None
    return True

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

def speciesToNode(v, ranges, labels):
    """
    Associate a species to a node number
    """
    ranges = np.asarray(ranges, dtype=float)  
    
    results = []
    
    mask = (v >= ranges[:, 0]) & (v <= ranges[:, 1])
    
    if np.any(mask):
        idx = np.where(mask)[0][0]
        matched_label =  labels[idx]
    else:
        matched_label = None
    
    results.append(matched_label)
    
    return results


def speciationPhreeqC(centralDict, commMtrxPart, beforeTrsptMtrx): 
    global phreeqc
    import phreeqpy.iphreeqc.phreeqc_dll as phreeqc_mod
    ref = time.perf_counter()
    
    if phreeqc is None: 
        phreeqc = phreeqc_mod.IPhreeqc()
        phreeqc.load_database(str(centralDict['chemPath']))
    
    initWorker = time.perf_counter() - ref

    mctTotals = centralDict.get('mctTotals') if centralDict.get('MultiCompoundTransport') else None

    if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation'):
        inputBlock = centralDict['crossDependencies']['speciation']['input']
        cmdToNameInput = dict(zip(inputBlock['inputPhreeqCCmd'], inputBlock['inputName']))

    def nodeSpeciation(commMtrxPart,commMtrx_primSpecies,spcChargeDefaut,spcChargeGeom,speciationCharge,solMod,currentPrimSpecies):
        

        
        kinetics = False
        phListUser = ['ph','pH','H']
        scriptMain =f"node n°{index}\n"
        if centralDict.get('molesStorage') and centralDict.get('cellWaterMass') is not None:
            # molesStorage : appat a la masse d'eau calculee au pas precedent (kg) ; SOLUTION_MODIFY la recalcule a
            # partir du bilan de O, l'appat n'est que le point de depart des iterations
            water = centralDict['cellWaterMass'][index]
        elif centralDict['water']:

                    water = centralDict['water'][index]
        else: water = 1
        
            
        bait = f"""
    SOLUTION 1
    -units mmol/kgw
    # Na 1
    # Cl 1
    -water {water}
    END
        """
        if centralDict['kinetics'] and not centralDict['solMod'] and centralDict['dtStep'] > 0:
            kinetics = True
            scriptMain += "\nKINETICS 1\n"
            scriptMain += f"{centralDict['kinetics']}\n"
            for k in centralDict['kineticsSpecies']:
                scriptMain += f'\n-m0 {commMtrx_primSpecies.loc[index,k]}'
            scriptMain += f'\n-step {centralDict["dtStep"]} {centralDict["timeUnit"]}\n'
            
            if centralDict['step_divide']:
                scriptMain += f"-step_divide {centralDict['step_divide']}\n"
                
        if centralDict['fixpH'] or centralDict['primarySpecies']['phases']:
            scriptMain +="\nEQUILIBRIUM_PHASES 1\n"
            if centralDict['fixpH']:
                scriptMain += f"Fix_ph {centralDict['fixpH']}\n"
            if centralDict['primarySpecies']['phases']: 
                for phase in centralDict['primarySpecies']['phases']:
                    if commMtrx_primSpecies.loc[index,phase] > centralDict['cutoffs']['phases'] and phase not in [centralDict['kineticsSpecies'] if centralDict['kinetics'] else []]:
                        scriptMain +=f"\t{phase} {centralDict['SI'][phase]} {commMtrx_primSpecies.loc[index,phase]}  {centralDict['mineralReversibility'][phase]}\n"

        if centralDict['primarySpecies']['surface']:
            scriptMain +="\nSURFACE 1\n"
            if centralDict['preliminarEquilibrium']: scriptMain += "equilibrate with solution 1\n"
            edl = False
            for surface in centralDict['primarySpecies']['surface']:
                if commMtrx_primSpecies.loc[index,surface] > centralDict['cutoffs']['surface'] :
                    scriptMain +=f"\t{surface} {commMtrx_primSpecies.loc[index,surface]} 100 1 \n"
                    edl = True
            if edl and centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation') and "DebyeLength" in centralDict['crossDependencies']['speciation']['input']['inputPhreeqCCmd'] and commMtrxPart.loc[index,cmdToNameInput["DebyeLength"]] > 0:
                scriptMain +=f"-diffuse_layer {commMtrxPart.loc[index,cmdToNameInput['DebyeLength']]}\n"
            else: scriptMain += "-no_edl\n"
            if centralDict['surfaceCounterIons']: scriptMain += "-only_counter_ions true\n"
                
            
        if centralDict['primarySpecies']['exchange']:
            scriptMain +="\nEXCHANGE 1\n"
            if centralDict['preliminarEquilibrium']: scriptMain += "equilibrate with solution 1\n"

            for exchange in centralDict['primarySpecies']['exchange']:
                if exchange in commMtrx_primSpecies and commMtrx_primSpecies.loc[index,exchange] > centralDict['cutoffs']['exchange']:
                        scriptMain +=f"\t{exchange} {commMtrx_primSpecies.loc[index,exchange]}\n"
                        
            if centralDict['acidicTrspt'] and not centralDict['rnvllmt']:
                if AciditeDiff.loc[index,'H+'] >0: scriptMain += f"\tAcide_in_H {AciditeDiff.loc[index,'H+']}\n"
                elif AciditeDiff.loc[index,'H+'] <0: scriptMain += f"\tAcide_out_Similication {-AciditeDiff.loc[index,'H+']}\n"
                if AciditeDiff.loc[index,'OH-'] >0: scriptMain += f"\tBase_in_OH {AciditeDiff.loc[index,'OH-']}\n"
                elif AciditeDiff.loc[index,'OH-'] <0: scriptMain += f"\tBase_out_Similianion {-AciditeDiff.loc[index,'OH-']}\n"

        if solMod:
            if centralDict['preliminarEquilibrium']:
                
                scriptMain +=f"\nSOLUTION 1 #node n°{index}\n-units mol/kgw\ntemp {centralDict['tempDefault']}\n"
                if 'H2O' in commMtrx_primSpecies:
                    scriptMain += f"-water {beforeTrsptMtrx.loc[index,'H2O']*18.015/1000}\n"
                else: 
                    
                    scriptMain += "-water 1\n"
                
                if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation'):
                    if '-la("e-")' in centralDict['crossDependencies']['speciation']['input']['inputPhreeqCCmd']:
                        pe = cmdToNameInput['-la("e-")']
                        scriptMain += f"\tpe {beforeTrsptMtrx.loc[index,pe]}\n"
                    if '-la("H+")' in centralDict['crossDependencies']['speciation']['input']['inputPhreeqCCmd']:
                        ph = cmdToNameInput['-la("H+")']
                        scriptMain += f"\tpH {beforeTrsptMtrx.loc[index,ph]}\n"


    
                elif centralDict["acidicTrspt"]:
                    scriptMain += f"\tpH {AcidicEcho.loc[index,'pH']}"
                else:
                    scriptMain += f"\tpH {centralDict['pHdefault']}" 

                if centralDict["acidicTrspt"] and AciditeDiff.loc[index,'H+'] > 0: scriptMain += f"\tSimilication {AciditeDiff.loc[index,'H+']}\n"
                if centralDict["acidicTrspt"] and AciditeDiff.loc[index,'OH-'] > 0 : scriptMain += f"\tSimilianion {AciditeDiff.loc[index,'OH-']}\n"                            
                scriptMain += "\n"
                perKgw = (beforeTrsptMtrx.loc[index,'H2O']*18.015/1000 if centralDict.get('molesStorage') and 'H2O' in commMtrx_primSpecies else 1)
                for species in centralDict['primarySpecies']['solution']:
                    if commMtrx_primSpecies.loc[index,species] > centralDict['cutoffs']['solution'] and species not in (centralDict['primarySpecies']['primarySpeciesPhantom'] + ['H','O','H2O']) :
                        scriptMain += f"\n\t{species} {commMtrx_primSpecies.loc[index,species] / perKgw}"

                scriptMain += "\n"
                        
                if centralDict['supplementarySolution'] : scriptMain += centralDict['supplementarySolution'] + '\n'
                if centralDict['preliminarEquilibrium']:
                    if centralDict['primarySpecies']['phases']: scriptMain += "save equilibrium_phases 2\n"
                    if centralDict['primarySpecies']['surface']: scriptMain += "save surface 2\n"
                    if centralDict['primarySpecies']['exchange']: scriptMain += "save exchange 2\n"
                    scriptMain += "\nend\n"

                scriptMain += "\nRUN_CELLS\n-cells 1\nEND\n"
            else:
                currentPrimSpecies = commMtrx_primSpecies.copy()
                firstLine, rest = scriptMain.split("\n", 1)
                scriptMain = firstLine + bait + rest

            if centralDict['kinetics'] and centralDict['dtStep'] > 0:
                kinetics = True
                scriptMain += "\nKINETICS 1\n"
                scriptMain += f"{centralDict['kinetics']}\n"
                for k in centralDict['kineticsSpecies']:
                    scriptMain += f'\n-m0 {commMtrx_primSpecies.loc[index,k]}'
                scriptMain += f'\n-step {centralDict["dtStep"]} {centralDict["timeUnit"]}\n'
                
                if centralDict['step_divide']:
                    scriptMain += f"-step_divide {centralDict['step_divide']}\n"

            scriptMain += "\nSOLUTION_MODIFY 1\n"
     
            if 'pH' in centralDict['systemSpeciation'] and 'pe' in centralDict['systemSpeciation']:
                scriptMain +=f'''
pH {commMtrxPart.loc[index,'pH']}
pe {commMtrxPart.loc[index,'pe']}\n'''

            scriptMain += f"-total_h {currentPrimSpecies.loc[index,'H2O']*2 + currentPrimSpecies.loc[index,'H'] }\n"
            scriptMain += f"-total_o {currentPrimSpecies.loc[index,'H2O'] + currentPrimSpecies.loc[index,'O'] }\n"
            
            if 'chargebalance' in centralDict['systemSpeciation']:
                scriptMain += f"-cb {commMtrxPart.loc[index,'chargebalance']}\n-totals\n"
            else:
                scriptMain += "-cb 0\n-totals\n"
                
            for species in centralDict['primarySpecies']['solution']:
                if currentPrimSpecies.loc[index,species] > centralDict['cutoffs']['solution'] and species not in (centralDict['primarySpecies']['primarySpeciesPhantom'] + ['H','O','H2O']) :
                    scriptMain += f"\t{species} {currentPrimSpecies.loc[index,species]}\n"

            if centralDict['preliminarEquilibrium']:
                scriptMain += "\n"
                if centralDict['primarySpecies']['phases']: scriptMain += "use equilibrium_phases 2\n"
                if centralDict['primarySpecies']['surface']: scriptMain += "use surface 2\n"
                if centralDict['primarySpecies']['exchange']: scriptMain += "use exchange 2\n"
                if kinetics: scriptMain += "use kinetics 1\n"
            scriptMain +="\nRUN_CELLS\n-cells 1\n"

        else:
            scriptMain +=f"""\nSOLUTION 1 #node n°{index}
-units mol/kgw
-water {water}
-temp {centralDict['tempDefault']}\n"""
            if centralDict['densityPhreeqC']: scriptMain += f"-density {centralDict['densityPhreeqC']}\n"

            if centralDict['preliminarEquilibrium']:       
                if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation'):
                    if '-la("e-")' in centralDict['crossDependencies']['speciation']['input']['inputPhreeqCCmd']:
                        pe = cmdToNameInput['-la("e-")']
                        scriptMain += f"\tpe {beforeTrsptMtrx.loc[index,pe]}\n"
                    if '-la("H+")' in centralDict['crossDependencies']['speciation']['input']['inputPhreeqCCmd']:
                        ph = cmdToNameInput['-la("H+")']
                        scriptMain += f"\tpH {beforeTrsptMtrx.loc[index,ph]}\n"

            else:
                if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation'):
                    if '-la("e-")' in centralDict['crossDependencies']['speciation']['input']['inputPhreeqCCmd']:
                        pe = cmdToNameInput['-la("e-")']
                        scriptMain += f"\tpe {commMtrxPart.loc[index,pe]}\n"
                    if '-la("H+")' in centralDict['crossDependencies']['speciation']['input']['inputPhreeqCCmd']:
                        ph = cmdToNameInput['-la("H+")']
                        scriptMain += f"\tpH {commMtrxPart.loc[index,ph]}\n"

            if speciationCharge:
                if spcChargeGeom and spcChargeGeom[0][0] in phListUser : scriptMain += "\tcharge"
                elif spcChargeDefaut and spcChargeDefaut[0] in phListUser : scriptMain += "\tcharge"
            if centralDict["acidicTrspt"] and AciditeDiff.loc[index,'H+'] > 0: scriptMain += f"\tSimilication {AciditeDiff.loc[index,'H+']}\n"
            if centralDict["acidicTrspt"] and AciditeDiff.loc[index,'OH-'] > 0 : scriptMain += f"\tSimilianion {AciditeDiff.loc[index,'OH-']}\n"                            
            scriptMain += "\n"
            for species in centralDict['primarySpecies']['solution']:
                if commMtrx_primSpecies.loc[index,species] > centralDict['cutoffs']['solution'] and species not in (centralDict['primarySpecies']['primarySpeciesPhantom'] + ['H','O','H2O']):
                    scriptMain += f"\t{species} {commMtrx_primSpecies.loc[index,species]}"
                    if speciationCharge:
                        if spcChargeGeom and spcChargeGeom[0][0] == species : scriptMain += "\tcharge\n"
                        elif spcChargeDefaut and spcChargeDefaut[0] == species : scriptMain += "\tcharge\n"
                        else: scriptMain += "\n"
                    else: scriptMain += "\n"
                    
            if centralDict['supplementarySolution'] : scriptMain += centralDict['supplementarySolution'] + '\n'
            if centralDict['preliminarEquilibrium']:
                if centralDict['primarySpecies']['phases']: scriptMain += "save equilibrium_phases 2\n"
                if centralDict['primarySpecies']['surface']: scriptMain += "save surface 2\n"
                if centralDict['primarySpecies']['exchange']: scriptMain += "save exchange 2\n"
                
                
                scriptMain += "end\nSOLUTION 2\n-units mol/kgw\n"
                if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation'):
                    if '-la("H+")' in centralDict['crossDependencies']['speciation']['input']['inputPhreeqCCmd']:
                        ph = cmdToNameInput['-la("H+")']
                        scriptMain += f"\tpH {commMtrxPart.loc[index,ph]}\n"
                    if '-la("e-")' in centralDict['crossDependencies']['speciation']['input']['inputPhreeqCCmd']:
                        pe = cmdToNameInput['-la("e-")']
                        scriptMain += f"\tpe {commMtrxPart.loc[index,pe]}\n"
                
                if speciationCharge:
                    if spcChargeGeom and spcChargeGeom[0][0] in phListUser : scriptMain += "\tcharge"
                    elif spcChargeDefaut and spcChargeDefaut[0] in phListUser : scriptMain += "\tcharge"
                scriptMain += '\n'
                for species in centralDict['primarySpecies']['solution']:
                    if currentPrimSpecies.loc[index,species] > centralDict['cutoffs']['solution'] and species not in (centralDict['primarySpecies']['primarySpeciesPhantom'] + ['H','O','H2O']):
                        scriptMain += f"\t{species} {currentPrimSpecies.loc[index,species]}"
                        if speciationCharge:
                            if spcChargeGeom and spcChargeGeom[0][0] == species : scriptMain += "\tcharge\n"
                            elif spcChargeDefaut and spcChargeDefaut[0] == species : scriptMain += "\tcharge\n"
                            else: scriptMain += "\n"
                        else: scriptMain += "\n"
                
                if centralDict['primarySpecies']['phases']: scriptMain += "use equilibrium_phases 2\n" 
                if centralDict['primarySpecies']['surface']: scriptMain += "use surface 2\n" 
                if centralDict['primarySpecies']['exchange']: scriptMain += "use exchange 2\n" 
                if kinetics: scriptMain += "use kinetics 1\n" 
        
        if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation') :
            punchNames = list(centralDict['crossDependencies']['speciation']['output']['outputName'])
            punchCmds = list(centralDict['crossDependencies']['speciation']['output']['outputPhreeqCCmd'])
            if mctTotals:
                punchNames += ['H(mol/kgw)', 'O(mol/kgw)']
                punchCmds += ['TOT("H")', 'TOT("O")']
            scriptMain += "\nUSER_PUNCH\n\t-headings"
            for val in punchNames:
                scriptMain += f"\t{val}"
            scriptMain += '\n'
            for it, val in enumerate(punchCmds, start = 1):
                scriptMain += f'\t{it} PUNCH {val} \n'

        scriptMain += "\nSELECTED_OUTPUT\n-reset false\n"
        molalities = [espece for espece in centralDict['systemSpecies']
                      if espece not in (['pH', 'pe', 'Potential'] + (centralDict['crossDependencies']['speciation']['total']['totalPhreeqCCmd'] if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation') else []))
                      and espece not in centralDict['primarySpecies']['phases']]
        if centralDict.get('molesStorage') and not centralDict['userVarBool'].get('water'):
            scriptMain += "-water true\n"
        if mctTotals:
            scriptMain += "-totals" + "".join(f"\t{t}" for t in mctTotals if t not in ('H', 'O'))
            molalities = [espece for espece in molalities if espece not in mctTotals]
            if molalities: scriptMain += "\n-molalities" + "".join(f"\t{espece}" for espece in molalities)
        else:
            scriptMain += "-molalities" + "".join(f"\t{espece}" for espece in molalities)

        if centralDict['primarySpecies']['phases']:
            scriptMain += "\n-equilibrium_phases"
            for phase in centralDict['primarySpecies']['phases']: scriptMain +=f"\t{phase}" 
        scriptMain += "\n"
        for val in centralDict['userVarBool'].keys():
            if centralDict['userVarBool'][val]:
                scriptMain += f"-{val} True\n"
        for val in centralDict['userVarList'].keys():
            if centralDict['userVarList'][val]:
                scriptMain += f"\n-{val}"
                for lst in centralDict['userVarList'][val]:
                    scriptMain += f"\t{lst}"
    
        scriptMain += "\nEND\n"
        return scriptMain

    warnings = 0
    scriptWarnings =""


    excluded = (
    centralDict['crossDependencies']['speciation']['total']['totalName']
    if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation') else [])
    

    if not beforeTrsptMtrx.empty :

        dico = {spc: [0]*len(beforeTrsptMtrx) for spc in centralDict['primarySpecies']['total']}

        ligne = 0
        for _, row in beforeTrsptMtrx.iterrows():
            for comp, conc in row.items() :
                if comp not in excluded:
                    for prim in centralDict['primToSecSpecies'][comp]:
                        if prim not in centralDict['primarySpecies']['primarySpeciesPhantom']:
                            try:
                                dico[prim][ligne] += centralDict['primToSecSpecies'][comp][prim] * conc
                            except: 
                                print(prim,comp,centralDict['primToSecSpecies'][comp])
                                sys.exit()
            ligne += 1        
        
        commMtrx_primSpecies = pd.DataFrame(dico, index = beforeTrsptMtrx.index)

        dico = {spc: [0]*len(commMtrxPart) for spc in centralDict['primarySpecies']['total']}
        ligne = 0
        for _, row in commMtrxPart.iterrows():
            for comp, conc in row.items():
                if comp not in excluded:
                    for prim in centralDict['primToSecSpecies'][comp] :
                        if prim not in centralDict['primarySpecies']['primarySpeciesPhantom']:
                            try:
                                dico[prim][ligne] += centralDict['primToSecSpecies'][comp][prim] * conc
                            except: 
                                print(prim,comp,centralDict['primToSecSpecies'][comp])
                                sys.exit()
            ligne += 1
        currentPrimSpecies = pd.DataFrame(dico, index = commMtrxPart.index)

        
    else:
        currentPrimSpecies = pd.DataFrame()
        ligne = 0
        dico = {spc: [0]*len(commMtrxPart) for spc in centralDict['primarySpecies']['total']}

        for _, row in commMtrxPart.iterrows():
            for comp, conc in row.items():
                if comp not in excluded:
                    for prim in centralDict['primToSecSpecies'][comp]:
                        try:
                            dico[prim][ligne] += centralDict['primToSecSpecies'][comp][prim] * conc
                        except: 
                            print(prim,comp,centralDict['primToSecSpecies'][comp],centralDict['primarySpecies']['total'])
                            sys.exit()
            ligne += 1
        
        commMtrx_primSpecies = pd.DataFrame(dico, index = commMtrxPart.index)

    pd.set_option('display.max_columns', None)
    if centralDict['acidicTrspt']:

        if centralDict['AcidicEcho'].empty:
            centralDict['AcidicEcho'] = pd.DataFrame(columns = ['H+','OH-','pH'], index = commMtrxPart.index)
            centralDict['AcidicEcho']['OH-'] = 0
            centralDict['AcidicEcho']['H+'] = 0
            centralDict['AcidicEcho']['pH'] = -np.log10(commMtrxPart['H+'])
            AciditeDiff = centralDict['AcidicEcho'].copy()
        else:
            centralDict['AcidicEcho'] = centralDict['AcidicEcho'].loc[commMtrxPart.index]
            AciditeDiff = commMtrxPart[['H+', 'OH-']] - centralDict['AcidicEcho'][['H+', 'OH-']]
        
        AciditeDiff.index = commMtrxPart.index
        
        mask = (AciditeDiff["H+"] > 0) & (AciditeDiff["OH-"] > 0)
        
        min_vals = AciditeDiff.loc[mask, ["H+", "OH-"]].min(axis=1)
        max_vals = AciditeDiff.loc[mask, ["H+", "OH-"]].max(axis=1)
        
        AciditeDiff.loc[mask, "H+"] = np.where(
            AciditeDiff.loc[mask, "H+"] == min_vals, 0, max_vals - min_vals
        )
        
        AciditeDiff.loc[mask, "OH-"] = np.where(
            AciditeDiff.loc[mask, "OH-"] == min_vals, 0, max_vals - min_vals
        )

    calcTime = 0
    abort = False
    sortiePhreeqCtotal = pd.DataFrame()
    resultats = []
    for index in commMtrxPart.index:
        if abort : break
        if centralDict['maillesChargeGeom']: specieChargeMailleListe = speciesToNode(index, centralDict['maillesChargeGeom'], centralDict['speciesChargeGeometry'])
        else: specieChargeMailleListe =  None
        
        try:
            ref = time.perf_counter()
            phreeqc.run_string(nodeSpeciation(commMtrxPart, commMtrx_primSpecies, centralDict['speciesCharge'],specieChargeMailleListe, centralDict['speciationCharge'],centralDict['solMod'],currentPrimSpecies))
            calcTime += time.perf_counter() - ref
        except Exception as e:  
            if centralDict['speciationCharge']:
                start = 1
                if centralDict['speciesChargeGeometry']:
                    scriptWarnings +=f"PhreeqC : node n°{index}, time step n°{centralDict['lStep']+1}, t={centralDict['tStep']}{centralDict['timeUnit']}, PID={os.getpid()} : assuming {specieChargeMailleListe[0][0]} as counter-charge species :\n {e}\n"
                elif centralDict['speciesCharge']:
                    scriptWarnings +=f"PhreeqC : node n°{index}, time step n°{centralDict['lStep']+1}, t={centralDict['tStep']}{centralDict['timeUnit']}, PID={os.getpid()} : assuming {centralDict['speciesCharge'][0]} as counter-charge species :\n {e}\n"
            else:
                start = 0
                scriptWarnings +=f"PhreeqC : node n°{index}, time step n°{centralDict['lStep']+1}, t={centralDict['tStep']}{centralDict['timeUnit']}, PID={os.getpid()} : assuming no counter-charge species :\n {e}\n"

            if centralDict['speciationCharge'] and centralDict['speciesChargeGeometry']:
                for k,spc in enumerate(specieChargeMailleListe[0][start:], start=start): 
                    try:
                        warnings +=1
                        scriptWarnings += f"PhreeqC : node n°{index}, time step n°{centralDict['lStep']+1}, t={centralDict['tStep']}{centralDict['timeUnit']}, PID={os.getpid()} : testing {spc} as geometrical dependent counter-balance species ...\n"
                        ref = time.perf_counter()
                        phreeqc.run_string(nodeSpeciation(commMtrxPart, commMtrx_primSpecies, None, spc, True, centralDict['solMod'],currentPrimSpecies))
                        calcTime += time.perf_counter() - ref
                        scriptWarnings +=  f"PhreeqC : node n°{index}, time step n°{centralDict['lStep']+1}, t={centralDict['tStep']}{centralDict['timeUnit']}, PID={os.getpid()} : success !\n"
                        break
                    except Exception as r:
                        scriptWarnings += f"PhreeqC : node n°{index}, time step n°{centralDict['lStep']+1}, t={centralDict['tStep']}{centralDict['timeUnit']}, PID={os.getpid()} : {r}\n"
                    if spc == specieChargeMailleListe[0][-1]:
                        scriptWarnings += f"PhreeqC : node n°{index}, time step n°{centralDict['lStep']+1}, t={centralDict['tStep']}{centralDict['timeUnit']}, PID={os.getpid()} : Fatal PhreeqC error. Aborted PhreeqC batch :\n"
                        scriptWarnings += nodeSpeciation(commMtrxPart, commMtrx_primSpecies, None, spc, True, centralDict['solMod'],currentPrimSpecies)
                        print('\nSpeciation batch aborted. See warning.log file.', end = '')
                        abort = True
                        
            elif centralDict['speciationCharge'] and centralDict['speciesCharge']:
                for k,spc in enumerate(centralDict['speciesCharge'][start:], start=start):
                    try:
                        warnings +=1
                        scriptWarnings +=f"PhreeqC : node n°{index}, time step n°{centralDict['lStep']+1}, t={centralDict['tStep']}{centralDict['timeUnit']}, PID={os.getpid()} : testing {spc} as counter-balance species ...\n"
                        ref = time.perf_counter()
                        phreeqc.run_string(nodeSpeciation(commMtrxPart, commMtrx_primSpecies, spc, None, True,centralDict['solMod'],currentPrimSpecies))
                        calcTime += time.perf_counter() - ref
                        scriptWarnings +=f"PhreeqC : node n°{index}, time step n°{centralDict['lStep']+1}, t={centralDict['tStep']}{centralDict['timeUnit']}, PID={os.getpid()} : success !\n"
                        break
                    except Exception as r: 
                        scriptWarnings +=f"PhreeqC : node n°{index}, time step n°{centralDict['lStep']+1}, t={centralDict['tStep']}{centralDict['timeUnit']}, PID={os.getpid()} : {r}\n"
                        
                    if spc == centralDict['speciesCharge'][-1]:
                        scriptWarnings +=f"PhreeqC : node n°{index}, time step n°{centralDict['lStep']+1}, t={centralDict['tStep']}{centralDict['timeUnit']}, PID={os.getpid()} : Fatal PhreeqC error. Aborted PhreeqC batch :\n"
                        scriptWarnings += nodeSpeciation(commMtrxPart, commMtrx_primSpecies, None, spc, True, centralDict['solMod'],currentPrimSpecies)
                        print('\nSpeciation batch aborted. See warning.log file.', end = '')
                        abort = True
                                
            else:
                scriptWarnings +=f"PhreeqC : node n°{index}, time step n°{centralDict['lStep']+1}, t={centralDict['tStep']}{centralDict['timeUnit']}, PID={os.getpid()} : Fatal PhreeqC error. Aborted PhreeqC batch :\n"
                scriptWarnings += nodeSpeciation(commMtrxPart, commMtrx_primSpecies, centralDict['speciesCharge'][0],specieChargeMailleListe, centralDict['speciationCharge'],centralDict['solMod'],currentPrimSpecies)
                print('\nSpeciation batch aborted. See warning.log file.', end = '')
                print('ici')
                abort = True
        
        if phreeqc.get_selected_output_array() and not abort:
            resultats.append(pd.DataFrame([phreeqc.get_selected_output_array()[-1]], columns=phreeqc.get_selected_output_array()[0]))
        elif not phreeqc.get_selected_output_array():
            warnings += 1
            scriptWarnings +=f"PhreeqC : node n°{index}, time step n°{centralDict['lStep']+1}, t={centralDict['tStep']}{centralDict['timeUnit']}, PID={os.getpid()} : no PhreeqC ouput ...\n"
            scriptWarnings +=f"PhreeqC : node n°{index}, time step n°{centralDict['lStep']+1}, t={centralDict['tStep']}{centralDict['timeUnit']}, PID={os.getpid()} : aborted PhreeqC batch :\n"
            scriptWarnings += nodeSpeciation(commMtrxPart, commMtrx_primSpecies, None, None, centralDict['speciationCharge'], centralDict['solMod'],currentPrimSpecies)
            abort = True

    if abort:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),warnings, scriptWarnings, abort,calcTime , initWorker
    else:
        sortiePhreeqCtotal = pd.concat(resultats, ignore_index=True)
        sortiePhreeqCtotal.index = commMtrxPart.index
        
        colonnesSpeciation  = []

        if mctTotals: colonnesSpeciation += [spc if spc in centralDict['primarySpecies']['phases'] else f"{spc}(mol/kgw)" if spc in mctTotals else f"m_{spc}(mol/kgw)" for spc in centralDict['systemSpecies'] if spc not in excluded]
        elif centralDict['primarySpecies']['phases']: colonnesSpeciation += [spc if spc in (centralDict['primarySpecies']['phases']) else f"m_{spc}(mol/kgw)" for spc in centralDict['systemSpecies'] if spc not in excluded ]
        else: colonnesSpeciation += [f"m_{spc}(mol/kgw)" for spc in centralDict['systemSpecies'] if spc not in excluded]
        
            
        
        
        crossSpc = centralDict['crossDependencies'].get('speciation') if centralDict['crossDependencies'] else None
        crossOut = crossSpc['output']['outputName'] if crossSpc else []
        crossIn = crossSpc['input']['inputName'] if crossSpc else []
        crossTrspt = (centralDict['crossDependencies']['transport']['total']
                      if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('transport') else [])
        waterKg = sortiePhreeqCtotal['mass_H2O'] if centralDict.get('molesStorage') else None
        speciesCols = iter(colonnesSpeciation)
        columns = {}
        for c in centralDict['systemSpeciation']:
            if c in crossOut:
                columns[c] = sortiePhreeqCtotal[c]
            elif c in crossIn or c in crossTrspt:
                columns[c] = centralDict['commMtrx'][c]
            else:
                col = next(speciesCols)
                columns[c] = sortiePhreeqCtotal[col] * waterKg if waterKg is not None and col.endswith('(mol/kgw)') else sortiePhreeqCtotal[col]
        if next(speciesCols, None) is not None:
            raise ValueError("PhreeqC output : more species columns than systemSpeciation entries (check systemSpecies)")
        commMtrxSpct = pd.DataFrame(columns, index=sortiePhreeqCtotal.index)
        
        
        

        
        if centralDict['acidicTrspt']:
            AcidicEcho = sortiePhreeqCtotal[['m_H+(mol/kgw)', 'm_OH-(mol/kgw)','pH']].copy()
            AcidicEcho.columns = ['H+','OH-','pH']
            AcidicEcho.index = commMtrxPart.index
            return commMtrxSpct, AcidicEcho, commMtrx_primSpecies, sortiePhreeqCtotal, warnings, scriptWarnings, abort,calcTime, initWorker

        else:
            return commMtrxSpct, pd.DataFrame(), commMtrx_primSpecies, sortiePhreeqCtotal, warnings, scriptWarnings, abort,calcTime, initWorker


def multiCompoundSetup(centralDict):
    """
    MultiCompoundTransport = True : commMtrx ne porte plus que des totaux :
        dependances croisees (chargebalance, pH, pe), totaux dissous par etat redox (Ca, Fe(+2), Fe(+3), S(+6), ...)
        pour tous les etats redox de la base des elements presents, H et O (eau comprise : TOT("H"), TOT("O")),
        puis les colonnes immobiles (phases, especes d'echange / de surface, especes cinetiques).
    Unites : moles par maille, rapportees a 1 kg d'eau initial (CI en mol/kgw avec 1 kg d'eau -> memes valeurs).
    Apres la chimie, totaux = TOT(x) x masse d'eau PhreeqC : la masse d'eau suit le bilan de O (hydratation,
    dissolution des hydrates) au lieu d'etre ramenee a 1 kg, tous les totaux sont conserves exactement.
    Seuls les totaux, H, O et chargebalance sont transportes ; pH et pe servent d'estimations a SOLUTION_MODIFY.

    Conversion unique, ici, des conditions initiales et des conditions aux limites (firstBoundary, secondBoundary).
    Chaque colonne est reconnue automatiquement, espece ou total :
      - H et O connus (colonne H2O, ou colonnes H et O) : decomposition lineaire (primToSecSpecies), H2O -> 2 H + O,
        totaux repris tels quels ('Fe' -> etat redox primaire : en reaction, PhreeqC repartit les etats redox a
        partir des bilans H / O) ; H et O sans H2O = totaux eau comprise
      - sinon (totaux analytiques) : solution initiale PhreeqC (pH, pe de la CI, 1 kg d'eau) qui donne les totaux
        par etat redox, H, O et chargebalance
    """
    from . import extractDB
    cd = centralDict

    def fail(message):
        print(f"\nMultiCompoundTransport : {message}")
        sys.exit()

    def warn(message):
        warningManager.warn(f"MultiCompoundTransport : {message}")

    crossSpc = cd['crossDependencies'].get('speciation') if cd['crossDependencies'] else None
    if not crossSpc or 'chargebalance' not in crossSpc['total']['totalName']:
        fail("the charge balance must be coupled : add 'chargebalance' to systemSpeciation, to the initial conditions "
             "and to crossDependencies = {'speciation' : ['chargebalance', 'pH', 'pe']}")
    for key in ('preliminarEquilibrium', 'acidicTrspt', 'NernstPlanck', 'donnan', 'activityGradient'):
        if cd.get(key):
            fail(f"{key} is not available (it needs the species)")
    if not cd['solMod']:
        cd['solMod'] = True
        warn("solMod forced to True (SOLUTION_MODIFY with the totals, -total_h, -total_o and -cb)")

    db = extractDB.extract_master_species(cd['chemPath'])
    masters = db['SOLUTION_MASTER_SPECIES']
    masterOf = dict(zip(masters, db['SOLUTION_MASTER_ELEMENT']))
    redox = {el: states for el, states in extractDB.group_redox_states(masters).items() if el not in ('H', 'O')}
    primaryState = {el: next((s for s in states if masterOf.get(s) == masterOf.get(el)), None) for el, states in redox.items()}
    element = lambda s: s.split('(')[0]
    present = {element(s) for s in cd['primarySpecies']['solution'] if s not in ('H', 'O', 'H2O')}
    totals = list(dict.fromkeys(s for s in masters if element(s) in present and element(s) not in ('H', 'O')
                                and (s in redox[element(s)] if element(s) in redox else s == element(s))))
    totals += ['H', 'O']

    crossNames = list(crossSpc['total']['totalName']) + list(cd['crossDependencies']['transport']['total'])
    surfaceSpecies = [s for s in db['SURFACE_SPECIES'] if s not in db['SOLUTION_MASTER_ELEMENT']]
    immobile = set(db['SURFACE_MASTER_SPECIES'] + surfaceSpecies + db['EXCHANGE_SPECIES'] + db['PHASES']
                   + list(cd['kineticsSpecies'] or []))
    cmdToName = dict(zip(crossSpc['input']['inputPhreeqCCmd'], crossSpc['input']['inputName']))
    pHName, peName = cmdToName.get('-la("H+")'), cmdToName.get('-la("e-")')

    def composition(name):
        if name == 'H2O':
            return {'H': 2, 'O': 1}
        if name in totals:
            return {name: 1}
        if name in redox and primaryState[name]:
            return {primaryState[name]: 1}
        return cd['primToSecSpecies'].get(name)

    def linear(tab):
        out = pd.DataFrame(0.0, index=tab.index, columns=totals)
        touched = set()
        for col in tab.columns:
            comp = composition(col)
            if comp is None:
                fail(f"'{col}' is neither a species of {cd['chemPath'].name} nor a total")
            for prim, coeff in comp.items():
                if prim not in totals:
                    fail(f"'{col}' decomposes into '{prim}', which is not a transported total")
                out[prim] += coeff * tab[col].astype(float)
                touched.add(prim)
        return out[[t for t in totals if t in touched]]

    def initialSolution(tab, pH, pe):
        import phreeqpy.iphreeqc.phreeqc_dll as phreeqc_mod
        solver = phreeqc_mod.IPhreeqc()
        solver.load_database(str(cd['chemPath']))
        punch = ("SELECTED_OUTPUT\n-reset false\n-totals" + "".join(f"\t{t}" for t in totals if t not in ('H', 'O'))
                 + '\nUSER_PUNCH\n-headings\tH\tO\tchargebalance\n10 PUNCH TOT("H"), TOT("O"), CHARGE_BALANCE\nEND\n')
        rows = []
        for k, index in enumerate(tab.index):
            given = {}
            for col in tab.columns:
                if col in masters:
                    comp = {col: 1}
                else:
                    comp = composition(col)
                    if comp is None:
                        fail(f"'{col}' is neither a species of {cd['chemPath'].name} nor a total")
                for prim, coeff in comp.items():
                    if prim not in ('H', 'O'):
                        given[prim] = given.get(prim, 0.0) + coeff * float(tab.at[index, col])
            script = (f"SOLUTION 1\n-units mol/kgw\n-water 1\n-temp {cd['tempDefault']}\n"
                      f"pH {pH[k]}\npe {pe[k]}\n")
            script += "".join(f"\t{name} {value}\n" for name, value in given.items() if value > cd['cutoffs']['solution'])
            try:
                solver.run_string(script + punch)
            except Exception as e:
                fail(f"PhreeqC initial solution failed (node {index}) :\n{e}\n{script}")
            out = solver.get_selected_output_array()
            res = dict(zip(out[0], out[-1]))
            rows.append({**{t: res[f"{t}(mol/kgw)"] for t in totals if t not in ('H', 'O')},
                         'H': res['H'], 'O': res['O'], 'chargebalance': res['chargebalance']})
        return pd.DataFrame(rows, index=tab.index)

    def convert(tab, what):
        cols = [c for c in tab.columns if c not in crossNames and c not in immobile and c not in cd['coord']]
        if 'H2O' in cols or {'H', 'O'} <= set(cols):
            return linear(tab[cols]), "species / totals decomposition"
        if 'H' in cols or 'O' in cols:
            fail(f"{what} : give both H and O totals (water included), or H2O")
        pH = tab[pHName].to_numpy() if pHName in tab else [cd['pHdefault']] * len(tab)
        pe = tab[peName].to_numpy() if peName in tab else [4] * len(tab)
        return initialSolution(tab[cols], pH, pe), "PhreeqC initial solution (pH, pe, totals)"

    ci = cd['commMtrx']
    tot, how = convert(ci, 'initial conditions')
    missing = [t for t in totals if t not in tot.columns]
    for t in missing:
        tot[t] = 0.0
    crossCols = [s for s in cd['systemSpeciation'] if s in crossNames]
    kept = [s for s in cd['systemSpeciation'] if s not in crossNames and s in immobile]
    comm = pd.concat([ci[cd['coord'] + crossCols], tot[totals], ci[kept]], axis=1)
    if 'chargebalance' in tot:
        comm['chargebalance'] = tot['chargebalance']

    speciation = crossCols + totals + kept
    ps = cd['primarySpecies']
    ps['solution'] = totals + ['H2O']
    ps['total'] = ps['solution'] + ps['exchange'] + ps['surface'] + ps['phases']
    cd['primToSecSpecies'].update({t: {t: 1} for t in totals})
    cd.update({
        'commMtrx': comm[cd['coord'] + speciation].copy(),
        'systemSpeciation': speciation,
        'systemSpecies': [s for s in speciation if s not in crossNames],
        'mctTotals': totals,
        'transportedSpecies': totals + ['chargebalance'],
        })
    cd['anythingButSpecies'] = [c for c in cd['commMtrx'].columns if c not in speciation]

    for key in ('firstBoundary', 'secondBoundary'):
        values = cd.get(key)
        if not values:
            continue
        unknown = [k for k in values if k not in crossNames and k not in immobile and composition(k) is None and k not in masters]
        if unknown:
            fail(f"{key} : {unknown} are neither totals nor species of the initial conditions (give the boundary "
                 f"with the same names as the initial conditions, or as totals)")
        tab = pd.DataFrame([{k: float(v) for k, v in values.items()}])
        if not [c for c in tab.columns if c not in crossNames and c not in immobile and c not in cd['coord']]:
            continue
        btot, _ = convert(tab, key)
        converted = {t: float(btot[t].iloc[0]) for t in btot.columns if t in totals}
        if 'chargebalance' in btot:
            converted['chargebalance'] = float(btot['chargebalance'].iloc[0])
        elif 'chargebalance' in values:
            converted['chargebalance'] = float(values['chargebalance'])
        else:
            warn(f"{key} has no chargebalance : Neumann condition for the charge balance on this side")
        cd[key] = converted

    for key in ('especeDiffCoeff', 'especePorosity'):
        stale = [s for s in (cd.get(key) or {}) if s not in cd['transportedSpecies']]
        if stale:
            warn(f"{key} : {stale} are not transported totals (ignored)")

    print(f"MultiCompoundTransport : initial conditions -> totals ({how}), "
          f"transported : {', '.join(totals)}, chargebalance")
    if missing:
        warn(f"totals not given in the initial conditions, set to 0 : {', '.join(missing)}")
    return centralDict


def spct(centralDict):
    startPhreeqC = time.time()
    print("PhreeqC", end=" ", flush=True)
    
    if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation'):
        phreeqcInput = [s for s in centralDict['systemSpeciation'] if s not in centralDict['crossDependencies']['transport']['total']]
    else: 
        phreeqcInput = centralDict['systemSpeciation']
    
    if centralDict['PIDnbr'] > 1:
        
        chunk_size = int(np.ceil(len(centralDict['commMtrx']) / centralDict['PIDnbr']))
        commMtrxSplit = [centralDict['commMtrx'][phreeqcInput].iloc[i:i + chunk_size] for i in range(0, len(centralDict['commMtrx'][phreeqcInput]), chunk_size)]


        if centralDict['preliminarEquilibrium']:
            
            assert len(centralDict['commMtrx'][phreeqcInput]) == len(centralDict["beforeTrsptMtrx"][phreeqcInput])

            beforeTrsptMtrx = centralDict["beforeTrsptMtrx"][phreeqcInput].copy()
            commMtrxSplitbeforeTrspt = [beforeTrsptMtrx[phreeqcInput].iloc[i:i + chunk_size] for i in range(0, len(centralDict['commMtrx'][phreeqcInput]), chunk_size)]
        else: 
            commMtrxSplitbeforeTrspt = [pd.DataFrame()] * len(commMtrxSplit)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=centralDict['PIDnbr']) as executor:
            futures = {}
            for i,chunk in enumerate(commMtrxSplit):

                
                
                fut = warningManager.submit(executor, speciationPhreeqC, centralDict, chunk, commMtrxSplitbeforeTrspt[i])
                futures[fut] = i
    
        results = [None] * len(commMtrxSplit)
        for future in as_completed(futures):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception:
                import traceback
                traceback.print_exc()
                raise
    
        
        df1, df2, df3, df4, intgr, strg, Bool, calc, initWorker = zip(*results)

        commMtrxSpct = pd.concat(df1, ignore_index=True)

        AcidicEcho = pd.concat(df2, ignore_index=True)
        commMtrx_primSpecies = pd.concat(df3, ignore_index=True)
        sortiePhreeqCtotal = pd.concat(df4, ignore_index=True)
        totalWarnings = sum(intgr)
        calcPrcsTime = sum(calc)
        calcWallClock = max(calc)
        abort = any(Bool)
        init = max(initWorker)
        totalScriptWarnings = "".join(strg)

    else:
        if centralDict['preliminarEquilibrium']:
            beforeTrspt = centralDict["beforeTrsptMtrx"][phreeqcInput].copy()
        else:
            beforeTrspt = pd.DataFrame()
        
        commMtrxSpct, AcidicEcho, commMtrx_primSpecies, sortiePhreeqCtotal, totalWarnings, totalScriptWarnings, abort, calcWallClock, init = speciationPhreeqC(centralDict,
                                                        centralDict['commMtrx'][phreeqcInput],beforeTrspt )
        calcPrcsTime = calcWallClock

    if totalWarnings:
        with open("warning.log", "a") as warningLog:
            warningLog.write(f"PhreeqC, time = {centralDict['tStep']}{centralDict['timeUnit']}, time step n°{centralDict['lStep']+1} : the {totalWarnings} following warnings occured ...\n")
            warningLog.write(f"{totalScriptWarnings}\n")
            centralDict['warningNbr'] += totalWarnings
    if abort:
        print('\nFatal PhreeqC error. Aborting run.', end = '')
        sys.exit()

    
    commMtrxSpct = pd.concat([centralDict['commMtrx'][centralDict['coord']],commMtrxSpct], axis=1)
    sortiePhreeqCtotal = pd.concat([centralDict['commMtrx'][centralDict['coord']],sortiePhreeqCtotal], axis=1)
    commMtrx_primSpecies = pd.concat([centralDict['commMtrx'][centralDict['coord']],commMtrx_primSpecies], axis=1)
    
    commMtrxSpct = commMtrxSpct[centralDict['commMtrx'].columns]
    if centralDict.get('molesStorage'):
        centralDict['cellWaterMass'] = sortiePhreeqCtotal['mass_H2O'].to_numpy(dtype=float)

    centralDict.update({
        "commMtrx": commMtrxSpct,
        "AcidicEcho": AcidicEcho,
        "PhreeqCCalcTime_WallClock": centralDict['PhreeqCCalcTime_WallClock']  + calcWallClock,
        "PhreeqCCalcTime_ProcessorTime": centralDict["PhreeqCCalcTime_ProcessorTime"] + calcPrcsTime,
        "PhreeqCInterfTime_WallClock": centralDict['PhreeqCInterfTime_WallClock'] + time.time() - startPhreeqC - calcWallClock - init,
        "PhreeqCInitTime" : centralDict["PhreeqCInitTime"] + init,
        "PhreeqCTotalTime" : centralDict['PhreeqCTotalTime'] + time.time() - startPhreeqC,
        })
    
    
    if centralDict['lStep'] == len(centralDict['dtpycte'])-1:
        resetPhreeqC()
    if outputManager.wanted(centralDict, 'speciation'):
        commMtrx_primSpecies.to_csv(outputManager.filePath(centralDict, 'primarySpecies', 'PrimarySpecies'), index=False, header=True, sep='\t')
        sortiePhreeqCtotal.to_csv(outputManager.filePath(centralDict, 'speciation', 'PhreeqC'), index=False, header=True, sep='\t')

    print(f"({writeTime((time.time() - startPhreeqC))})") 
    return centralDict