import numpy as np
import time 
import pandas as pd
import sys 
import importlib.util
import os
import concurrent.futures
from concurrent.futures import as_completed

from pathlib import Path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

with open("store.txt", 'r', encoding='utf-8') as fichier:
    inputPath =  fichier.readline().strip() 
    nameInput =  fichier.readline().strip()

module_name = os.path.splitext(nameInput)[0]
file_path = os.path.join(inputPath, nameInput)
spec = importlib.util.spec_from_file_location(module_name, file_path)

PICCTS_input = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PICCTS_input)

def writeTime(tps, arr=2):
    if tps >= 3600 * 24:
        return f"{tps / (3600 * 24):.{arr}f} d"
    elif tps >= 3600:
        return f"{tps / 3600:.{arr}f} h"
    elif tps >= 60:
        return f"{tps / 60:.{arr}f} min"
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
    import phreeqpy.iphreeqc.phreeqc_dll as phreeqc_mod
    ref = time.perf_counter()
    phreeqc = phreeqc_mod.IPhreeqc()
    phreeqc.load_database(str(centralDict['chemPath']))
    initWorker = time.perf_counter() - ref
    
    def nodeSpeciation(commMtrxPart,commMtrx_primSpecies,spcChargeDefaut,spcChargeGeom,speciationCharge,solMod,currentPrimSpecies): # complexationSurface, echangeIon, phases,primarySpeciesSurf,primarySpeciesAq, primarySpeciesAq, primarySpeciesPha en variables 'globales'

        phListUser = ['ph','pH','H']
        scriptMain =f"node n°{index}\n"
        if centralDict['water']:
            for key in centralDict['water']:
                if index <= key:
                    water = centralDict['water'][key]
                    break
        else: water = 1
        
            
        bait = f"""
    SOLUTION 1
    -units mmol/kgw
    # Na 1
    # Cl 1
    -water {water}
    END
    
    RUN_CELLS
    -cells 1
    END               
        """
        if centralDict['kinetics'] and not centralDict['solMod'] :
            scriptMain += "\nKINETICS 1\n"
            for p in centralDict['kinetics']:
                if commMtrx_primSpecies.loc[index,p] > centralDict['cutoffs']['phases']:
                    scriptMain += f"{p}\n-m0 {commMtrx_primSpecies.loc[index,p]}\n"
                
            scriptMain += f'\n-step {centralDict["dtStep"]} {centralDict["timeUnit"]}\n' #kinetics
            if centralDict['step_divide']:
                scriptMain += f"-step_divide {centralDict['step_divide']}\n"
                
        if centralDict['fixpH'] or centralDict['primarySpecies']['phases']:
            scriptMain +="\nEQUILIBRIUM_PHASES 1\n"
            if centralDict['fixpH']:
                scriptMain += f"Fix_ph {centralDict['fixpH']}\n"
            if centralDict['primarySpecies']['phases']: 
                for phase in centralDict['primarySpecies']['phases']:
                    if commMtrx_primSpecies.loc[index,phase] > centralDict['cutoffs']['phases']:
                        scriptMain +=f"\t{phase} {centralDict['SI'][phase]} {commMtrx_primSpecies.loc[index,phase]}  {centralDict['mineralReversibility'][phase]}\n"

        if centralDict['primarySpecies']['surface']:
            scriptMain +="\nSURFACE 1\n"
            if centralDict['preliminarEquilibrium']: scriptMain += "equilibrate with solution 1\n"
            edl = False
            for surface in centralDict['primarySpecies']['surface']:
                if commMtrx_primSpecies.loc[index,surface] > centralDict['cutoffs']['surface'] :
                    scriptMain +=f"\t{surface} {commMtrx_primSpecies.loc[index,surface]} 100 1 \n"
                    edl = True # prevent edl equilibrium when no surface site is present
            if edl and centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation') and "DebyeLength" in centralDict['crossDependencies']['speciation']['input']:
                scriptMain +=f"-diffuse_layer {commMtrxPart.loc[index,'DebyeLength']}\n"
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

        if solMod: # transport respect electroneutrality
            if centralDict['preliminarEquilibrium']:
                
                scriptMain +=f"\nSOLUTION 1 #node n°{index}\n-units mol/kgw\ntemp {centralDict['tempDefault']}\n"
                if 'H2O' in commMtrx_primSpecies:
                    scriptMain += f"-water {beforeTrsptMtrx.loc[index,'H2O']*18.015/1000}\n"
                else: 
                    scriptMain += "-water 1\n"
                
                if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation'):
                    if '-la("e-")' in centralDict['crossDependencies']['speciation']['input']:
                        pe = '-la("e-")'
                        scriptMain += f"\tpe {beforeTrsptMtrx.loc[index,pe]}\n"
                    if '-la("H+")' in centralDict['crossDependencies']['speciation']['input']:
                        ph = '-la("H+")'
                        scriptMain += f"\tpH {beforeTrsptMtrx.loc[index,ph]}\n"


    
                elif centralDict["acidicTrspt"]:
                    scriptMain += f"\tpH {AcidicEcho.loc[index,'pH']}"
                else:
                    scriptMain += f"\tpH {centralDict['pHdefault']}" 

                if centralDict["acidicTrspt"] and AciditeDiff.loc[index,'H+'] > 0: scriptMain += f"\tSimilication {AciditeDiff.loc[index,'H+']}\n"
                if centralDict["acidicTrspt"] and AciditeDiff.loc[index,'OH-'] > 0 : scriptMain += f"\tSimilianion {AciditeDiff.loc[index,'OH-']}\n"                            
                scriptMain += "\n"
                for species in centralDict['primarySpecies']['solution']:
                    if commMtrx_primSpecies.loc[index,species] > centralDict['cutoffs']['solution'] and species not in (centralDict['primarySpecies']['primarySpeciesPhantom'] + ['H','O','H2O']) :
                        scriptMain += f"\n\t{species} {commMtrx_primSpecies.loc[index,species]}"

                scriptMain += "\n"
                        
                if centralDict['supplementarySolution'] : scriptMain += centralDict['supplementarySolution'] + '\n'
                if centralDict['preliminarEquilibrium']:
                    if centralDict['primarySpecies']['phases']: scriptMain += "save equilibrium_phases 2\n"
                    if centralDict['primarySpecies']['surface']: scriptMain += "save surface 2\n"
                    if centralDict['primarySpecies']['exchange']: scriptMain += "save exchange 2\n"
                    scriptMain += "\nend\n"

                scriptMain += "\nRUN_CELLS\n-cells 1\nEND\n"
            else:
                scriptMain += bait

            if centralDict['kinetics'] :
                scriptMain += "\nKINETICS 1\n"
                for p in centralDict['kinetics']:
                    if commMtrx_primSpecies.loc[index,p] > centralDict['cutoffs']['phases']:
                        scriptMain += f"{p}\n-m0 {currentPrimSpecies.loc[index,p]}"
                    
                scriptMain += f'\n-step {centralDict["dtStep"]} {centralDict["timeUnit"]}\n' #kinetics
                if centralDict['step_divide']:
                    scriptMain += f"-step_divide {centralDict['step_divide']}\n"

            scriptMain += "\nSOLUTION_MODIFY 1\n"
     
            if 'pH' in centralDict['systemSpeciation'] and 'pe' in centralDict['systemSpeciation']:
                scriptMain +=f'''
pH {commMtrxPart.loc[index,'pH']}
pe {commMtrxPart.loc[index,'pe']}\n'''
            
            scriptMain += f"-total_h {currentPrimSpecies.loc[index,'H2O']*2 + currentPrimSpecies.loc[index,'H'] }\n"
            scriptMain += f"-total_o {currentPrimSpecies.loc[index,'H2O'] + currentPrimSpecies.loc[index,'O'] }\n"
            
            scriptMain += "-cb 0\n-totals\n"
            for species in centralDict['primarySpecies']['solution']:
                if currentPrimSpecies.loc[index,species] > centralDict['cutoffs']['solution'] and species not in (centralDict['primarySpecies']['primarySpeciesPhantom'] + ['H','O','H2O']) :
                    scriptMain += f"\t{species} {currentPrimSpecies.loc[index,'H2O']*18.015/1000 * currentPrimSpecies.loc[index,species]}\n"
            if centralDict['preliminarEquilibrium']:
                scriptMain += "\n"
                if centralDict['primarySpecies']['phases']: scriptMain += "use equilibrium_phases 2\n"
                if centralDict['primarySpecies']['surface']: scriptMain += "use surface 2\n"
                if centralDict['primarySpecies']['exchange']: scriptMain += "use exchange 2\n"
                if centralDict['kinetics']: scriptMain += "use kinetics 1\n"
            scriptMain +="\nRUN_CELLS\n-cells 1\n"

        else:
            scriptMain +=f"""\nSOLUTION 1 #node n°{index}
-units mol/kgw
-water {water}
temp {centralDict['tempDefault']}\n"""
            if centralDict['preliminarEquilibrium']:       
                if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation'):
                    if '-la("e-")' in centralDict['crossDependencies']['speciation']['input']:
                        pe = '-la("e-")'
                        scriptMain += f"\tpe {beforeTrsptMtrx.loc[index,pe]}\n"
                    if '-la("H+")' in centralDict['crossDependencies']['speciation']['input']:
                        ph = '-la("H+")'
                        scriptMain += f"\tpH {beforeTrsptMtrx.loc[index,ph]}\n"

            else:
                if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation'):
                    if '-la("e-")' in centralDict['crossDependencies']['speciation']['input']:
                        pe = '-la("e-")'
                        scriptMain += f"\tpe {commMtrxPart.loc[index,pe]}\n"
                    if '-la("H+")' in centralDict['crossDependencies']['speciation']['input']:
                        ph = '-la("H+")'
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
                    if '-la("H+")' in centralDict['crossDependencies']['speciation']['input']:
                        ph = '-la("H+")'
                        scriptMain += f"\tpH {commMtrxPart.loc[index,ph]}\n"
                    if '-la("e-")' in centralDict['crossDependencies']['speciation']['input']:
                        pe = '-la("e-")'
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
                if centralDict['kinetics']: scriptMain += "use kinetics 1\n" 
        
        if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation') :
            scriptMain += "\nUSER_PUNCH\n\t-headings"
            for val in centralDict['crossDependencies']['speciation']['output']:
                scriptMain += f"\t{val}"
            scriptMain += '\n'
            for it, val in enumerate(centralDict['crossDependencies']['speciation']['output'], start = 1):
                scriptMain += f'\t{it} PUNCH {val} \n'

        scriptMain += "\nSELECTED_OUTPUT\n-reset false\n"
        if 'pH' in centralDict['systemSpeciation'] and 'pe' in centralDict['systemSpeciation']:
            scriptMain += """
            -pH True
            -pe True\n"""
        scriptMain +="-molalities"
        for espece in centralDict['systemSpeciation']:
            if espece not in (['pH', 'pe', 'Potential'] + (centralDict['crossDependencies']['speciation']['total'] if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation') else [])):
                if not centralDict['primarySpecies']['phases']: scriptMain += f"\t{espece}"
                else:
                    if espece in centralDict['primarySpecies']['phases']: pass
                    else: scriptMain += f"\t{espece}"
    
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
    centralDict['crossDependencies']['speciation']['input']
    if centralDict['crossDependencies']
    and centralDict['crossDependencies'].get('speciation')
    else [])


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
                                dico[prim][ligne] += centralDict['primToSecSpecies'][comp][prim] * conc
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
                                dico[prim][ligne] += centralDict['primToSecSpecies'][comp][prim] * conc
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
                            print(prim,comp,centralDict['primToSecSpecies'][comp])
                            dico[prim][ligne] += centralDict['primToSecSpecies'][comp][prim] * conc
                            sys.exit()
            ligne += 1
        
        commMtrx_primSpecies = pd.DataFrame(dico, index = commMtrxPart.index)

    pd.set_option('display.max_columns', None)


    if centralDict['acidicTrspt']:

        if centralDict['AcidicEcho'].empty:
            centralDict['AcidicEcho'] = pd.DataFrame(columns = ['H+','OH-','pH'], index = commMtrxPart.index)
            centralDict['AcidicEcho']['OH-'] = 0
            centralDict['AcidicEcho']['H+'] = 0
            centralDict['AcidicEcho']['pH'] = -np.log10(commMtrxPart['H+']) # environ.
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
        # print(index, end = ' ')
        if abort : break
        if centralDict['maillesChargeGeom']: specieChargeMailleListe = speciesToNode(index, centralDict['maillesChargeGeom'], centralDict['speciesChargeGeometry']) # associe une liste d'espèces de contre-charge en fonction de la maille
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

            if centralDict['speciesChargeGeometry']:
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
                        print('Speciation batch aborted. See warning.log file.')
                        abort = True
                        
            elif centralDict['speciesCharge']:
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
                        print('Speciation batch aborted. See warning.log file.')
                        abort = True
                                
            else:
                scriptWarnings +=f"PhreeqC : node n°{index}, time step n°{centralDict['lStep']+1}, t={centralDict['tStep']}{centralDict['timeUnit']}, PID={os.getpid()} : Fatal PhreeqC error. Aborted PhreeqC batch :\n"
                scriptWarnings += nodeSpeciation(commMtrxPart, commMtrx_primSpecies, centralDict['speciesCharge'][0],specieChargeMailleListe, centralDict['speciationCharge'],centralDict['solMod'],currentPrimSpecies)
                print('Speciation batch aborted. See warning.log file.')
                abort = True
        
        if phreeqc.get_selected_output_array() and not abort:
            resultats.append(pd.DataFrame([phreeqc.get_selected_output_array()[-1]], columns=phreeqc.get_selected_output_array()[0]))
            # if index == 3825:
            #     print(phreeqc.get_selected_output_array())
            #     sys.exit()
        elif not phreeqc.get_selected_output_array():
            warnings += 1
            scriptWarnings +=f"PhreeqC : node n°{index}, time step n°{centralDict['lStep']+1}, t={centralDict['tStep']}{centralDict['timeUnit']}, PID={os.getpid()} : no PhreeqC ouput ...\n"
            scriptWarnings += nodeSpeciation(commMtrxPart, commMtrx_primSpecies, None, None, centralDict['speciationCharge'], centralDict['solMod'],currentPrimSpecies)
            abort = True

    if abort:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),warnings, scriptWarnings, abort,calcTime , initWorker
    else:
        sortiePhreeqCtotal = pd.concat(resultats, ignore_index=True)
  
        sortiePhreeqCtotal.index = commMtrxPart.index
        
        colonnesSpeciation  = [] # headers with phreeqc formalism, in piccts order
        excluded = ['pH','pe'] #+ (centralDict['crossDependencies']['speciation']['total'] if centralDict['crossDependencies'].get('speciation') else []) 
        
        if centralDict['primarySpecies']['phases']: colonnesSpeciation += [spc if spc in (centralDict['primarySpecies']['phases']) else f"m_{spc}(mol/kgw)" for spc in centralDict['systemSpeciation'] if spc not in excluded ]
        else: colonnesSpeciation += [f"m_{spc}(mol/kgw)" for spc in centralDict['systemSpeciation'] if spc not in excluded]
        if 'pH' in centralDict['systemSpeciation'] and 'pe' in centralDict['systemSpeciation'] : colonnesSpeciation += ['pH','pe']



        #coupled parameters
        if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation') :
            DC = [x for x in centralDict['crossDependencies']['speciation']['input'] if x not in centralDict['crossDependencies']['speciation']['output']]

            sortiePhreeqCtotal = sortiePhreeqCtotal.join(centralDict['commMtrx'][DC])
            
        

        finalCol = colonnesSpeciation + (centralDict['crossDependencies']['speciation']['total'] if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation') else [])
        commMtrxSpct = sortiePhreeqCtotal[finalCol].copy()

        commMtrxSpct.columns = centralDict['systemSpeciation'] + (centralDict['crossDependencies']['speciation']['total'] if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation') else [])

        if centralDict['acidicTrspt']:
            AcidicEcho = sortiePhreeqCtotal[['m_H+(mol/kgw)', 'm_OH-(mol/kgw)','pH']].copy()
            AcidicEcho.columns = ['H+','OH-','pH']
            AcidicEcho.index = commMtrxPart.index
            return commMtrxSpct, AcidicEcho, commMtrx_primSpecies, sortiePhreeqCtotal, warnings, scriptWarnings, abort,calcTime, initWorker
        
        else:
            return commMtrxSpct, pd.DataFrame(), commMtrx_primSpecies, sortiePhreeqCtotal, warnings, scriptWarnings, abort,calcTime, initWorker 

    
def spct(centralDict):
    startPhreeqC = time.time()
    print("PhreeqC", end=" ", flush=True)

    if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation'):
        phreeqcInput = centralDict['systemSpeciation'] + list(centralDict['crossDependencies']['speciation']['input'])
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
            commMtrxSplitbeforeTrspt = [pd.DataFrame() * chunk_size ]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=centralDict['PIDnbr']) as executor:
            futures = {}
            for i,chunk in enumerate(commMtrxSplit):

                with open("saveCommMtrx.txt", "a", encoding="utf-8") as f:
                    f.write(f"=== Comm Mtrx after transport, n°{i}, lStep = {centralDict['lStep']}  ===\n")
                    f.write(chunk.to_string())
                    f.write("\n\n")
                
                    f.write(f"=== Comm Mtrx before transport, n°{i}, lStep = {centralDict['lStep']} ===\n")
                    f.write(commMtrxSplitbeforeTrspt[i].to_string())
                    f.write("\n\n")
                
                #
                fut = executor.submit(speciationPhreeqC, centralDict, chunk, commMtrxSplitbeforeTrspt[i])
                futures[fut] = i
                #
    
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
        init = max(initWorker) # shall be parallelized as orchestra ..
        totalScriptWarnings = "".join(strg)

    else:
        if centralDict['preliminarEquilibrium']: 
            beforeTrspt = centralDict["beforeTrsptMtrx"][phreeqcInput].copy()
        else:
            beforeTrspt = pd.DataFrame()
        commMtrxSpct, AcidicEcho, commMtrx_primSpecies, sortiePhreeqCtotal, totalWarnings, totalScriptWarnings, abort,calcWallClock, init = speciationPhreeqC(centralDict,
                                                        centralDict['commMtrx'][phreeqcInput],beforeTrspt )
        calcPrcsTime = calcWallClock
        
    
    if totalWarnings:
        with open("warning.log", "a") as warningLog:
            warningLog.write(f"PhreeqC, time = {centralDict['tStep']}{centralDict['timeUnit']}, time step n°{centralDict['lStep']+1} : the {totalWarnings} following warnings occured ...\n")
            warningLog.write(f"{totalScriptWarnings}\n")
    if abort:
        print('Fatal PhreeqC error. Aborting run.')
        sys.exit()
            
    if centralDict['crossDependencies'] and centralDict['crossDependencies'].get('speciation'):
        col = [s for s in centralDict["anythingButSpecies"] if s not in centralDict['crossDependencies']['speciation']['total']]

    else:
        col = centralDict["anythingButSpecies"] 
    
    commMtrxSpct = pd.concat([centralDict['commMtrx'][col],commMtrxSpct], axis=1)
    sortiePhreeqCtotal = pd.concat([centralDict['commMtrx'][centralDict["anythingButSpecies"]],sortiePhreeqCtotal], axis=1)
    commMtrx_primSpecies = pd.concat([centralDict['commMtrx'][['x', 'y', 'z'][:centralDict['geometry']]],commMtrx_primSpecies], axis=1)


    commMtrxSpct = commMtrxSpct[centralDict['commMtrx'].columns]
    commMtrxSpct.to_csv('ci-eq.txt', index=False, header=True, sep='\t')

    centralDict.update({
        "commMtrx": commMtrxSpct,
        "AcidicEcho": AcidicEcho,
        "PhreeqCCalcTime_WallClock": centralDict['PhreeqCCalcTime_WallClock']  + calcWallClock,
        "PhreeqCCalcTime_ProcessorTime": centralDict["PhreeqCCalcTime_ProcessorTime"] + calcPrcsTime,
        "PhreeqCInterfTime_WallClock": centralDict['PhreeqCInterfTime_WallClock'] + time.time() - startPhreeqC - calcWallClock - init,
        "PhreeqCInitTime" : centralDict["PhreeqCInitTime"] + init,
        "PhreeqCTotalTime" : centralDict['PhreeqCTotalTime'] + time.time() - startPhreeqC,
        })


    if centralDict["firstStepEquilibrium"]==True:
        commMtrx_primSpecies.to_csv(os.path.join(centralDict['paths']['PrimarySpecies'], f"PrimarySpecies_{centralDict['lStep']}.txt"), index=False, header=True, sep='\t')
        sortiePhreeqCtotal.to_csv(os.path.join(centralDict['paths']['Speciation'], f"PhreeqC_{centralDict['lStep']}.txt"), index=False, header=True, sep='\t')
    else:
        commMtrx_primSpecies.to_csv(os.path.join(centralDict['paths']['PrimarySpecies'], f"PrimarySpecies_{centralDict['lStep']+1}.txt"), index=False, header=True, sep='\t')
        sortiePhreeqCtotal.to_csv(os.path.join(centralDict['paths']['Speciation'], f"PhreeqC_{centralDict['lStep']+1}.txt"), index=False, header=True, sep='\t')

    print(f"({writeTime((time.time() - startPhreeqC))})") 

    return centralDict 