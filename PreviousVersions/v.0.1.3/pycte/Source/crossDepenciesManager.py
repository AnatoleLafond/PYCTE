from itertools import product
import sys

def crossDep(centralDict, crossDependencies, systemSpeciation=None):
    'link cross dependencies (CD) with respect to coupled modules'
    'we need to differentiate input CD and output CD'
    'systemSpeciation (optionnel) : noms du systeme, sert a aligner les noms des CD sur ce formalisme'

    knownNames = set(systemSpeciation) if systemSpeciation else set()

    def resolveName(keyword, cells, args):
        '''nom de la CD au formalisme de systemSpeciation
        - grandeur simple (pH, mu, ...) : alias present dans systemSpeciation, sinon le mot donne par l'utilisateur
        - grandeur indexee (EDL, TOT, ...) : <prefixe>_<arg1>_<arg2>..., ex. EDL('Na') -> EDL_Na'''
        candidates = [keyword] + list(cells)
        if not args:
            for alias in candidates:
                if alias in knownNames:
                    return alias
            return keyword
        suffix = '_'.join(args)
        for alias in candidates:
            prefix = alias if alias.endswith('_') else alias + '_'
            if prefix + suffix in knownNames:
                return prefix + suffix
        return f"{cells[0]}_{suffix}"

    def inCells(keyword, cells):
        'le mot correspond a une cellule, au "_" final pres (edl <-> edl_, TOT_ <-> TOT)'
        return keyword in cells or keyword.rstrip('_') in {c.rstrip('_') for c in cells}

    def fillCrossDepDic(coupledParam, inputOrOutput, crossDepDict, CD, nameDict=None):
        '''remplit crossDepDict[inputOrOutput] avec les commandes du solveur (cells[0])
        et, si nameDict est fourni, nameDict[inputOrOutput] avec le nom associe (meme indice)'''
        for cpld in coupledParam:
            if isinstance(cpld, dict):
                for key, val in cpld.items():
                    for cells in CD[inputOrOutput]:
                        if inCells(key, cells):
                            if isinstance(val, str):
                                combos = [(val,)]
                            elif any(isinstance(v, list) for v in val):
                                val_normalized = [v if isinstance(v, list) else [v] for v in val]
                                combos = list(product(*val_normalized))
                            else:
                                combos = [(v,) for v in val]
                            for combo in combos:
                                crossDepDict[inputOrOutput].append(
                                    f'{cells[0]}(' + ','.join(f"'{item}'" for item in combo) + ')')
                                if nameDict is not None:
                                    nameDict[inputOrOutput].append(resolveName(key, cells, combo))
                            break
            else:
                for cells in CD[inputOrOutput]:
                    if inCells(cpld, cells):
                        crossDepDict[inputOrOutput].append(cells[0])
                        if nameDict is not None:
                            nameDict[inputOrOutput].append(resolveName(cpld, cells, ()))
                        break
        return crossDepDict

    def pairByCmd(cmds, names):
        '''dedoublonne sur la commande (ordre de 1re apparition, comme dict.fromkeys)
        en gardant le nom associe -> les deux listes restent alignees indice par indice'''
        paired = {}
        for cmd, name in zip(cmds, names):
            paired.setdefault(cmd, name)
        return list(paired.values()), list(paired.keys())

    transportCD = {}
    speciationCD = {}

    if centralDict["couplingInfo"][1] == 'PhreeqC':
        speciationCD = {
            'input': [
                ['tk', 'temperature(K)', 't', 'temp', 'tempK', 'tK', 'temperatureK', 'temperature'],
                ['tc', 'temperature(C)', 'tC', 'tempC', 'temperatureC'],
                ['DebyeLength', 'DebyeLength(m)', 'debyeLength'],
                ['-la("H+")', 'ph', 'pH'],
                ['CHARGE_BALANCE', 'chargebalance', 'chargeBalance'],
                ['-la("e-")', 'pe', 'Eh', 'eh'],
            ],
            "output": [
                ['tk', 'temperature(K)', 't', 'temp', 'tempK', 'tK', 'temperatureK', 'temperature'],
                ['tc', 'temperature(C)', 'tC', 'tempC', 'temperatureC'],
                ['viscos', 'viscosity(mPa·s)', 'v', 'visc'],
                ['MU', 'mu', 'ionicStrength(mol/kgw)', 'IS', 'is'],
                ['EDL', 'EDL_', 'edl_'],
                ['TOT', 'TotAq_', 'tot_aq', 'TOT_'],
                ['-la("H+")', 'ph', 'pH'],
                ['CHARGE_BALANCE', 'chargebalance', 'chargeBalance'],
                ['-la("e-")', 'pe', 'Eh', 'eh'],
            ]}

    elif centralDict["couplingInfo"][1] == 'xGEMS':
        speciationCD = {
            'input': [
                ['temperature(K)', 'tk', 't', 'temp', 'tempK', 'tK', 'temperatureK', 'temperature'],
                ['temperature(C)', 'tc', 'tC', 'tempC', 'temperatureC'],
                ['pressure', 'p', 'P'],
            ],
            "output": [
                ['temperature(K)', 'tk', 't', 'temp', 'tempK', 'tK', 'temperatureK', 'temperature'],
                ['temperature(C)', 'tc', 'tC', 'tempC', 'temperatureC'],
                ['pressure', 'p', 'P'],
            ]}

    if centralDict["couplingInfo"][2] == 'PFLOTRAN':
        transportCD = {
            'input': [
                ['temperature(K)', 'tk', 't', 'temp', 'tempK', 'tK', 'temperatureK', 'temperature'],
                ['temperature(C)', 'tc', 'tC', 'tempC', 'temperatureC'],
                ['pressure', 'p', 'P'],
                ['porosity'],
            ],
            "output": [
                ['temperature(K)', 'tk', 't', 'temp', 'tempK', 'tK', 'temperatureK', 'temperature'],
                ['temperature(C)', 'tc', 'tC', 'tempC', 'temperatureC'],
                ['pressure', 'p', 'P'],
                ['porosity'],
            ]}

    crossDepSpc = {'input': [], 'output': []}
    nameSpc = {'input': [], 'output': []}
    crossDepTrspt = {'input': [], 'output': []}

    speciationSolver = centralDict["couplingInfo"][1]
    isPhreeqC = speciationSolver == 'PhreeqC'

    if speciationSolver == 'ORCHESTRA':
        crossDepSpc['input'] = crossDependencies.get('speciation', [])
        crossDepSpc['output'] = crossDependencies.get('speciation', [])
    elif speciationSolver != 'nativeKinetics':
        for io in ('input', 'output'):
            fillCrossDepDic(crossDependencies.get('speciation', []), io, crossDepSpc, speciationCD, nameSpc)

    if centralDict["couplingInfo"][2] == 'COMSOL':
        crossDepTrspt['input'] = crossDependencies.get('transport', [])
        crossDepTrspt['output'] = crossDependencies.get('transport', [])
    else:
        fillCrossDepDic(crossDependencies.get('transport', []), 'input', crossDepTrspt, transportCD)
        fillCrossDepDic(crossDependencies.get('transport', []), 'output', crossDepTrspt, transportCD)

    if isPhreeqC:
        inputName, inputCmd = pairByCmd(crossDepSpc['input'], nameSpc['input'])
        outputName, outputCmd = pairByCmd(crossDepSpc['output'], nameSpc['output'])
        totalName, totalCmd = pairByCmd(crossDepSpc['input'] + crossDepSpc['output'],
                                        nameSpc['input'] + nameSpc['output'])
        speciationBlock = {
            'input': {'inputName': inputName, 'inputPhreeqCCmd': inputCmd},
            'output': {'outputName': outputName, 'outputPhreeqCCmd': outputCmd},
            'total': {'totalName': totalName, 'totalPhreeqCCmd': totalCmd},
        }
    else:
        speciationBlock = {
            'input': list(dict.fromkeys(crossDepSpc['input'])),
            'output': list(dict.fromkeys(crossDepSpc['output'])),
            'total': list(dict.fromkeys(crossDepSpc['input'] + crossDepSpc['output'])),
        }

    test = {
        'speciation': speciationBlock,
        'transport': {
            'input': list(dict.fromkeys(crossDepTrspt['input'])),
            'output': list(dict.fromkeys(crossDepTrspt['output'])),
            'total': list(dict.fromkeys(crossDepTrspt['output'] + crossDepTrspt['input']))},
        'totalCrossDep': list(dict.fromkeys(crossDepSpc['input'] + crossDepSpc['output'] + crossDepTrspt['output'] + crossDepTrspt['input'])),
    }
    return test