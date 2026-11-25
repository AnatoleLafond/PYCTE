from itertools import product
import math
import re

FARADAY = 96485.33212
GAS_CONSTANT = 8.314462618
EH_PER_PE_K = math.log(10) * GAS_CONSTANT / FARADAY
ATM = 101325.0

PUNCH = {
    'Eh': f'-LA("e-")*LOG(10)*{GAS_CONSTANT}*TK/{FARADAY}',
    'pressure': f'PRESSURE*{ATM}',
}

_TK = ['tk', 'temperature(K)', 't', 'temp', 'tempK', 'tK', 'temperatureK', 'temperature']
# temperatures in K only: the °C names (former crossDependency tc) are refused by crossDep
_REMOVED_TC = ('tc', 'temperature(C)', 'tC', 'tempC', 'temperatureC')
_PRESSURE = ['pressure', 'p', 'P']
_PH = ['-la("H+")', 'ph', 'pH']
_PE = ['-la("e-")', 'pe']
_EH = ['Eh', 'eh', 'Eh(V)']
_CB = ['CHARGE_BALANCE', 'chargebalance', 'chargeBalance']
_PHREEQC_OUTPUT = [
    _TK, _PRESSURE,
    ['viscos', 'viscosity(mPa·s)', 'v', 'visc'],
    ['MU', 'mu', 'ionicStrength(mol/kgw)', 'IS', 'is'],
    ['EDL', 'EDL_', 'edl_'],
    ['TOT', 'TotAq_', 'tot_aq', 'TOT_'],
    _PH, _CB, _PE, _EH,
]

SPECIATION_TABLES = {
    'PhreeqC': {
        'input': [_TK, _PRESSURE, ['DebyeLength', 'DebyeLength(m)', 'debyeLength'], _PH, _CB, _PE, _EH],
        'output': _PHREEQC_OUTPUT},
    'PhreeqcRM': {
        'input': [_TK, _PRESSURE, _CB],
        'output': _PHREEQC_OUTPUT},
    'xGEMS': {
        'input': [
            ['temperature(K)'] + [a for a in _TK if a != 'temperature(K)'],
            _PRESSURE],
        'output': [
            ['temperature(K)'] + [a for a in _TK if a != 'temperature(K)'],
            _PRESSURE,
            ['pH', 'ph'],
            ['pe'],
            _EH,
            ['ionicStrength', 'IS', 'is', 'mu', 'MU'],
            ['systemMass', 'systemMass(kg)'],
            ['systemVolume', 'systemVolume(m3)']]},
}

TRANSPORT_TABLES = {
    'PFLOTRAN': {
        'input': [
            ['temperature(K)'] + [a for a in _TK if a != 'temperature(K)'],
            _PRESSURE, ['porosity']],
        'output': [
            ['temperature(K)'] + [a for a in _TK if a != 'temperature(K)'],
            _PRESSURE, ['porosity']]},
}

GENERIC_OUTPUT = ('PhreeqC', 'PhreeqcRM', 'xGEMS')
TIME_INPUT = ('PhreeqC', 'PhreeqcRM')

_IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z0-9_]*$')
_CALL = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)$")


def parseCommand(cmd):
    """Command -> (function name, arguments), e.g. SI('Calcite') -> ('SI', ['Calcite'])."""
    match = _CALL.match(cmd)
    if not match:
        return cmd, []
    args = [a.strip().strip("'\"") for a in match.group(2).split(',')] if match.group(2).strip() else []
    return match.group(1), args


def byQuantity(block, io='input'):
    """{solver quantity: column} of block['input'], ['output'] or ['total']."""
    return {block['quantity'][name]: name for name in block[io]}


def crossDep(centralDict, crossDependencies, coupledVariables=None):
    """crossDependencies -> per solver kind {'input', 'output', 'total', 'quantity'}, plus 'names' and unknowns."""
    knownNames = set(coupledVariables) if coupledVariables else set()
    for kind in ('speciation', 'transport'):
        words = [w for kw in crossDependencies.get(kind) or [] for w in (kw if isinstance(kw, dict) else [kw])]
        removed = [w for w in words if w in _REMOVED_TC]
        if removed:
            raise ValueError(f"crossDependencies['{kind}'] : {removed} : the temperature in °C is no longer coupled, "
                             "every temperature is in K : couple 'tk' (or 'temperature(K)') with a column in K")

    def resolveName(keyword, cells, args):
        '''Column name of a crossDependency (alias in coupledVariables, or <prefix>_<args>).'''
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
        '''Whether the word matches a cell, up to a trailing '_'.'''
        return keyword in cells or keyword.rstrip('_') in {c.rstrip('_') for c in cells}

    def combinations(val):
        if val is None:
            return [()]
        if isinstance(val, str):
            return [(val,)]
        if any(isinstance(v, list) for v in val):
            return list(product(*[v if isinstance(v, list) else [v] for v in val]))
        return [(v,) for v in val] or [()]

    def fill(coupledParam, table, cmds, names, unmatched, generic):
        '''Solver commands and column names of the keywords; the others go to `unmatched`.'''
        for cpld in coupledParam:
            if isinstance(cpld, dict):
                for key, val in cpld.items():
                    cells = next((c for c in table if inCells(key, c)), None)
                    if cells is None and generic and _IDENTIFIER.match(key):
                        cells = [key]
                    if cells is None:
                        unmatched.append(key)
                        continue
                    for combo in combinations(val):
                        cmds.append(f'{cells[0]}(' + ','.join(f"'{item}'" for item in combo) + ')' if combo
                                    else cells[0])
                        names.append(resolveName(key, cells, combo))
            else:
                cells = next((c for c in table if inCells(cpld, c)), None)
                if cells is None:
                    unmatched.append(cpld)
                    continue
                cmds.append(cells[0])
                names.append(resolveName(cpld, cells, ()))

    def pairByCmd(cmds, names):
        '''Removes duplicated commands, keeping the first name.'''
        paired = {}
        for cmd, name in zip(cmds, names):
            paired.setdefault(cmd, name)
        return list(paired.values()), list(paired.keys())

    def block(keywords, table, solver, outputFirst):
        cmds = {'input': [], 'output': []}
        names = {'input': [], 'output': []}
        ignored = []
        if table:
            unmatched = {'input': [], 'output': []}
            for io in ('input', 'output'):
                fill(keywords, table[io], cmds[io], names[io], unmatched[io],
                     generic=io == 'output' and solver in GENERIC_OUTPUT)
            ignored = [kw for kw in dict.fromkeys(unmatched['input']) if kw in unmatched['output']]
            if solver in TIME_INPUT and 'time' in ignored:
                ignored.remove('time')
                cmds['input'].append('time')
                names['input'].append('time')
        else:
            for io in ('input', 'output'):
                cmds[io] = list(keywords)
                names[io] = list(keywords)
        order = ('output', 'input') if outputFirst else ('input', 'output')
        inputName, _ = pairByCmd(cmds['input'], names['input'])
        outputName, _ = pairByCmd(cmds['output'], names['output'])
        totalName, _ = pairByCmd(cmds[order[0]] + cmds[order[1]], names[order[0]] + names[order[1]])
        quantity = {}
        for cmd, name in zip(cmds['input'] + cmds['output'], names['input'] + names['output']):
            quantity.setdefault(name, cmd)
        return {'input': inputName, 'output': outputName, 'total': totalName, 'quantity': quantity}, ignored

    def known(table):
        '''Quantities of a table with their aliases (error messages).'''
        table = table or {}
        inputs = {cells[0] for cells in table.get('input', [])}
        outputs = {cells[0] for cells in table.get('output', [])}
        quantities = {}
        for cells in table.get('input', []) + table.get('output', []):
            mark = '' if cells[0] in inputs and cells[0] in outputs else ' (input)' if cells[0] in inputs else ' (output)'
            quantities.setdefault(cells[0], ' / '.join(c for c in cells if not c.startswith('-')) + mark)
        return list(quantities.values())

    speciationSolver = centralDict["couplingInfo"][1]
    transportSolver = centralDict["couplingInfo"][2]
    speciationTable = SPECIATION_TABLES.get(speciationSolver)
    transportTable = TRANSPORT_TABLES.get(transportSolver)
    speciationBlock, ignoredSpeciation = block(crossDependencies.get('speciation') or [], speciationTable,
                                               speciationSolver, outputFirst=False)
    transportBlock, ignoredTransport = block(crossDependencies.get('transport') or [], transportTable,
                                             transportSolver, outputFirst=True)
    knownSpeciation = known(speciationTable) + (['time'] if speciationSolver in TIME_INPUT else [])
    if speciationSolver in GENERIC_OUTPUT:
        knownSpeciation.append("{function: [arguments]} (output)")
    return {
        'speciation': speciationBlock,
        'transport': transportBlock,
        'names': list(dict.fromkeys(speciationBlock['total'] + transportBlock['total'])),
        'ignoredSpeciation': ignoredSpeciation,
        'ignoredTransport': ignoredTransport,
        'knownSpeciation': knownSpeciation,
        'knownTransport': known(transportTable),
    }
