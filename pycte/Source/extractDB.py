import re
import pandas as pd
import time
import numpy as np
import concurrent.futures
import os
from charset_normalizer import from_path
import glob
import json
try:
    from . import warningManager
except ImportError:
    import warningManager

_encodings = {}


def databaseEncoding(filepath):
    """Encoding of a database file (charset_normalizer: UTF-8, latin-1 / cp1252 ...), detected once per file version."""
    path = os.path.abspath(str(filepath))
    key = (path, os.path.getmtime(path))
    if key not in _encodings:
        best = from_path(path).best()
        _encodings[key] = best.encoding if best else 'utf-8'
    return _encodings[key]


def asciiPath(filepath):
    """Path a solver library can open: on Windows, the C libraries (PhreeqcRM, xGEMS) read file names in the ANSI code
    page, so a path with an accent (Données, Thèse) is not found. Its 8.3 short form is ASCII; where short names are
    disabled, a copy in an ASCII temporary folder; ValueError if neither exists."""
    import sys
    path = os.path.abspath(str(filepath))
    if path.isascii() or sys.platform != 'win32':
        return path
    import ctypes
    import shutil
    import tempfile

    def short(p):
        buffer = ctypes.create_unicode_buffer(32768)
        n = ctypes.windll.kernel32.GetShortPathNameW(p, buffer, len(buffer))
        return buffer.value if 0 < n < len(buffer) else p

    if short(path).isascii():
        return short(path)
    folder = short(tempfile.mkdtemp(prefix='pycte_'))
    if not folder.isascii():
        raise ValueError(f"{path} : the solver cannot open a path with non-ASCII characters (accents), and no ASCII "
                         "copy could be made : move the files to a folder whose path has no accent")
    copy = os.path.join(folder, os.path.basename(path).encode('ascii', 'replace').decode().replace('?', '_'))
    shutil.copy2(path, copy)
    return copy

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

_primaryPatterns = {}
_lastPattern = []          # [tuple of primary species, its pattern]: the same tuple for every column of a database


class _PrimaryMatcher:
    """matcher(formula, index): the master species read at `index` of the formula, or None. The same as
    re.compile('(?:' + '|'.join(re.escape(ps) for ps in sorted(primarySpecies, key=len, reverse=True)) + ')(?![a-z])')
    .match(formula, index).group(0): the first species, longest first, written there and not followed by a lower-case
    letter ('S' does not match the start of 'Sr', 'Si' or 'Sn': a master species ends at an element boundary), found
    among the species starting with that character, in the same order. Without the regular expression, whose
    compilation took ~30 ms for the hundreds of phases of a database."""

    def __init__(self, primarySpecies):
        # no species: the expression '(?:)(?![a-z])' matches the empty string, as the species '' would
        self.ordered = sorted(primarySpecies, key=len, reverse=True) or ['']
        self.byFirst = {}

    def __call__(self, formula, index):
        first = formula[index]
        candidates = self.byFirst.get(first)
        if candidates is None:
            candidates = self.byFirst[first] = [ps for ps in self.ordered if not ps or ps[0] == first]
        for ps in candidates:
            if formula.startswith(ps, index):
                end = index + len(ps)
                if not ('a' <= formula[end:end + 1] <= 'z'):
                    return ps
        return None


def _primaryPattern(primarySpecies):
    if _lastPattern and _lastPattern[0] is primarySpecies:
        return _lastPattern[1]
    key = tuple(primarySpecies)
    pattern = _primaryPatterns.get(key)
    if pattern is None:
        pattern = _primaryPatterns[key] = _PrimaryMatcher(primarySpecies)
    if isinstance(primarySpecies, tuple):
        # a tuple cannot change: the next call with the same object skips building and hashing the key
        _lastPattern[:] = [primarySpecies, pattern]
    return pattern


def decomposingIntoPrimSpecies(formula, primarySpecies,redoxPrim,redox_dict, unparsed=None):
    """Composition of `formula` on primary species; unmatched parts are appended to `unparsed`."""
    def multiply_dict(d, factor):
        return {k: v * factor for k, v in d.items()}
    def merge_dicts(a, b):
        for k, v in b.items():
            a[k] = a.get(k, 0) + v
        return a

    charge_match = re.search(r'([+-]\d*)$', formula)
    if charge_match and formula in primarySpecies:
        # master species whose name ends like a charge (phase 'Illite_Imt-2' of ThermoChimie): itself
        return {formula: 1}, None
    if charge_match:
        charge = charge_match.group(1)
        formula = formula[:charge_match.start()]
    else:
        charge = None

    formula = re.sub(r'__([0-9]+)', r')\1', formula)

    pattern = _primaryPattern(primarySpecies)

    index = 0
    tokens = []
    skippedAt = []

    while index < len(formula):
        name = pattern(formula, index)
        if name is not None:
            j = index + len(name)
            coef_match = re.match(r'\d+', formula[j:]) 
            if coef_match:
                count = int(coef_match.group())
                j += len(coef_match.group()) 
            else:
                count = 1 
            tokens.append(('master', name, count)) 
            index = j 
            continue

        if formula[index] == '(':
            tokens.append(('(',))
            index += 1
            continue
        elif formula[index] == ')': 
            j = index + 1
            while j < len(formula) and formula[j].isdigit():
                j += 1
            multiplicateur = int(formula[index+1:j]) if j > index+1 else 1 
        
            tokens.append((')', multiplicateur))
            index = j
            continue

        skippedAt.append(index)
        index += 1
    if unparsed is not None:
        fragments = []
        for k, pos in enumerate(skippedAt):
            if k and pos == skippedAt[k - 1] + 1:
                fragments[-1] += formula[pos]
            else:
                fragments.append(formula[pos])
        # lower case only parts ('(g)', 'aq', the 'e' of 'e-') and signs are not elements
        unparsed.extend(f for f in fragments if any(ch.isupper() for ch in f))
    stack = []
    current = {}

    gotcha = False
    # redox state of the species: the state whose species list holds it, charge written 'Cu+' or 'Cu+1'
    species = _canonical_charge(formula + charge if charge else formula)
    for token in tokens:

        gotcha = False
        if token[0] == 'master':
            if redox_dict and token[1] in redox_dict:
                for key in redox_dict[token[1]]:
                    if key in redoxPrim:
                        for sec in redoxPrim[key]:
                            if species == _canonical_charge(sec):
                                name, count = key, token[2]
                                gotcha = True
                        

            if not gotcha : name, count = token[1], token[2]
            current[name] = current.get(name, 0) + count

        elif token[0] == '(':
            stack.append(current)
            current = {}
        elif token[0] == ')':
            multiplicateur = token[1]
            current = multiply_dict(current, multiplicateur)
            prev = stack.pop()
            current = merge_dicts(prev, current)

    return current, charge


def _strip_comment(line):
    """Line without its # comment and spaces."""
    return line.split("#", 1)[0].strip()


_strippedLines = {}


def _dbLines(filepath, encoding):
    """Lines of a database file without their # comment and spaces, empty lines left out: what every reader below
    iterates on. Read once per file version (path, modification time, size) and encoding, instead of once per reader
    (and once per coupled phase in phreeqcDBextraction)."""
    path = os.path.abspath(str(filepath))
    try:
        key = (path, os.path.getmtime(path), os.path.getsize(path), encoding)
    except OSError:
        key = None
    lines = _strippedLines.get(key) if key is not None else None
    if lines is None:
        with open(filepath, "r", encoding=encoding) as f:
            lines = tuple(stripped for stripped in map(_strip_comment, f) if stripped)
        if key is not None:
            if len(_strippedLines) >= 8:
                _strippedLines.clear()
            _strippedLines[key] = lines
    return lines


def extract_master_redox_map(filepath, redox_dict):
    """SOLUTION_MASTER_SPECIES -> {redox state: master species}, e.g. {'S(+6)': 'SO4-2'}."""
    redox_map = {}
    in_block = False

    for stripped in _dbLines(filepath, databaseEncoding(filepath)):
        upper = stripped.upper()
        first_token = upper.split()[0]

        if first_token == "SOLUTION_MASTER_SPECIES":
            in_block = True
            continue
        if in_block and (
            first_token in ("SOLUTION_SPECIES", "PHASES", "REACTION",
                            "KINETICS", "RATES", "END", "SIT")
            or first_token.startswith(("SURFACE", "EXCHANGE"))
        ):
            break

        if not in_block:
            continue

        parts = stripped.split()
        if len(parts) < 2:
            continue

        state = parts[0]
        master = parts[1]

        for base, states in redox_dict.items():
            if state in states:
                redox_map[state] = master
                break

    return redox_map

_canonicalCharges = {}


def _canonical_charge(species):
    """Species name with its charge written sign + number : 'Cu+' and 'Cu+1' -> 'Cu+1', 'CO3--' -> 'CO3-2'."""
    canonical = _canonicalCharges.get(species)
    if canonical is None:
        m = re.match(r'^(.+?)([+-]\d+|[+-]+)$', species)
        if m:
            base, charge = m.groups()
            n = int(charge[1:]) if charge[1:].isdigit() else len(charge)
            canonical = f"{base}{charge[0]}{n}"
        else:
            canonical = species
        _canonicalCharges[species] = canonical
    return canonical


def _reaction_species(text):
    """Species of a reaction side (or of a whole reaction, '=' included), charges in canonical form. Whole names are
    compared : 'Cu+' (master of Cu(+1)) is not found in 'Cu+2 + H2O', nor 'UO2+' in 'UO2+2 + H2O'."""
    return {_canonical_charge(sp) for _, sp in _parse_reaction_side(text.replace('=', ' '))}


def extract_secondary_species(filepath, redox_dict, redox_map):

    
    secondary = {}

    species_block_keywords = ("SOLUTION_SPECIES", "SURFACE_SPECIES", "EXCHANGE_SPECIES")
    phases_block_keyword = "PHASES"
    closing_only_keywords = (
        "SOLUTION_MASTER_SPECIES", "EXCHANGE_MASTER_SPECIES", "SURFACE_MASTER_SPECIES",
        "REACTION", "KINETICS", "RATES", "END",
    )
    param_prefixes = (
        "log_k", "delta_h", "-analytic", "-vm", "-gamma", "-llnl",
        "-no_check", "-mole_bal", "-add_logk", "-t_c", "-p_c", "-omega", "-cd_music",
    )

    def add_secondary(state, sp):
        if state not in secondary:
            secondary[state] = []
        if sp not in secondary[state]:
            secondary[state].append(sp)

    states = []
    for state, master in redox_map.items():
        base = state[:state.index("(")] if "(" in state else state
        # master species of the other redox states of the same element (Cu+2 + e- = Cu+ : Cu+ is Cu(+1))
        others = {_canonical_charge(m) for s, m in redox_map.items() if s != state and s.split("(")[0] == base}
        states.append((state, master, base, _canonical_charge(master), others))

    block_type = None
    current_phase_name = None
    for stripped in _dbLines(filepath, databaseEncoding(filepath)):
        upper = stripped.upper()
        first_token = upper.split()[0]

        if first_token in species_block_keywords:
            block_type = "species"
            current_phase_name = None
            continue

        if first_token == phases_block_keyword:
            block_type = "phases"
            current_phase_name = None
            continue

        if block_type is not None and first_token in closing_only_keywords:
            block_type = None
            current_phase_name = None
            continue

        if block_type is None:
            continue

        if block_type == "species":
            if stripped.lower().startswith(param_prefixes):
                continue
            if "=" not in stripped:
                continue

            lhs, rhs = stripped.split("=", 1)
            if not rhs.strip():
                continue
            rhs_species = rhs.strip().split()[0]
            lhs_species = _reaction_species(lhs)
            rhs_canonical = _canonical_charge(rhs_species)

            for state, master, base, canonical, others in states:
                if canonical not in lhs_species:
                    continue
                if base not in rhs_species:
                    continue
                if rhs_canonical in others or rhs_canonical == canonical:
                    continue

                add_secondary(state, rhs_species)

        elif block_type == "phases":
            if stripped.lower().startswith(param_prefixes):
                continue

            if "=" not in stripped:
                current_phase_name = stripped.split()[0]
                continue

            if current_phase_name is None:
                continue

            reaction_species = _reaction_species(stripped)
            for state, master, base, canonical, others in states:
                if canonical in reaction_species:
                    add_secondary(state, current_phase_name)

    for sp in secondary:
        secondary[sp] += [redox_map[sp]]
    return secondary

def group_redox_states(species_list):
    redox = {}
    for sp in species_list:
        if "(" in sp:
            base = sp[:sp.index("(")]
        else:
            base = sp
        if base not in redox:
            redox[base] = []
        if sp not in redox[base]:
            redox[base].append(sp)

    return {k: [s for s in v if re.search(r'\d', s)] for k, v in redox.items() if any(re.search(r'\d', s) for s in v)}

def _is_block_keyword_line(upper_line, keywords):

    first_token = upper_line.split()[0] if upper_line.split() else ""
    return first_token in keywords  


def _strip_coeff(term):
    m = re.match(r'^[0-9]*\.?[0-9]+(?=[A-Za-z(])', term)
    return term[m.end():] if m else term


def _split_terms(side):
    terms = []
    for raw in re.split(r'\s+\+\s+', side.strip()):
        tok = raw.strip()
        if not tok:
            continue
        terms.append(_strip_coeff(tok))
    return terms

_masterSpecies = {}


def extract_master_species(filepath, encoding=None):
    """Master species, phases ... of a database ({block: [names]}). Read once per file version (path, modification
    time, size) and encoding, as _dbLines: the run reads it twice (database extraction, MultiCompoundTransport); each
    call gets its own lists."""
    encoding = encoding or databaseEncoding(filepath)
    path = os.path.abspath(str(filepath))
    try:
        key = (path, os.path.getmtime(path), os.path.getsize(path), encoding)
    except OSError:
        key = None
    found = _masterSpecies.get(key) if key is not None else None
    if found is None:
        found = _extractMasterSpecies(filepath, encoding)
        if key is not None:
            if len(_masterSpecies) >= 8:
                _masterSpecies.clear()
            _masterSpecies[key] = found
    return {block: list(names) for block, names in found.items()}


def _extractMasterSpecies(filepath, encoding):
    masterSpecies = {
        'SOLUTION_MASTER_SPECIES': [],
        'SOLUTION_MASTER_ELEMENT': [],
        'SURFACE_MASTER_SPECIES': [],
        'EXCHANGE_SPECIES': [],
        'EXCHANGE_MASTER_SPECIES': [],
        'EXCHANGE_MASTER_ELEMENT': [],
        'PHASES': [],
        'SURFACE_SPECIES': [],
    }
    stop_keywords = {
        "SOLUTION_SPECIES", 'SOLUTION_MASTER_SPECIES', "SURFACE_SPECIES",
        'SURFACE_MASTER_SPECIES', "EXCHANGE_MASTER_SPECIES", "EXCHANGE_SPECIES",
        "EQUILIBRIUM_PHASES", 'PHASES', "REACTION", "KINETICS", "RATES", "END", 'SIT'
    }

    def _is_surface_species(sp, surface_master_prefixes):
        """Whether `sp` contains a site name of SURFACE_MASTER_SPECIES."""
        return any(prefix in sp for prefix in surface_master_prefixes)


    lines = _dbLines(filepath, encoding)
    current_block = None
    for line in lines:
        upper_line = line.upper()
        if upper_line in masterSpecies:
            current_block = upper_line
            continue
        if current_block and _is_block_keyword_line(upper_line, stop_keywords):
            current_block = None
            continue
        if current_block == 'SOLUTION_MASTER_SPECIES':
            parts = line.split()
            if len(parts) >= 2:
                masterSpecies['SOLUTION_MASTER_SPECIES'].append(parts[0])
                masterSpecies['SOLUTION_MASTER_ELEMENT'].append(parts[1])
        elif current_block == 'SURFACE_MASTER_SPECIES':
            parts = line.split()
            if parts:
                masterSpecies['SURFACE_MASTER_SPECIES'].append(parts[0])
        elif current_block == 'EXCHANGE_MASTER_SPECIES':
            parts = line.split()
            if len(parts) >= 2:
                masterSpecies['EXCHANGE_MASTER_ELEMENT'].append(parts[1])

    current_block = None
    pending_phase_name = None
    for line in lines:
        upper_line = line.upper()
        if upper_line in masterSpecies:
            current_block = upper_line
            continue
        if current_block and _is_block_keyword_line(upper_line, stop_keywords):
            current_block = None
            continue

        if current_block == "PHASES":
            if any(line.lower().startswith(kw) for kw in ("log_k", "delta_h", "-analytic", "-vm")):
                continue
            if "=" in line:
                if pending_phase_name:
                    masterSpecies["PHASES"].append(pending_phase_name)
                    pending_phase_name = None
            else:
                pending_phase_name = line.split()[0]

        elif current_block in ('SOLUTION_MASTER_SPECIES', 'SURFACE_MASTER_SPECIES'):
            continue

        elif current_block:
            if line.lower().startswith("log_k"):
                continue
            if "=" in line:
                lhs, rhs = line.split("=", 1)
                lhs_last = lhs.strip().split()[-1]
                rhs_first = rhs.strip().split()[0]
                if lhs_last == rhs_first:
                    continue
                if not rhs_first.replace(".", "").isdigit():
                    if current_block != "SURFACE_SPECIES" or _is_surface_species(
                        rhs_first, masterSpecies['SURFACE_MASTER_SPECIES']
                    ):
                        masterSpecies[current_block].append(rhs_first)
            parts = line.split()
            if parts and current_block != "EXCHANGE_SPECIES":
                if current_block != "SURFACE_SPECIES" or _is_surface_species(
                    parts[0], masterSpecies['SURFACE_MASTER_SPECIES']
                ):
                    masterSpecies[current_block].append(parts[0])

    return masterSpecies


_PHREEQC_KEYWORDS = {
    "SOLUTION_MASTER_SPECIES", "SOLUTION_SPECIES", "PHASES",
    "EXCHANGE_MASTER_SPECIES", "EXCHANGE_SPECIES",
    "SURFACE_MASTER_SPECIES", "SURFACE_SPECIES",
    "RATES", "END", "SIT", "PITZER", "LLNL_AQUEOUS_MODEL_PARAMETERS",
    "GAS_BINARY_PARAMETERS", "MEAN_GAMMAS", "NAMED_EXPRESSIONS",
    "CALCULATE_VALUES", "ISOTOPES", "ISOTOPE_RATIOS", "ISOTOPE_ALPHAS",
    "SOLID_SOLUTIONS", "EQUILIBRIUM_PHASES", "KINETICS", "REACTION",
    "SOLUTION", "EXCHANGE", "SURFACE", "GAS_PHASE", "KNOBS", "USE",
    "SAVE", "SELECTED_OUTPUT", "USER_PUNCH", "USER_PRINT", "USER_GRAPH",
    "PRINT", "TITLE", "DATABASE", "INCLUDE$", "COPY", "DELETE", "DUMP", "MIX",
}
_PHASE_PARAMS = {
    "log_k", "logk", "delta_h", "deltah", "analytic", "analytical_expression",
    "a_e", "vm", "add_logk", "add_constant", "t_c", "p_c", "omega",
}


def _parse_reaction_side(side):
    """Reaction side -> [(coefficient, species)], e.g. '2 H+ + CO3-2' -> [(2.0, 'H+'), (1.0, 'CO3-2')]."""
    terms = []
    sign, coef = 1.0, None
    for tok in side.split():
        m = re.fullmatch(r'([+-]?)(\d+\.?\d*|\.\d+)?(.*)', tok)
        s, num, sp = m.groups()
        if not sp:
            if s == '-':
                sign = -sign
            if num:
                try:
                    coef = float(tok.lstrip('+-'))
                except ValueError:
                    coef = float(num)
            continue
        c = float(num) if num else 1.0
        if coef is not None:
            c *= coef
        if s == '-':
            c = -c
        terms.append((sign * c, sp))
        sign, coef = 1.0, None
    return terms


def extract_solution_species(filepath):
    """Names of the species defined in SOLUTION_SPECIES (first species of the right-hand side of each reaction)."""
    encoding = databaseEncoding(filepath)
    names = set()
    in_block = False
    for stripped in _dbLines(filepath, encoding):
        first = stripped.split()[0].upper()
        if first == "SOLUTION_SPECIES":
            in_block = True
            continue
        if first in _PHREEQC_KEYWORDS or first.endswith(("_MODIFY", "_RAW")):
            in_block = False
            continue
        if in_block and "=" in stripped:
            rhs = stripped.split("=", 1)[1].split()
            if rhs:
                names.add(rhs[0])
    return names


def extract_phase_reactions(filepath):

    encoding = databaseEncoding(filepath)

    reactions = {}
    in_phases = False
    name = None
    for stripped in _dbLines(filepath, encoding):
        first = stripped.split()[0]
        upper_first = first.upper()

        if upper_first == "PHASES":
            in_phases, name = True, None
            continue
        if in_phases and (upper_first in _PHREEQC_KEYWORDS
                          or upper_first.endswith(("_MODIFY", "_RAW"))):
            in_phases, name = False, None
            continue
        if not in_phases:
            continue

        if first.startswith("-") or first.lower() in _PHASE_PARAMS:
            continue
        if "=" in stripped:
            if name is not None:
                lhs, rhs = stripped.split("=", 1)
                reactions[name] = (_parse_reaction_side(lhs), _parse_reaction_side(rhs))
                name = None
            continue
        name = first

    return reactions


def decomposingPhaseIntoPrimSpecies(reaction, primarySpecies, redoxPrim, redox_dict, inverse_redox, unparsed=None):

    lhs, rhs = reaction
    total = {}
    for sign, terms in ((1, rhs), (-1, lhs[1:])):
        for coef, sp in terms:
            composition, _ = decomposingIntoPrimSpecies(sp, primarySpecies, redoxPrim, redox_dict, unparsed)
            if _canonical_charge(sp) in inverse_redox:
                redox_name = inverse_redox[_canonical_charge(sp)]
                element = redox_name.split('(')[0]
                if element not in ('H', 'O') and element in composition and redox_name not in composition:
                    composition[redox_name] = composition.pop(element)
            for prim, n in composition.items():
                total[prim] = total.get(prim, 0) + sign * coef * n

    return {prim: (int(round(n)) if abs(n - round(n)) < 1e-9 else round(n, 10))
            for prim, n in total.items() if abs(n) > 1e-9}


def _protonate_exchange_master(master):

    m = re.fullmatch(r'(.+?)(-+|-\d+)', master)
    if not m:
        return master
    base, charge = m.groups()
    n = int(charge[1:]) if charge[1:].isdigit() else len(charge)
    return base + 'H' + (str(n) if n > 1 else '')


def phreeqcDBextraction(centralDict, commMtrx):

    phasesAsPrimary = centralDict.get('phasesAsPrimary', True)
    
    totPrimSpecies = extract_master_species(centralDict['chemPath'])

    totPrimSpecies['SURFACE_SPECIES'] = [
    s for s in totPrimSpecies['SURFACE_SPECIES']
    if s not in totPrimSpecies['SOLUTION_MASTER_ELEMENT']
]

    totPrimSpecies['EXCHANGE_MASTER_ELEMENT'] = [_protonate_exchange_master(s) for s in totPrimSpecies['EXCHANGE_MASTER_ELEMENT']]
    
    phaseNames = list(totPrimSpecies['PHASES'])
    # a column named both as an aqueous species and as a phase (Cd(OH)2, CdSO4, Pb(OH)2 in phreeqc.dat) is the phase
    named = [c for c in commMtrx.columns if c in phaseNames]
    solutionSpecies = extract_solution_species(centralDict['chemPath']) if named else set()
    ambiguous = [c for c in named if c in solutionSpecies]
    if ambiguous:
        warningManager.warn(f"pycte : {ambiguous} are both aqueous species (SOLUTION_SPECIES) and phases (PHASES) of "
                            f"{centralDict['chemPath']} : these columns are taken as the phases (moles of mineral, not "
                            "transported) ; to couple the aqueous species, give the phase another name in a copy of "
                            "the database")
    if not phasesAsPrimary:
        totPrimSpecies['PHASES'] = []
    
    
    redox_dict = group_redox_states((totPrimSpecies['SOLUTION_MASTER_SPECIES']))
    redox_map = extract_master_redox_map(centralDict['chemPath'], redox_dict)

    
    # H and O are counted as totals (H, O, H2O), not per redox state, whatever the database calls the states
    # (H(1) in phreeqc.dat, H(+1) in ThermoChimie): otherwise H+ would decompose into H(1), missing from -total_h
    for state in [s for s in redox_map if s.split('(')[0] in ('H', 'O')]:
        del redox_map[state]
    if 'H' in redox_dict:
        del redox_dict['H']
    if 'O' in redox_dict:
        del redox_dict['O']

    
    secondary = extract_secondary_species(centralDict['chemPath'], redox_dict, redox_map)

    sol = []
    phases = []
    surf = []
    exch = []

    
    def stringCell(conc, comp):
        raise TypeError(f"initial conditions : column {comp!r} holds the text {conc!r} instead of a number")

    primToSecSpecies = {}
    unparsed = {}           # species -> parts of its name matching no master species (decomposition incomplete)
    if len(commMtrx):
        # a tuple: its decomposition pattern is built once for every column (_primaryPattern)
        primList = tuple(totPrimSpecies['SOLUTION_MASTER_SPECIES']+
            totPrimSpecies['SURFACE_MASTER_SPECIES']+totPrimSpecies['EXCHANGE_SPECIES']+ totPrimSpecies['EXCHANGE_MASTER_SPECIES']+totPrimSpecies['EXCHANGE_MASTER_ELEMENT']+totPrimSpecies['PHASES'])
        cells =commMtrx.to_numpy(dtype=object) if any(dt.kind not in 'biufc' for dt in commMtrx.dtypes) else None
        for j, comp in enumerate(commMtrx.columns):
            if cells is not None and isinstance(cells[0, j], str):
                stringCell(cells[0, j], comp)
            fragments = []
            composition, _ = decomposingIntoPrimSpecies(comp, primList, secondary, redox_dict, fragments)
            primToSecSpecies.update({comp : composition})
            if fragments:
                unparsed[comp] = fragments
        if cells is not None:
            for i in range(1, len(cells)):
                for j, comp in enumerate(commMtrx.columns):
                    if isinstance(cells[i, j], str):
                        stringCell(cells[i, j], comp)

    for exMaster in totPrimSpecies['EXCHANGE_MASTER_ELEMENT']:
        primToSecSpecies[exMaster] = {exMaster: 1}
    

    inverse_redox = {_canonical_charge(species): element for element, species in redox_map.items()}
    for species, composition in primToSecSpecies.items():
        if _canonical_charge(species) in inverse_redox:
            redox_name = inverse_redox[_canonical_charge(species)]
            element = redox_name.split('(')[0]
    
            if element in composition and redox_name not in composition:
                composition[redox_name] = composition.pop(element)

    masterOf = dict(zip(totPrimSpecies['SOLUTION_MASTER_SPECIES'], totPrimSpecies['SOLUTION_MASTER_ELEMENT']))
    for comp in primToSecSpecies:
        if comp in redox_dict:
            primary = next((s for s in redox_dict[comp] if masterOf.get(s) == masterOf.get(comp)), None)
            if primary:
                primToSecSpecies[comp] = {primary: 1}
                unparsed.pop(comp, None)

    if not phasesAsPrimary:
        phaseReactions = extract_phase_reactions(centralDict['chemPath'])
        primList = tuple(totPrimSpecies['SOLUTION_MASTER_SPECIES']+totPrimSpecies['SURFACE_MASTER_SPECIES']
                         +totPrimSpecies['EXCHANGE_SPECIES']+totPrimSpecies['PHASES'])
        missing = [comp for comp in primToSecSpecies if comp in phaseNames and comp not in phaseReactions]
        if missing:
            raise ValueError(f"Dissolution reaction not found in PHASES for : {missing}")
        for comp in primToSecSpecies:
            if comp in phaseReactions:
                fragments = []
                primToSecSpecies[comp] = decomposingPhaseIntoPrimSpecies(
                    phaseReactions[comp], primList, secondary, redox_dict, inverse_redox, fragments)
                unparsed.pop(comp, None)
                if fragments:
                    unparsed[comp] = fragments
    
                   
    def withoutStates(names):
        prefixes = {t[:k] for t in names for k, ch in enumerate(t) if ch == '('}
        return [s for s in names if s not in prefixes]

    solutionMaster = withoutStates(totPrimSpecies['SOLUTION_MASTER_SPECIES'])
    surfaceMaster = withoutStates(totPrimSpecies['SURFACE_MASTER_SPECIES'])
    if centralDict['preliminaryEquilibrium']:
        exchangeMaster = withoutStates(totPrimSpecies['EXCHANGE_MASTER_SPECIES'])
    else:
        exchangeMaster = withoutStates(totPrimSpecies['EXCHANGE_SPECIES'])
    phaseMaster = withoutStates(totPrimSpecies['PHASES'])
    
    for sp, primDic in primToSecSpecies.items():
        for prim in primDic:
            if prim in surfaceMaster and prim not in surf:
                surf += [prim]
            elif prim in (solutionMaster +['H','O']) and prim not in sol:
                sol += [prim]
            elif prim in phaseMaster and prim not in phases:
                phases += [prim]
            elif prim in exchangeMaster and prim not in exch:

                exch += [prim]

    for exMaster in totPrimSpecies['EXCHANGE_MASTER_ELEMENT']:
        if exMaster not in exch:
            exch += [exMaster]


    fixed = totPrimSpecies['SURFACE_MASTER_SPECIES']+totPrimSpecies['SURFACE_SPECIES']+totPrimSpecies['EXCHANGE_MASTER_SPECIES']+totPrimSpecies['EXCHANGE_SPECIES']+phaseNames
    trspt = [s for s in centralDict['systemSpecies'] if s not in fixed]

    if centralDict['nonTrivialDecomposition']:
        for ky in primToSecSpecies:
            for subky in centralDict['nonTrivialDecomposition']:
                if subky == ky:
                      primToSecSpecies[ky] = centralDict['nonTrivialDecomposition'][ky]
                      unparsed.pop(ky, None)


    primToSecSpecies['H2O'] = {'H2O':1}
    unparsed.pop('H2O', None)
    if unparsed and not totPrimSpecies['SOLUTION_MASTER_SPECIES']:
        # no master species at all: not a PhreeqC database (e.g. the .inp file of ORCHESTRA)
        raise ValueError(f"pycte : {centralDict['chemPath']} has no SOLUTION_MASTER_SPECIES block : it is not a PhreeqC "
                         "database (chemPath of chemistryModule 'phreeqc' / 'phreeqcrm', e.g. phreeqc.dat)")
    if unparsed:
        raise ValueError(
            "pycte : part of the name of these species matches no master species of the database, their "
            "decomposition into primary species would miss it :\n"
            + "\n".join(f"  {sp} : {', '.join(parts)}" for sp, parts in unparsed.items())
            + f"\nCheck these names against {centralDict['chemPath']} (SOLUTION_MASTER_SPECIES, EXCHANGE_SPECIES, "
            "SURFACE_MASTER_SPECIES, PHASES), or give their decomposition with "
            "nonTrivialDecomposition = {species: {primary species: coefficient}}.")

    return primToSecSpecies, sol, phases, surf, exch, fixed, trspt
    
    
def gemsDBextraction(centralDict):
    print("Extracting xGEMS database", end=" ", flush=True)
    # absolute paths, no chdir: a relative chemPath ('gems/x-dat.lst', 'x-dat.lst') is read from the run folder
    taskPath = os.path.abspath(str(centralDict['chemPath']))
    folder = os.path.dirname(taskPath)

    with open(taskPath, 'r', encoding='utf-8', errors='ignore') as f:
        task_content = f.read()

    file_list = re.findall(r'["\']([^"\']+)["\']', task_content)
    if not file_list:
        raise FileNotFoundError(f"No file found in: {taskPath}")

    dch_candidates = [fn for fn in file_list if 'dch' in fn.lower()]
    dch_filename = os.path.join(folder, dch_candidates[0] if dch_candidates else file_list[0])

    if not os.path.isfile(dch_filename):
        raise FileNotFoundError(f"Cannot find 'dch' file: {dch_filename}")

    with open(dch_filename, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    centralDict['gemsGrid'] = _dchGrid(content, os.path.basename(dch_filename))

    if dch_filename.lower().endswith('.json'):
        icnl_list, dcnl_list, ccdc_list, A_matrix = _parse_dch_json(content)
    elif dch_filename.lower().endswith('.dat'):
        icnl_list, dcnl_list, ccdc_list, A_matrix = _parse_dch_dat(content)
    else:
        raise ValueError(f"I can only parse '.json' and '.dat' files, and not '{dch_filename}' extension, sorry. Please convert.")

    assert len(dcnl_list) == len(A_matrix), "Problem in stoechiometry matrix ? Number of DC not equal to the number of lines in matrix A."
    assert all(len(row) == len(icnl_list) for row in A_matrix), "Problem in stoechiometry matrix ? Number of IC not equal to the number of columns in matrix A."
    assert len(dcnl_list) == len(ccdc_list), "Number of DC not equal to number of ccDC."

    stoichio_dict = {}
    for dc_name, row in zip(dcnl_list, A_matrix):
        stoichio_dict[dc_name] = {ic: v for ic, v in zip(icnl_list, row) if v != 0}

    valid_classes = {'S', 'T', 'W', 'G', 'O'}
    dc_class_dict = {
        dc_name: code
        for dc_name, code in zip(dcnl_list, ccdc_list)
        if code in valid_classes
    }
    species_by_class = {c: [] for c in valid_classes}
    for dc_name, code in dc_class_dict.items():
        species_by_class[code].append(dc_name)

    centralDict.update({
        'transportedSpecies': species_by_class['S'] + species_by_class['W'] + species_by_class['T'],
        # MultiCompoundTransport: totals of the mobile species (aqueous, water), immobile species kept as they are
        # (solids, gases)
        'mobileSpecies': species_by_class['S'] + species_by_class['W'] + species_by_class['T'],
        'independentComponents': icnl_list,
        # species of the GEMS system; coupledVariables stays the list of the coupled variables (crossDependencies included)
        'gemsSpecies': dcnl_list,
        'primToSecSpecies': stoichio_dict,
        'dcClass': dc_class_dict,
        'speciesByClass': species_by_class,
        'fixedSpecies': species_by_class['O'] + species_by_class['G'],
    })

    return centralDict


def _dchGrid(content, name):
    """Temperature (K) and pressure (Pa) range of the thermodynamic data of a DCH file (TKval, Pval grids and their
    tolerances Ttol, Ptol): {'T': (lo, hi), 'P': (lo, hi), 'file': name}, open bounds as GEMS3K checks them (a value
    on a bound is 'beyond the interpolation range'); None if the file does not give them."""
    if name.lower().endswith('.json'):
        try:
            data = json.loads(content)
        except ValueError:
            return None
        def find(obj, key):
            if isinstance(obj, dict):
                if key in obj:
                    return obj[key]
                return next((v for v in (find(o, key) for o in obj.values()) if v is not None), None)
            if isinstance(obj, list):
                return next((v for v in (find(o, key) for o in obj) if v is not None), None)
            return None
        raw = {k: find(data, k) for k in ('TKval', 'Pval', 'Ttol', 'Ptol')}
        values = {k: ([float(x) for x in v] if isinstance(v, list) else [float(v)]) if v is not None else None
                  for k, v in raw.items()}
    else:
        number = re.compile(r'-?\d+(\.\d*)?([eE][-+]?\d+)?$')
        def numbers(tag):
            m = re.search(r'<' + tag + r'>(.*?)(?=<|\Z)', content, re.DOTALL)
            if not m:
                return None
            found = [float(t) for line in m.group(1).splitlines() if not line.strip().startswith('#')
                     for t in line.split() if number.match(t)]
            return found or None
        values = {k: numbers(k) for k in ('TKval', 'Pval', 'Ttol', 'Ptol')}
    if any(v is None for v in values.values()):
        return None
    return {'T': (min(values['TKval']) - values['Ttol'][0], max(values['TKval']) + values['Ttol'][0]),
            'P': (min(values['Pval']) - values['Ptol'][0], max(values['Pval']) + values['Ptol'][0]),
            'TKval': values['TKval'], 'Pval': values['Pval'], 'file': name}


def _parse_dch_dat(content):
    def extract_list(tag):
        pattern = re.compile(r'<' + re.escape(tag) + r'>(.*?)<[^>]+>', re.DOTALL)
        m = pattern.search(content)
        return re.findall(r"'([^']+)'", m.group(1)) if m else None

    def extract_matrix(tag):
        pattern = re.compile(r'<' + re.escape(tag) + r'>(.*?)<[^>]+>', re.DOTALL)
        m = pattern.search(content)
        if not m:
            return None
        matrix = []
        for line in m.group(1).strip().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            tokens = line.split()
            if all(re.fullmatch(r'-?\d+(\.\d+)?([eE][-+]?\d+)?', t) for t in tokens):
                # fractional coefficients kept (C-S-H end-members: 0.6667 Si ...), whole ones as int
                matrix.append([int(v) if v.is_integer() else v for v in map(float, tokens)])
        return matrix

    icnl_list = extract_list('ICNL')
    dcnl_list = extract_list('DCNL')
    ccdc_list = extract_list('ccDC')
    A_matrix = extract_matrix('A')
    return icnl_list, dcnl_list, ccdc_list, A_matrix


def _parse_dch_json(content):
    data = json.loads(content)
    if isinstance(data, list):
        data = data[0]
    dch = data.get('dch', data)

    icnl_list = dch['ICNL']
    dcnl_list = dch['DCNL']
    ccdc_list = dch['ccDC']

    nIC = dch['nIC']
    nDC = dch['nDC']
    A_flat = dch['A']
    A_matrix = [
        [float(A_flat[i * nIC + j]) for j in range(nIC)]
        for i in range(nDC)
    ]
    return icnl_list, dcnl_list, ccdc_list, A_matrix

def OrchestraDBextraction(centralDict):
    print("Extracting ORCHESTRA database", end=" ", flush=True)

    from . import parse_reactions 

    reactions  = parse_reactions.build_reaction_dict(centralDict['chemPath'])
    components = parse_reactions.build_components(centralDict['chemPath'])
    stoich     = parse_reactions.build_stoich_matrix(
                     reactions, components, centralDict['systemSpecies'],
                     filepath=centralDict['chemPath'])

    stoichReduced = parse_reactions.align_stoich(
    stoich,
    centralDict['systemSpecies'],
    centralDict['primarySpecies'][-1]) 

    centralDict.update({
        "stoichReduced" : stoichReduced,
        'reactions': reactions,
        "primary_entities": components,
        "primToSecSpecies" : stoich , })
    return centralDict


def checkDecomposition(centralDict):

    known = (set(centralDict['primarySpecies']['total'])
             | set(centralDict['primarySpecies']['primarySpeciesPhantom']))

    orphans = {sp: [p for p in comp if p not in known]
               for sp, comp in centralDict['primToSecSpecies'].items()
               if any(p not in known for p in comp)}

    if orphans:
        raise ValueError(
            "Secondary species cannot be broken down in primary species :\n"
            + "\n".join(f"  {sp} -> {k}" for sp, k in orphans.items())
            + f"\nPlease check your secondary species in {centralDict['chemPath']}."
        )

    return centralDict


def extract(centralDict):
    startExtract = time.perf_counter()
    
    if centralDict['couplingInfo'][1] == 'PhreeqC':
        print("Extracting PhreeqC database", end=" ", flush=True)
        primToSecSpecies, sol, phases, surf, exch, fixed, trspt = phreeqcDBextraction(centralDict,centralDict['commMtrx'][centralDict['systemSpecies']])

    elif centralDict['couplingInfo'][1] == 'ORCHESTRA':
        centralDict.update(OrchestraDBextraction(centralDict))

    elif centralDict['couplingInfo'][1] == 'xGEMS':
        centralDict.update(gemsDBextraction(centralDict))
    
    
    if centralDict['couplingInfo'][1] == 'PhreeqC':

        
        centralDict.update({
                            "primToSecSpecies" : primToSecSpecies,
                            "transportedSpecies" : trspt, 
                            "primarySpecies" : {
                            'solution' : list(dict.fromkeys(sol+['H2O'])),
                            'phases' : phases,
                            'surface': surf,
                            'exchange': exch,
                            'total' :(list(dict.fromkeys(sol+['H2O']))+exch+surf+phases),
                            'primarySpeciesPhantom' : ['pH','ph','pe'],
                            }})    

        checkDecomposition(centralDict)

    
    centralDict["extractDBTime"] = time.perf_counter() - startExtract

    print(f"({writeTime((time.perf_counter() - startExtract))})") 
    return centralDict