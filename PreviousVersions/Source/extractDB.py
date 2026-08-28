import re
import sys
import pandas as pd
import time
import numpy as np
import concurrent.futures
import os
from charset_normalizer import from_path
import glob


def writeTime(tps, arr=2):
    if tps >= 3600 * 24:
        return f"{tps / (3600 * 24):.{arr}f} d"
    elif tps >= 3600:
        return f"{tps / 3600:.{arr}f} h"
    elif tps >= 60:
        return f"{tps / 60:.{arr}f} min"
    else:
        return f"{tps:.{arr}f} sec"

def readInputFile(txtPath, coord,  inputHeaders = None) :
    dataframe = pd.read_csv(txtPath,sep=r"\s+",comment="%",dtype=float)
    if len(dataframe.columns) != len(coord + inputHeaders):
        print(f"input file : {len(dataframe.columns)} columns")
        print(f"coupled variables : {len(coord + inputHeaders)}")
        print(coord +inputHeaders)
        sys.exit()
    if (coord + inputHeaders) != dataframe.columns.tolist():
        dataframe = pd.read_csv(txtPath,sep=r"\s+",comment="%",dtype=float,names=coord + inputHeaders)
    return dataframe
    
# Decompose secundary species into primary species (e.g. Na2S2O3 into 2Na, 2S and 3O)
def decomposingIntoPrimSpecies(formula, primarySpecies,redoxPrim,redox_dict):
    def multiply_dict(d, factor): 
        return {k: v * factor for k, v in d.items()}
    def merge_dicts(a, b): 
        for k, v in b.items():
            a[k] = a.get(k, 0) + v
        return a

    charge_match = re.search(r'([+-]\d*)$', formula)
    if charge_match:
        charge = charge_match.group(1)
        formula = formula[:charge_match.start()]
    else:
        charge = None

    formula = re.sub(r'__([0-9]+)', r')\1', formula) 

    primaryspecies_trie = sorted(primarySpecies, key=len, reverse=True)
    pattern = re.compile('|'.join(re.escape(ps) for ps in primaryspecies_trie)) 

    index = 0
    tokens = []
    
    
    
    while index < len(formula): 
        m = pattern.match(formula, index)
        if m:
            name = m.group(0)
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

        index += 1
    stack = []
    current = {}

    gotcha = False
    for token in tokens:

        gotcha = False
        if token[0] == 'master':
            if charge : species = formula + charge
            else: species = formula

            if redox_dict and token[1] in redox_dict:
                for key in redox_dict[token[1]]:
                    if key in redoxPrim:
                        for sec in redoxPrim[key]:
                            if species == sec:
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


def extract_master_redox_map(filepath, redox_dict):
    """
    Lit SOLUTION_MASTER_SPECIES et retourne un mapping:
    { 'S(+4)': 'SO3-2', 'S(+6)': 'SO4-2', 'Fe(+3)': 'Fe+3', ... }
    """
    redox_map = {}
    in_block = False

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            upper = stripped.upper()

            if "SOLUTION_MASTER_SPECIES" in upper:
                in_block = True
                continue
            if in_block and any(kw in upper for kw in (
                "SOLUTION_SPECIES", "PHASES", "SURFACE", "EXCHANGE",
                "REACTION", "KINETICS", "RATES", "END"
            )):
                break

            if not in_block:
                continue

            parts = stripped.split()
            if len(parts) < 2:
                continue

            state = parts[0]   
            master = parts[1]  
            for c in state:
                if not c.isdigit() : continue
            
            for base, states in redox_dict.items():
                if state in states:
                    redox_map[state] = master
                    break

    return redox_map

def _charge_variants(species):
    """
    Retourne les notations alternatives d'une espèce chargée, pour gérer le fait
    que PHREEQC mélange parfois la notation numérique (Fe+2, CO3-2) et la notation
    répétée (Fe++, CO3--) selon les bases de données.
    """
    variants = {species}
    m = re.match(r'^(.+?)([+-]\d+|[+-]+)$', species)
    if not m:
        return variants
    base, charge = m.groups()
    sign = charge[0]
    n = int(charge[1:]) if charge[1:].isdigit() else len(charge)
    variants.add(f"{base}{sign * n}") 
    variants.add(f"{base}{sign}{n}")  
    return variants


def _matches_master(master, text):
    """Vrai si master (ou une variante de notation de charge) apparaît dans text."""
    return any(v in text for v in _charge_variants(master))


def extract_secondary_species(filepath, redox_dict, redox_map):
    """
    Extrait, pour chaque état redox de redox_map, les espèces secondaires
    correspondantes dans une base PHREEQC.

    Blocs pris en compte :
      - SOLUTION_SPECIES / SURFACE_SPECIES / EXCHANGE_SPECIES
        (réactions "maître = secondaire ...")
      - PHASES
        (bloc du type :
            NomDePhase
                formule + ... = produits...
                log_k     ...
                delta_h   ...
         -> le NOM DE LA PHASE est ajouté comme "espèce secondaire" du/des
            état(s) redox dont l'espèce maître apparaît dans la réaction,
            ex: secondary["Fe(+2)"] contiendra "Siderite" si la réaction
            produit Fe+2)
    """
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

    block_type = None    
    current_phase_name = None
    result = from_path(filepath).best()
    
    with open(filepath, "r", encoding=result.encoding) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            upper = stripped.upper()
            first_token = upper.split()[0]

            if any(kw in upper for kw in species_block_keywords):
                block_type = "species"
                current_phase_name = None
                continue

            if first_token == phases_block_keyword:
                block_type = "phases"
                current_phase_name = None
                continue

            if block_type is not None and any(kw in upper for kw in closing_only_keywords):
                block_type = None
                current_phase_name = None
                continue

            if block_type is None:
                continue

            if block_type == "species":
                if any(stripped.lower().startswith(kw) for kw in param_prefixes):
                    continue
                if "=" not in stripped:
                    continue

                lhs, rhs = stripped.split("=", 1)
                rhs_species = rhs.strip().split()[0]

                for state, master in redox_map.items():
                    base = state[:state.index("(")] if "(" in state else state

                    if not _matches_master(master, lhs):
                        continue
                    if base not in rhs_species:
                        continue

                    other_masters = [m for s, m in redox_map.items() if s != state and s.startswith(base)]
                    if rhs_species in other_masters or rhs_species == master:
                        continue

                    add_secondary(state, rhs_species)

            elif block_type == "phases":
                if any(stripped.lower().startswith(kw) for kw in param_prefixes):
                    continue

                if "=" not in stripped:
                    current_phase_name = stripped.split()[0]
                    continue

                if current_phase_name is None:
                    continue

                for state, master in redox_map.items():
                    if not _matches_master(master, stripped):
                        continue
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
    """
    Vrai seulement si upper_line EST un mot-clé de bloc (avec ses éventuels
    arguments après un espace), pas si le mot-clé apparaît comme simple
    sous-chaîne (ex: 'SIT' dans 'HALLOYSITE').
    """
    first_token = upper_line.split()[0] if upper_line.split() else ""
    return first_token in keywords  


def _strip_coeff(term):
    """'2X_aH' -> 'X_aH', '3H2O' -> 'H2O', 'Sr+2' -> 'Sr+2' (inchangé)"""
    m = re.match(r'^[0-9]*\.?[0-9]+(?=[A-Za-z(])', term)
    return term[m.end():] if m else term


def _split_terms(side):
    """Découpe un membre de réaction PHREEQC en termes, en respectant les charges collées (Sr+2, 2H+)."""
    terms = []
    for raw in re.split(r'\s+\+\s+', side.strip()):
        tok = raw.strip()
        if not tok:
            continue
        terms.append(_strip_coeff(tok))
    return terms

def extract_master_species(filepath, encoding="utf-8"):
    masterSpecies = {
        'SOLUTION_MASTER_SPECIES': [],
        'SOLUTION_MASTER_ELEMENT': [],
        'SURFACE_MASTER_SPECIES': [],
        'EXCHANGE_SPECIES': [],
        'EXCHANGE_MASTER_SPECIES': [],
        'PHASES': [],
        'SURFACE_SPECIES': [],
    }
    stop_keywords = {
        "SOLUTION_SPECIES", 'SOLUTION_MASTER_SPECIES', "SURFACE_SPECIES",
        'SURFACE_MASTER_SPECIES', "EXCHANGE_MASTER_SPECIES", "EXCHANGE_SPECIES",
        "EQUILIBRIUM_PHASES", 'PHASES', "REACTION", "KINETICS", "RATES", "END", 'SIT'
    }

    def _is_surface_species(sp, surface_master_prefixes):
        """Vrai si sp contient l'un des noms de site de SURFACE_MASTER_SPECIES (ex: 'X_a' dans 'CaX_a2')."""
        return any(prefix in sp for prefix in surface_master_prefixes)


    current_block = None
    with open(filepath, "r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
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

    current_block = None
    pending_phase_name = None
    with open(filepath, "r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
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
                continue  # déjà traité au passage 1

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


def phreeqcDBextraction(centralDict, commMtrx):
    
    totPrimSpecies = extract_master_species(centralDict['chemPath'])

    totPrimSpecies['SURFACE_SPECIES'] = [
    s for s in totPrimSpecies['SURFACE_SPECIES']
    if s not in totPrimSpecies['SOLUTION_MASTER_ELEMENT']
]
    
    redox_dict = group_redox_states((totPrimSpecies['SOLUTION_MASTER_SPECIES']))
    redox_map = extract_master_redox_map(centralDict['chemPath'], redox_dict)

    
    rmv = ['H','H+','H(0)','H(+1)','O(0)','O','O(-2)'] # phreeqc limitations ..
    for cle in rmv:
        redox_map.pop(cle, None) 
    if 'H' in redox_dict:
        del redox_dict['H']
    if 'O' in redox_dict:
        del redox_dict['O']

    
    secondary = extract_secondary_species(centralDict['chemPath'], redox_dict, redox_map)

    sol = []
    phases = []
    surf = []
    exch = []

    
    primToSecSpecies = {}
    for _, row in commMtrx.iterrows():
        for comp, conc in row.items():
            if isinstance(conc, str):
                print("\nTypeError: '<' not supported between instances of 'str' and 'int'")
                print(conc,comp)
                sys.exit()
            composition, _ = decomposingIntoPrimSpecies(comp, (totPrimSpecies['SOLUTION_MASTER_SPECIES']+
            totPrimSpecies['SURFACE_MASTER_SPECIES']+totPrimSpecies['EXCHANGE_SPECIES']+totPrimSpecies['PHASES']), secondary, redox_dict)
            primToSecSpecies.update({comp : composition})
    

    inverse_redox = {species: element for element, species in redox_map.items()}
    for species, composition in primToSecSpecies.items():
        if species in inverse_redox:
            redox_name = inverse_redox[species]      # ex. "N(+3)"
            element = redox_name.split('(')[0]       # ex. "N"
    
            if element in composition and redox_name not in composition:
                composition[redox_name] = composition.pop(element)
    
                   
    # we delete the 'base' species of redox species 
    solutionMaster = [s for s in totPrimSpecies['SOLUTION_MASTER_SPECIES']
    if not any(t.startswith(s + "(") for t in totPrimSpecies['SOLUTION_MASTER_SPECIES'])]
    surfaceMaster = [s for s in totPrimSpecies['SURFACE_MASTER_SPECIES']
    if not any(t.startswith(s + "(") for t in totPrimSpecies['SURFACE_MASTER_SPECIES'])]
    if centralDict['preliminarEquilibrium']:
        exchangeMaster = [s for s in totPrimSpecies['EXCHANGE_MASTER_SPECIES']
        if not any(t.startswith(s + "(") for t in totPrimSpecies['EXCHANGE_MASTER_SPECIES'])]
    else:
        exchangeMaster = [s for s in totPrimSpecies['EXCHANGE_SPECIES']
        if not any(t.startswith(s + "(") for t in totPrimSpecies['EXCHANGE_SPECIES'])]
    phaseMaster = [s for s in totPrimSpecies['PHASES']
    if not any(t.startswith(s + "(") for t in totPrimSpecies['PHASES'])]


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

    fixed = totPrimSpecies['SURFACE_MASTER_SPECIES']+totPrimSpecies['SURFACE_SPECIES']+totPrimSpecies['EXCHANGE_MASTER_SPECIES']+totPrimSpecies['EXCHANGE_SPECIES']+totPrimSpecies['PHASES']
    trspt = [s for s in centralDict['systemSpeciation'] if s not in fixed]

    if centralDict['nonTrivialDecomposition']:
        for ky in primToSecSpecies:
            for subky in centralDict['nonTrivialDecomposition']:
                if subky == ky:
                      primToSecSpecies[ky] = centralDict['nonTrivialDecomposition'][ky]

    
    primToSecSpecies['H2O'] = {'H2O':1}

    return primToSecSpecies, sol, phases, surf, exch, fixed, trspt
    
    
def gemsDBextraction(centralDict):
    print("Extracting xGEMS database", end=" ", flush=True)
    folder = os.path.dirname(centralDict['chemPath'])
    os.chdir(folder)

    matches = glob.glob("*dch.dat")
    if not matches:
        raise FileNotFoundError(f"Aucun fichier '*dch.dat' trouvé dans {folder}")
    filename = matches[0]

    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    def extract_list(tag, content):
        pattern = re.compile(r'<' + re.escape(tag) + r'>(.*?)<[^>]+>', re.DOTALL)
        m = pattern.search(content)
        return re.findall(r"'([^']+)'", m.group(1)) if m else None

    def extract_matrix(tag, content):
        pattern = re.compile(r'<' + re.escape(tag) + r'>(.*?)<[^>]+>', re.DOTALL)
        m = pattern.search(content)
        if not m:
            return None
        matrix = []
        for line in m.group(1).strip().splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                continue
            tokens = line.split()
            if all(re.fullmatch(r'-?\d+(\.\d+)?([eE][-+]?\d+)?', t) for t in tokens):
                matrix.append([int(float(t)) for t in tokens])
        return matrix

    icnl_list = extract_list('ICNL', content)
    dcnl_list = extract_list('DCNL', content)
    ccdc_list = extract_list('ccDC', content)
    A_matrix = extract_matrix('A', content)

    assert len(dcnl_list) == len(A_matrix), "Nb DC != Nb lignes de A"
    assert all(len(row) == len(icnl_list) for row in A_matrix), "Nb IC != Nb colonnes de A"
    assert len(dcnl_list) == len(ccdc_list), "Nb DC != Nb codes ccDC"

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
        'transportedSpecies': species_by_class['S']+species_by_class['W']+species_by_class['T'],
        'independentComponents': icnl_list,
        'systemSpeciation': dcnl_list,
        'primToSecSpecies': stoichio_dict,
        'dcClass': dc_class_dict,        
        'speciesByClass': species_by_class, 
        'fixedSpecies' : species_by_class['O'],
    })

    return centralDict

def OrchestraDBextraction(centralDict):
    print("Extracting ORCHESTRA database", end=" ", flush=True)
    
    
    primToSecSpecies = {}
    for _, row in centralDict['commMtrx'][centralDict['systemSpeciation']].iterrows():
        for comp, conc in row.items():
            if isinstance(conc, str):
                print("\nTypeError: '<' not supported between instances of 'str' and 'int'")
                print(conc,comp)
                sys.exit()
            composition, _ = decomposingIntoPrimSpecies(comp, centralDict['primarySpecies'][-1], None,None)
            primToSecSpecies.update({comp : composition})

    centralDict.update({"primToSecSpecies" : primToSecSpecies,})
    return centralDict


def extract(centralDict):
    startExtract = time.time()
    
    # calling db functions shall be generic ...
    if centralDict['PIDextract'] > 1 and centralDict['couplingInfo'][1] == 'PhreeqC':
        print("Extracting PhreeqC database", end=" ", flush=True)
        chunk_size = int(np.ceil(len(centralDict['commMtrx']) / centralDict['PIDextract']))

        commMtrxSplit = [centralDict['commMtrx'][centralDict['systemSpeciation']].iloc[i:i + chunk_size] for i in range(0, len(centralDict['commMtrx']), chunk_size)]
    
        with concurrent.futures.ProcessPoolExecutor(max_workers=centralDict['PIDnbr']) as executor:
            futures = []
            for i,chunk in enumerate(commMtrxSplit):
                futures.append(executor.submit(phreeqcDBextraction, centralDict, chunk ))
                
                
        results = []
        results = [f.result() for f in futures]
    
        primToSecSpecieS, soL, phaseS, surF, excH, fixeD, trspT = zip(*results)
        
        sol = list(set(item for sublist in soL for item in sublist))
        phases = list(set(item for sublist in phaseS for item in sublist))
        surf = list(set(item for sublist in surF for item in sublist))
        exch = list(set(item for sublist in excH for item in sublist))
        trspt = list(set(item for sublist in trspT for item in sublist))
        fixed = fixeD[0]
        primToSecSpecies = primToSecSpecieS[0] # they are all equivalent


    elif centralDict['couplingInfo'][1] == 'PhreeqC':
        print("Extracting PhreeqC database", end=" ", flush=True)
        primToSecSpecies, sol, phases, surf, exch, fixed, trspt = phreeqcDBextraction(centralDict,centralDict['commMtrx'][centralDict['systemSpeciation']])

    elif centralDict['couplingInfo'][1] == 'ORCHESTRA':
        centralDict.update(OrchestraDBextraction(centralDict))

    elif centralDict['couplingInfo'][1] == 'xGEMS':
        centralDict.update(gemsDBextraction(centralDict))
    
    
    if centralDict['couplingInfo'][1] == 'PhreeqC':

        
        centralDict.update({
                            "primToSecSpecies" : primToSecSpecies ,
                            "transportedSpecies" : trspt, 
                            "primarySpecies" : {
                            'solution' : list(set(sol+['H2O'])),
                            'phases' : phases,
                            'surface': surf , 
                            'exchange': exch, 
                            'total' :(list(set(sol+['H2O']))+exch+surf+phases),
                            'primarySpeciesPhantom' : ['pH','ph','pe'],
                            }})    

    
    centralDict["extractDBTime"] = time.time() - startExtract

    print(f"({writeTime((time.time() - startExtract))})") 
    return centralDict

      