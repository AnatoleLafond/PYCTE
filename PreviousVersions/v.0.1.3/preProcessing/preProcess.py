import sys
import mph

'''
This script will set comsol accordingly to the PICCTS coupling requirements
Initial comsol file must only contain the geometry (no study, no data, etc., see javaformatting.mph)
'''

import re

from pathlib import Path
import chardet



outputFiles = [r'clay_DRO.pqo',r'source_DRO.pqo']
outputFilesBoundary =  [r'clay_DRO.pqo']
porosity = [0.24]

Comsolpath = r"DRO.mph"
geometry = 1 # 1 for 1D, 2 for 2D etc
interface = "npe" # tds, npe, tdsPorous


# 1 .pqo per domain
# 1 porosity per domain
# as much boundary condition as possible


def get_last_block(contenu, bloc_header_pattern):
    """
    You know that PhreeqC may return several outputs within one output file (.pqo).
    here we only take the last one. Depending on you system this may not be the most appropriate approach, but it is the most convienent one ..
    
    """

    matches = list(re.finditer(bloc_header_pattern, contenu, re.IGNORECASE))
    if not matches:
        return None
    last_match = matches[-1] # change here if you want to get the ith occurence of the output block
    return contenu[last_match.start():]


def parse_phreeqc_output(file):
    """
    It will parse a phreeqc output block
    """
    encoding = chardet.detect(Path(file).read_bytes())['encoding'] or 'utf-8'
    with open(file, "r", encoding=encoding) as f:
        contenu = f.read()

    sim_markers = list(re.finditer(
        r'-+\s*(?:Beginning of|BEGINNING OF)\s+(?:run|RUN|simulation|SIMULATION).*?-+',
        contenu, re.IGNORECASE
    ))
    # print(sim_markers)
    if sim_markers:

        last_block = contenu[sim_markers[-1].start():]
        # print(last_block[:500])
        # print(last_block)
        # sys.exit()
    else:
        # Nouveau : on essaie d'abord de repartir du début de la dernière étape
        step_matches = list(re.finditer(r'Reaction step \d+\.', contenu, re.IGNORECASE))
        if not step_matches:
            step_matches = list(re.finditer(
                r'-+\s*Beginning of batch-reaction calculations\.?\s*-+',
                contenu, re.IGNORECASE
            ))
    
        if step_matches:
            last_block = contenu[step_matches[-1].start():]
        else:
            dist_matches = list(re.finditer(
                r'-+\s*Distribution of species\s*-+',
                contenu, re.IGNORECASE
            ))
            if dist_matches:
                search_start = dist_matches[-1].start()
                phase_matches = list(re.finditer(
                    r'(?:^[ \t]*[A-Za-z][\w()\-]+[ \t]+[-\d.]+[ \t]+[-\d.]+[ \t]+[-\d.]+[ \t]+[\d.eE+\-]+[ \t]+[\d.eE+\-]+[ \t]+[-\d.eE+\-]+[ \t]*\n)+',
                    contenu[:search_start], re.MULTILINE
                ))
                if phase_matches:
                    last_block = contenu[phase_matches[-1].start():]
                else:
                    last_block = contenu[dist_matches[-1].start():]
            else:
                last_block = contenu
    # print(last_block)
    # sys.exit()
    result = {
        'phases'  : {},
        'surface' : {},
        'solution': {},
        'species' : {}
    }

    # phases
    phase_lines = re.finditer(
        r'^[ \t]*([A-Za-z][\w()\-]+)[ \t]+([-\d.]+)[ \t]+([-\d.]+)[ \t]+([-\d.]+)[ \t]+([\d.eE+\-]+)[ \t]+([\d.eE+\-]+)[ \t]+([-\d.eE+\-]+)[ \t]*$',
        last_block, re.MULTILINE
    )
    for m in phase_lines:
        name, si_t, log_iap, log_k, mol_i, mol_f, delta = m.groups()
        result['phases'][name] = {
            'si_target'  : float(si_t),
            'log_iap'    : float(log_iap),
            'log_k'      : float(log_k),
            'moles_init' : float(mol_i),
            'moles_final': float(mol_f),
            'delta_moles': float(delta)
        }

    # surface
    surface_block = re.search(
        r'Surface composition.*?\n(.*?)(?=-{20,}Solution composition)',
        last_block, re.DOTALL | re.IGNORECASE
    )
    if surface_block:
        block = surface_block.group(1)
        for site_m in re.finditer(
            r'^(\w+_\w+)\s*\n\s+([\d.eE+\-]+)\s+moles',
            block, re.MULTILINE
        ):
            site_name, moles = site_m.group(1), float(site_m.group(2))
            site_start = site_m.end()
            next_site  = re.search(r'^\w+_\w+\s*\n', block[site_start:], re.MULTILINE)
            site_end   = site_start + next_site.start() if next_site else len(block)
            site_block = block[site_start:site_end]

            species = {}
            for sm in re.finditer(
                r'^\t(\S+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([-\d.eE+]+)',
                site_block, re.MULTILINE
            ):
                sp, mol, frac, molal, logm = sm.groups()
                species[sp] = {
                    'moles': float(mol), 'mole_fraction': float(frac),
                    'molality': float(molal), 'log_molality': float(logm)
                }
            result['surface'][site_name] = {'moles': moles, 'species': species}

    # solution
    sol_block = re.search(
        r'Solution composition.*?\n\s+Elements\s+Molality\s+Moles\s*\n(.*?)(?=\n-{10,})',
        last_block, re.DOTALL | re.IGNORECASE
    )
    if sol_block:
        for line in sol_block.group(1).strip().split('\n'):
            parts = line.split()
            if len(parts) == 3:
                try:
                    result['solution'][parts[0]] = {
                        'molality': float(parts[1]),
                        'moles'   : float(parts[2])
                    }
                except ValueError:
                    pass

    # solution but sec. species
    dist_block = re.search(
        r'Distribution of species.*?\n.*?\n.*?\n(.*?)(?=\n-{20,}[A-Z]|\Z)',
        last_block, re.DOTALL | re.IGNORECASE
    )
    if dist_block:
        for m in re.finditer(
            r'^\s{3,}(\S+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)',
            dist_block.group(1), re.MULTILINE
        ):
            sp, molal, act, lm, la, gam = m.groups()
            result['species'][sp] = {
                'molality'    : float(molal),
                'activity'    : float(act),
                'log_molality': float(lm),
                'log_activity': float(la),
                'gamma'       : float(gam)
            }

    return result

# outputFiles = ['eausynthétiqueCOX.pqo','eausynthétiqueCOX.pqo','eausynthétiqueCOX.pqo','source.pqo']
# outputFilesBoundary = ['eausynthétiqueCOX.pqo','source.pqo']




outputData = {'domain' : { f'domain{i}' : {'systemSpeciation' : [],
                              'transportedSpecies' : [],
                              'fixedSpecies' : [],
                              'solutionSpecies' : [],
                              'phasesSpecies' : [],
                              'surfSpecies' : [],
                              'exchSpecies' : [],
                              'solutionMolality' : {},
                              'phaseMolality' : {},
                              'surfaceMolality' : {},
                              'exchangeMolality' : {},} for i in range(len(outputFiles))},
              'boundary' : { f'boundary{i}' : {'systemSpeciation' : [],
                                            'transportedSpecies' : [],
                                            'fixedSpecies' : [],
                                            'solutionSpecies' : [],
                                            'phasesSpecies' : [],
                                            'surfSpecies' : [],
                                            'exchSpecies' : [],
                                            'solutionMolality' : {},
                                            'phaseMolality' : {},
                                            'surfaceMolality' : {},
                                            'exchangeMolality' : {},} for i in range(len(outputFilesBoundary))},
              }

if outputFilesBoundary:
    for it, output in enumerate(outputFilesBoundary):
        data = parse_phreeqc_output(output)
    
        for x, contenu in data['surface'].items():
            for sp in contenu['species']:
    
                outputData['boundary'][f'boundary{it}']['systemSpeciation'] += [sp]
                outputData['boundary'][f'boundary{it}']['surfSpecies'] += [sp]
                outputData['boundary'][f'boundary{it}']['fixedSpecies'] += [sp]
                outputData['boundary'][f'boundary{it}']['surfaceMolality'].update({sp : contenu['species'][sp]['molality'] })
        
        for sp in data['phases'].items():
            outputData['boundary'][f'boundary{it}']['systemSpeciation'] += [sp[0]]
            outputData['boundary'][f'boundary{it}']['fixedSpecies'] += [sp[0]]
            outputData['boundary'][f'boundary{it}']['phasesSpecies'] += [sp[0]]
            outputData['boundary'][f'boundary{it}']['phaseMolality'].update({sp[0] : sp[1]['moles_final']})
    
        for sp, v in list(data['species'].items()):
            outputData['boundary'][f'boundary{it}']['systemSpeciation'] += [sp]
            outputData['boundary'][f'boundary{it}']['solutionSpecies'] += [sp]
            
            outputData['boundary'][f'boundary{it}']['solutionMolality'].update({sp : v['molality'] })
            
        for sp in outputData['boundary'][f'boundary{it}']['systemSpeciation']:
            if sp not in outputData['boundary'][f'boundary{it}']['fixedSpecies']:
                outputData['boundary'][f'boundary{it}']['transportedSpecies'] += [sp]

for it, output in enumerate(outputFiles):
    data = parse_phreeqc_output(output)

    for x, contenu in data['surface'].items():
        for sp in contenu['species']:

            outputData['domain'][f'domain{it}']['systemSpeciation'] += [sp]
            outputData['domain'][f'domain{it}']['surfSpecies'] += [sp]
            outputData['domain'][f'domain{it}']['fixedSpecies'] += [sp]
            outputData['domain'][f'domain{it}']['surfaceMolality'].update({sp : contenu['species'][sp]['molality'] })
    
    for sp in data['phases'].items():
        outputData['domain'][f'domain{it}']['systemSpeciation'] += [sp[0]]
        outputData['domain'][f'domain{it}']['fixedSpecies'] += [sp[0]]
        outputData['domain'][f'domain{it}']['phasesSpecies'] += [sp[0]]
        outputData['domain'][f'domain{it}']['phaseMolality'].update({sp[0] : sp[1]['moles_final']})

    for sp, v in list(data['species'].items()):
        outputData['domain'][f'domain{it}']['systemSpeciation'] += [sp]
        outputData['domain'][f'domain{it}']['solutionSpecies'] += [sp]
        
        outputData['domain'][f'domain{it}']['solutionMolality'].update({sp : v['molality'] })
        
    for sp in outputData['domain'][f'domain{it}']['systemSpeciation']:
        if sp not in outputData['domain'][f'domain{it}']['fixedSpecies']:
            outputData['domain'][f'domain{it}']['transportedSpecies'] += [sp]

tot = []
fixed = []
tr = []
for domain ,data in outputData['domain'].items():
    for sp in data['systemSpeciation']:
        if sp not in tot: tot += [sp]
    for sp in data['fixedSpecies']:
        if sp not in fixed: fixed += [sp]

for domain ,data in outputData['boundary'].items():
    for sp in data['systemSpeciation']:
        if sp not in tot: tot += [sp]
    for sp in data['fixedSpecies']:
        if sp not in fixed: fixed += [sp]    


for sp in tot:
    if sp not in fixed:
        tr += [sp]
        
outputData.update({'totals': {'systemSpeciationTotal' : tot,
                   'fixedSpeciesTotal' : fixed,
                   'transportedSpeciesTotal' : tr}})



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
   
# assuming phase, surface and exchange species as fixed species
totalspc = Convert_Species_PhreeqC_to_COMSOL(outputData['totals']['systemSpeciationTotal'])
spcTransported = Convert_Species_PhreeqC_to_COMSOL(outputData['totals']['transportedSpeciesTotal'])
spcFixed = Convert_Species_PhreeqC_to_COMSOL(outputData['totals']['fixedSpeciesTotal'])


interf = {'npe': ['npe','NernstPlanck'],
          'tds': ['tds','DilutedSpecies'],
          'tdsPorous': ['tds','DilutedSpeciesInPorousMedia'],}


separateur = "\t"
ligne_header =  separateur.join(['x', 'y', 'z'][:geometry] + totalspc)
ligne_valeurs = separateur.join(["0"] * len(totalspc+['x','y','z'][:geometry]) )

with open("int.txt", "w") as f:
    f.write("%" + ligne_header + "\n")
    f.write(ligne_valeurs + "\n")


client = mph.start()
model = client.load(Comsolpath)
javamodel = model.java

javamodel.component("comp1").variable().create("var1");
javamodel.component("comp1").variable("var1").set("DiffCoeff", "1e-9 [m^2/s]");
if interf[interface][0] == 'npe':
    javamodel.component("comp1").variable("var1").set("poroAnion", "0.1");
    javamodel.component("comp1").variable("var1").set("poroCation", "0.2");
    javamodel.component("comp1").variable("var1").set("poroNeutral", "0.2");

#create interpolate function
javamodel.component("comp1").func().create("int1", "Interpolation")
javamodel.component("comp1").func("int1").set("source", "file")
javamodel.component("comp1").func("int1").set("filename", "int.txt")
javamodel.component("comp1").func("int1").set("defvars", "on") # spatial coord as arguments

# #setup solver
javamodel.study("std1").create("time", "Transient");
javamodel.study("std1").setGenConv(False)
javamodel.study("std1").setGenPlots(False)
javamodel.study("std1").feature("time").set("tunit", "fs")
javamodel.study("std1").feature("time").set("tlist", "range(0,0.1,1)")


javamodel.component("comp1").physics().create(interf[interface][0], interf[interface][1], "geom1");
javamodel.component("comp1").physics().create('tds2', interf['tds'][1], "geom1");
javamodel.component("comp1").physics("tds2").label("fixed_species");


javamodel.component("comp1").physics(interf[interface][0]).feature("init1").label('piccts');
for a in range(len(outputData['domain'])): # 1 per domain + 1 for piccts. First init is piccts
    javamodel.component("comp1").physics(interf[interface][0]).feature().duplicate(f"init{a+2}", "init1")
    javamodel.component("comp1").physics('tds2').feature().duplicate(f"init{a+2}", "init1")
    if outputFiles[a] in outputFiles[:a]: outputFiles[a] = [outputFiles[a] + f'_{a}'] 
    javamodel.component("comp1").physics(interf[interface][0]).feature(f"init{a+2}").label(Path(outputFiles[a]).stem);
    javamodel.component("comp1").physics(interf[interface][0]).feature(f"init{a+2}").selection().set(a+1) # 1 init per domain
    javamodel.component("comp1").physics('tds2').feature(f"init{a+2}").label(Path(outputFiles[a]).stem);
    javamodel.component("comp1").physics('tds2').feature(f"init{a+2}").selection().set(a+1) # 1 init per domain
    
if len(outputData['domain']) > 1 :
    for a in range(len(outputData['domain'])): # 1 per domain 
        if interf[interface][0] == 'tds':
            javamodel.component("comp1").physics(interf[interface][0]).feature().duplicate(f"cdm{a+2}", "cdm1")
        elif interf[interface][1] == 'DilutedSpeciesInPorousMedia':
            javamodel.component("comp1").physics("tds").feature().duplicate("porous{a+2}", "porous1");
        # elif 

for i in range(len(outputData['boundary'])):
    javamodel.component("comp1").physics(interf[interface][0]).create(f"conc{i+1}", "Concentration", 0)
    javamodel.component("comp1").physics(interf[interface][0]).feature(f"conc{i+1}").selection().set(i+1);

if interf[interface][1] == 'DilutedSpeciesInPorousMedia':
    # javamodel.component("comp1").physics("tds").feature("porous1").feature("fluid1").set("FluidDiffusivityModelType", "UserDefined");
    javamodel.component("comp1").physics("tds").feature(f"porous{it+1}").feature("pm1").set("poro_mat", "userdef");
    javamodel.component("comp1").physics("tds").feature(f"porous{it+1}").feature("pm1").set("poro", float(porosity[it]));
    
elif interf[interface][1] == 'NernstPlanck': # homogeneous domain
    javamodel.component("comp1").physics("npe").feature().duplicate("cdm2", "cdm1");

# print(len(outputData['domain']))

# model.save()    
# sys.exit()

it_fixed = it_trspt = 0
for i, spc in enumerate(totalspc, start = 0):
    # one pqo per domain + 1 domain for piccts
    # one interface (tds) solely desgined for fixed species that may be deleted after
    
    # first interface tags, then generic tags
    
    if spc in spcFixed:
        it_fixed +=1
        for it,domain in enumerate(outputData['domain']):
            javamodel.component("comp1").physics("tds2").field("concentration").component(it_fixed, spc); # start at 1
            javamodel.component("comp1").physics("tds2").feature("cdm1").set(f"D_{spc}", ["0", "0", "0", "0", "0", "0", "0", "0", "0"]);
            if outputData['totals']['systemSpeciationTotal'][i] in outputData['domain'][domain]['surfaceMolality']:
                javamodel.component("comp1").physics("tds2").feature(f"init{it+2}").setIndex("initc", f"{outputData['domain'][domain]['surfaceMolality'][outputData['totals']['systemSpeciationTotal'][i]]} [mol/L]", it_fixed-1); # begin at 0
            elif outputData['totals']['systemSpeciationTotal'][i] in outputData['domain'][domain]['exchangeMolality']:
                javamodel.component("comp1").physics("tds2").feature(f"init{it+2}").setIndex("initc", f"{outputData['domain'][domain]['exchangeMolality'][outputData['totals']['systemSpeciationTotal'][i]]} [mol/L]", it_fixed-1); # begin at 0
            elif outputData['totals']['systemSpeciationTotal'][i] in outputData['domain'][domain]['phaseMolality']:
                javamodel.component("comp1").physics("tds2").feature(f"init{it+2}").setIndex("initc", f"{outputData['domain'][domain]['phaseMolality'][outputData['totals']['systemSpeciationTotal'][i]]} [mol/L]", it_fixed-1); # begin at 0
 
    if spc not in spcFixed:
        it_trspt += 1
        javamodel.component("comp1").physics(interf[interface][0]).field("concentration").component(it_trspt, spc);
        javamodel.component("comp1").physics(interf[interface][0]).feature("init1").setIndex("initc", spc+'i', it_trspt-1);
        
        for it,domain in enumerate(outputData['domain']):
            if outputData['totals']['systemSpeciationTotal'][i] in outputData['domain'][domain]['solutionMolality']:
                javamodel.component("comp1").physics(interf[interface][0]).feature(f"init{it+2}").setIndex("initc", f"{outputData['domain'][domain]['solutionMolality'][outputData['totals']['systemSpeciationTotal'][i]]} [mol/L]", it_trspt-1);


        for it, bd in enumerate(outputData['boundary']):
            if outputData['totals']['systemSpeciationTotal'][i] in outputData['boundary'][bd]['solutionMolality']:
                javamodel.component("comp1").physics(interf[interface][0]).feature(f"conc{it+1}").setIndex("species", '1', it_trspt-1)
                javamodel.component("comp1").physics(interf[interface][0]).feature(f"conc{it+1}").setIndex("c0",  f"{outputData['boundary'][bd]['solutionMolality'][outputData['totals']['systemSpeciationTotal'][i]]} [mol/L]", it_trspt-1);


        
    if interf[interface][0] == 'tds':
        # if spc not in spcFixed:
            # javamodel.component("comp1").physics("tds").field("concentration").component(it_trspt, spc); # start at 1


        # for it,domain in enumerate(outputData['domain']):
        #     if outputData['totals']['systemSpeciationTotal'][i] in outputData['domain'][domain]['solutionMolality']:
        #         javamodel.component("comp1").physics("tds").feature(f"init{it+2}").setIndex("initc", f"{outputData['domain'][domain]['solutionMolality'][outputData['totals']['systemSpeciationTotal'][i]]} [mol/L]", it_trspt-1); # begin at 0

        # for it, bd in enumerate(outputData['boundary']):

        #     if outputData['totals']['systemSpeciationTotal'][i] in outputData['boundary'][bd]['solutionMolality']:
        #         javamodel.component("comp1").physics("tds").feature(f"conc{it+1}").setIndex("species", '1', it_trspt-1)
        #         javamodel.component("comp1").physics("tds").feature(f"conc{it+1}").setIndex("c0",  f"{outputData['boundary'][bd]['solutionMolality'][outputData['totals']['systemSpeciationTotal'][i]]} [mol/L]", it_trspt-1);

        if interf[interface][1] == 'DilutedSpecies':
            if spc not in spcFixed:
                javamodel.component("comp1").physics("tds").feature("cdm1").set(f"D_{spc}", ["DiffCoeff", "0", "0", "0", "DiffCoeff", "0", "0", "0", "DiffCoeff"]);
        
        if interf[interface][1] == 'DilutedSpeciesInPorousMedia':
            for it,domain in enumerate(outputData['domain']):
                javamodel.component("comp1").physics("tds").feature(f"porous{it+1}").feature("fluid1").set("FluidDiffusivityModelType", "UserDefined")
                if spc not in spcFixed:
                    javamodel.component("comp1").physics("tds").feature(f"porous{it+1}").feature("fluid1").set(f"DF_{spc}", ["DiffCoeff", "0", "0", "0", "DiffCoeff", "0", "0", "0", "DiffCoeff"]);

    elif interf[interface][0] == "npe":
        # three porosity (cation, anion, neutral) within one interface

        # for it,domain in enumerate(outputData['domain']):
        #     if spc in outputData['domain'][domain]['solutionMolality']:
        #         javamodel.component("comp1").physics("npe").feature(f"init{it+2}").setIndex("initc", f"{outputData['domain'][domain]['solutionMolality'][outputData['totals']['systemSpeciationTotal'][i]]} [mol/L]", it_trspt-1); # begin at 0

        if spc in outputData['totals']['transportedSpeciesTotal']: 
            if 'minus' in spc:
                if isinstance(spc[-1], str):
                    c = 1
                else: c = spc[-1]
                javamodel.component("comp1").physics("npe").feature("sp1").setIndex("z", f"-{c}", it_trspt-1); # begin at 0
                javamodel.component("comp1").physics("npe").feature("cdm1").set(f"D_{spc}", ["DiffCoeffAnion/poroAnion", "0", "0", "0", "DiffCoeffAnion/poroAnion", "0", "0", "0", "DiffCoeffAnion/poroAnion"]);
        
            elif "plus" in spc:
                if isinstance(spc[-1], str):
                    c = 1
                else: c = spc[-1]
                javamodel.component("comp1").physics("npe").feature("sp1").setIndex("z", f"+{c}", it_trspt-1); # begin at 0
                javamodel.component("comp1").physics("npe").feature("cdm1").set(f"D_{spc}", ["DiffCoeffCation/poroCation", "0", "0", "0", "DiffCoeffCation/poroCation", "0", "0", "0", "DiffCoeffCation/poroCation"]);
        
            else:
                javamodel.component("comp1").physics("npe").feature("cdm1").set(f"D_{spc}", ["DiffCoeffNeutral/poroNeutral", "0", "0", "0", "DiffCoeffNeutral/poroNeutral", "0", "0", "0", "DiffCoeffNeutral/poroNeutral"]);
            
            # for homogeneous domain
            javamodel.component("comp1").physics("npe").feature("cdm2").set(f"D_{spc}", ["D_source", "0", "0", "0", "D_source", "0", "0", "0", "D_source"]);

model.save()
sys.exit()

model.solve()

for i in range(geometry):
    javamodel.component("comp1").func("int1").setEntry("columnType", f"col{i+1}", "arg")
    javamodel.component("comp1").func("int1").setIndex("argunit", "m", i)

for i,spc in enumerate(totalspc, start = geometry+1):
    javamodel.component("comp1").func("int1").setEntry("columnType", f"col{i}", "value");
    javamodel.component("comp1").func("int1").setIndex("fununit", "mol/L", i-geometry-1)
    javamodel.component("comp1").func("int1").setEntry("funcnames", f"col{i}", f"{spc}i")
        
javamodel.result().export().create("data1", "Data")
javamodel.result().export("data1").set("data", "dset1")
javamodel.result().export("data1").setIndex("looplevelinput", "first", 0);
javamodel.result().export("data1").setIndex("looplevelinput", "last", 0);
javamodel.result().export().duplicate("data2", "data1");

for i,spc in enumerate(totalspc, start = geometry+1):
    javamodel.result().export("data2").setIndex("expr", spc, i-geometry-1)
    if spc in spcFixed:
        javamodel.result().export("data1").setIndex("expr", spc+"i", i-geometry-1)
    else: 
        javamodel.result().export("data1").setIndex("expr", spc, i-geometry-1)
    javamodel.result().export("data1").setIndex("unit", "mol/L", i-geometry-1)
    javamodel.result().export("data1").setIndex("descr", outputData['totals']['systemSpeciationTotal'][i-geometry-1], i-geometry-1)
    javamodel.result().export("data2").setIndex("unit", "mol/L", i-geometry-1)
    javamodel.result().export("data2").setIndex("descr", outputData['totals']['systemSpeciationTotal'][i-geometry-1], i-geometry-1)

javamodel.result().export("data2").set("filename", "piccts_ic.txt")            #name for output
javamodel.result().export("data2").run()                                #get txt output
javamodel.component("comp1").func("int1").set("filename", "piccts_ic.txt")     #change interpolation input


print('System speciation is:')
print(outputData['totals']['systemSpeciationTotal'])
model.save()
