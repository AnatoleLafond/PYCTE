import itertools
import sys
# xGEMS module: an `xgems` the script imported before pycte.run() (another version or build, chosen by its sys.path,
# possibly with a patched ChemicalEngine) is used as it is; otherwise the xGEMS compiled with pycte, otherwise xGEMS
# installed separately. See xgemsModule
if 'xgems' in sys.modules:
    xgems = sys.modules['xgems']
else:
    try:
        from .. import PyxGEMS as xgems     # xGEMS compiled with pycte (setup.py, sources in pycte/gems_src)
    except ImportError:
        try:
            import xgems                    # xGEMS installed separately (conda install -c conda-forge xgems)
        except ImportError as err:
            raise ImportError("chemistryModule = 'xGEMS' : this pycte installation has no compiled xGEMS (no pycte "
                              "wheel for this platform and Python version, and no C++ compiler when pycte was "
                              "installed). Install a C++ compiler (e.g. Visual Studio Build Tools on Windows) and "
                              "reinstall pycte from its sources : pip install --force-reinstall --no-deps --no-binary "
                              "pycte pycte, or install xGEMS with conda : conda install -c conda-forge xgems") from err
import pandas as pd
import numpy as np
import time 
import importlib.util
import os
from pathlib import Path
try:
    from . import crossDependenciesManager
    from . import extractDB
    from . import outputManager
    from . import warningManager
    from . import display
    from . import frames
except ImportError:
    import crossDependenciesManager
    import extractDB
    import outputManager
    import warningManager
    import display
    import frames

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


gemsStatus= {
0: "No GEM re-calculation needed",
1: "Need GEM calculation with LPP (automatic) initial approximation (AIA)",
2: "OK after GEM calculation with LPP AIA",
3: "Bad (not fully trustful) result after GEM calculation with LPP AIA",
4: "Failure (no result) in GEM calculation with LPP AIA",
5: "Need GEM calculation with no-LPP (smart) IA, SIA using the previous speciation",
6: "OK after GEM calculation with SIA",
7: "Bad (not fully trustful) result after GEM calculation with SIA",
8: "Failure (no result) in GEM calculation with SIA",
9: "Terminal error in GEMS3K (e.g., memory corruption). Restart required.",
    }

def nodeConditions(centralDict, commMtrx, quantity):
    """Temperature (K) and pressure (Pa) of each node: coupled columns or tempDefault / pressureDefault."""
    columnOf = {q: name for name, q in quantity.items()}

    def column(q):
        name = columnOf[q]
        if name not in commMtrx.columns:
            raise ValueError(f"xGEMS : crossDependencies couples {name!r} but the coupled variables have no column "
                             f"{name!r} (add it to coupledVariables and to the initial conditions)")
        return commMtrx[name].to_numpy(dtype=float)

    if 'temperature(K)' in columnOf:
        temperature = column('temperature(K)')
    else:
        temperature = np.full(len(commMtrx), float(centralDict.get('tempDefault', 298.15)))
    if 'pressure' in columnOf:
        pressure = column('pressure')
    else:
        pressure = np.full(len(commMtrx), float(centralDict.get('pressureDefault', 101325.0)))
    return temperature, pressure


def checkConditions(centralDict, temperature, pressure, nodes):
    """Stops before the speciation when a node is outside the T / P range of the thermodynamic data of the DCH file
    (gemsGrid, extractDB._dchGrid): GEMS3K would fail on it ('no result') or extrapolate, and say so only in
    ipmlog.txt."""
    grid = centralDict.get('gemsGrid')
    if not grid:
        return
    for name, values, key, unit, setting in (('temperature', temperature, 'T', 'K', 'tempDefault (K)'),
                                             ('pressure', pressure, 'P', 'Pa', 'pressureDefault (Pa)')):
        lo, hi = grid[key]
        out = np.flatnonzero(~((values > lo) & (values < hi)))
        if out.size:
            k = out[0]
            points = grid['TKval'] if key == 'T' else grid['Pval']
            raise ValueError(f"xGEMS : {name} {values[k]:.6g} {unit} at node {nodes[k]} ({out.size} node(s)) is outside "
                             f"the range of the thermodynamic data of {grid['file']} : {lo:.6g} < {name} < {hi:.6g} "
                             f"{unit} (grid {', '.join(f'{p:g}' for p in points)} {unit} and its tolerance). Export from "
                             f"GEMS-Selektor a DCH file whose grid covers the run, or change {setting} or the coupled "
                             "column")


def multiCompoundSetup(centralDict, transportedGiven=False):
    """MultiCompoundTransport: commMtrx, boundaries and renewal solutions converted to component totals."""
    cd = centralDict
    ics = list(cd['independentComponents'])
    mobile, immobile = set(cd['mobileSpecies']), set(cd['fixedSpecies'])
    cross = cd['crossDependencies']['names']

    def totals(table):
        out = pd.DataFrame(0.0, index=table.index, columns=ics)
        for col in table.columns:
            if col in ics:
                out[col] += table[col].astype(float)
            else:
                for ic, coeff in cd['primToSecSpecies'][col].items():
                    out[ic] += coeff * table[col].astype(float)
        return out

    species = [c for c in cd['coupledVariables'] if c not in cross]
    unknown = [c for c in species if c not in ics and c not in mobile and c not in immobile]
    if unknown:
        raise ValueError(f"xGEMS, MultiCompoundTransport : {unknown} are neither species nor independent components of "
                         f"the GEMS system {cd['chemPath']}")
    kept = [c for c in species if c in immobile]
    crossCols = [c for c in cd['coupledVariables'] if c in cross]
    comm = cd['commMtrx']
    tot = totals(comm[[c for c in species if c not in immobile]])
    # an immobile species named as an independent component: its column is the species amount, the total of the
    # mobile species in that component must be zero
    shared = [c for c in kept if c in ics]
    mixed = [c for c in shared if (tot[c].abs() > 0).any()]
    if mixed:
        raise ValueError(f"xGEMS, MultiCompoundTransport : {mixed} are both immobile species and independent components "
                         "present in the mobile species : their totals cannot be coupled under the same name")
    totalsCols = [c for c in ics if c not in shared]
    converted = pd.concat([comm[cd['coord']], tot[totalsCols], comm[kept], comm[crossCols]], axis=1)
    coupled = totalsCols + kept + crossCols
    cd.update({'commMtrx': converted, 'coupledVariables': coupled, 'systemSpecies': totalsCols + kept})
    cd['anythingButSpecies'] = [c for c in converted.columns if c not in coupled]
    if not transportedGiven:
        cd['transportedSpecies'] = [c for c in cd['transportedIC'] if c not in shared]

    def convertValues(values, key):
        ignored = [k for k in values if k not in ics and k not in mobile and k not in immobile and k not in cross]
        if ignored:
            warningManager.warn(f"xGEMS, MultiCompoundTransport : {key} : {ignored} are neither species nor independent "
                                "components of the GEMS system, ignored")
        mobileKeys = [k for k in values if (k in ics and k not in shared) or k in mobile]
        table = pd.DataFrame([{k: float(values[k]) for k in mobileKeys}])
        converted = {ic: float(v) for ic, v in totals(table).iloc[0].items() if ic in totalsCols}
        converted.update({k: float(values[k]) for k in values if k in immobile or k in cross})
        return converted

    for key in ('firstBoundary', 'secondBoundary'):
        values = cd.get(key)
        if not values:
            continue
        cd[key] = convertValues(values, key)
    for k, group in enumerate(cd.get('renewal') or []):
        if group['concentration']:
            group['concentration'] = convertValues(group['concentration'], f"renewal, group {k}")

    print(f"MultiCompoundTransport : initial conditions -> totals of {', '.join(ics)}, transported : "
          f"{', '.join(cd['transportedSpecies'])}")
    return centralDict


_stoichFrames = {}


def _stoichFrame(primToSecSpecies, index, columns, species=None):
    """pd.DataFrame(0.0, index=index, columns=columns) where stoich.at[comp, prim] = coeff for each comp of `species`
    (default: index) and each (prim, coeff) of primToSecSpecies[comp]. Built once per species, components and
    decompositions (the same frame, built the same way, at every call of speciation_xGEMS: read only). For an Index
    (immutable, the same object at every call), the frame found is kept for the same objects (frames.constant): the
    key of the names and decompositions is not rebuilt at every call."""
    if isinstance(index, pd.Index):
        return frames.constant('gemsStoich', (primToSecSpecies, index, columns, species),
                               lambda: _stoichFrameOf(primToSecSpecies, index, columns, species))
    return _stoichFrameOf(primToSecSpecies, index, columns, species)


def _stoichFrameOf(primToSecSpecies, index, columns, species=None):
    indexLabels = frames.labels(index)
    species = indexLabels if species is None else frames.labels(species)
    key = (tuple(indexLabels), tuple(frames.labels(columns)),
           tuple((comp, tuple(primToSecSpecies[comp].items())) for comp in species))
    frame = _stoichFrames.get(key)
    if frame is None:
        frame = pd.DataFrame(0.0, index=index, columns=columns)
        for comp in species:
            for prim, coeff in primToSecSpecies[comp].items():
                frame.at[comp, prim] = coeff
        if len(_stoichFrames) >= 16:
            _stoichFrames.clear()
        _stoichFrames[key] = frame
    return frame


def xgemsModule():
    """xGEMS module of the speciation: the `xgems` imported by the script, even after pycte, or the one chosen at the
    import of this module."""
    return sys.modules.get('xgems', xgems)


# xGEMS modules (id) that made a ChemicalEngine in this process: GEMS3K opens its log (ipmlog.txt) in the working folder
# of its first TNode, here the folder of the .lst (speciation_xGEMS), once per process
_enginesMade = set()
_directPaths = {}                   # .lst path -> what _directEnginePath gives (cleared by workingCopy, at each run)


def _directEnginePath(enginePath):
    """Path given to ChemicalEngine without changing the working folder (two os.chdir per call, ~0.7 ms on a disk
    watched by an antivirus): the absolute path of the .lst, when it is ASCII (GEMS3K opens no path with accents on
    Windows), short enough, and every file it names is a relative name, that GEMS3K reads from the folder of the .lst
    (GEMS3KGenerator: impex_dir + '/' + name), the same file as from that folder as working folder. None otherwise."""
    key = str(enginePath)
    if key not in _directPaths:
        path = os.path.abspath(key)
        direct = None
        if path.isascii():
            try:
                text = Path(path).read_text(encoding='utf-8', errors='ignore')
            except OSError:
                text = ''
            names = [t.strip('"\'') for t in text.split() if not t.startswith('-')]
            if (names and all(n and not os.path.isabs(n) and not os.path.splitdrive(n)[0] for n in names)
                    and len(path) + max(map(len, names)) < 240):
                direct = path
        _directPaths[key] = direct
    return _directPaths[key]


def workingCopy(chemPath, runFolder):
    import re
    import shutil
    _directPaths.clear()
    source = Path(chemPath)
    work = Path(runFolder) / "gemsWork"
    work.mkdir(exist_ok=True)
    names = re.findall(r'["\']([^"\']+)["\']', source.read_text(encoding='utf-8', errors='ignore'))
    for name in [source.name] + names:
        if (source.parent / name).is_file():
            shutil.copy2(source.parent / name, work / name)
    return work / source.name


def speciation_xGEMS(centralDict,commMtrx):
    enginePath = Path(centralDict.get('gemsEnginePath') or centralDict['chemPath'])
    module = xgemsModule()
    direct = _directEnginePath(enginePath) if id(module) in _enginesMade else None
    if direct is not None:
        # not the first engine of this process: the .lst by its absolute path, from the run folder
        ref = time.perf_counter()
        engine = module.ChemicalEngine(direct)
        init = time.perf_counter() - ref
    else:
        chem_dir = enginePath.parent
        _cwd = os.getcwd()
        try:
            os.chdir(chem_dir)
            ref = time.perf_counter()
            # the file name in its folder (the current one): GEMS3K, a C++ library, does not open a path with accents on
            # Windows (Données), and the files the .lst names are found next to it
            name = enginePath.name
            engine = module.ChemicalEngine(name if name.isascii() else extractDB.asciiPath(enginePath))
            init = time.perf_counter() - ref
        finally:
            os.chdir(_cwd)
        _enginesMade.add(id(module))
    
    crossSpc = centralDict['crossDependencies']['speciation']
    crossTotal = crossSpc['total']

    # output: the columns of commMtrx; MultiCompoundTransport (commMtrx holds totals): every species of the GEMS system,
    # converted back into totals below
    gemsColumns = (list(centralDict['gemsSpecies']) + [c for c in commMtrx.columns if c in crossTotal]
                   if centralDict['MultiCompoundTransport'] else commMtrx.columns)
    # columns of pd.DataFrame(columns=gemsColumns) (the Index given, or the one pandas builds from the list), without
    # building that empty frame of object columns (~0.5 ms), built below only when a row is not made of floats
    outputGemsColumns = (gemsColumns if isinstance(gemsColumns, pd.Index)
                         else pd.Index(list(gemsColumns), tupleize_cols=False))
    quantity = crossSpc['quantity']
    temperature, pressure = nodeConditions(centralDict, commMtrx, quantity)
    checkConditions(centralDict, temperature, pressure, commMtrx.index)
    if crossTotal:
        commMtrx.drop(columns=crossTotal, inplace=True)

    MultiCompoundTransport = centralDict['MultiCompoundTransport'] 
    totals = None
    if centralDict['lStep'] ==0 :
    
        notThere = (centralDict['coord']+centralDict['speciesByClass']['T']+centralDict['speciesByClass']['W']+centralDict['speciesByClass']['O']+centralDict['speciesByClass']['G']
                    + centralDict['crossDependencies']['speciation']['total'])
    
        l = [s for s in list(centralDict['commMtrx'].columns) if s not in notThere]
        if set(l).issubset(centralDict['speciesByClass']['S']):
            if centralDict['MultiCompoundTransport'] :
                MultiCompoundTransport = False
            

    if MultiCompoundTransport :
        commMtrx_primSpecies = commMtrx.copy()
        
        ic_list = centralDict['independentComponents']
        species_list = commMtrx.columns
        
        already_primary = [s for s in species_list.tolist() if s in ic_list]

        to_decompose = [s for s in species_list.tolist() if s not in ic_list]

        stoich_matrix = _stoichFrame(centralDict['primToSecSpecies'], to_decompose, ic_list)

        decomposed = commMtrx[to_decompose].values @ stoich_matrix.values
        decomposed_df = pd.DataFrame(decomposed, columns=ic_list, index=commMtrx.index)
        
        result_df = pd.DataFrame(0.0, columns=ic_list, index=commMtrx.index)
        
        for s in already_primary:
            result_df[s] += commMtrx[s]
        
        result_df += decomposed_df
        
        commMtrx_primSpecies = result_df
        

        
    else:
        ic_list = centralDict['independentComponents']
        species_list = commMtrx.columns
        
        # one row per species column of commMtrx, by name: the coupled species may be any subset of the GEMS species,
        # in any order (the rows used to follow the order of the GEMS system, the columns that of the initial conditions)
        unknownSpecies = [s for s in species_list.tolist() if s not in centralDict['primToSecSpecies']]
        if unknownSpecies:
            raise ValueError(f"xGEMS : coupled variable(s) {unknownSpecies} are neither species of the GEMS system "
                             f"{centralDict['chemPath']} nor crossDependencies")
        stoich_matrix = _stoichFrame(centralDict['primToSecSpecies'], species_list, centralDict['independentComponents'])


        result = commMtrx.values @ stoich_matrix.values

        commMtrx_primSpecies = pd.DataFrame(result, columns=ic_list, index=commMtrx.index)
        # its columns are ic_list: when the names are unique, commMtrx_primSpecies[independentComponents] takes them
        # all in their order, the values of result (float64 array, only read)
        if (isinstance(result, np.ndarray) and result.dtype == np.float64 and result.ndim == 2
                and len(set(ic_list)) == len(ic_list)):
            totals = result

    gemsStatusList = []
    gemsIterations = []

    # output row of a node, by column name: species amounts of the GEMS system, pressure and temperature
    gemsNames = [engine.speciesName(i) for i in range(engine.numSpecies())]
    stateNames = ('pressure', 'temperature(K)')
    engineOutputs = {c: crossDependenciesManager.parseCommand(quantity[c]) for c in outputGemsColumns
                     if c in crossSpc['output'] and c not in gemsNames and quantity[c] not in stateNames}
    missing = {c: name for c, (name, _) in engineOutputs.items() if not callable(getattr(engine, name, None))}
    if missing:
        raise ValueError(f"xGEMS : crossDependencies {missing} are not methods of xgems.ChemicalEngine")
    unknown = [c for c in outputGemsColumns if c not in gemsNames and quantity.get(c, c) not in stateNames
               and c not in engineOutputs]
    if unknown:
        raise ValueError(f"xGEMS : column(s) {unknown} of the coupled variables are not species of the GEMS system "
                         f"{centralDict['chemPath']} (nor pressure, temperature(K))")

    calcTime = 0

    # rows of commMtrx_primSpecies, temperature and pressure: the nodes of commMtrx, in the same order
    if totals is None:
        totals = commMtrx_primSpecies[centralDict['independentComponents']].to_numpy(dtype=float)
    rows = []
    # key of each output column in the values of a node, found once (not by iterating the columns at every node)
    outputColumns = list(outputGemsColumns)
    # value of a key in a node: dict(zip(gemsNames, speciesAmounts())) updated with the pressure, the temperature and
    # the engine outputs, read without building that dict: the amount at the last position of the name, unless the
    # key is one of the updates
    speciesPosition = {name: i for i, name in enumerate(gemsNames)}
    outputGems = None
    lookup = [quantity.get(c, c) for c in outputColumns]
    nColumns = len(outputColumns)
    floats = itertools.repeat(float)
    # every key a species name and none of the keys of `updates` (pressure, temperature(K), engine outputs): the row is
    # nodeAmounts at the positions of the keys, taken at once (the same float64 values; a row is always made of floats
    # then, as the check below finds)
    updateKeys = {'pressure', 'temperature(K)'}.union(quantity[c] for c in engineOutputs)
    take = (np.array([speciesPosition[key] for key in lookup], dtype=np.intp)
            if all(key in speciesPosition and key not in updateKeys for key in lookup) else None)
    for k, index in enumerate(commMtrx_primSpecies.index):
        ref = time.perf_counter()
        state = (temperature[k], pressure[k], list(totals[k]))
        status = engine.equilibrate(*state)
        if status in (4, 8):
            # no result: the engine still holds the previous node's speciation. One retry from the other initial
            # approximation (4: automatic IA failed -> smart IA from that speciation; 8: the reverse), then back
            first = status
            (engine.setWarmStart if first == 4 else engine.setColdStart)()
            status = engine.equilibrate(*state)
            (engine.setColdStart if first == 4 else engine.setWarmStart)()
            if status in (2, 6):
                warningManager.warn(f"GEMS, node n°{index}, time step n°{centralDict['lStep']+1}, "
                                    f"t={centralDict['tStep']}{centralDict['timeUnit']}, PID={os.getpid()} : "
                                    f"{gemsStatus[first]} ; retried with the other initial approximation : {gemsStatus[status]}")

        calcTime += time.perf_counter() - ref # good approximation

        nodeAmounts = engine.speciesAmounts()
        updates = {'pressure': engine.pressure(), 'temperature(K)': engine.temperature()}
        updates.update({quantity[c]: float(getattr(engine, name)(*args)) for c, (name, args) in engineOutputs.items()})
        if (take is not None and rows is not None and len(nodeAmounts) == len(gemsNames)
                and isinstance(nodeAmounts, np.ndarray) and nodeAmounts.dtype == np.float64 and nodeAmounts.ndim == 1):
            rows.append(nodeAmounts[take])
            gemsStatusList += [status]
            gemsIterations += [engine.numIterations()]
            continue
        if len(nodeAmounts) == len(gemsNames):
            row = [updates[key] if key in updates else nodeAmounts[speciesPosition[key]] for key in lookup]
        else:
            values = dict(zip(gemsNames, nodeAmounts))
            values.update(updates)
            row = [values[key] for key in lookup]
        if rows is not None and len(row) == nColumns and all(map(isinstance, row, floats)):
            rows.append(row)
        else:
            if outputGems is None:
                outputGems = pd.DataFrame(columns=gemsColumns)
            for previous in rows or []:
                # a row taken at once: as the list of its values (numpy floats, as nodeAmounts[position])
                outputGems.loc[len(outputGems)] = list(previous) if isinstance(previous, np.ndarray) else previous
            rows = None
            outputGems.loc[len(outputGems)] = row

        gemsStatusList += [status]
        gemsIterations += [engine.numIterations()]
    if rows:
        outputGems = pd.DataFrame(np.array(rows, dtype=np.float64), columns=outputGemsColumns)
    elif outputGems is None:
        outputGems = pd.DataFrame(columns=gemsColumns)

    outputGems.index = commMtrx.index
        
    gemsSpecies = set(centralDict['gemsSpecies'])
    amounts = [c for c in outputGems.columns.tolist() if c in gemsSpecies]
    if (not centralDict['MultiCompoundTransport'] and outputGems.columns.is_unique
            and frames.allFloat64(outputGems)):
        # clip(lower=0) of the amounts on the array (NaN and values >= 0 kept, the others set to 0, as pandas does),
        # instead of assigning the columns one by one
        clipped = outputGems.to_numpy(copy=True)
        positions = outputGems.columns.get_indexer(amounts)
        selected = clipped[:, positions]
        clipped[:, positions] = np.where(np.isnan(selected) | (selected >= 0), selected, 0)
        outputGems = pd.DataFrame(clipped, index=outputGems.index, columns=outputGems.columns)
    else:
        outputGems[amounts] = outputGems[amounts].clip(lower=0)

    if centralDict['MultiCompoundTransport']:
        
        commMtrxCrossDep = outputGems[crossTotal].copy()
        outputGems.drop(columns=crossTotal, inplace=True)
        
        outputSpecies = outputGems.copy()
        ic_list = centralDict['independentComponents']
        species_list = outputGems.columns

        # totals of the mobile species (aqueous, water); the immobile species of the coupled variables (solids,
        # gases) stay as they are (see multiCompoundSetup)
        stoich_matrix = _stoichFrame(centralDict['primToSecSpecies'], species_list, ic_list, centralDict['mobileSpecies'])
        result = outputGems.values @ stoich_matrix.values

        outputGems = pd.DataFrame(result, columns=ic_list, index=commMtrx.index)
        immobile = [c for c in centralDict['coupledVariables'] if c in centralDict['fixedSpecies']]
        outputGems[immobile] = outputSpecies[immobile].copy()
        outputGems = pd.concat([outputGems,commMtrxCrossDep], axis=1)

        return outputGems,outputSpecies,gemsStatusList,gemsIterations, calcTime, init
    else:
        
        return outputGems,commMtrx_primSpecies,gemsStatusList,gemsIterations, calcTime, init


# keys of centralDict read by speciation_xGEMS and the functions it calls (tests/test_worker_dicts.py checks that the
# list holds every key they read): with speciationThreads > 1, the workers receive these keys only, pickled at every
# call, instead of the whole dictionary (dtpycte, ~370 kB for 100 000 steps, the transport settings and arrays, ...)
WORKER_KEYS = frozenset({
    'MultiCompoundTransport', 'chemPath', 'commMtrx', 'coord', 'coupledVariables', 'crossDependencies', 'fixedSpecies',
    'gemsEnginePath', 'gemsGrid', 'gemsSpecies', 'independentComponents', 'lStep', 'mobileSpecies', 'pressureDefault',
    'primToSecSpecies', 'speciesByClass', 'tStep', 'tempDefault', 'timeUnit',
})


def _workerDict(centralDict):
    """centralDict as the workers read it: the WORKER_KEYS (in the order of centralDict), commMtrx without its rows
    (speciation_xGEMS reads its column names only; each worker is given its block of nodes apart)."""
    light = {k: v for k, v in centralDict.items() if k in WORKER_KEYS}
    comm = light.get('commMtrx')
    if isinstance(comm, pd.DataFrame):
        light['commMtrx'] = comm.iloc[:0]
    return light


def spct(centralDict):
    display.stepPrint("xGEMS", end=" ", flush=True)
    startGems = time.perf_counter()
    if centralDict.get('xGEMSModule') is None:
        module = xgemsModule()
        centralDict['xGEMSModule'] = getattr(module, '__file__', repr(module))
        warningManager.info(f"xGEMS : module {centralDict['xGEMSModule']}")

    crossDep = centralDict['crossDependencies']
    if crossDep['names']:
        transportOnly = [c for c in crossDep['transport']['total'] if c not in crossDep['speciation']['total']]
        gemsInput = [s for s in centralDict['commMtrx'].columns.tolist()
                     if s in centralDict['coupledVariables'] + crossDep['speciation']['input'] and s not in transportOnly]
    else:
        gemsInput = [s for s in centralDict['commMtrx'].columns.tolist() if s not in centralDict['coord']]


    if centralDict['speciationThreads'] > 1:
    
        chunk_size = int(np.ceil(len(centralDict['commMtrx']) / centralDict['speciationThreads']))
    
        gemsFrame = centralDict['commMtrx'][gemsInput]             # selected once, sliced into the blocks
        commMtrxSplit = [gemsFrame.iloc[i:i + chunk_size] for i in range(0, len(centralDict['commMtrx']), chunk_size)]

        # processes kept from one call to the next (started once per run, stopped by warningManager.shutdown at its
        # end), block k always in process k: a new pool at every call started its processes (Python, pycte and xGEMS
        # imports, ~0.8 s per call) at every step. Each call still builds its own ChemicalEngine (speciation_xGEMS)
        executors = warningManager.pinnedExecutors(centralDict['speciationThreads'], group='xGEMS')
        light = _workerDict(centralDict)            # what the workers read, not the whole dictionary!
        futures = [warningManager.submit(executors[k], speciation_xGEMS, light, chunk)
                   for k, chunk in enumerate(commMtrxSplit)]
        results = [f.result() for f in futures]
    
        output,primSpc, status, iteration, calc, initWorker = zip(*results)
    
        commMtrxSpct = pd.concat(output, ignore_index=True)
        commMtrx_primSpecies = pd.concat(primSpc, ignore_index=True)
        calcPrcsTime = sum(calc)
        calcWallClock = max(calc)
        init = max(initWorker)
        
        gemsStatusList = [s for block in status for s in block]
        gemsIterations = [n for block in iteration for n in block]
        
    else:
        # frames.columnsOf: commMtrx[gemsInput] built from its array, unless speciation_xGEMS drops the
        # crossDependencies from it in place (pandas then checks the frame a selection was taken from)
        gemsFrame = (centralDict['commMtrx'][gemsInput] if crossDep['speciation']['total']
                     else frames.columnsOf(centralDict['commMtrx'], gemsInput))
        commMtrxSpct, commMtrx_primSpecies, gemsStatusList, gemsIterations, calcWallClock, init = speciation_xGEMS(centralDict,gemsFrame)
        calcPrcsTime = calcWallClock

    # primary species: written to the speciation output only, assembled only when it is written
    speciationOutput = outputManager.wanted(centralDict, 'speciation')
    # coordinates and copied columns: frames.columnsOf (commMtrx[names], only read by the concat)
    coords = frames.columnsOf(centralDict['commMtrx'], ['x', 'y', 'z'][:centralDict['geometry']])
    commMtrxSpct = pd.concat([coords,commMtrxSpct], axis=1)
    if speciationOutput:
        commMtrx_primSpecies = pd.concat([coords,commMtrx_primSpecies], axis=1)



    for copied in ([c for c in crossDep['transport']['total'] if c not in crossDep['speciation']['total']],
                   [c for c in crossDep['speciation']['input'] if c not in crossDep['speciation']['output']]):
        if copied:
            copiedFrame = frames.columnsOf(centralDict['commMtrx'], copied)
            commMtrxSpct = pd.concat([commMtrxSpct, copiedFrame], axis=1)
            if speciationOutput:
                commMtrx_primSpecies = pd.concat([commMtrx_primSpecies, copiedFrame], axis=1)
        
    
    failed = []
    for i,status in enumerate(gemsStatusList):
        if status in (2, 6):
            continue
        message = (f"GEMS, node n°{i}, time step n°{centralDict['lStep']+1}, t={centralDict['tStep']}{centralDict['timeUnit']}, "
                   f"PID={os.getpid()} : {gemsStatus.get(status, f'status {status}')} ({gemsIterations[i]} iterations)")
        if status in (4, 8, 9):
            failed.append(message)
        else:
            warningManager.warn(message)
    if failed:
        for message in failed:
            warningManager.warn(message)
        raise RuntimeError(f"xGEMS : no result for {len(failed)} node(s), see warning.log :\n" + "\n".join(failed[:10]))


    if outputManager.wanted(centralDict, 'speciation'):
        outputManager.write(centralDict, commMtrx_primSpecies, 'primarySpecies', 'PrimarySpecies')
        outputManager.write(centralDict, commMtrxSpct, 'speciation', 'xGEMS')
    display.stepPrint()
    display.stepPrint(commMtrxSpct)
    
    centralDict.update({
        "commMtrx": commMtrxSpct,
        "xGEMSCalcTime_WallClock": centralDict["xGEMSCalcTime_WallClock"] + calcWallClock,
        "xGEMSCalcTime_ProcessorTime": centralDict["xGEMSCalcTime_ProcessorTime"] + calcPrcsTime,
        "xGEMSInterfTime_WallClock": centralDict["xGEMSInterfTime_WallClock"] + time.perf_counter() - startGems - calcWallClock - init,
        "xGEMSInitTime" : centralDict["xGEMSInitTime"] + init,
        "xGEMSTotalTime" : centralDict["xGEMSTotalTime"] + time.perf_counter() - startGems,})
 
    
    display.stepPrint(f"({writeTime((time.perf_counter() - startGems))})") 


    return centralDict