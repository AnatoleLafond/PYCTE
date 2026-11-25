import os
import re
import time

import numpy as np
import pandas as pd

try:
    from . import crossDependenciesManager
    from . import extractDB
    from . import outputManager
    from . import phreeqc
    from . import timeUnits
    from . import warningManager
    from . import display
    from . import frames
except ImportError:
    import crossDependenciesManager
    import extractDB
    import outputManager
    import phreeqc
    import timeUnits
    import warningManager
    import display
    import frames

writeTime = phreeqc.writeTime

TIME_UNITS = timeUnits.SECONDS


def _reprs(values):
    """[repr(v) for v in values.tolist()] of a 1-D float64 array: written at once by pycte._floatText (same text), or by
    repr without it."""
    if phreeqc._floatText is not None and values.dtype == np.float64 and values.ndim == 1:
        return phreeqc._floatText.texts(values)
    return list(map(repr, values.tolist()))


REACTANTS = (
    ('exchange', 'EXCHANGE', 'InitialExchanges2Module', '-exchange'),
    ('phases', 'EQUILIBRIUM_PHASES', 'InitialEquilibriumPhases2Module', '-equilibrium_phases'),
    ('kinetics', 'KINETICS', 'InitialKinetics2Module', '-kinetics'),
    ('surface', 'SURFACE', 'InitialSurfaces2Module', '-surface'),
)

_state = None


def resetPhreeqcRM(verbose=False):
    """Releases the PhreeqcRM instance and its processes (end of run)."""
    global _state
    if _state is None:
        return False
    workers = _state.get('workers')
    _state = None
    if workers is not None:
        workers.close()
    if verbose:
        print(f"PhreeqcRM released (PID={os.getpid()})")
    return True


def _checked(rm, result, what):
    """Result of a PhreeqcRM method; RuntimeError if it failed."""
    if result is not None and result < 0:
        raise RuntimeError(f"PhreeqcRM : {what} failed :\n{rm.GetErrorString().strip()}")
    return result


def _run(rm, calls):
    """Runs the calls on rm until one fails -> (results, RunCells times)."""
    results, wall, cpu = [], 0.0, 0.0
    for method, args, what in calls:
        ref, refProcess = time.perf_counter(), time.process_time()
        result = getattr(rm, method)(*args)
        if method == 'RunCells':
            wall += time.perf_counter() - ref
            cpu += time.process_time() - refProcess
        if isinstance(result, int):
            _checked(rm, result, what)
        results.append(result)
    return results, (wall, cpu)


def _worker(connection, cells, spec):
    """Worker process: PhreeqcRM instance of `cells` cells running the calls it receives."""
    try:
        rm = _create(cells, spec)
        connection.send(('ok', None))
    except Exception as error:
        connection.send(('error', error))
        return
    while True:
        calls = connection.recv()
        if calls is None:
            break
        try:
            connection.send(('ok', _run(rm, calls)))
        except Exception as error:
            connection.send(('error', error))


class _Workers:
    """Processes each holding the PhreeqcRM instance of a block of cells (speciationThreads > 1)."""

    def __init__(self, n, processes, spec):
        import multiprocessing
        size, extra = divmod(n, processes)
        stops = [int(s) for s in np.cumsum([size + (k < extra) for k in range(processes)])]
        self.bounds = list(zip([0] + stops[:-1], stops))
        self.connections, self.processes = [], []
        for a, b in self.bounds:
            parent, child = multiprocessing.Pipe()
            process = multiprocessing.Process(target=_worker, args=(child, b - a, spec), daemon=True)
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)
        try:
            self._receive()
        except BaseException:
            self.close()
            raise

    def _receive(self):
        replies, error = [], None
        for connection in self.connections:
            try:
                status, value = connection.recv()
            except (EOFError, OSError):
                status, value = 'error', RuntimeError("PhreeqcRM : a worker process stopped")
            if status == 'error' and error is None:
                error = value
            replies.append(value)
        if error is not None:
            raise error
        return replies

    def run(self, calls):
        """Sends each process the calls of its cells -> [reply of _run]."""
        for connection, (a, b) in zip(self.connections, self.bounds):
            try:
                connection.send(calls(a, b))
            except (OSError, ValueError):
                raise RuntimeError("PhreeqcRM : a worker process stopped")
        return self._receive()

    def close(self):
        for connection in self.connections:
            try:
                connection.send(None)
            except (OSError, ValueError):
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        for connection in self.connections:
            connection.close()
        self.connections, self.processes = [], []


def _ranges(cells):
    """[0, 1, 2, 5] -> '0-2 5'."""
    parts, start, previous = [], None, None
    for cell in cells:
        if start is None:
            start = previous = cell
        elif cell == previous + 1:
            previous = cell
        else:
            parts.append(f"{start}-{previous}" if previous > start else f"{start}")
            start = previous = cell
    if start is not None:
        parts.append(f"{start}-{previous}" if previous > start else f"{start}")
    return " ".join(parts)


def timeFactor(unit):
    """Seconds in one timeUnit."""
    return timeUnits.secondsPerUnit(unit)


def totalsMode(centralDict):
    """Whether the totals (SOLUTION_MODIFY) path is used: solMod or MultiCompoundTransport."""
    return bool(centralDict.get('solMod') or centralDict.get('MultiCompoundTransport'))


def crossSpeciation(centralDict):
    """Speciation crossDependencies in PhreeqC format, or None."""
    block = centralDict['crossDependencies']['speciation']
    return block if block['total'] else None


def extract(centralDict):
    """Database extraction, as for chemistryModule = 'phreeqc'."""
    start = time.perf_counter()
    print("Extracting PhreeqC database", end=" ", flush=True)
    primToSecSpecies, sol, phases, surf, exch, fixed, trspt = extractDB.phreeqcDBextraction(
        centralDict, centralDict['commMtrx'][centralDict['systemSpecies']])
    solution = list(dict.fromkeys(sol + ['H2O']))
    centralDict.update({
        "primToSecSpecies": primToSecSpecies,
        "transportedSpecies": trspt,
        "primarySpecies": {
            'solution': solution,
            'phases': phases,
            'surface': surf,
            'exchange': exch,
            'total': solution + exch + surf + phases,
            'primarySpeciesPhantom': ['pH', 'ph', 'pe'],
        }})
    extractDB.checkDecomposition(centralDict)
    centralDict["extractDBTime"] = time.perf_counter() - start
    print(f"({writeTime(centralDict['extractDBTime'])})")
    return centralDict


def _kinetics(centralDict):
    """(KINETICS blocks, {species: rate name}, kinetic phases) of the configuration."""
    if not centralDict.get('kinetics'):
        return None, {}, []
    species = centralDict.get('kineticsSpecies') or []
    blocks = phreeqc.kineticsBlocks(centralDict['kinetics']) if species else None
    rates = phreeqc.kineticsRates(blocks, species) if blocks else {}
    kineticPhases = [s for s in species if s in centralDict['primarySpecies']['phases']]
    return blocks, rates, kineticPhases


def _checkOptions(centralDict, species, totals):
    """ValueError for the options and species PhreeqcRM does not handle."""
    cd = centralDict
    reasons = [key for key in ('preliminaryEquilibrium', 'speciationCharge') if cd.get(key)]
    crossSpc = crossSpeciation(cd)
    if crossSpc and not totals:
        reasons.append('crossDependencies of the speciation without solMod = True or MultiCompoundTransport = True')
    if not cd.get('phasesAsPrimary', True):
        reasons.append('phasesAsPrimary = False')
    if cd.get('surfaceCounterIons'):
        reasons.append('surfaceCounterIons (diffuse layer)')
    if reasons:
        raise ValueError(f"PhreeqcRM : {', '.join(reasons)} not available ; use chemistryModule = 'phreeqc' for "
                         "these options")


def _bootstrap(centralDict, species, kinBlocks, kinRates):
    """InitialPhreeqc input holding every element of the system."""
    cd = centralDict
    used = {p for s in species for p, coeff in cd['primToSecSpecies'].get(s, {}).items() if coeff}
    text = "SOLUTION 1\n-units mol/kgw\n" + "".join(f"\t{e} 1e-10\n" for e in _elements(cd)) + "END\n"
    exchange = [x for x in cd['primarySpecies']['exchange'] if x in used]
    if exchange:
        text += "EXCHANGE 1\n" + "".join(f"\t{x} 1e-10\n" for x in exchange) + "END\n"
    surfaces = [m for m in cd['primarySpecies']['surface'] if m in used]
    if surfaces:
        text += "SURFACE 1\n" + "".join(f"\t{m} 1e-10 100 1\n" for m in surfaces) + "-no_edl\nEND\n"
    phases = [p for p in cd['primarySpecies']['phases'] if p in used]
    # pseudo-phases of fixedPhases with their reactant (NaOH ...) : FindComponents sees the elements of the reactant.
    # A redox pseudo-phase (reaction e- = e-, Fix_pe) is left out : FindComponents reads the elements of the phase
    # formula and fails on e- ('unexpected character, e') ; its reactant (O2) only holds O and H, components anyway.
    # It is still written in EQUILIBRIUM_PHASES at every step (_reactantsText).
    fixed = [f for f in phreeqc.fixedPhaseSpecs(cd.get('fixedPhases')) if f[4].split('=')[0].strip() != 'e-']
    allFixed = [f[0] for f in phreeqc.fixedPhaseSpecs(cd.get('fixedPhases'))]
    if phases or (cd.get('fixpH') and 'Fix_ph' not in allFixed) or fixed:
        text += "EQUILIBRIUM_PHASES 1\n" + "".join(f"\t{p} 0 0\n" for p in phases)
        text += "\tFix_ph 7 0\n" if cd.get('fixpH') and 'Fix_ph' not in allFixed else ""
        text += "".join(f"\t{name} {si} {reactant + ' ' if reactant else ''}0\n" for name, si, reactant, _, _ in fixed)
        text += "END\n"
    if cd.get('kinetics'):
        body = (phreeqc.kineticsText(kinBlocks, {rate: 1e-10 for rate in kinRates.values()}) if kinBlocks
                else str(cd['kinetics']))
        text += f"KINETICS 1\n{body}\nEND\n"
    return text


def _charge(name):
    """Charge at the end of a species name ('SO4-2' -> -2)."""
    match = re.search(r'([+-])(\d*)$', name)
    if not match:
        return 0.0
    return (1.0 if match.group(1) == '+' else -1.0) * (int(match.group(2)) if match.group(2) else 1)


def _surfaceMasterCharges(chemPath, masters):
    """{surface master: charge of its master species} from the database."""
    formulas, current = {}, None
    with open(chemPath, encoding='latin-1') as f:
        for line in f:
            keyword = phreeqc._blockKeyword(line)
            if keyword is not None:
                current = keyword
                continue
            words = line.split('#', 1)[0].split()
            if current == 'SURFACE_MASTER_SPECIES' and len(words) >= 2 and words[0] in masters:
                formulas[words[0]] = words[1]
    charges = {}
    for master in masters:
        formula = formulas.get(master)
        if formula is None or not formula.startswith(master):
            raise ValueError(f"PhreeqcRM : master species of the surface {master} not found in SURFACE_MASTER_SPECIES "
                             f"of {chemPath}")
        charges[master] = _charge(formula)
    return charges


def _surfaceColumns(centralDict, species):
    """Surface species columns -> {column: (sites, charge, charge of the sites in master form)}."""
    cd = centralDict
    masters = [m for m in cd['primarySpecies']['surface']
               if any(m in cd['primToSecSpecies'].get(s, {}) for s in species)]
    if not masters:
        return [], {}
    siteCharges = _surfaceMasterCharges(cd['chemPath'], masters)
    columns = {}
    for s in species:
        composition = cd['primToSecSpecies'].get(s, {})
        sites = {m: c for m, c in composition.items() if m in siteCharges and c}
        if not sites or s in masters:
            continue
        columns[s] = (sites, _charge(s), sum(count * siteCharges[m] for m, count in sites.items()))
    return masters, columns


def _elements(centralDict):
    """Dissolved elements of the system (H and O excluded)."""
    elements = list(dict.fromkeys(s.split('(')[0] for s in centralDict['primarySpecies']['solution']))
    return [e for e in elements if e not in ('H', 'O', 'H2O', 'E', 'e', 'Alkalinity', 'pH', 'ph', 'pe')]


def _spec(centralDict, species, kinBlocks, kinRates, totals):
    """Data of _create: database, totals mode, KNOBS, RATES, elements, temperature."""
    cd = centralDict
    rates = cd.get('kineticRates')
    if rates:
        rates = str(rates).strip('\n')
        words = rates.split()
        if not words or words[0].upper() != 'RATES':
            rates = "RATES\n" + rates
    return {
        'chemPath': str(cd['chemPath']),
        'totals': totals,
        'knobs': phreeqc.knobsText(cd.get('knobs')),
        'rates': rates or None,
        'bootstrap': _bootstrap(cd, species, kinBlocks, kinRates),
        'temperature': float(cd.get('tempDefault', 298.15)) - 273.15,     # °C for PhreeqcRM, tempDefault in K
    }


def _phreeqcrm():
    """phreeqcrm module. Its SWIG wrappers raise DeprecationWarnings at import ('builtin type SwigPyPacked has no
    __module__ attribute'), which were counted as warnings of the run: silenced. Missing package: ImportError with the
    fix, when the run is set up."""
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            import phreeqcrm
    except ImportError as err:
        raise ImportError("chemistryModule = 'PhreeqcRM' : the phreeqcrm package is not installed (it comes with "
                          "pycte) : pip install phreeqcrm (macOS older than 15 : pip install phreeqcrm==0.0.18)") from err
    return phreeqcrm


def _create(n, spec):
    """PhreeqcRM instance (one thread) of n cells."""
    phreeqcrm = _phreeqcrm()
    totals = spec['totals']
    rm = phreeqcrm.PhreeqcRM(n, 1)
    rm.SetErrorHandlerMode(0)
    rm.SetScreenOn(False)
    rm.SetComponentH2O(not totals)
    rm.SetSpeciesSaveOn(not totals)
    rm.SetUnitsSolution(2)
    for setter in (rm.SetUnitsExchange, rm.SetUnitsPPassemblage, rm.SetUnitsKinetics, rm.SetUnitsSurface,
                   rm.SetUnitsGasPhase, rm.SetUnitsSSassemblage):
        setter(1)
    rm.UseSolutionDensityVolume(False)
    ones = [1.0] * n
    rm.SetPorosity(ones)
    rm.SetSaturationUser(ones)
    rm.SetRepresentativeVolume(ones)
    rm.SetDensityUser(ones)
    # ASCII form of the path: the PhreeqcRM library does not open a path with accents on Windows (Données)
    _checked(rm, rm.LoadDatabase(extractDB.asciiPath(spec['chemPath'])), f"loading the database {spec['chemPath']}")
    _checked(rm, rm.RunString(True, True, True, "PRINT\n-warnings 0\nEND\n"), "PRINT -warnings 0")
    knobs = spec['knobs']     # same keyword as phreeqc.py ; kept by the workers for every RunCells
    if knobs:
        _checked(rm, rm.RunString(True, True, False, knobs), f"knobs (KNOBS block)\n{knobs}")
    if spec['rates']:
        if rm.RunString(True, True, False, spec['rates'] + "\nEND\n") < 0:
            raise ValueError(f"kineticRates : PhreeqcRM could not read the RATES block :\n{rm.GetErrorString()}")
    _checked(rm, rm.RunString(False, True, False, spec['bootstrap']), "definition of the elements of the system")
    _checked(rm, rm.FindComponents(), "FindComponents")
    _checked(rm, rm.InitialSolutions2Module([1] * n), "InitialSolutions2Module")
    rm.SetTemperature([spec['temperature']] * n)
    rm.SetPressure([1.0] * n)
    return rm


def setup(centralDict):
    """Checks the options and creates the PhreeqcRM instance(s) of the run."""
    global _state
    cd = centralDict
    start = time.perf_counter()
    totals = totalsMode(cd)
    species = list(cd['systemSpecies'])
    _checkOptions(cd, species, totals)
    kinBlocks, kinRates, kineticPhases = _kinetics(cd)

    n = len(cd['commMtrx'])
    processes = min(n, max(1, int(cd.get('speciationThreads', 1) or 1)))
    resetPhreeqcRM()
    spec = _spec(cd, species, kinBlocks, kinRates, totals)
    rm = _create(n if processes == 1 else 1, spec)

    components = [str(c) for c in rm.GetComponents()]
    elements = [c for c in components if c not in ('H2O', 'H', 'O', 'Charge')]
    surfaceMasters, surfaceColumns = _surfaceColumns(cd, species)
    reactants = set(cd['primarySpecies']['exchange']) | set(cd['primarySpecies']['phases']) | set(surfaceMasters)
    phantom = set(cd['primarySpecies']['primarySpeciesPhantom'])
    dissolved = set(cd['primarySpecies']['solution']) | {'H', 'O', 'H2O'}
    allowed = reactants | phantom
    if totals:
        allowed |= dissolved
        names, index = [], {}
    else:
        names = [str(s) for s in rm.GetSpeciesNames()]
        index = {s: k for k, s in enumerate(names)}
    unknown = []
    for s in species:
        if s in index:
            continue
        composition = {p: c for p, c in cd['primToSecSpecies'].get(s, {}).items() if c}
        valid = allowed | dissolved if s in surfaceColumns else allowed
        if not composition or any(p not in valid for p in composition):
            unknown.append(f"{s} -> {composition}")
    if unknown:
        kind = "totals or species" if totals else "aqueous species of PhreeqcRM"
        raise ValueError(f"PhreeqcRM : these columns are neither {kind} (elements {', '.join(elements)}) nor "
                         "exchange species, surface species or phases :\n  " + "\n  ".join(unknown))
    masterColumns = [m for m in surfaceMasters if m in species]
    if masterColumns:
        raise ValueError(f"PhreeqcRM : {masterColumns} are surface masters : give the surface species as columns "
                         "(e.g. Hfo_wOH, Hfo_wOH2+, Hfo_wO-, Hfo_wOCa+), their sum is the number of sites")
    if surfaceMasters:
        surfaceNotListed = [str(s) for s in rm.GetSurfaceSpecies() if str(s) not in species]
        if surfaceNotListed:
            warningManager.warn(f"PhreeqcRM : surface species of the system absent from coupledVariables : "
                                f"{', '.join(dict.fromkeys(surfaceNotListed))} ; their sites and content are not "
                                "kept from one speciation to the next")
    released = {e.split('(')[0] for s in surfaceColumns for e in cd['primToSecSpecies'][s] if e not in surfaceMasters}
    missing = [e for e in released if e not in ('H', 'O') and e not in elements]
    if missing:
        raise ValueError(f"PhreeqcRM : elements {missing} of the surface species are not components of PhreeqcRM "
                         f"({components})")

    _state = {
        'rm': rm,
        'workers': None,
        'bounds': [(0, n)],
        'n': n,
        'totals': totals,
        'species': species,
        'kinBlocks': kinBlocks,
        'kinRates': kinRates,
        'kineticPhases': kineticPhases,
        'surfaceMasters': surfaceMasters,
        'surfaceColumns': surfaceColumns,
        'components': components,
        'defined': {key: np.zeros(n, dtype=bool) for key, _, _, _ in REACTANTS},
        'selectedOutput': None,
    }
    if totals:
        missing = [e for e in _elements(cd) if e not in components]
        if missing:
            raise ValueError(f"PhreeqcRM : elements {missing} are not components of PhreeqcRM ({components})")
        _state.update({
            'elementRow': {e: components.index(e) for e in elements},
            'elementOf': {p: p.split('(')[0] for p in cd['primarySpecies']['solution']
                          if p.split('(')[0] in elements},
            'hydrogen': [p for p in cd['primarySpecies']['solution'] if p == 'H' or p.startswith('H(')],
            'oxygen': [p for p in cd['primarySpecies']['solution'] if p == 'O' or p.startswith('O(')],
        })
    else:
        gfw = dict(zip(components, rm.GetGfw()))
        aqueous = [s for s in species if s in index]
        notListed = [s for s in names if s != 'H2O' and s not in aqueous]
        if notListed:
            warningManager.warn(f"PhreeqcRM : aqueous species of the system absent from coupledVariables : "
                                f"{', '.join(notListed)} ; they are not transported and their amount is not kept "
                                "from one speciation to the next (as with chemistryModule = 'phreeqc')")
        _state.update({
            'aqueous': aqueous,
            'aqueousIndex': np.array([index[s] for s in aqueous], dtype=int),
            'nSpecies': len(names),
            'h2o': index['H2O'],
            'waterMoles': 1000.0 / float(gfw['H2O']),
        })
        if surfaceColumns:
            stoich = rm.GetSpeciesStoichiometry()
            matrix = np.zeros((len(components), len(names)))
            for k, name in enumerate(names):
                for key, coeff in dict(stoich[name]).items():
                    matrix[components.index(str(key)), k] = coeff
            _state['speciesStoichiometry'] = matrix
    if processes > 1:
        workers = _Workers(n, processes, spec)
        _state.update({'rm': None, 'workers': workers, 'bounds': workers.bounds})
    cd['PhreeqcRMInitTime'] = cd.get('PhreeqcRMInitTime', 0) + time.perf_counter() - start
    return centralDict


def _addSurfaces(centralDict, st, comm, siteDefined, conc, content):
    """Adds to conc what the surface species give back to the solution."""
    components = st['components']
    for s, (sites, charge, siteCharge) in st['surfaceColumns'].items():
        amount = comm[s].to_numpy(dtype=float)
        mask = np.logical_and.reduce([siteDefined[m] for m in sites])
        conc[components.index('Charge')] += amount * charge - np.where(mask, amount * siteCharge, 0.0)
        if content:
            for p, coeff in centralDict['primToSecSpecies'][s].items():
                if p not in sites and coeff:
                    conc[components.index(p.split('(')[0])] += amount * coeff


def _speciesConcentrations(centralDict, st, comm, n):
    """Species mode: species concentrations (species x cells)."""
    cd = centralDict
    water = (np.asarray(cd['water'], dtype=float) if cd.get('water') is not None and cd.get('water') is not False
             else np.ones(n))
    values = comm[st['aqueous']].to_numpy(dtype=float)
    values = np.where(values > cd['cutoffs']['solution'], values, 0.0)
    conc = np.zeros((st['nSpecies'], n))
    conc[st['aqueousIndex']] = (values * water[:, None]).T
    conc[st['h2o']] = water * st['waterMoles']
    return conc


def _speciesComponents(centralDict, st, comm, siteDefined, n):
    """Species mode with surfaces: components of the species plus surface contributions."""
    species = _speciesConcentrations(centralDict, st, comm, n)
    conc = st['speciesStoichiometry'] @ species
    _addSurfaces(centralDict, st, comm, siteDefined, conc, content=True)
    components = st['components']
    water = species[st['h2o']]
    conc[components.index('H2O')] = water
    conc[components.index('H')] -= 2 * water
    conc[components.index('O')] -= water
    return conc


def _componentConcentrations(centralDict, st, comm, primary, siteDefined, n):
    """Totals mode: component concentrations for SetConcentrations (as SOLUTION_MODIFY)."""
    cd = centralDict
    components = st['components']
    zeros = np.zeros(n)

    def column(name):
        return np.asarray(primary[name], dtype=float) if name in primary else zeros

    conc = np.zeros((len(components), n))
    conc[components.index('H')] = sum((column(p) for p in st['hydrogen']), zeros) + 2 * column('H2O')
    conc[components.index('O')] = sum((column(p) for p in st['oxygen']), zeros) + column('H2O')
    names = crossDependenciesManager.byQuantity(cd['crossDependencies']['speciation'], 'input')
    charge = names.get('CHARGE_BALANCE')
    if charge in comm.columns:
        conc[components.index('Charge')] = comm[charge].to_numpy(dtype=float)
    cutoff = cd['cutoffs']['solution']
    for p, element in st['elementOf'].items():
        if p in primary:
            values = np.asarray(primary[p], dtype=float)
            conc[st['elementRow'][element]] += np.where(values > cutoff, values, 0.0)
    # the charge of the surface species goes back to the solution with the coupled charge balance ; without it the
    # solution gets charge 0, the whole system being neutral (as -cb 0 in phreeqc.py)
    if charge in comm.columns:
        _addSurfaces(cd, st, comm, siteDefined, conc, content=False)
    return conc


def _reactantsText(centralDict, st, primary, n):
    """InitialPhreeqc input of the reactants of each cell."""
    cd = centralDict
    cutoffs = cd['cutoffs']
    defined = {key: np.zeros(n, dtype=bool) for key, _, _, _ in REACTANTS}
    blocks = [[] for _ in range(n)]

    exchange = [(x, np.asarray(primary[x], dtype=float)) for x in cd['primarySpecies']['exchange'] if x in primary]
    # amounts written once per column, repr(float(values[i])) (_reprs): the same text, without converting a numpy
    # scalar per cell
    for x, values in exchange:
        amounts = _reprs(values)
        for i in np.flatnonzero(values > cutoffs['exchange']).tolist():
            if not defined['exchange'][i]:
                blocks[i].append(f"EXCHANGE {i + 1}\n")
                defined['exchange'][i] = True
            blocks[i].append(f"\t{x} {amounts[i]}\n")

    fixed = phreeqc.fixedPhaseSpecs(cd.get('fixedPhases'))
    if cd.get('fixpH') or fixed or cd['primarySpecies']['phases']:
        phases = [(p, np.asarray(primary[p], dtype=float)) for p in cd['primarySpecies']['phases']
                  if p in primary and p not in st['kineticPhases']]
        lines = [[] for _ in range(n)]
        if cd.get('fixpH') and 'Fix_ph' not in [f[0] for f in fixed]:
            line = f"Fix_ph {cd['fixpH']}\n"
            for i in range(n):
                lines[i].append(line)
        # pseudo-phases of fixedPhases, rewritten at every call in every cell (same as phreeqc.py) : the reservoir
        # of reactant is the same at every step and the target SI is held, as Fix_H+ / Fix_pe in a PhreeqC TRANSPORT
        for name, si, reactant, amount, _ in fixed:
            line = f"{name} {si} {reactant + ' ' if reactant else ''}{amount}\n"
            for i in range(n):
                lines[i].append(line)
        # lines of the phases, phase by phase: f"\t{p} {SI} {amount!r}  {reversibility}\n" for the cells above the
        # cutoff, "" for the others (a line is never empty: "" adds nothing to the text of a cell)
        phaseLines = []
        for p, values in phases:
            above = (values > cutoffs['phases']).tolist()
            if any(above):
                head, tail = f"\t{p} {cd['SI'][p]} ", f"  {cd['mineralReversibility'][p]}\n"
                phaseLines.append([head + amount + tail if there else "" for amount, there in zip(_reprs(values), above)])
        # text of each cell: its fixpH / fixedPhases lines, then the lines of its phases in the order of the phases
        texts = map("".join, zip(*phaseLines)) if phaseLines else [""] * n
        for i, text in enumerate(texts):
            if lines[i] or text:
                blocks[i].append(f"EQUILIBRIUM_PHASES {i + 1}\n")
                blocks[i].extend(lines[i])
                if text:
                    blocks[i].append(text)
                defined['phases'][i] = True

    if cd.get('kinetics') and cd['dtStep'] > 0:
        kinBlocks, kinRates = st['kinBlocks'], st['kinRates']
        amounts = {k: np.asarray(primary[k], dtype=float) for k in kinRates}
        # initial amounts (engine.py): M0 of the rate laws, -m = current amount (see phreeqc.kineticsText)
        initial = cd.get('kineticsM0')
        m0 = ({k: initial[k].to_numpy(dtype=float) for k in kinRates if k in initial}
              if initial is not None and len(initial) == n else None)
        for i in range(n):
            if kinBlocks:
                body = phreeqc.kineticsText(kinBlocks, {kinRates[k]: float(amounts[k][i]) for k in kinRates},
                                            {kinRates[k]: float(m0[k][i]) for k in m0} if m0 else None)
            else:
                body = str(cd['kinetics'])
            blocks[i].append(f"KINETICS {i + 1}\n{body}\n")
            if cd.get('step_divide'):
                blocks[i].append(f"-step_divide {cd['step_divide']}\n")
            defined['kinetics'][i] = True

    sites = [(m, np.asarray(primary[m], dtype=float)) for m in st['surfaceMasters'] if m in primary]
    siteDefined = {m: values > cutoffs['surface'] for m, values in sites}
    siteValues = [(m, _reprs(values), siteDefined[m].tolist()) for m, values in sites]
    for i in range(n):
        lines = [f"\t{m} {amounts[i]} 100 1\n" for m, amounts, there in siteValues if there[i]]
        if lines:
            blocks[i].append(f"SURFACE {i + 1}\n")
            blocks[i].extend(lines)
            blocks[i].append("-no_edl\n")
            defined['surface'][i] = True

    return ["".join(block) for block in blocks], defined, siteDefined


def _stepCalls(step, a, b):
    """PhreeqcRM calls of a step for the cells a to b - 1."""
    calls = [(step['transfer'], (step['conc'][:, a:b].ravel(),), step['transfer'])]
    text = "".join(step['blocks'][a:b])
    if text:
        calls.append(('RunString', (False, True, False, text + "END\n"), "definition of the reactants"))
    deletions = ""
    for key, _, transfer, option in REACTANTS:
        defined = step['defined'][key][a:b]
        if defined.any():
            calls.append((transfer, (np.where(defined, np.arange(a + 1, b + 1), -1).tolist(),), transfer))
        removed = np.flatnonzero(step['previous'][key][a:b] & ~defined)
        if removed.size:
            deletions += f"{option} {_ranges(removed.tolist())}\n"
    if deletions:
        calls.append(('RunString', (True, False, False, f"DELETE\n{deletions}END\n"), "DELETE"))
    if step['selected'] is not None:
        calls += [('RunString', (True, False, False, step['selected']), "SELECTED_OUTPUT"),
                  ('SetSelectedOutputOn', (True,), "SetSelectedOutputOn")]
    for method, values in (('SetTemperature', step['temperature']), ('SetPressure', step['pressure'])):
        if values is not None:
            calls.append((method, (values[a:b].tolist(),), method))
    calls.append(('SetTimeStep', (step['dt'],), "SetTimeStep"))
    return calls


RUN_CALLS = [('RunCells', (), "RunCells"), ('GetSelectedOutputHeadings', (), "GetSelectedOutputHeadings"),
             ('GetSelectedOutput', (), "GetSelectedOutput")]


def _execute(st, calls):
    """Runs calls(a, b) on the instance or on each worker process."""
    if st['workers'] is None:
        return [_run(st['rm'], calls(0, st['n']))]
    return st['workers'].run(calls)


def _selectedOutputText(centralDict, st, kineticsRan):
    """SELECTED_OUTPUT of phreeqc.py for PhreeqcRM."""
    cd = centralDict
    phases = cd['primarySpecies']['phases']
    mctTotals = cd.get('mctTotals') if cd.get('MultiCompoundTransport') else None
    crossSpc = crossSpeciation(cd)
    text = ""
    if crossSpc:
        punchNames = list(crossSpc['output'])
        punchCmds = [crossDependenciesManager.PUNCH.get(crossSpc['quantity'][n], crossSpc['quantity'][n]) for n in punchNames]
        if mctTotals:
            punchNames += ['H(mol/kgw)', 'O(mol/kgw)']
            punchCmds += ['TOT("H")', 'TOT("O")']
        text += "USER_PUNCH 1\n-headings" + "".join(f"\t{name}" for name in punchNames) + "\n"
        text += "".join(f"{k} PUNCH {cmd}\n" for k, cmd in enumerate(punchCmds, start=1))
    text += "SELECTED_OUTPUT 1\n-reset false\n"
    if cd.get('molesStorage') and not (cd.get('userVarBool') or {}).get('water'):
        text += "-water true\n"
    molalities = [s for s in st['species'] if s not in phases]
    if mctTotals:
        totals = [t for t in mctTotals if t not in ('H', 'O')]
        if totals:
            text += "-totals" + "".join(f"\t{t}" for t in totals) + "\n"
        molalities = [s for s in molalities if s not in mctTotals]
    if molalities:
        text += "-molalities" + "".join(f"\t{s}" for s in molalities) + "\n"
    if phases:
        text += "-equilibrium_phases" + "".join(f"\t{p}" for p in phases) + "\n"
    kineticOutput = [s for s in st['kineticPhases'] if s not in ((cd.get('userVarList') or {}).get('kinetic_reactants') or [])]
    if kineticsRan and kineticOutput:
        text += "-kinetic_reactants" + "".join(f"\t{s}" for s in kineticOutput) + "\n"
    for key, value in (cd.get('userVarBool') or {}).items():
        if value:
            text += f"-{key} True\n"
    for key, values in (cd.get('userVarList') or {}).items():
        if values:
            text += f"-{key}" + "".join(f"\t{v}" for v in values) + "\n"
    return text + "END\n"


def _fatal(centralDict, message):
    """Raises the PhreeqcRM error (written to warning.log by engine.main)."""
    cd = centralDict
    raise RuntimeError(f"PhreeqcRM : time step n°{cd['lStep'] + 1}, t={cd['tStep']}{cd['timeUnit']}, "
                       f"PID={os.getpid()} : fatal PhreeqcRM error, batch aborted :\n{message}")


def spct(centralDict):
    cd = centralDict
    startRM = time.perf_counter()
    display.stepPrint("PhreeqcRM", end=" ", flush=True)
    init = 0.0
    if _state is None or _state['n'] != len(cd['commMtrx']):
        before = cd.get('PhreeqcRMInitTime', 0)
        setup(cd)
        init = cd['PhreeqcRMInitTime'] - before
        cd['PhreeqcRMInitTime'] = before
    st = _state
    n = st['n']
    comm = cd['commMtrx']
    species = st['species']

    stoich = phreeqc.stoichMatrix(cd, species)
    primary = phreeqc.primaryColumnsOf(comm, species, stoich)
    blocks, defined, siteDefined = _reactantsText(cd, st, primary, n)
    kineticsRan = bool(cd.get('kinetics')) and cd['dtStep'] > 0
    if st['totals']:
        conc, transfer = _componentConcentrations(cd, st, comm, primary, siteDefined, n), 'SetConcentrations'
    elif st['surfaceColumns']:
        conc, transfer = _speciesComponents(cd, st, comm, siteDefined, n), 'SetConcentrations'
    else:
        conc, transfer = _speciesConcentrations(cd, st, comm, n), 'SpeciesConcentrations2Module'
    selected = _selectedOutputText(cd, st, kineticsRan)

    # temperature of the cells, in °C for PhreeqcRM: column coupled as tk (K), whatever its alias; otherwise
    # tempDefault (K), set once by setup
    names = crossDependenciesManager.byQuantity(cd['crossDependencies']['speciation'], 'input')
    temp = atm = None
    if 'tk' in names:
        temp = comm[names['tk']].to_numpy(dtype=float) - 273.15
    if 'pressure' in names:
        atm = comm[names['pressure']].to_numpy(dtype=float) / crossDependenciesManager.ATM

    step = {
        'transfer': transfer,
        'conc': conc,
        'blocks': blocks,
        'defined': defined,
        'previous': st['defined'],
        'selected': selected if selected != st['selectedOutput'] else None,
        'temperature': temp,
        'pressure': atm,
        'dt': float(cd['dtStep']) * timeFactor(cd['timeUnit']),
    }
    try:
        _execute(st, lambda a, b: _stepCalls(step, a, b))
        st['defined'] = defined
        st['selectedOutput'] = selected
        ref = time.perf_counter()
        replies = _execute(st, lambda a, b: RUN_CALLS)
    except RuntimeError as error:
        _fatal(cd, f"{error}\n{''.join(blocks)[:2000]}")
    if st['workers'] is None:
        calcWallClock, calcPrcsTime = replies[0][1]
    else:
        calcWallClock = time.perf_counter() - ref
        calcPrcsTime = sum(times[1] for _, times in replies)

    headings = [str(h) for h in replies[0][0][1]]
    parts = [np.asarray(results[2], dtype=float).reshape(len(headings), b - a).T
             for (results, _), (a, b) in zip(replies, st['bounds'])]
    output = parts[0] if len(parts) == 1 else np.vstack(parts)
    rmOutput = pd.DataFrame(output, columns=frames.columnIndex(headings), index=comm.index)

    crossOut = cd['crossDependencies']['speciation']['output']
    crossIn = cd['crossDependencies']['speciation']['input']
    crossTrspt = cd['crossDependencies']['transport']['total']
    mctTotals = cd.get('mctTotals') if cd.get('MultiCompoundTransport') else None
    waterKg = rmOutput['mass_H2O'] if cd.get('molesStorage') else None
    kineticPhases = st['kineticPhases']
    phases = cd['primarySpecies']['phases']
    def source(c):
        """New value of the commMtrx column c : (rmOutput or comm, column, multiplied by the water mass)."""
        if c in crossOut:
            return rmOutput, c, False
        if c in crossIn or c in crossTrspt or c not in species:
            return comm, c, False
        if c in kineticPhases:
            return (rmOutput, f"k_{c}", False) if kineticsRan else (comm, c, False)
        if c in phases:
            return rmOutput, c, False
        col = f"{c}(mol/kgw)" if mctTotals and c in mctTotals else f"m_{c}(mol/kgw)"
        return rmOutput, col, waterKg is not None

    if phreeqc._floatFrame(rmOutput) and phreeqc._floatFrame(comm):
        arrays = {False: (rmOutput.to_numpy(), {name: j for j, name in enumerate(rmOutput.columns)}),
                  True: (comm.to_numpy(), {name: j for j, name in enumerate(comm.columns)})}
        water = arrays[False][0][:, arrays[False][1]['mass_H2O']] if waterKg is not None else None
        values = np.empty((len(comm.columns), n))
        for j, c in enumerate(comm.columns):
            frame, col, perWater = source(c)
            array, position = arrays[frame is comm]
            values[j] = array[:, position[col]] * water if perWater else array[:, position[col]]
        commMtrxSpct = pd.DataFrame(values.T, index=comm.index, columns=comm.columns)
    else:
        columns = {}
        for c in comm.columns:
            frame, col, perWater = source(c)
            columns[c] = frame[col] * waterKg if perWater else frame[col]
        commMtrxSpct = pd.DataFrame(columns, index=comm.index)
    if waterKg is not None:
        cd['cellWaterMass'] = waterKg.to_numpy(dtype=float)

    if outputManager.wanted(cd, 'speciation'):
        coords = frames.columnsOf(comm, cd['coord'])
        primSpecies = pd.DataFrame(primary, index=comm.index)
        outputManager.write(cd, pd.concat([coords, primSpecies], axis=1), 'primarySpecies', 'PrimarySpecies')
        outputManager.write(cd, pd.concat([coords, rmOutput], axis=1), 'speciation', 'PhreeqcRM')

    elapsed = time.perf_counter() - startRM
    cd.update({
        "commMtrx": commMtrxSpct,
        "PhreeqcRMCalcTime_WallClock": cd.get('PhreeqcRMCalcTime_WallClock', 0) + calcWallClock,
        "PhreeqcRMCalcTime_ProcessorTime": cd.get('PhreeqcRMCalcTime_ProcessorTime', 0) + calcPrcsTime,
        "PhreeqcRMInterfTime_WallClock": cd.get('PhreeqcRMInterfTime_WallClock', 0) + elapsed - calcWallClock - init,
        "PhreeqcRMInitTime": cd.get('PhreeqcRMInitTime', 0) + init,
        "PhreeqcRMTotalTime": cd.get('PhreeqcRMTotalTime', 0) + elapsed,
    })
    display.stepPrint(f"({writeTime(elapsed)})")
    return centralDict
