"""pycte interface to nativeSpeciation: spct(centralDict) equilibrates commMtrx, split between workers."""

import concurrent.futures
import difflib
import os
import time

import numpy as np
import pandas as pd

from .chemistryDataframe import Config, equilibrate_dataframe, _stack_u, _system
from .nativeSpeciation import H2O_PER_KG
try:
    from . import outputManager
    from . import warningManager
    from . import display
except ImportError:
    import outputManager
    import warningManager
    import display


_PROCESS_CALL_COST = 0.01
_PROCESS_START_COST = 2.0
_speciationProcesses = None


def resetProcesses():
    """End of run: the choice between worker processes and a single process is made again at the next run."""
    global _speciationProcesses
    _speciationProcesses = None


def _chooseProcesses(centralDict, calcWall, nProcs):
    """After a single-process call with warm start: worker processes kept only if their gain over the remaining steps
    exceeds their cost."""
    global _speciationProcesses
    remaining = max(len(centralDict.get('dtpycte') or []) - centralDict.get('lStep', 0), 1)
    gain = remaining * (calcWall * (1.0 - 1.0 / nProcs) - _PROCESS_CALL_COST) - _PROCESS_START_COST
    _speciationProcesses = gain > 0
    if not _speciationProcesses:
        warningManager.info(f"nativeSpeciation : speciationThreads = {nProcs} not used : a speciation call takes "
                            f"{1e3 * calcWall:.1f} ms in a single process, less than the cost of {nProcs} worker "
                            f"processes (~{1e3 * _PROCESS_CALL_COST:.0f} ms per call and ~{_PROCESS_START_COST:.0f} s "
                            "to start) : single-process speciation (same results)")


def writeTime(tps, arr=2):
    if tps >= 3600 * 24:
        return f"{tps / (3600 * 24):.{arr}f} d"
    elif tps >= 3600:
        return f"{tps / 3600:.{arr}f} h"
    elif tps >= 60:
        return f"{tps / 60:.{arr}f} min"
    elif tps < 1:
        return f"{tps * 1000:.{arr}f} msec"
    else:
        return f"{tps:.{arr}f} sec"


def buildConfig(centralDict) -> Config:
    """centralDict['chemistry'] (only 'elements' required) -> solver Config."""
    chem = centralDict.get('chemistry', {})
    database = chem.get('chemPath', centralDict.get('chemPath'))
    if database is None:
        raise KeyError("no 'chemPath' in centralDict nor in "
                       "centralDict['chemistry']")
    return Config(
        database=database,
        comp=bool(centralDict.get('MultiCompoundTransport', False)),
        elements=tuple(chem['elements']),
        minerals=tuple(chem.get('minerals', ())),
        gases=dict(chem.get('gases', {})),
        sorbed_species=tuple(chem.get('sorbedSpecies', ())),
        cec=chem.get('cec', 0.0),
        exchanger=chem.get('exchanger', 'X'),
        temperature=chem.get('temperature', 298.15),
        ph=chem.get('ph', None),
        ph_reagent=chem.get('phReagent', None),
        diagnostics=dict(chem.get('diagnostics', {})),
        species=tuple(chem.get('species', ())),
        totals_output=chem.get('totalsOutput', 'dissolved'),
        redox=chem.get('redox', False),
        pe=chem.get('pe', 4.0),
        tol=chem.get('tol', 1e-10),
        charge_balance=bool(chem.get('chargeBalance', False)),
        water=bool(centralDict.get('molesStorage', False)),
    )


def checkColumns(centralDict):
    """Species mode: a coupled species (systemSpecies) without a role in the chemical system is transported as an inert
    tracer, untouched by the speciation (MultiCompoundTransport: multiCompoundSetup refuses it): warning listing them
    (a misspelled name shows there; tracers or species of elements left out of chemistry['elements'] are allowed)."""
    cfg = buildConfig(centralDict)
    if cfg.comp:
        return
    temperature = 298.15 if isinstance(cfg.temperature, str) else float(cfg.temperature)
    system, _ = _system(cfg, cfg.database_source(), temperature)
    roles = (set(cfg.species or system.species) | set(cfg.elements) | set(cfg.minerals) | set(cfg.sorbed_species)
             | set(cfg.diagnostics.values()) | {'H', 'H2O'}
             | {v for v in (cfg.cec, cfg.temperature, cfg.ph, *cfg.gases.values()) if isinstance(v, str)})
    unknown = [c for c in centralDict['systemSpecies'] if c not in roles]
    if unknown:
        close = {c: difflib.get_close_matches(c, sorted(system.species), n=2, cutoff=0.6) for c in unknown}
        warningManager.warn("nativeSpeciation : these coupled variables are neither species of the chemical system "
                            f"(elements {', '.join(cfg.elements)}) nor columns of the chemistry setting (minerals, "
                            "sorbedSpecies, diagnostics, cec, temperature, ph, gases) : the speciation leaves them "
                            "unchanged (inert tracers) : "
                            + ", ".join(c + (f" (did you mean {' or '.join(close[c])} ?)" if close[c] else "")
                                        for c in unknown)
                            + ". If they are chemical species, check their names and chemistry['elements']")


def multiCompoundSetup(centralDict, transportedGiven=False):
    """MultiCompoundTransport: commMtrx, boundaries and renewal solutions converted to component totals."""
    cd = centralDict
    cfg = buildConfig(cd)
    temperature = 298.15 if isinstance(cfg.temperature, str) else float(cfg.temperature)
    system, _ = _system(cfg, cfg.database_source(), temperature)
    water = bool(cd.get('molesStorage'))
    elements = list(cfg.elements)
    totalsCols = elements + ([] if cfg.charge_balance or 'H' in elements else ['H']) + (['H2O'] if water else [])
    master = {e: system.db.master[e].species if e in system.db.master else e for e in elements}
    jCols = [system.index[master[e]] for e in elements] + ([system.index['H+']] if 'H' in totalsCols else [])
    rowOf = {n: i for i, n in enumerate(system.species) if system.is_aq[i]}

    position = {c: k for k, c in enumerate(totalsCols)}

    def totals(table, freeWater=H2O_PER_KG):
        out = np.zeros((len(table), len(totalsCols)))
        if water:
            out[:, position['H2O']] = table['H2O'].astype(float).to_numpy() if 'H2O' in table else freeWater
        for col in table.columns:
            if col == 'H2O':
                continue
            values = table[col].to_numpy(dtype=float)
            if col in totalsCols:
                out[:, position[col]] += values
                continue
            i = rowOf[col]
            out[:, :len(jCols)] += np.outer(values, system.nu[i, jCols])
            if water:
                out[:, position['H2O']] += values * system.nu_free[i, 0]
        return pd.DataFrame(out, index=table.index, columns=totalsCols)

    dropped = [c for c in cd['coupledVariables'] if c not in rowOf and c != 'H2O' and c in system.db.species
               and system.db.species[c].kind == 'aq']
    if dropped:
        raise ValueError(f"nativeSpeciation, MultiCompoundTransport : {dropped} are aqueous species outside the chemical "
                         f"system of chemistry['elements'] = {elements} : their amounts cannot be converted to totals")
    comm = cd['commMtrx']
    converted = [c for c in cd['coupledVariables'] if c in rowOf or c in totalsCols]
    tot = totals(comm[converted])
    kept = [c for c in cd['coupledVariables'] if c not in converted]
    coord = [c for c in comm.columns if c not in cd['coupledVariables']]
    cd['commMtrx'] = pd.concat([comm[coord], tot, comm[kept]], axis=1)
    cd['coupledVariables'] = totalsCols + kept
    cross = cd.get('crossDependencies', {}).get('names', [])
    cd['systemSpecies'] = [c for c in cd['coupledVariables'] if c not in cross]
    cd['anythingButSpecies'] = coord
    given = list(cd.get('transportedSpecies') or []) if transportedGiven else []
    if given:
        cd['transportedSpecies'] = ([c for c in given if c in kept]
                                    + (totalsCols if any(c in converted for c in given) else []))
    else:
        cd['transportedSpecies'] = list(totalsCols)

    def convertValues(values):
        mobile = [k for k in values if k in rowOf or k in totalsCols]
        converted = totals(pd.DataFrame([{k: float(values[k]) for k in mobile}])).iloc[0].to_dict()
        converted.update({k: values[k] for k in values if k not in mobile})
        return converted

    for key in ('firstBoundary', 'secondBoundary'):
        values = cd.get(key)
        if not values:
            continue
        cd[key] = convertValues(values)
    for group in cd.get('renewal') or []:
        if group['concentration']:
            group['concentration'] = convertValues(group['concentration'])

    print(f"MultiCompoundTransport : initial conditions -> totals of {', '.join(totalsCols)}, transported : "
          f"{', '.join(cd['transportedSpecies'])}")
    return centralDict


def speciation_native(centralDict, commMtrx, u_start=None, want_totals=True):
    """Equilibrates one slice of commMtrx (tuple of speciation_xGEMS, plus warm start and water mass). want_totals: the
    frame of the component totals (written to the speciation output only), otherwise None."""
    cfg = buildConfig(centralDict)
    output, info = equilibrate_dataframe(commMtrx, cfg, n_procs=1, u_start=u_start, want_totals=want_totals)
    return (output, info['component_totals'], info['statuses'],
            info['iterations'], info['solve_time_wall'], info['init_time'], info['u'],
            info['water_mass'])


# keys of centralDict read by speciation_native and the functions it calls (buildConfig) in a worker process
# (tests/test_worker_dicts.py checks that the list holds every key they read): with speciationThreads > 1 the workers
# receive these keys only, instead of the whole dictionary (dtpycte, ~370 kB for 100 000 steps, the transport settings
# and arrays, the state of the other solvers ...), pickled at every call for every process
WORKER_KEYS = frozenset({'MultiCompoundTransport', 'chemPath', 'chemistry', 'molesStorage'})


def _workerDict(centralDict):
    """centralDict as the workers read it: the WORKER_KEYS, in the order of centralDict (the same objects)."""
    return {k: v for k, v in centralDict.items() if k in WORKER_KEYS}


def _task(args):
    """Worker entry point, at module level so it can be pickled."""
    return speciation_native(*args)


def spct(centralDict):
    display.stepPrint("Native speciation", end=" ", flush=True)
    startNative = time.perf_counter()

    commMtrx = centralDict['commMtrx']
    coord = list(centralDict.get('coord', ['x', 'y', 'z']))
    if 'geometry' in centralDict:
        coord = coord[:centralDict['geometry']]
    coord = [c for c in coord if c in commMtrx.columns]
    requested = max(1, int(centralDict.get('speciationThreads', 1)))
    uStart = centralDict.get('nativeSpeciationU')
    if not (isinstance(uStart, np.ndarray) and uStart.ndim == 2 and len(uStart) == len(commMtrx)):
        uStart = None
    nProcs = requested if _speciationProcesses else 1
    # component totals built only for the speciation output (the only place they are written)
    wantTotals = outputManager.wanted(centralDict, 'speciation')

    if nProcs > 1:
        chunk_size = int(np.ceil(len(commMtrx) / nProcs))
        starts = range(0, len(commMtrx), chunk_size)
        split = [commMtrx.iloc[i:i + chunk_size] for i in starts]
        uSplit = [None if uStart is None else uStart[i:i + chunk_size] for i in starts]

        light = _workerDict(centralDict)            # what the workers read, not the whole dictionary
        executor = warningManager.executor(nProcs)
        futures = [warningManager.submit(executor, _task, (light, chunk, u, wantTotals))
                   for chunk, u in zip(split, uSplit)]
        results = [f.result() for f in futures]

        output, primSpc, status, iteration, calc, initWorker, uBlocks, masses = zip(*results)
        commMtrxSpct = pd.concat(output)
        commMtrx_primSpecies = pd.concat(primSpc) if wantTotals else None
        statusList = [s for block in status for s in block]
        iterations = [n for block in iteration for n in block]
        calcPrcsTime = sum(calc)
        calcWallClock = max(calc)
        init = max(initWorker)
        uNodes = _stack_u(uBlocks)
        waterMass = np.concatenate(masses)
    else:
        (commMtrxSpct, commMtrx_primSpecies, statusList, iterations,
         calcWallClock, init, uNodes, waterMass) = speciation_native(centralDict, commMtrx, uStart, wantTotals)
        calcPrcsTime = calcWallClock
        if requested > 1 and _speciationProcesses is None and uStart is not None:
            _chooseProcesses(centralDict, calcWallClock, requested)

    if not commMtrxSpct.columns.equals(commMtrx.columns):
        commMtrxSpct = commMtrxSpct[commMtrx.columns]

    # a node that did not converge (every initial guess tried) stops the run: its row would otherwise keep the values
    # before the speciation, coupled as if they were a result
    lStep = int(centralDict.get('lStep', 0))
    failed = [f"nativeSpeciation, node n°{i}, time step n°{lStep + 1}, "
              f"t={centralDict.get('tStep', 0.0)}{centralDict.get('timeUnit', '')} : {message}"
              for i, message in enumerate(statusList) if message]
    if failed:
        for message in failed:
            warningManager.warn(message)
        raise RuntimeError(f"nativeSpeciation : no convergence at {len(failed)} node(s), see warning.log :\n"
                           + "\n".join(failed[:10]))

    if outputManager.wanted(centralDict, 'speciation'):
        if coord:
            commMtrx_primSpecies = pd.concat(
                [commMtrx[coord], commMtrx_primSpecies], axis=1)
        outputManager.write(centralDict, commMtrx_primSpecies, 'primarySpecies', 'PrimarySpecies')
        outputManager.write(centralDict, commMtrxSpct, 'speciation', 'nativeSpeciation')

    if centralDict.get('molesStorage') and centralDict.get('MultiCompoundTransport'):
        centralDict['cellWaterMass'] = np.where(np.isfinite(waterMass), waterMass,
                                                centralDict['cellWaterMass'])
    elif centralDict.get('molesStorage'):
        centralDict['cellWaterMass'] = commMtrxSpct['H2O'].to_numpy(dtype=float) / H2O_PER_KG

    elapsed = time.perf_counter() - startNative
    centralDict.update({
        "commMtrx": commMtrxSpct,
        "nativeSpeciationU": uNodes,
        "nativeSpeciationCalcTime_WallClock":
            centralDict.get("nativeSpeciationCalcTime_WallClock", 0.0) + calcWallClock,
        "nativeSpeciationCalcTime_ProcessorTime":
            centralDict.get("nativeSpeciationCalcTime_ProcessorTime", 0.0) + calcPrcsTime,
        # interface: what is neither calculation nor initialisation (the chemical system built at the first call is
        # initialisation, as for the other solvers; it was counted in both)
        "nativeSpeciationInterfTime_WallClock":
            centralDict.get("nativeSpeciationInterfTime_WallClock", 0.0)
            + elapsed - calcWallClock - init,
        "nativeSpeciationInitTime": centralDict.get("nativeSpeciationInitTime", 0.0) + init,
        "nativeSpeciationTotalTime": centralDict.get("nativeSpeciationTotalTime", 0.0) + elapsed,
    })

    display.stepPrint(f"({writeTime(elapsed)})")
    return centralDict


if __name__ == "__main__":
    import tempfile

    N = 200
    x = np.linspace(0.0, 1.0, N)
    outdir = tempfile.mkdtemp()
    for sub in ("speciation", os.path.join("speciation", "primarySpecies")):
        os.makedirs(os.path.join(outdir, sub), exist_ok=True)

    centralDict = {
        'commMtrx': pd.DataFrame({
            "x": x, "y": 0.0,
            "Na": 1e-2 + 5e-3 * x, "Ca": 1e-3 * (1.0 + x),
            "Cl": 1.2e-2 + 5e-3 * x, "C": 2e-3 * np.ones(N),
            "S": 1e-3 * np.ones(N), "Calcite": 1e-2 * np.ones(N),
            "CO2(g)": -3.5 + 0.5 * x, "CEC": 1e-3 * np.ones(N),
            "NaX": 0.0, "CaX2": 0.0, "pH": 0.0, "I": 0.0,
        }),
        'coord': ['x', 'y', 'z'],
        'geometry': 2,
        'speciationThreads': 1,
        'lStep': 0,
        'tStep': 0.0,
        'timeUnit': 'd',
        'firstStepEquilibrium': True,
        'MultiCompoundTransport': True,
        'chemPath': "phreeqc.dat",
        'paths': {'speciation': os.path.join(outdir, "speciation"),
                  'primarySpecies': os.path.join(outdir, "speciation", "primarySpecies")},
        'output': {'speciation': {0, 1, 2}},
        'chemistry': {
            'elements': ("Na", "Ca", "Cl", "C", "S"),
            'minerals': ("Calcite",),
            'gases': {"CO2(g)": "CO2(g)"},
            'sorbedSpecies': ("NaX", "CaX2"),
            'cec': "CEC",
            'diagnostics': {"pH": "pH", "I": "I"},
        },
    }

    for pid in (1, 2):
        centralDict['speciationThreads'] = pid
        before = centralDict['commMtrx'].copy()
        centralDict = spct(centralDict)
        after = centralDict['commMtrx']
        print(f"  speciationThreads={pid}: {len(after)} nodes, columns unchanged: "
              f"{list(after.columns) == list(before.columns)}, "
              f"pH range {after['pH'].min():.3f} to {after['pH'].max():.3f}")
        print(f"  cumulated: calc {writeTime(centralDict['nativeSpeciationCalcTime_WallClock'])}, "
              f"init {writeTime(centralDict['nativeSpeciationInitTime'])}, "
              f"interface {writeTime(centralDict['nativeSpeciationInterfTime_WallClock'])}, "
              f"total {writeTime(centralDict['nativeSpeciationTotalTime'])}")
        centralDict['commMtrx'] = before
        centralDict['lStep'] += 1

    print(f"\nfiles written under {outdir}")
    for sub in ("speciation", os.path.join("speciation", "primarySpecies")):
        print(f"  {sub}: {sorted(os.listdir(os.path.join(outdir, sub)))}")