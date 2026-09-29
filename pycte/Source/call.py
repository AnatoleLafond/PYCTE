"""
pycte interface to the native speciation solver.

Same shape as gems.py: a worker function that takes centralDict plus a slice of
the communication matrix, and an spct(centralDict) entry point that splits the
work, runs it, writes the outputs and returns centralDict updated.

What differs from gems.py, and why
----------------------------------
* No engine to instantiate per worker: the database is read once per process
  and cached, together with the chemical system built from it.
* The coordinate columns need not be removed and put back. The solver updates
  the columns it knows and copies the rest through, so the matrix keeps its
  columns and their order on its own.
* The stoichiometric decomposition of species onto components is the
  primToSecSpecies step of xGEMS; here the matrix is the one the chemical
  system already holds, so nothing has to be assembled by hand.

Keys read in centralDict
------------------------
Required : commMtrx, chemPath, chemistry
Optional : coord (default ['x','y','z']), geometry, PIDnbr (1), lStep (0),
           tStep (0.0), timeUnit (''), paths, warningLog ('warning.log'),
           firstStepEquilibrium (False), MultiCompoundTransport (False)

Every column the configuration does not name is copied through untouched, so
the whole communication matrix can be handed over as it is.

Keys written
------------
    commMtrx, nativeCalcTime_WallClock, nativeCalcTime_ProcessorTime,
    nativeInterfTime_WallClock, nativeInitTime, nativeTotalTime

See call_template.py for a filled-in centralDict.
"""

import concurrent.futures
import os
import time

import numpy as np
import pandas as pd

from .chemistryDataframe import Config, equilibrate_dataframe
try:
    from . import outputManager
    from . import warningManager
except ImportError:
    import outputManager
    import warningManager


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
    """Turn the chemistry section of centralDict into a solver Config.

    Expected in centralDict['chemistry'] (every key optional but 'elements'):

        elements        ('Na', 'Ca', 'Cl', 'C', 'S')   defines the system
        minerals        ('Calcite',)          columns holding moles of mineral
        gases           {'CO2(g)': 'CO2(g)'}  gas -> log10 P, value or column
        sorbedSpecies   ('NaX', 'CaX2')       output columns, element mode
        cec             'CEC' or 1e-3         column or constant
        exchanger       'X' or None
        temperature     25.0 or 'temperature(C)'
        ph              None, a value, or a column
        phReagent       None, 'Cl', 'Na'
        diagnostics     {'pH': 'pH', 'I': 'I'}
        species         ()                    species columns, empty = auto
        totalsOutput    'dissolved' or 'system'
        redox, pe, tol

    MultiCompoundTransport drives the encoding of the matrix, exactly as in
    gems.py and phreeqc.py: True means the columns carry element totals
    (compounds are transported), False means they carry species that must be
    decomposed onto the components.
    """
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
        temperature=chem.get('temperature', 25.0),
        ph=chem.get('ph', None),
        ph_reagent=chem.get('phReagent', None),
        diagnostics=dict(chem.get('diagnostics', {})),
        species=tuple(chem.get('species', ())),
        totals_output=chem.get('totalsOutput', 'dissolved'),
        redox=chem.get('redox', False),
        pe=chem.get('pe', 4.0),
        tol=chem.get('tol', 1e-10),
    )


def speciation_native(centralDict, commMtrx):
    """Equilibrate one slice of the communication matrix.

    Returns the same tuple shape as speciation_xGEMS: updated matrix, component
    totals, per-node statuses, per-node iteration counts, solve time, and the
    time spent building the database and the chemical system.
    """
    cfg = buildConfig(centralDict)

    ref = time.perf_counter()
    cfg.database_source()
    init = time.perf_counter() - ref

    output, info = equilibrate_dataframe(commMtrx, cfg, n_procs=1)
    return (output, info['component_totals'], info['statuses'],
            info['iterations'], info['solve_time_wall'], init)


def _task(args):
    """Worker entry point, at module level so it can be pickled."""
    return speciation_native(*args)


def spct(centralDict):
    print("Native speciation", end=" ", flush=True)
    startNative = time.time()

    commMtrx = centralDict['commMtrx']
    coord = list(centralDict.get('coord', ['x', 'y', 'z']))
    if 'geometry' in centralDict:
        coord = coord[:centralDict['geometry']]
    coord = [c for c in coord if c in commMtrx.columns]
    nProcs = max(1, int(centralDict.get('PIDnbr', 1)))

    if nProcs > 1:
        chunk_size = int(np.ceil(len(commMtrx) / nProcs))
        split = [commMtrx.iloc[i:i + chunk_size]
                 for i in range(0, len(commMtrx), chunk_size)]

        light = {k: v for k, v in centralDict.items() if k != 'commMtrx'}
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=nProcs) as executor:
            futures = [warningManager.submit(executor, _task, (light, chunk)) for chunk in split]
        results = [f.result() for f in futures]

        output, primSpc, status, iteration, calc, initWorker = zip(*results)
        commMtrxSpct = pd.concat(output)
        commMtrx_primSpecies = pd.concat(primSpc)
        statusList = [s for block in status for s in block]
        iterations = [n for block in iteration for n in block]
        calcPrcsTime = sum(calc)
        calcWallClock = max(calc)
        init = max(initWorker)
    else:
        (commMtrxSpct, commMtrx_primSpecies, statusList, iterations,
         calcWallClock, init) = speciation_native(centralDict, commMtrx)
        calcPrcsTime = calcWallClock

    commMtrxSpct = commMtrxSpct[commMtrx.columns]
    if coord:
        commMtrx_primSpecies = pd.concat(
            [commMtrx[coord], commMtrx_primSpecies], axis=1)

    lStep = int(centralDict.get('lStep', 0))
    if any(statusList):
        with open(centralDict.get('warningLog', "warning.log"), "a") as log:
            for i, message in enumerate(statusList):
                if message:
                    log.write(
                        f"nativeSpeciation, node n°{i}, time step n°{lStep + 1}"
                        f", t={centralDict.get('tStep', 0.0)}"
                        f"{centralDict.get('timeUnit', '')}, PID={os.getpid()}"
                        f" : {iterations[i]} iterations : {message}\n")

    if outputManager.wanted(centralDict, 'speciation'):
        commMtrx_primSpecies.to_csv(
            outputManager.filePath(centralDict, 'primarySpecies', 'PrimarySpecies'),
            index=False, header=True, sep='\t')
        commMtrxSpct.to_csv(
            outputManager.filePath(centralDict, 'speciation', 'nativeSpeciation'),
            index=False, header=True, sep='\t')

    elapsed = time.time() - startNative
    centralDict.update({
        "commMtrx": commMtrxSpct,
        "nativeSpeciationCalcTime_WallClock":
            centralDict.get("nativeSpeciationCalcTime_WallClock", 0.0) + calcWallClock,
        "nativeSpeciationCalcTime_ProcessorTime":
            centralDict.get("nativeSpeciationCalcTime_ProcessorTime", 0.0) + calcPrcsTime,
        "nativeSpeciationInterfTime_WallClock":
            centralDict.get("nativeSpeciationInterfTime_WallClock", 0.0)
            + elapsed - calcWallClock - init,
        "nativeSpeciationInitTime": centralDict.get("nativeSpeciationInitTime", 0.0) + init,
        "nativeSpeciationTotalTime": centralDict.get("nativeSpeciationTotalTime", 0.0) + elapsed,
    })

    print(f"({writeTime(elapsed)})")
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
        'PIDnbr': 1,
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
        centralDict['PIDnbr'] = pid
        before = centralDict['commMtrx'].copy()
        centralDict = spct(centralDict)
        after = centralDict['commMtrx']
        print(f"  PIDnbr={pid}: {len(after)} nodes, columns unchanged: "
              f"{list(after.columns) == list(before.columns)}, "
              f"pH range {after['pH'].min():.3f} to {after['pH'].max():.3f}")
        print(f"  cumulated: calc {writeTime(centralDict['nativeCalcTime_WallClock'])}, "
              f"init {writeTime(centralDict['nativeInitTime'])}, "
              f"interface {writeTime(centralDict['nativeInterfTime_WallClock'])}")
        centralDict['commMtrx'] = before
        centralDict['lStep'] += 1

    print(f"\nfiles written under {outdir}")
    for sub in ("speciation", os.path.join("speciation", "primarySpecies")):
        print(f"  {sub}: {sorted(os.listdir(os.path.join(outdir, sub)))}")