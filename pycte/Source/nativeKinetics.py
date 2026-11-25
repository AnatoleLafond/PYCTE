import numpy as np
import pandas as pd
import time
import os
import contextlib
from scipy.integrate import solve_ivp
from scipy.linalg import null_space
from scipy.sparse import bsr_matrix
try:
    from . import outputManager
    from . import timeUnits
    from . import warningManager
    from . import display
    from . import frames
except ImportError:
    import outputManager
    import timeUnits
    import warningManager
    import display
    import frames

@contextlib.contextmanager
def redirect_stdout_fd(filepath, t, dt, lStep, timeUnit):
    stdout_fd = 1
    stderr_fd = 2
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    with open(filepath, 'a', encoding='utf-8') as f:
        print(f'\nPICCTS: time-step n°{lStep}, t={t} {timeUnit}, dt={dt} {timeUnit}: \n', file=f)
        os.dup2(f.fileno(), stdout_fd)
        os.dup2(f.fileno(), stderr_fd)
        try:
            yield
        finally:
            os.dup2(saved_stdout, stdout_fd)
            os.dup2(saved_stderr, stderr_fd)
            os.close(saved_stdout)
            os.close(saved_stderr)


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


def _prepare(species, reactions):

    idx = {name: i for i, name in enumerate(species)}

    unknown = set()
    for rx in reactions:
        for key in ("rate_law", "consumes", "products"):
            for sp, _ in rx.get(key, []):
                if sp not in idx:
                    unknown.add(sp)
    if unknown:
        raise KeyError(
            "Species reported in 'kineticReactions' but not "
            f"in 'coupledVariables' : {sorted(unknown)}"
        )

    preparedRx = []
    for rx in reactions:
        preparedRx.append({
            "rate_law": [(idx[sp], n) for sp, n in rx["rate_law"]],
            "consumes": [(idx[sp], c) for sp, c in rx.get("consumes", rx["rate_law"])],
            "products": [(idx[sp], y) for sp, y in rx.get("products", [])],
            "k": rx["k"],
        })
    return idx, preparedRx


def _rate_and_partials(C, rate_law, k):
    r = k
    for m_idx, n in rate_law:
        r *= C[m_idx] ** n

    partials = {}
    for m_idx, n in rate_law:
        rest = k
        for i_idx, ni in rate_law:
            if i_idx == m_idx:
                continue
            rest *= C[i_idx] ** ni
        partials[m_idx] = n * (C[m_idx] ** (n - 1)) * rest
    return r, partials


def make_rhs_and_jac(n_species, prepared_reactions):
    def rhs(t, C):
        dC = np.zeros(n_species)
        for rx in prepared_reactions:
            r, _ = _rate_and_partials(C, rx["rate_law"], rx["k"])
            for sp_idx, coeff in rx["consumes"]:
                dC[sp_idx] -= coeff * r
            for sp_idx, yld in rx["products"]:
                dC[sp_idx] += yld * r
        return dC

    def jac(t, C):
        J = np.zeros((n_species, n_species))
        for rx in prepared_reactions:
            _, partials = _rate_and_partials(C, rx["rate_law"], rx["k"])
            for m_idx, dr_dCm in partials.items():
                for sp_idx, coeff in rx["consumes"]:
                    J[sp_idx, m_idx] -= coeff * dr_dCm
                for sp_idx, yld in rx["products"]:
                    J[sp_idx, m_idx] += yld * dr_dCm
        return J

    return rhs, jac


def check_jacobian(rhs, jac, C0, n, eps=1e-6):

    J_analytic = jac(0.0, C0)
    J_fd = np.zeros((n, n))
    f0 = rhs(0.0, C0)
    for m in range(n):
        Cp = np.asarray(C0, dtype=float).copy()
        Cp[m] += eps
        J_fd[:, m] = (rhs(0.0, Cp) - f0) / eps
    return np.max(np.abs(J_analytic - J_fd))


def _power(c, n):
    """c**n on max(c, 0)."""
    return np.maximum(c, 0.0) ** n


def _slope(c, n):
    """Derivative of _power (0 below 0)."""
    use = (c > 0) | ((c == 0) & (n >= 1))
    slope = np.zeros_like(c, dtype=float)
    slope[use] = n * c[use] ** (n - 1)
    return slope


def make_block_rhs_and_jac(n_cells, n_species, prepared_reactions):

    def blocks(C):
        J = np.zeros((n_cells, n_species, n_species))
        for rx in prepared_reactions:
            rate_law = rx["rate_law"]
            for m_idx, n in rate_law:
                rest = np.full(n_cells, float(rx["k"]))
                for i_idx, ni in rate_law:
                    if i_idx == m_idx:
                        continue
                    rest = rest * _power(C[:, i_idx], ni)
                dr = _slope(C[:, m_idx], n) * rest
                for sp_idx, coeff in rx["consumes"]:
                    J[:, sp_idx, m_idx] -= coeff * dr
                for sp_idx, yld in rx["products"]:
                    J[:, sp_idx, m_idx] += yld * dr
        return J

    def rhs(t, Y):
        C = Y.reshape(n_cells, n_species)
        dC = np.zeros_like(C)
        for rx in prepared_reactions:
            rate = np.full(n_cells, float(rx["k"]))
            for m_idx, n in rx["rate_law"]:
                rate = rate * _power(C[:, m_idx], n)
            for sp_idx, coeff in rx["consumes"]:
                dC[:, sp_idx] -= coeff * rate
            for sp_idx, yld in rx["products"]:
                dC[:, sp_idx] += yld * rate
        return dC.ravel()

    def jac(t, Y):
        C = Y.reshape(n_cells, n_species)
        return bsr_matrix(
            (blocks(C), np.arange(n_cells), np.arange(n_cells + 1)),
            shape=(n_cells * n_species, n_species * n_cells),
        )

    return rhs, jac


def _integrate_block(rows, dt, prepared_reactions, n_species, method, rtol, atol):

    n_cells = rows.shape[0]
    rhs, jac = make_block_rhs_and_jac(n_cells, n_species, prepared_reactions)

    if method == "LSODA":
        extraKw = {"lband": n_species - 1, "uband": n_species - 1}
    elif method in ("RK45", "RK23", "DOP853"):
        extraKw = {}
    else:
        extraKw = {"jac": jac}

    refCpu = time.thread_time()
    sol = solve_ivp(
        fun=rhs,
        t_span=(0.0, dt),
        y0=rows.ravel(),
        method=method,
        t_eval=[dt],
        rtol=rtol,
        atol=atol,
        **extraKw,
    )
    cpu = time.thread_time() - refCpu

    if not sol.success:
        raise RuntimeError(f"Kinetic integration failed : {sol.message}")

    return sol.y[:, -1].reshape(n_cells, n_species), cpu


def kineticsSolve(centralDict, commMtrxPart):
    """Integrates the kinetic reactions over dtStep; rate constants k in s-1 (SI) whatever timeUnit."""
    refInit = time.perf_counter()

    species = commMtrxPart.columns.tolist()
    n_species = len(species)
    dt = timeUnits.toSeconds(centralDict['dtStep'], centralDict['timeUnit'])

    method = centralDict.get('kineticMethod', 'LSODA')
    rtol = centralDict.get('kineticRtol', 1e-9)
    atol = centralDict.get('kineticAtol', 1e-14)
    chunkSize = centralDict.get('kineticChunkSize', None)

    _, prepared = _prepare(species, centralDict['kineticReactions'])

    rows = commMtrxPart.to_numpy(dtype=float, copy=True)
    nrCells = rows.shape[0]

    if chunkSize is None or chunkSize >= nrCells:
        bounds = [(0, nrCells)]
    else:
        edges = list(range(0, nrCells, int(chunkSize))) + [nrCells]
        bounds = list(zip(edges[:-1], edges[1:]))

    initTime = time.perf_counter() - refInit

    with redirect_stdout_fd("kinetics_output.log", centralDict['tStep'],
                            centralDict['dtStep'], centralDict['lStep'],
                            centralDict['timeUnit']):
        ref = time.perf_counter()
        outputarray = np.empty_like(rows)
        cpuTime = 0.0
        for lo, hi in bounds:
            res, cpu = _integrate_block(rows[lo:hi], dt, prepared, n_species,
                                        method, rtol, atol)
            outputarray[lo:hi] = res
            cpuTime += cpu

        # values below 0 are set to 0; beyond the integration tolerances it is a mass created, reported below
        clipped = outputarray < -(atol + rtol * np.abs(rows).max(axis=0, initial=0.0))
        worst = {sp: float(outputarray[:, j].min()) for j, sp in enumerate(species) if clipped[:, j].any()}
        outputarray = np.maximum(outputarray, 0.0)
        calcTime = time.perf_counter() - ref

    where = f"nativeKinetics, time step n°{centralDict['lStep'] + 1}, t={centralDict['tStep']}{centralDict['timeUnit']}"
    if worst:
        warningManager.warn(f"{where} : {int(clipped.sum())} negative value(s) set to 0 (mass created), lowest : "
                            + ", ".join(f"{sp} {value:.3g}" for sp, value in worst.items()))

    outputKinetics = pd.DataFrame(outputarray, columns=species,
                                  index=commMtrxPart.index)

    if centralDict.get('kineticCheckMB', False):
        # the invariants are the combinations of species the reactions keep constant: A -> 2 B keeps 2A + B (the sum
        # A + B changes), a decay A -> (nothing) keeps nothing
        tol = centralDict.get('kineticTolMB', 1e-9)
        invariants = _conservedCombinations(n_species, prepared)
        diff = np.abs((outputarray - rows) @ invariants)
        if diff.size and (diff > tol).any():
            cell, comb = np.unravel_index(np.argmax(diff), diff.shape)
            combination = " + ".join(f"{w:.3g} {sp}" for w, sp in zip(invariants[:, comb], species) if abs(w) > 1e-12)
            message = (f"{where} : mass balance not kept, {combination} changed by {diff.max():.3e} "
                       f"in node n°{cell} (kineticTolMB = {tol})")
            warningManager.warn(message)
            raise RuntimeError(message)

    return outputKinetics, calcTime, cpuTime, initTime


def _conservedCombinations(n_species, prepared_reactions):
    """Orthonormal basis of the species combinations conserved by the reactions."""
    stoich = np.zeros((n_species, len(prepared_reactions)))
    for j, rx in enumerate(prepared_reactions):
        for sp_idx, coeff in rx["consumes"]:
            stoich[sp_idx, j] -= coeff
        for sp_idx, yld in rx["products"]:
            stoich[sp_idx, j] += yld
    return null_space(stoich.T)


def spct(centralDict):
    startKinetics = time.perf_counter()
    display.stepPrint("nativeKinetics", end=" ", flush=True)

    if centralDict['dtStep'] <= 0:
        commMtrxKin = centralDict['commMtrx'].copy()
        calcWallClock = calcPrcsTime = init = 0.0
    else:
        # columns selected by frames.columnsOf (the frames of commMtrx[names], built from its array)
        commMtrxKin, calcWallClock, calcPrcsTime, init = kineticsSolve(
            centralDict, frames.columnsOf(centralDict['commMtrx'], centralDict['systemSpecies']))

        speciesSet = set(centralDict['systemSpecies'])
        others = [c for c in centralDict['commMtrx'].columns.tolist() if c not in speciesSet]
        commMtrxKin = pd.concat(
            [frames.columnsOf(centralDict['commMtrx'], others), commMtrxKin],
            axis=1)
        commMtrxKin = commMtrxKin[centralDict['commMtrx'].columns]

    if outputManager.wanted(centralDict, 'speciation'):
        outputManager.write(centralDict, commMtrxKin, 'speciation', 'nativeKinetics')

    elapsed = time.perf_counter() - startKinetics

    centralDict.update({
        "commMtrx": commMtrxKin,
        "nativeKineticsCalcTime_WallClock":
            centralDict.get("nativeKineticsCalcTime_WallClock", 0.0) + calcWallClock,
        "nativeKineticsCalcTime_ProcessorTime":
            centralDict.get("nativeKineticsCalcTime_ProcessorTime", 0.0) + calcPrcsTime,
        "nativeKineticsInterfTime_WallClock":
            centralDict.get("nativeKineticsInterfTime_WallClock", 0.0)
            + elapsed - calcWallClock - init,
        "nativeKineticsInitTime":
            centralDict.get("nativeKineticsInitTime", 0.0) + init,
        "nativeKineticsTotalTime":
            centralDict.get("nativeKineticsTotalTime", 0.0) + elapsed,
    })

    display.stepPrint(f"({writeTime(elapsed)})")

    return centralDict