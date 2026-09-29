import numpy as np
import pandas as pd
import time
import sys
import os
import contextlib
from scipy.integrate import solve_ivp
from scipy.sparse import bsr_matrix

@contextlib.contextmanager
def redirect_stdout_fd(filepath, t, dt, lStep, timeUnit):
    stdout_fd = 1
    stderr_fd = 2
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    with open(filepath, 'a') as f:
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
            "Species reporyed in 'kineticReactions' but not "
            f"in 'systemSpeciation' : {sorted(unknown)}"
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
        # dr/dCm = n * Cm^(n-1) * (k * prod_{i != m} Ci^ni)
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

def make_block_rhs_and_jac(n_cells, n_species, prepared_reactions):


    def blocks(C):
        J = np.zeros((n_cells, n_species, n_species))
        for rx in prepared_reactions:
            rate_law = rx["rate_law"]
            for m_idx, n in rate_law:
                # dr/dCm = n * Cm^(n-1) * (k * prod_{i != m} Ci^ni)
                rest = np.full(n_cells, float(rx["k"]))
                for i_idx, ni in rate_law:
                    if i_idx == m_idx:
                        continue
                    rest = rest * C[:, i_idx] ** ni
                dr = n * C[:, m_idx] ** (n - 1) * rest
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
                rate = rate * C[:, m_idx] ** n
            for sp_idx, coeff in rx["consumes"]:
                dC[:, sp_idx] -= coeff * rate
            for sp_idx, yld in rx["products"]:
                dC[:, sp_idx] += yld * rate
        return dC.ravel()

    def jac(t, Y):
        C = Y.reshape(n_cells, n_species)
        # BSR : un bloc par maille sur la diagonale -> construction directe,
        # sans passer par un assemblage générique.
        return bsr_matrix(
            (blocks(C), np.arange(n_cells), np.arange(n_cells + 1)),
            shape=(n_cells * n_species, n_species * n_cells),
        )

    return rhs, jac


def _integrate_block(rows, dt, prepared_reactions, n_species, method, rtol, atol):

    n_cells = rows.shape[0]
    rhs, jac = make_block_rhs_and_jac(n_cells, n_species, prepared_reactions)

    if method == "LSODA":
        # Jacobien : bande = n_species-1 de part et d'autre (bloc-diagonal)
        extraKw = {"lband": n_species - 1, "uband": n_species - 1}
    elif method in ("RK45", "RK23", "DOP853"):
        extraKw = {}                      # explicites : pas de Jacobien
    else:                                  # BDF, Radau
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
    species = list(centralDict['systemSpeciation'])
    n_species = len(species)
    dt = centralDict['dtStep']

    method = centralDict.get('kineticMethod', 'LSODA')
    rtol = centralDict.get('kineticRtol', 1e-9)
    atol = centralDict.get('kineticAtol', 1e-14)
    chunkSize = centralDict.get('kineticChunkSize', None)

    _, prepared = _prepare(species, centralDict['kineticReactions'])

    rows = commMtrxPart[species].to_numpy(dtype=float, copy=True)
    nrCells = rows.shape[0]

    if chunkSize is None or chunkSize >= nrCells:
        bounds = [(0, nrCells)]
    else:
        edges = list(range(0, nrCells, int(chunkSize))) + [nrCells]
        bounds = list(zip(edges[:-1], edges[1:]))

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

        outputarray = np.maximum(outputarray, 0.0)
        calcTime = time.perf_counter() - ref

    outputKinetics = pd.DataFrame(outputarray, columns=species,
                                  index=commMtrxPart.index)

    if centralDict.get('kineticCheckMB', False):
        tol = centralDict.get('kineticTolMB', 1e-9)
        diff = np.abs(outputarray.sum(axis=1) - rows.sum(axis=1))
        if (diff > tol).any():
            with open("warning.log", "a") as warningLog:
                warningLog.write(
                    f"\n MB issue ?, tol={tol}\n")
                warningLog.write(f"ecart max = {diff.max():.3e}\n")
            print('KINETICS: pblm MB, tol = ', tol)
            sys.exit()

    return outputKinetics, calcTime, cpuTime


def spct(centralDict):
    print("nativeKinetics", end=" ", flush=True)
    startKinetics = time.time()
    if centralDict['dtStep'] <= 0:
        commMtrxKin, calcWallClock, calcPrcsTime = centralDict['commMtrx'].copy(), 0,0
    else:
        commMtrxKin, calcWallClock, calcPrcsTime = kineticsSolve(
        centralDict, centralDict['commMtrx'][centralDict['systemSpeciation']])

        commMtrxKin = pd.concat(
            [centralDict['commMtrx'][centralDict["anythingButSpecies"]], commMtrxKin],
            axis=1)
        commMtrxKin = commMtrxKin[centralDict['commMtrx'].columns]
    
    if centralDict['output'] and centralDict['output'].get('speciation') and (centralDict['lStep']+1) in centralDict['output']['speciation'] :
       commMtrxKin.to_csv(os.path.join(centralDict['paths']['Speciation'], f"ORCHESTRA_{centralDict['lStep']+1}.txt"), index=False, header=True, sep='\t')

    
    centralDict.update({
        "commMtrx": commMtrxKin,
        "KINETICSInterfTime_WallClock": centralDict.get('KINETICSInterfTime_WallClock', 0.0)
            + time.time() - startKinetics - calcWallClock,
        "KINETICSCalcTime_WallClock": centralDict.get('KINETICSCalcTime_WallClock', 0.0)
            + calcWallClock,
        "KINETICSCalcTime_ProcessorTime": centralDict.get('KINETICSCalcTime_ProcessorTime', 0.0)
            + calcPrcsTime,
        "KINETICSTotalTime": centralDict.get('KINETICSTotalTime', 0.0)
            + time.time() - startKinetics,
    })

    print(f"({writeTime((time.time() - startKinetics))})")

    return centralDict

