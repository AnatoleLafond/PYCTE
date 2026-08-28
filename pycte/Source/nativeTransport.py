import numpy as np
import pandas as pd
import time
import importlib.util
import os
import concurrent.futures
import sys
import time
def writeTime(tps, arr=2):
    if tps >= 3600 * 24:
        return f"{tps / (3600 * 24):.{arr}f} d"
    elif tps >= 3600:
        return f"{tps / 3600:.{arr}f} h"
    elif tps >= 60:
        return f"{tps / 60:.{arr}f} min"
    elif tps < 1:
        return f"{tps*1000 :.{arr}f} msec"
    else:
        return f"{tps:.{arr}f} sec"
def basicTransport(centralDict, commMtrx, trsptedSpecies):
    def boundaries(df,col, sp): # the user may enter anything but correct boundary conditions, we need to select what is possible and what is not

        if sp in centralDict['firstBoundary'] and centralDict['boundaryConditions'][0] != 'closed':
            if centralDict['FickDiffusion'] or not centralDict['velocity']:
                if centralDict['boundaryConditions'][0] == 'constant':
                    df.iloc[0, col] = centralDict['firstBoundary'][sp]
                elif centralDict['boundaryConditions'][0] == 'flux':
                    print('to be done..')
                    sys.exit()
            elif centralDict['velocity'] and centralDict['velocity'] > 0 :
                df.iloc[0, col] = centralDict['firstBoundary'][sp]
        if sp in centralDict['secondBoundary'] and centralDict['boundaryConditions'][1] != 'closed':
            if centralDict['FickDiffusion'] or not centralDict['velocity']:
                if centralDict['boundaryConditions'][1] == 'constant':
                    df.iloc[-1, col] = centralDict['secondBoundary'][sp]
                elif centralDict['boundaryConditions'][1] == 'flux':
                    print('to be done..')
            elif centralDict['velocity'] and centralDict['velocity'] < 0 :
                df.iloc[-1, col] = centralDict['secondBoundary'][sp]

        return df

    def subCycling(function, state, timeStep, n, **kwargs):
        dt_sub = timeStep / n

        for _ in range(n):
            state = function(state, dt_sub, **kwargs)

        return state
    def precompute_diffusion_transmissivity(lo, hi):

        poro = np.asarray(centralDict['porosity'][lo:hi])
        area = np.asarray(centralDict['area'][lo:hi])
        poreD = np.asarray(centralDict['diffCoeff'][lo:hi])  
        cond = poro * area * poreD                             

        dx_half = np.asarray(centralDict['dxHalfCell'][lo:hi])  
        n_trim = len(cond)
        T_right = np.zeros(n_trim)
        T_left  = np.zeros(n_trim)

        with np.errstate(divide='ignore', invalid='ignore'):
            R = dx_half / np.where(cond > 0, cond, 1.0)
        R[cond <= 0] = np.inf
        R_sum = R[:-1] + R[1:]

        T_interface = np.zeros(n_trim - 1)
        finite_R = np.isfinite(R_sum) & (R_sum > 0)
        T_interface[finite_R] = 1000.0 / R_sum[finite_R]
        T_right[:-1] = T_interface
        T_left[1:]   = T_interface
        return T_right, T_left
    def FickDiffusion(c, T_right, T_left):
        """Rapide : a appeler pour chaque espece, chaque sous-pas."""
        diff = c[:-1] - c[1:]
        flux_right = np.zeros(len(c))
        flux_left  = np.zeros(len(c))
        flux_right[:-1] = -diff * T_right[:-1]
        flux_left[1:]   =  diff * T_left[1:]
        return flux_left + flux_right
    def diffusion(C_old, C_new, col, sp):
        c = C_old.iloc[:, sp].to_numpy()
        c_new = c.copy()
        if centralDict['diffusionScheme'] == "fick":
            T_right, T_left = precompute_diffusion_transmissivity(0, n)
            diff = FickDiffusion(c, T_right, T_left)

        c_new = c + ( dt * diff / (centralDict['nodeSize'] * centralDict['porosity']) ) / 1000
        C_new[col] = c_new
        return C_new
    def upwindAdvection(c,C_old):
        grad = np.zeros(len(C_old))
        for i in range(1, len(C_old)-1):

            if centralDict['velocity'] >= 0:
                grad[i] = 1000*(c[i] - c[i-1]) / centralDict['f_right'][i-1]
            elif centralDict['velocity'] :
                grad[i] = 1000*(c[i+1] - c[i]) / centralDict['f_right'][i]
        return grad

    def advection(C_old, C_new, col, sp):
        c = C_old.iloc[:, sp].to_numpy()
        if centralDict['advectionScheme'] == "upwind":
            c_new = c.copy()
            grad = upwindAdvection(c,C_old)
            c_new = c - centralDict['velocity'] * dt * grad/1000
        C_new[col] = c_new

        return C_new
    _diffusion_cache = {}
    def diffusionStep(c, dt_sub, T_right, T_left, nodeSize_trim, porosity_trim):
        diff = FickDiffusion(c, T_right, T_left)

        c_new = c + (dt_sub * diff / (nodeSize_trim * porosity_trim)) / 1000
        return c_new
    def AdvDispEqu(C_old, C_new, col, sp):
        c = C_old.iloc[:, sp].to_numpy()
        n_local = len(C_old)
        # --- Advection ---
        if centralDict['advectionScheme'] == "upwind":
            grad = upwindAdvection(c, C_old)
        else:
            grad = np.zeros(n_local)
        c_new = c - centralDict['velocity'] * dt * grad / 1000
        if centralDict['diffusionScheme'] == "fick":
            ghost_left  = 1 if (centralDict['velocity'] and centralDict['velocity'] < 0) else 0
            ghost_right = 1 if (centralDict['velocity'] and centralDict['velocity'] > 0) else 0
            lo = ghost_left
            hi = n_local - ghost_right
            c_trim         = c_new[lo:hi]
            nodeSize_trim  = centralDict['nodeSize'][lo:hi]
            porosity_trim  = np.asarray(centralDict['porosity'][lo:hi])
            cache_key = (lo, hi)
            if cache_key not in _diffusion_cache:
                _diffusion_cache[cache_key] = precompute_diffusion_transmissivity(lo, hi)
            T_right, T_left = _diffusion_cache[cache_key]
            c_diff_trim = subCycling(
                diffusionStep,
                state=c_trim,
                timeStep=dt,
                n=centralDict['subCyclingDiff'],
                T_right=T_right,
                T_left=T_left,

                nodeSize_trim=nodeSize_trim,
                porosity_trim=porosity_trim,
            )
            c_new[lo:hi] += (c_diff_trim - c_trim)
        C_new[col] = c_new
        return C_new
    if centralDict['timeUnit'] == 'y':
        dt = centralDict['dtStep']*3600*24*365.25
    elif centralDict['timeUnit'] == 'd':
        dt = centralDict['dtStep']*3600*24
    elif centralDict['timeUnit'] == 'h':
        dt = centralDict['dtStep']*3600
    elif centralDict['timeUnit'] == 'm' or centralDict['timeUnit'] == 'min':
        dt = centralDict['dtStep']*60
    else: dt = centralDict['dtStep']
    DiffCoeff = centralDict['diffCoeff']
    if not centralDict['FickDiffusion'] and centralDict['velocity'] and centralDict['velocity'] < 0:
        commMtrx = pd.concat([commMtrx.iloc[[0]], commMtrx], ignore_index=True)
    if not centralDict['FickDiffusion'] and centralDict['velocity'] and centralDict['velocity'] > 0:
        commMtrx.loc[len(commMtrx)] = commMtrx.iloc[-1]
    C_old = commMtrx[trsptedSpecies].copy()
    C_new = commMtrx[trsptedSpecies].copy()
    n = len(C_old)
    for sp, col in enumerate(trsptedSpecies) : # j'ai inverse col et sp ..
        C_old = boundaries(C_new, sp, col)
        if centralDict['FickDiffusion']:
            C_new = diffusion(C_old, C_new, col, sp)
        elif centralDict['advection']:
            C_new = advection(C_old, C_new, col, sp)
        elif centralDict['ADE']:
            C_new = AdvDispEqu(C_old, C_new, col, sp)
        else:
            print('please choose a transport process')
            sys.exit()
        C_new = boundaries(C_new, sp, col)
    if centralDict['velocity'] and centralDict['velocity'] < 0:
        C_new = C_new.iloc[1:]
    if centralDict['velocity'] and centralDict['velocity'] > 0:
        C_new = C_new.iloc[:-1]
    return C_new, None

def trspt(centralDict):

    startTransport = time.time()
    print("Native transport", end=" ", flush=True)
    xSave = centralDict["commMtrx"]['x'].copy()

    initialColumns = list(centralDict['commMtrx'].columns)
    commMtrxOther = centralDict["commMtrx"][([s for s in initialColumns if s not in centralDict['transportedSpecies']])].copy()
    if centralDict['PIDnbr'] > 1:

        chunk_size = int(np.ceil(len(centralDict['transportedSpecies']) / centralDict['PIDnbr']))
        taille, reste = divmod(len(centralDict['transportedSpecies']), chunk_size)
        lst_split = [
            centralDict['transportedSpecies'][i:i + chunk_size]
            for i in range(0, len(centralDict['transportedSpecies']), chunk_size)
        ]

        with concurrent.futures.ProcessPoolExecutor(max_workers=centralDict['PIDnbr']) as executor:
            futures = []
            for i,chunk in enumerate(lst_split):
                futures.append(executor.submit(basicTransport, centralDict, centralDict['commMtrx'], chunk))


        results = []
        results = [f.result() for f in futures]

        df1, intgr2 = zip(*results)

        commMtrxTrspt = pd.concat(df1,axis=1)
        commMtrxTrspt = pd.concat([commMtrxTrspt,commMtrxOther], axis=1)

        commMtrxTrspt = commMtrxTrspt[initialColumns]
    else:
        C_new, _ = basicTransport(centralDict, centralDict['commMtrx'], centralDict['transportedSpecies'])
        commMtrxTrspt = pd.concat([C_new,commMtrxOther], axis=1)
        commMtrxTrspt = commMtrxTrspt[initialColumns]

    commMtrxTrspt['x'] = xSave.copy()
    # t = centralDict['nativeTransportClockTime'] + time.time() - startTransport

    if centralDict['output'] and centralDict['output'].get('transport') and (centralDict['lStep']+1) in centralDict['output']['transport'] :
       commMtrxTrspt.to_csv(os.path.join(centralDict['paths']['Transport'], f"nativeTransport_{centralDict['lStep']+1}.txt"), index=False, header=True, sep='\t')


    centralDict.update({
        "commMtrx": commMtrxTrspt,
        "nativeTransportCalcTime_WallClock": centralDict["nativeTransportCalcTime_WallClock"] + time.time() - startTransport,
        # "nativeTransportCalcTime_ProcessorTime": centralDict["nativeTransportCalcTime_ProcessorTime"]+ t,
        # "nativeTransportInterfTime_WallClock": centralDict["nativeTransportInterfTime_WallClock"]+ t,
        "nativeTransportTotalTime": centralDict["nativeTransportCalcTime_WallClock"] + time.time() - startTransport,
        # "nativeTransportInitTime": centralDict["nativeTransportTotalTime"] + t,
        })
    print(f"({writeTime((time.time() - startTransport))})")
    return centralDict