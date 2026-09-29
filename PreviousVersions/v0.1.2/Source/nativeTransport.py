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
    else:
        return f"{tps:.{arr}f} sec"


def basicTransport(trsptDict, commMtrx, trsptedSpecies):
    def boundaries(df,col, sp): # the user may enter anything but correct boundary conditions, we need to select what is possible and what is not
    
        if sp in trsptDict['firstBoundary'] and trsptDict['boundaryConditions'][0] != 'closed':
            if trsptDict['FickDiffusion'] or not trsptDict['velocity']:
                if trsptDict['boundaryConditions'][0] == 'constant':
                    df.iloc[0, col] = trsptDict['firstBoundary'][sp]
                elif trsptDict['boundaryConditions'][0] == 'flux':
                    print('to be done..')
            elif trsptDict['velocity'] and trsptDict['velocity'] > 0 :
                df.iloc[0, col] = trsptDict['firstBoundary'][sp]

        if sp in trsptDict['secondBoundary'] and trsptDict['boundaryConditions'][1] != 'closed':
            if trsptDict['FickDiffusion'] or not trsptDict['velocity']:
                if trsptDict['boundaryConditions'][1] == 'constant':
                    df.iloc[-1, col] = trsptDict['secondBoundary'][sp]
                elif trsptDict['boundaryConditions'][1] == 'flux':
                    print('to be done..')
            elif trsptDict['velocity'] and trsptDict['velocity'] < 0 :
                df.iloc[-1, col] = trsptDict['secondBoundary'][sp]
            
        return df
    
    def subCycling(function, state, timeStep, n, **kwargs):

        dt_sub = timeStep / n
        
        for _ in range(n):
            state = function(state, dt_sub, **kwargs)
        
        return state
        
        
    def FickDiffusion(c,C_old,f_right,f_left):
        flux_right = np.zeros(len(C_old))
        flux_left = np.zeros(len(C_old))
        for i in range(len(C_old)):
            if i==0:
                flux_right[i] = -1000*(c[i] - c[i+1]) / (f_right[i] * 0.5*(trsptDict['porosity'][i]*trsptDict['area'][i] + trsptDict['area'][i+1]*trsptDict['porosity'][i+1])/(trsptDict['porosity'][i]*trsptDict['area'][i] * trsptDict['area'][i+1]*trsptDict['porosity'][i+1]))
                flux_left[i] = 0

            elif i==len(C_old)-1: 
                flux_left[i] = 1000*(c[i-1] - c[i]) / (f_left[i] * 0.5*(trsptDict['porosity'][i]*trsptDict['area'][i] + trsptDict['area'][i-1]*trsptDict['porosity'][i-1])/(trsptDict['porosity'][i]*trsptDict['area'][i] * trsptDict['area'][i-1]*trsptDict['porosity'][i-1]))
                flux_right[i] = 0

            else:
                flux_right[i] = -1000*(c[i] - c[i+1]) / (f_right[i] * 0.5*(trsptDict['porosity'][i]*trsptDict['area'][i] + trsptDict['area'][i+1]*trsptDict['porosity'][i+1])/(trsptDict['porosity'][i]*trsptDict['area'][i] * trsptDict['area'][i+1]*trsptDict['porosity'][i+1]))
                flux_left[i] = 1000*(c[i-1] - c[i]) / (f_left[i] * 0.5*(trsptDict['porosity'][i]*trsptDict['area'][i] + trsptDict['area'][i-1]*trsptDict['porosity'][i-1])/(trsptDict['porosity'][i]*trsptDict['area'][i] * trsptDict['area'][i-1]*trsptDict['porosity'][i-1]))
        return  (flux_left + flux_right)


    def diffusion(C_old, C_new, col, sp):    
        
        f_left  = np.zeros(n)
        f_right = np.zeros(n)
        
        f_left[1:]  = trsptDict['x'][1:] - trsptDict['x'][:-1] 
        f_right[:-1]= trsptDict['x'][1:] - trsptDict['x'][:-1]

        c = C_old.iloc[:, sp].to_numpy()
        c_new = c.copy()
        if trsptDict['diffusionScheme'] == "fick":
            diff = FickDiffusion(c,C_old,f_right,f_left)
        
        c_new = c + ( dt  * diff * (trsptDict['diffCoeff'] + trsptDict['dispersivity']*abs(trsptDict['velocity'])) / trsptDict['nodeSize'] ) / 1000

        C_new[col] = c_new

        return C_new
    
    def upwindAdvection(c,C_old):
        grad = np.zeros(len(C_old))
        for i in range(1, len(C_old)-1):
            if trsptDict['velocity'] >= 0:
                grad[i] = 1000*(c[i] - c[i-1]) / trsptDict['dx'][i-1] 
            elif trsptDict['velocity'] :
                grad[i] = 1000*(c[i+1] - c[i]) / trsptDict['dx'][i] 
        return grad
    
    def advection(C_old, C_new, col, sp):

        c = C_old.iloc[:, sp].to_numpy()
        if trsptDict['advectionScheme'] == "upwind":
            c_new = c.copy()
            grad = upwindAdvection(c,C_old)
            c_new = c - trsptDict['velocity'] * dt * grad/1000

        C_new[col] = c_new
    
        return C_new
    
    def diffusionStep(c, dt_sub, C_old_trim, f_right, f_left, nodeSize_trim):
        diff = FickDiffusion(c, C_old_trim, f_right, f_left)
        c_new = c + ((trsptDict['diffCoeff'] + trsptDict['dispersivity']*abs(trsptDict['velocity']))
                     * dt_sub * diff / nodeSize_trim) / 1000
        
        return c_new
    
    def AdvDispEqu(C_old, C_new, col, sp):
        
        c = C_old.iloc[:, sp].to_numpy()
        n_local = len(C_old)
        
        # --- Advection ---
        if trsptDict['advectionScheme'] == "upwind":
            grad = upwindAdvection(c, C_old)
        else:
            grad = np.zeros(n_local)
    
        c_new = c - trsptDict['velocity'] * dt * grad / 1000
        
        if trsptDict['diffusionScheme'] == "fick":

            ghost_left  = 1 if (trsptDict['velocity'] and trsptDict['velocity'] < 0) else 0
            ghost_right = 1 if (trsptDict['velocity'] and trsptDict['velocity'] > 0) else 0
            lo = ghost_left
            hi = n_local - ghost_right

            c_trim         = c_new[lo:hi]
            C_old_trim     = C_old.iloc[lo:hi].reset_index(drop=True)
            f_left_trim    = trsptDict['f_left'][lo:hi]
            f_right_trim   = trsptDict['f_right'][lo:hi]
            nodeSize_trim  = trsptDict['nodeSize'][lo:hi]

            c_diff_trim = subCycling(
                diffusionStep,
                state=c_trim,
                timeStep=dt,
                n=trsptDict['subCyclingDiff'],
                C_old_trim=C_old_trim,
                f_right=f_right_trim,
                f_left=f_left_trim,
                nodeSize_trim=nodeSize_trim
            )
        
            c_new[lo:hi] += (c_diff_trim - c_trim)

        
        #c_new = c - trsptDict['velocity'] * dt * grad/1000

        #c_new[lo:hi] += ((trsptDict['diffCoeff'] + trsptDict['dispersivity']*trsptDict['velocity']) * dt  * diff_trim / nodeSize) / 1000

        C_new[col] = c_new
        return C_new

    if trsptDict['timeUnit'] == 'd':
        dt = trsptDict['dtStep']*3600*24
    elif trsptDict['timeUnit'] == 'h':
        dt = trsptDict['dtStep']*3600
    elif trsptDict['timeUnit'] == 'm' or trsptDict['timeUnit'] == 'min':
        dt = trsptDict['dtStep']*60
    else: dt = trsptDict['dtStep']

    DiffCoeff = trsptDict['diffCoeff']

    
    if not trsptDict['FickDiffusion'] and trsptDict['velocity'] and trsptDict['velocity'] < 0:
        commMtrx = pd.concat([commMtrx.iloc[[0]], commMtrx], ignore_index=True)
    
    
    if not trsptDict['FickDiffusion'] and trsptDict['velocity'] and trsptDict['velocity'] > 0:
        commMtrx.loc[len(commMtrx)] = commMtrx.iloc[-1]
   
    C_old = commMtrx[trsptedSpecies].copy()
    C_new = commMtrx[trsptedSpecies].copy()
    n = len(C_old)
    
    for sp, col in enumerate(trsptedSpecies) : # j'ai inversé col et sp ..

        C_old = boundaries(C_new, sp, col)

        if trsptDict['FickDiffusion']:
            C_new = FickDiffusion(C_old, C_new, col, sp)
        elif trsptDict['advection']:
            C_new = advection(C_old, C_new, col, sp)
        elif trsptDict['ADE']:
            C_new = AdvDispEqu(C_old, C_new, col, sp)

        else:
            print('please choose a transport process')
            sys.exit()

        C_new = boundaries(C_new, sp, col)
    

    if trsptDict['velocity'] and trsptDict['velocity'] < 0:
        C_new = C_new.iloc[1:] 
    
    if trsptDict['velocity'] and trsptDict['velocity'] > 0:
        C_new = C_new.iloc[:-1]

    pd.set_option('display.max_columns', None)

    return C_new, None
 
    
 
def trspt(centralDict):
    
    startTransport = time.time()
    print("Native transport", end=" ", flush=True)
    # print(centralDict['transportedSpecies'])
    # sys.exit()
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


    t = centralDict['nativeTransportClockTime'] + time.time() - startTransport # quite no difference between interface and raw transport calculation ...
    centralDict.update({
        "commMtrx": commMtrxTrspt,
        "nativeTransportCalcTime_WallClock": centralDict["nativeTransportCalcTime_WallClock"] + t,
        "nativeTransportCalcTime_ProcessorTime": centralDict["nativeTransportCalcTime_ProcessorTime"]+ t,
        "nativeTransportInterfTime_WallClock": centralDict["nativeTransportInterfTime_WallClock"]+ t,
        "nativeTransportTotalTime": centralDict["nativeTransportTotalTime"] + t,
        })

    print(f"({writeTime((time.time() - startTransport))})") 

    return centralDict 