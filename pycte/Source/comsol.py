import time
import pandas as pd
import importlib.util
import os
from pathlib import Path
import numpy as np
try:
    from . import outputManager
    from . import warningManager
    from . import display
    from . import timeUnits
except ImportError:
    import outputManager
    import warningManager
    import display
    import timeUnits

_comsol = {}
_session = {}

# messages of FlexNet (COMSOL licence manager, lib/win64/csnutil.dll) when every licence of a feature is taken: the only
# errors initComsol waits on; any other start-up or loading error (no COMSOL, licence expired or unreachable, wrong
# tag, unreadable .mph) stops the run at once
LICENCE_BUSY = ("licensed number of users already reached", "users are queued for this feature",
                "in the queue for this feature", "license server system temporarily busy")

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

_secondsPerUnit = timeUnits.secondsPerUnit

def _reactionInput(centralDict):
    comm = centralDict['commMtrx']
    species = list(centralDict['transportedSpecies'])
    perSecond = _secondsPerUnit(centralDict['timeUnit'])
    cols = {}
    for key, prefix in (('transportSource', 'S_'), ('transportSink', 'K_')):
        values = centralDict.get(key)
        for sp in species:
            if isinstance(values, pd.DataFrame) and sp in values.columns:
                cols[prefix + sp] = values[sp].to_numpy(dtype=float) / perSecond
            else:
                cols[prefix + sp] = np.zeros(len(comm))
    return pd.concat([comm.reset_index(drop=True), pd.DataFrame(cols)], axis=1)

def _dirichletMask(centralDict, javamodel, comm):
    species = list(centralDict['transportedSpecies'])
    names = Convert_Species_PhreeqC_to_COMSOL(species)
    mask = pd.DataFrame(False, index=centralDict['commMtrx'].index, columns=species)
    coords = comm.iloc[:, 0].to_numpy(dtype=float)
    tol = 1e-9 * max(float(np.ptp(coords)), 1e-300)
    comp = javamodel.component(f"{centralDict['comsolTags'][0]}")
    for tag in comp.physics().tags():
        physics = comp.physics(tag)
        if 'Concentration' not in [str(physics.feature(f).getType()) for f in physics.feature().tags()]:
            continue
        fields = [str(v) for v in physics.field('concentration').component()]
        vertices = np.array(comp.geom(str(comp.geom().tags()[0])).getVertexCoord(), dtype=float)
        for f in physics.feature().tags():
            feature = physics.feature(f)
            if str(feature.getType()) != 'Concentration' or not feature.isActive() or 0 not in np.atleast_1d(np.array(feature.selection().dimension(), dtype=int)):
                continue
            flags = [str(v) == '1' for v in feature.getStringArray('species')]
            nodes = np.zeros(len(coords), dtype=bool)
            for point in feature.selection().entities(0):
                nodes |= np.abs(coords - vertices[0, int(point) - 1]) <= tol
            for sp, name in zip(species, names):
                if name in fields and flags[fields.index(name)]:
                    mask.loc[nodes, sp] = True
    return mask

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

def _protectJvmSignals():
    import faulthandler
    if faulthandler.is_enabled():
        faulthandler.disable()

def _newClient(cores):
    import jpype
    import mph
    import mph.session
    if jpype.isJVMStarted():
        from com.comsol.model.util import ModelUtil
        client = mph.Client.__new__(mph.Client)
        client.version = mph.discovery.backend()['name']
        client.standalone = False
        client.port = None
        client.host = None
        client.java = ModelUtil
    else:
        client = mph.Client(cores=cores, host=None)
    mph.session.client = client
    return client

def _startClient(cores=None):
    import mph
    import mph.session
    _protectJvmSignals()
    cores = cores or None
    client = _session.get('client')
    if client is None:
        client = mph.session.client
    if client is None:
        # client-server by default (COMSOL computes in its own process): a stand-alone session loads COMSOL's OpenMP
        # runtime (libiomp5md) in the Python process, and the one of numpy / scipy built with MKL then aborts the run
        # (OMP Error #15, e.g. nativeKinetics + COMSOL). mph.option('session', 'stand-alone') still selects it
        if mph.option('session') in ('client-server', 'platform-dependent'):
            client = _newClient(cores)
        else:
            client = mph.start(cores=cores)
    _session['client'] = client
    if not client.standalone and client.port is None:
        server = mph.Server(cores=cores)
        try:
            client.connect(server.port)
        except Exception:
            _session['serverOutput'] = _abortServer(server)
            raise
        _session['server'] = server
    return client

def _abortServer(server):
    """Stops a COMSOL server whose session failed; returns what it wrote after its port line: a server that cannot check
    out its licence starts, says why ('Closing server because the license ...') and closes."""
    process = getattr(server, 'process', None)
    output = ''
    if process is not None:
        try:
            process.wait(timeout=10)
        except Exception:
            process.kill()
        try:
            output = process.stdout.read() if process.stdout is not None else ''
        except Exception:
            pass
    _closeServer(server)
    return (output or '').strip()

def _resetSession():
    """After a failed start-up or model loading: client disconnected (its server may be gone), server stopped, so that the
    next _startClient starts a new server; returns the output of the stopped server."""
    client = _session.pop('client', None)
    server = _session.pop('server', None)
    output = _session.pop('serverOutput', '')
    if client is not None and not client.standalone and client.port:
        try:
            client.disconnect()
        except Exception:
            client.host = client.port = None
    if server is not None:
        output = _abortServer(server) or output
    return output

def _closeServer(server):
    process = getattr(server, 'process', None)
    if process is None:
        return
    try:
        process.wait(timeout=60)
    except Exception:
        process.kill()
        process.wait()
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass

def shutdownComsol():
    client = _session.pop('client', None)
    server = _session.pop('server', None)
    _comsol.clear()
    if client is None:
        return
    try:
        client.clear()
        if not client.standalone and client.port:
            if server is None:
                import mph.session
                server = getattr(mph.session, 'server', None)
            client.disconnect()
            if server is not None:
                _closeServer(server)
    except Exception as r:
        warningManager.warn(f"COMSOL : the COMSOL session could not be closed : {r}")

def _interpolationColumns(javamodel, centralDict):
    """(kind, function name) of each column of the interpolation function comsolTags[1] as the model defines it (read
    before its file is replaced: COMSOL then adapts the columns to the new file); (None, None) if it is not an
    interpolation table read from a file."""
    try:
        interpolation = javamodel.component(centralDict['comsolTags'][0]).func(centralDict['comsolTags'][1])
        return ([str(v) for v in interpolation.getStringMatrix('columnType')[0]][1::2],
                [str(v) for v in interpolation.getStringMatrix('funcnames')[0]][1::2])
    except Exception:
        return None, None


def _checkColumns(centralDict, javamodel, layout):
    """inputTransport.txt (read by the interpolation function comsolTags[1]) and the data1 export (read back by pycte)
    have no header: columns are matched by position, whatever their names in the model. Their numbers are checked
    against the model (a wrong count would shift every species), and the position table goes to warning.log so that
    the order can be checked: the names of the model follow no rule pycte could check."""
    comp, func = centralDict['comsolTags'][0], centralDict['comsolTags'][1]
    coord = list(centralDict['coord'])
    columns = list(centralDict['commMtrx'].columns)
    reactionTerms = bool(centralDict.get('comsolReactionTerms'))
    written = list(_reactionInput(centralDict).columns) if reactionTerms else columns
    kinds, names = layout
    try:
        exported = [str(e) for e in javamodel.result().export('data1').getStringArray('expr') if str(e).strip()]
    except Exception:
        exported = None
    problems = []
    if kinds is not None:
        if len(kinds) != len(written):
            problems.append(f"the function {func} of {comp} has {len(kinds)} columns, pycte writes {len(written)} in "
                            f"inputTransport.txt : {', '.join(written)}")
        elif kinds.count('arg') != len(coord):
            problems.append(f"the function {func} of {comp} has {kinds.count('arg')} argument column(s), the initial "
                            f"conditions {len(coord)} coordinate(s) ({', '.join(coord)})")
    species = len(columns) - len(coord)
    transported = len(centralDict['transportedSpecies'])
    allowed = [species] + ([species + transported, species + 2 * transported] if reactionTerms else [])
    if exported is not None and len(exported) not in allowed:
        problems.append(f"the export data1 has {len(exported)} expressions, pycte reads back {species} columns after "
                        f"the coordinates : {', '.join(columns[len(coord):])}"
                        + (f" (+ {transported} I_ integrals, + {transported} R_ amounts)" if reactionTerms else ""))
    if problems:
        raise ValueError(f"COMSOL model {centralDict['trsptPath']} : the columns are exchanged by position, without "
                         "header, and their numbers differ :\n  " + "\n  ".join(problems))
    table = [f"  {k + 1} : {col} -> {func} {names[k] if names else '?'} -> data1 "
             f"{exported[k - len(coord)] if exported and len(coord) <= k < len(coord) + len(exported) else '(coordinate)'}"
             for k, col in enumerate(columns)]
    warningManager.info("COMSOL : columns exchanged by position (initial conditions -> interpolation function -> data1 "
                        "export), check that each line names the same quantity :\n" + "\n".join(table))


def initComsol(centralDict):
    import mph
    scriptWarning = ""
    warning = 0
    waitingLicence = 0
    init = 0
    inputPath = os.path.join(centralDict['inputPath'], 'inputTransport.txt')
    licenceTaken = False
    licence = False
    startWaiting = time.perf_counter()         
    while not licence:
        ref = time.perf_counter()
        try:
            client = _startClient(centralDict['transportThreads'])
            model = client.load(f"{centralDict['trsptPath']}")
            javamodel = model.java
            layout = _interpolationColumns(javamodel, centralDict)     # read before the new file updates its columns
            javamodel.component(f"{centralDict['comsolTags'][0]}").func(f"{centralDict['comsolTags'][1]}").set("filename", f"{inputPath}")
            init += time.perf_counter() - ref
            licence = True
            if licenceTaken:
                waitingLicence = time.perf_counter() - startWaiting  
                print("Licence released :)", end=" ", flush=True)
        except Exception as r:
            serverOutput = _resetSession()
            detail = f"{r}\n{serverOutput}".strip()
            if not any(message in detail.lower() for message in LICENCE_BUSY):
                raise RuntimeError(f"COMSOL could not be started or could not load {centralDict['trsptPath']} :\n{detail}"
                                   f"\n(tags comsolTags = {centralDict['comsolTags']})") from None
            if not licenceTaken:
                print(detail)
                print("Waiting for a licence (every COMSOL licence is taken) ...", flush=True)
                licenceTaken = True
            time.sleep(3)
    if licenceTaken:
        warning += 1
        print(f"({writeTime(waitingLicence)} of waiting)", flush=True)
        scriptWarning += f"COMSOL init : {writeTime(waitingLicence)} of waiting\n"
    _checkColumns(centralDict, javamodel, layout)
    _comsol.pop('handles', None)            # Java objects of a previous model: looked up again on this one
    _comsol['comsolClient'] = client
    _comsol['comsolModel'] = model
    _comsol['comsolJavaModel'] = javamodel
    return centralDict, warning, scriptWarning, waitingLicence, init


def _handle(key, make):
    """Java object of the loaded model (interpolation function, time feature of the study, export), looked up once per
    model: each lookup (component(...).func(...), study(...).feature(...), result().export(...)) is a round trip to
    the COMSOL server, a few ms each, ~10 of them at every step. Same objects, same calls on them as before."""
    handles = _comsol.setdefault('handles', {})
    found = handles.get(key)
    if found is None:
        found = handles[key] = make()
    return found


def _studies(model):
    """Java objects of the studies that mph's model.solve() runs, in its order, resolved once per model: mph resolves
    them again at every call (tags, labels, search by label: ~10 round trips to the server)."""
    def resolve():
        studies = model / 'studies'
        nodes = studies.children()
        if not nodes:
            raise RuntimeError('No studies defined in the model.')
        return [node.java_if_exists() for node in nodes]
    return _handle(('studies',), resolve)


def _solve(model):
    """model.solve() of mph (every study of the model, in its order), on the study objects resolved once."""
    for study in _studies(model):
        study.run()


def closeComsol(centralDict):
    client = _comsol.get('comsolClient')
    if client is not None:
        try:
            client.clear()
        except Exception as r:
            warningManager.warn(f"COMSOL : the COMSOL client could not be closed : {r}")
    _comsol.clear()

def transportComsol(centralDict):

    scriptWarning = ""
    warning = 0
    abort = False
    calcTime = 0
    init = 0
    waitingLicence = 0

    inputPath = os.path.join(centralDict['inputPath'], 'inputTransport.txt')

    reactionTerms = bool(centralDict.get('comsolReactionTerms'))
    if reactionTerms:
        outputManager.writeText(inputPath, _reactionInput(centralDict), header=False)
    else:
        # same bytes as to_csv(index=False, header=False, sep='\t'), written faster for float64 columns
        outputManager.writeText(inputPath, centralDict['commMtrx'], header=False)

    if _comsol.get('comsolModel') is None:
        centralDict, initWarning, initScriptWarning, waitingLicence, initTime = initComsol(centralDict)
        warning += initWarning
        scriptWarning += initScriptWarning
        init += initTime

    _protectJvmSignals()
    model = _comsol['comsolModel']
    javamodel = _comsol['comsolJavaModel']

    comp, func, study = (f"{tag}" for tag in centralDict['comsolTags'][:3])
    try:
        ref = time.perf_counter()
        interpolation = _handle(('func', comp, func), lambda: javamodel.component(comp).func(func))
        interpolation.set("filename", f"{inputPath}")
        interpolation.refresh()
        init += time.perf_counter() - ref

        # COMSOL notation: s, min, h, d, a (year)
        timeUnit = timeUnits.shortName(centralDict['timeUnit']).replace('y', 'a')

        ref = time.perf_counter()
        timeFeature = _handle(('time', study), lambda: javamodel.study(study).feature("time"))
        timeFeature.set("tunit", f"{timeUnit}")
        timeFeature.set("tlist", f"range(0,{centralDict['dtStep']},{centralDict['dtStep']})")
        init += time.perf_counter() - ref
    except Exception as r:
        print(r)
        warning += 1
        scriptWarning += f"COMSOL, time = {centralDict['tStep']}{centralDict['timeUnit']}, time-step n° {centralDict['lStep']+1} : Fatal error occurred :\n{r}\n"
        abort = True

    if not abort:
        try:
            ref = time.perf_counter()
            _solve(model)
            calcTime += time.perf_counter() - ref
        except Exception as r:
            # failed model saved in the run folder: the original model (e.g. one from an installed example)
            # stays untouched
            model.save(os.path.join(centralDict['inputPath'],
                                    os.path.splitext(os.path.basename(centralDict['trsptPath']))[0] + "_failed.mph"))
            print(r)
            warningManager.block(f"COMSOL, t={centralDict['tStep']}{centralDict['timeUnit']}, time-step n° {centralDict['lStep']+1} : {r}\n")
            abort = True

    if not abort:




        outputPath = os.path.join(centralDict['inputPath'], 'outputTransport.txt')
        ref = time.perf_counter()
        data1 = _handle(('export', "data1"), lambda: javamodel.result().export("data1"))
        data1.setIndex("looplevelinput", "last", 0)
        data1.set("exporttype", "text")
        data1.set("filename", f"{outputPath}")
        data1.run()
        init += time.perf_counter() - ref

        exports = [(centralDict['lStep'] + 1, "last")] + ([(0, "first")] if centralDict['lStep'] == 0 else [])
        for step, level in exports:
            if not outputManager.wanted(centralDict, 'transport', step):
                continue
            for tag in centralDict['outputComsol']:
                formats = [("text", f'Transport{tag}', "txt"), ("vtu", f'Transport{tag}_vtu', "vtu")]
                for exportType, pathKey, ext in formats:
                    if pathKey not in centralDict['paths']:
                        continue
                    ref = time.perf_counter()
                    path = os.path.join(centralDict['paths'][pathKey], f'COMSOL_{step}.{ext}')
                    export = _handle(('export', f"{tag}"), lambda: javamodel.result().export(f"{tag}"))
                    export.setIndex("looplevelinput", level, 0)
                    export.set("exporttype", exportType)
                    export.set("filename", f"{path}")
                    export.run()
                    init += time.perf_counter() - ref

        columns = list(centralDict['commMtrx'].columns)
        integrals = [f"I_{sp}" for sp in centralDict['transportedSpecies']] if reactionTerms else []
        commMtrx_Comsol = pd.read_csv(outputPath,sep=r"\s+",comment="%",header=None)
        amounts = [f"R_{sp}" for sp in centralDict['transportedSpecies']] if reactionTerms and commMtrx_Comsol.shape[1] == len(columns) + 2 * len(integrals) else []
        commMtrx_Comsol.columns = columns + integrals + amounts
        if reactionTerms:
            if centralDict.get('transportSource') is not None or centralDict.get('transportSink') is not None:
                species = list(centralDict['transportedSpecies'])
                index = centralDict['commMtrx'].index
                I = commMtrx_Comsol[integrals].to_numpy(dtype=float) / _secondsPerUnit(centralDict['timeUnit'])
                centralDict['transportReactionIntegral'] = pd.DataFrame(I, index=index, columns=species)
                if amounts:
                    centralDict['transportReactionAmount'] = pd.DataFrame(commMtrx_Comsol[amounts].to_numpy(dtype=float), index=index, columns=species)
                    centralDict['transportFixed'] = _dirichletMask(centralDict, javamodel, commMtrx_Comsol)
            else:
                centralDict.pop('transportReactionIntegral', None)
            commMtrx_Comsol = commMtrx_Comsol[columns]

        return commMtrx_Comsol, warning, scriptWarning,waitingLicence, abort,calcTime, init

    else:
        return pd.DataFrame(), warning, scriptWarning, waitingLicence, abort, 0, init

def reportComsol(centralDict, warning, scriptWarning, abort):
    step = f"COMSOL, time = {centralDict['tStep']}{centralDict['timeUnit']}, time step n°{centralDict['lStep']+1}"
    if warning:
        warningManager.block(f"{step} : the {warning} following warnings occurred ...\n {scriptWarning}", warning)
    if abort:
        closeComsol(centralDict)
        raise RuntimeError(f"{step} : fatal COMSOL error (COMSOL message above in warning.log)")

def _checkNodes(centralDict, comm):
    """The data1 export replaces commMtrx: when it gives other nodes (the mesh nodes of the model: other number, other
    coordinates or another order), the coupling goes on on the nodes of COMSOL (int1 interpolates its input by
    coordinates). Said once in warning.log; settings that address nodes by their row in the initial conditions would
    then point to other nodes: error."""
    if centralDict.get('comsolNodesChanged'):
        return
    before = centralDict['commMtrx']
    coord = list(centralDict['coord'])
    old, new = before[coord].to_numpy(dtype=float), comm[coord].to_numpy(dtype=float)
    span = float(np.max(np.ptp(old, axis=0))) if len(old) > 1 else 1.0
    if len(new) == len(old) and np.all(np.abs(new - old) <= 1e-6 * max(span, 1e-300)):
        return
    centralDict['comsolNodesChanged'] = True
    what = (f"{len(comm)} nodes instead of the {len(before)} of the initial conditions" if len(comm) != len(before)
            else f"the {len(comm)} nodes of the initial conditions in another order or at other coordinates")
    byRow = ([name for name in ('renewal', 'speciesChargeGeometry') if centralDict.get(name)]
             + (['molesStorage (water mass of each node)'] if centralDict.get('molesStorage') else []))
    if byRow:
        raise RuntimeError(f"COMSOL : the export data1 of {Path(centralDict['trsptPath']).name} gives {what} : "
                           f"{', '.join(byRow)} address the nodes by their row in the initial conditions and would "
                           "point to other nodes. Export data1 at the points of the initial conditions (same "
                           "coordinates, same order), or give initial conditions on the nodes of the COMSOL mesh")
    warningManager.warn(f"COMSOL : the export data1 of {Path(centralDict['trsptPath']).name} gives {what} : from "
                        "step 1 on, the coupling runs on the nodes of COMSOL (outputs and postProcess included)")


def trspt(centralDict):
    display.stepPrint("COMSOL", end=" ", flush=True)
    startComsol = time.perf_counter()

    comm, warning, scriptWarning,waitingLicence, abort, calcTime, init = transportComsol(centralDict)

    reportComsol(centralDict, warning, scriptWarning, abort)

    if centralDict["preliminaryEquilibrium"]:
    
        if len(comm) != len(centralDict['commMtrx']):
            print('Mtrx post transport length different from input: post trspt: ', len(comm), ' input: ', len(centralDict['commMtrx']))
            print('2nd try')
            comm, warning, scriptWarning,waitingLicence, abort, calcTime, init = transportComsol(centralDict)

            reportComsol(centralDict, warning, scriptWarning, abort)

            if len(comm) != len(centralDict['commMtrx']):
                raise RuntimeError(f"COMSOL : {len(comm)} nodes returned instead of {len(centralDict['commMtrx'])} "
                                   "(preliminaryEquilibrium needs the nodes of the initial conditions), twice")
    if not abort and len(comm):
        _checkNodes(centralDict, comm)

    centralDict.update({
            "commMtrx": comm,
            "COMSOLWaitingTime" : centralDict['COMSOLWaitingTime'] + waitingLicence,
            "COMSOLCalcTime_WallClock": centralDict["COMSOLCalcTime_WallClock"] + calcTime,
            "COMSOLCalcTime_ProcessorTime": centralDict["COMSOLCalcTime_ProcessorTime"] + calcTime,
            "COMSOLInterfTime_WallClock": centralDict['COMSOLInterfTime_WallClock'] + time.perf_counter() - startComsol - calcTime - init,
            "COMSOLInitTime" : centralDict["COMSOLInitTime"] + init,
            "COMSOLTotalTime" : centralDict["COMSOLTotalTime"] + time.perf_counter() - startComsol,
            
            })


    display.stepPrint(f"({writeTime((time.perf_counter()-startComsol))})")

    return centralDict