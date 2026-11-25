import numpy as np
import time 
import pandas as pd
import sys 
import os
from pathlib import Path
import concurrent.futures
import contextlib
import functools
try:
    from .. import PyORCHESTRA              # ORCHESTRA compiled with pycte (setup.py, sources in pycte/orchestra_src)
except ImportError:
    try:
        import PyORCHESTRA                  # PyORCHESTRA installed separately from its sources
        if not hasattr(getattr(PyORCHESTRA, 'ORCHESTRA', None), 'set_and_calculate_multi_2'):
            raise ImportError(f"{getattr(PyORCHESTRA, '__file__', 'PyORCHESTRA')} is not the ORCHESTRA solver of "
                              "H. Meeussen (no ORCHESTRA.set_and_calculate_multi_2) : the PyPI project PyORCHESTRA "
                              "is unrelated, pip uninstall PyORCHESTRA")
    except ImportError as err:
        raise ImportError("chemistryModule = 'ORCHESTRA' : this pycte installation has no compiled ORCHESTRA (no pycte "
                          "wheel for this platform and Python version, and no C++ compiler when pycte was installed). "
                          "Install a C++ compiler (e.g. Visual Studio Build Tools on Windows) and reinstall pycte from its "
                          "sources : pip install --force-reinstall --no-deps --no-binary pycte pycte. Do not "
                          "'pip install PyORCHESTRA' from PyPI : that name is an unrelated project.") from err
import faulthandler
if 'jpype' not in sys.modules:
    faulthandler.enable()
import ctypes
import tempfile
try:
    from . import outputManager
    from . import warningManager
    from . import display
    from . import frames
except ImportError:
    import outputManager
    import warningManager
    import display
    import frames

solver = None
solverCells = 0     # number of cells the solver was initialised with
outputArray = None
initWorker = 0
firstCalc = True
SIGNED_COMPONENTS = ('H', 'E')     # proton and electron excess: their totals may be negative

try:
    _crt = ctypes.CDLL("ucrtbase") if os.name == "nt" else ctypes.CDLL(None)
except OSError:
    _crt = None


def _dtypes(df):
    """Column dtypes of df, without building a Series."""
    try:
        return df._mgr.get_dtypes()
    except AttributeError:
        return df.dtypes.to_numpy()


_indexer = frames.indexer        # columns.get_indexer(names), kept for the same Index and names


def _takeColumns(df, names):
    """df[names] (new frame, same values, same block and memory layout): frames.columnsOf."""
    return frames.columnsOf(df, names)


def _takeValues(df, names):
    """df[names].to_numpy(): the same array (values and layout), without building the frame (frames.takeArray) for a
    frame of one float64 block."""
    names = list(names)
    taken = frames.takeArray(df, names)
    return taken if taken is not None else df[names].to_numpy()


def _positions(df, names):
    """Positions of `names` in df when df has unique float64 columns and holds them all, else None."""
    if not (df.columns.is_unique and frames.allFloat64(df)):
        return None
    positions = _indexer(df.columns, names)
    return positions if np.all(positions >= 0) else None


def _literbulkValues(frame, colAq, waterMass, cross):
    """literbulk input of the stoichiometric product, without pandas: the colAq columns of `frame` times the water mass
    of each node, the crossDependencies of the speciation left out. Same values (the same products) and the same
    array layout as `frame.loc[:, colAq] = frame.loc[:, colAq].mul(waterMass, axis=0)`, `frame.drop(columns=cross)`,
    `frame.to_numpy()` (an F-ordered view of the column-major block): (values, kept column names, {name: position}).
    None outside the plain case (unique float64 columns, known names, one water mass per node or a single one), where
    the pandas path runs (and raises its own errors)."""
    columns = frame.columns
    if not (columns.is_unique and frames.allFloat64(frame)):
        return None
    if len(set(colAq)) != len(colAq):
        return None
    # every name of cross in the columns, the kept rows and columns: kept for the same Index (_literbulkLayout, below)
    known, keep, keptColumns, rowOf = _literbulkLayout(columns, cross)
    if not known:
        return None
    aq = _indexer(columns, colAq)
    if np.any(aq < 0):
        return None
    water = np.asarray(waterMass)
    if water.dtype.kind not in 'fiu' or not (water.ndim == 0 or (water.ndim == 1 and len(water) == len(frame))):
        return None
    block = frame.to_numpy().T.copy()               # columns x nodes, C-ordered like the block of frame.copy()
    block[aq] = block[aq] * water
    if keep is not None:
        block = block[keep]
    values = block.T
    return values, keptColumns, rowOf


def _flushC():
    sys.stdout.flush()
    sys.stderr.flush()
    if _crt is not None:
        _crt.fflush(None)


_capture = None


def _captureFile():
    """Temporary file receiving the ORCHESTRA console output, created once per process and emptied at every use:
    truncated only when something was written into it (truncating costs ~0.2 ms on some disks), empty and read from
    the start in both cases; already empty at position 0 after a use that wrote nothing: no seek then (_captureWritten)."""
    global _capture
    if _capture is None or _capture.closed:
        _capture = tempfile.TemporaryFile()
    elif _captureWritten[0] or os.lseek(_capture.fileno(), 0, os.SEEK_CUR) != 0:
        _capture.seek(0)
        _capture.truncate()
        _captureWritten[0] = False
    return _capture


@contextlib.contextmanager
def captureOutput(label):
    """ORCHESTRA console output -> warning.log only."""
    _flushC()
    saved = os.dup(1), os.dup(2)
    capture = _captureFile()
    os.dup2(capture.fileno(), 1)
    os.dup2(capture.fileno(), 2)
    try:
        yield
    finally:
        _flushC()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])
        # nothing written (the position shared by the file and fds 1 and 2 still 0): the text read would be empty
        text = ""
        if os.lseek(capture.fileno(), 0, os.SEEK_END):
            capture.seek(0)
            text = capture.read().decode(errors="replace").replace("\r\n", "\n").strip()
            _captureWritten[0] = True
        if text:
            warningManager.info(f"ORCHESTRA, {label} :\n{text}")


def _stepLabel(t, dt, lStep, timeUnit):
    return f"t={t}{timeUnit}, dt={dt}{timeUnit}, time-step n° {lStep+1}"

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

def resetOrchestra():
    global solver, solverCells, outputArray, initWorker, firstCalc
    with captureOutput("end of run"):
        if solver is not None:
            try:
                solver.stop()
            except Exception:
                pass
        solver = None
    solverCells = 0
    outputArray = None
    initWorker = 0
    firstCalc = True


def _init_worker(chemPath, inVars, outVars,t,dt,l,unit):
    global solver, initWorker
    with captureOutput(_stepLabel(t, dt, l, unit)):
        ref = time.perf_counter()
        solver = PyORCHESTRA.ORCHESTRA()
        solver.initialise(str(chemPath), 1, inVars, outVars)
        initWorker = time.perf_counter() - ref

_checkedOutputs = set()


def checkOutputVariables(outputVariables, columns):
    """Each ORCHESTRA output variable naming a column must name its own column (checked once per pair of lists)."""
    key = (tuple(outputVariables), tuple(columns))
    if key in _checkedOutputs:
        return
    _checkOutputVariables(outputVariables, columns)
    _checkedOutputs.add(key)


def _checkOutputVariables(outputVariables, columns):
    if len(outputVariables) < len(columns):
        raise ValueError(f"ORCHESTRA : {len(outputVariables)} variables in outputVariableOrchestra for {len(columns)} "
                         f"coupled columns {columns}")
    known = set(columns)
    wrong = []
    for k, (variable, column) in enumerate(zip(outputVariables, columns)):
        named = variable if variable in known else (variable[:-4] if variable.endswith('.con') else None)
        if named in known and named != column:
            wrong.append(f"  position {k} : {variable!r} fills column {column!r}")
    if wrong:
        raise ValueError("ORCHESTRA : outputVariableOrchestra must follow the order of coupledVariables (each output "
                         "variable fills the column at the same position) :\n" + "\n".join(wrong))


def _waterMass(centralDict, nrCells):
    """waterMass for nrCells nodes."""
    waterMass = centralDict['waterMass']
    values = np.asarray(waterMass, dtype=float)
    if values.ndim == 0 or len(values) == nrCells:
        return waterMass
    if len(values) and np.all(values == values[0]):
        warningManager.info(f"ORCHESTRA, time-step n° {centralDict['lStep']+1} : waterMass resized from "
                            f"{len(values)} to {nrCells} nodes (uniform value {values[0]:g})")
        centralDict['waterMass'] = [float(values[0])] * nrCells
        return centralDict['waterMass']
    raise ValueError(f"ORCHESTRA : waterMass has {len(values)} values (not uniform) but the grid has {nrCells} nodes ; "
                     "give a single value, a uniform list or one value per node of the current grid")


def speciationOrchestra(centralDict,commMtrxPart):
    global solver, solverCells, outputArray, initWorker, firstCalc
    pd.set_option('display.max_columns', None)
    """
    Site quantity is natively defined in the .inp file. This file could change depending on the node number ...
    side note:
        Sorbed attribute is .solid
        Speciation is Na+.con, Na.diss en solution, Na.tot total.
        Species attributes are defined by speciesAttributes keyword. It may refer to the .inp file.
    """
    
    # commMtrxPart is read only, but by the pandas path of literbulk below, which works on its copy (the fast paths read
    # the same values, and build arrays of the same layout, from the frame itself)
    inputVariableOrchestra = centralDict['inputVariableOrchestra']

    if centralDict['couplingFormalism'] == 'literbulk' :

        waterMass = _waterMass(centralDict, len(commMtrxPart))
        if centralDict.get('saturationColumn'):
            waterMass = (np.asarray(waterMass, dtype=float)
                         * centralDict['commMtrx'][centralDict['saturationColumn']].to_numpy(dtype=float))
        cross = centralDict['crossDependencies']['speciation']['total']
        literbulk = _literbulkValues(commMtrxPart, centralDict['colAq'], waterMass, cross)
        if literbulk is None:
            commMtrxPart = commMtrxPart.copy()
            commMtrxPart.loc[:, centralDict['colAq']] = (commMtrxPart.loc[:, centralDict['colAq']].mul(waterMass, axis=0))

            if cross:
                commMtrxPart.drop(columns=cross, inplace=True)
            values = commMtrxPart.to_numpy()
            column = lambda name: commMtrxPart[name].to_numpy()
            keptColumns = commMtrxPart.columns
        else:
            values, keptColumns, rowOf = literbulk
            column = lambda name: values[:, rowOf[name]]

        if 'O' in centralDict['stoich']:
            del centralDict['stoich']['O']


        product = values @ centralDict['stoich'].to_numpy()
        comm = centralDict['commMtrx']
        crossPositions = _positions(comm, cross) if cross else None
        # names derived from the settings only, the same at every call: built once (_LiterbulkNames)
        names_ = _literbulkNames(centralDict, cross)
        if crossPositions is not None and product.dtype == np.float64 and len(comm) == len(product):
            species, names = names_.species, names_.names
            if (not outputManager.wanted(centralDict, 'speciation') and product.ndim == 2
                    and product.shape[1] == len(species) and names_.unique and names_.covered):
                # the speciation output is not written: the columns of the frame below as a dict {name: column}, the
                # same values, from which the rows given to ORCHESTRA are taken below (as for 'tot')
                commValues = comm.to_numpy()
                commMtrx_primSpecies = {name: commValues[:, p] for name, p in zip(cross, crossPositions.tolist())}
                commMtrx_primSpecies.update((name, product[:, j]) for j, name in enumerate(species))
            else:
                # the frame pd.concat([comm[cross], product]) builds, assembled at once (float64 columns only)
                commMtrx_primSpecies = pd.DataFrame(np.hstack([comm.to_numpy()[:, crossPositions], product]),
                                                    columns=frames.columnIndex(names))
        else:
            species, names = names_.species, names_.names
            crossArrays = (_crossArrays(comm, cross)
                           if cross and not outputManager.wanted(centralDict, 'speciation') else None)
            if (crossArrays is not None and product.dtype == np.float64 and product.ndim == 2
                    and product.shape[1] == len(species) and len(comm) == len(product)
                    and names_.unique and names_.covered):
                # a commMtrx with integer columns (e.g. time): the columns of the concat below as a dict {name: column},
                # the same values, the rows given to ORCHESTRA taken from them below (each column converted to float64
                # as to_numpy(dtype=float64) converts the column of the concat)
                commMtrx_primSpecies = crossArrays
                commMtrx_primSpecies.update((name, product[:, j]) for j, name in enumerate(species))
            else:
                commMtrx_primSpecies = pd.DataFrame(
                    product,
                    columns=centralDict["inputVariableOrchestraSpecies"]
                    )

                if cross:
                    commMtrx_primSpecies = pd.concat([comm[cross].reset_index(drop=True),commMtrx_primSpecies], axis=1)


        # mineral amount given back to ORCHESTRA: an output '<mineral>[s].m1' (former special case of Calcite only)
        # makes the column '<mineral>[s]' an input variable
        minerals = names_.minerals
        missing = [m for m in minerals if m not in keptColumns]
        if missing:
            raise ValueError(f"ORCHESTRA literbulk : outputVariableOrchestra gives {[m + '.m1' for m in missing]}, "
                             f"whose amount is given back to ORCHESTRA from the column(s) {missing} : not in "
                             "coupledVariables")
        for mineral in minerals:
            commMtrx_primSpecies[mineral] = column(mineral)
        if minerals:
            inputVariableOrchestra = names_.inputs

    elif centralDict['couplingFormalism'] == 'tot':
        # species (cells x systemSpecies) @ stoichReduced (systemSpecies x components) -> component totals
        components = centralDict['primarySpecies'][-1]       # e.g. ['Ca', 'Cl', 'K', 'N', 'Na']
        totals = _takeValues(commMtrxPart, centralDict['systemSpecies']) @ centralDict['stoichReduced'].to_numpy()
        names_ = _totNames(components, inputVariableOrchestra)     # from the settings only: built once
        signed = names_.signed                                          # not clipped to 0
        if (outputManager.wanted(centralDict, 'speciation') or totals.dtype != np.float64
                or not names_.componentsUnique or not names_.inputsUnique):
            commMtrx_primSpecies = pd.DataFrame(np.where(signed | (totals >= 0), totals, 0),
                                                columns=components, index=commMtrxPart.index)

            cross = centralDict['crossDependencies']['speciation']['total']
            if cross:
                commMtrx_primSpecies[cross] = commMtrxPart[cross]    # pH, pe, ...: no clipping (pe < 0 possible)


            source = [v if v in commMtrx_primSpecies.columns else v.rsplit('.', 1)[0]
                      for v in centralDict["inputVariableOrchestra"]]
            commMtrx_primSpecies = commMtrx_primSpecies[source]
            commMtrx_primSpecies.columns = centralDict["inputVariableOrchestra"]
        else:
            clipped = np.where(np.isnan(totals) | signed | (totals >= 0), totals, 0)
            columns = {c: clipped[:, j] for j, c in enumerate(components)}
            cross = centralDict['crossDependencies']['speciation']['total']
            crossPositions = _positions(commMtrxPart, cross) if cross else None
            if crossPositions is not None:
                # unique float64 columns: the column of each name in the array of the frame, the values of
                # commMtrxPart[c].to_numpy() (the rows given to ORCHESTRA are copied from them below)
                partValues = commMtrxPart.to_numpy()
                for c, p in zip(cross, crossPositions.tolist()):
                    columns[c] = partValues[:, p]
            else:
                for c in cross:
                    columns[c] = commMtrxPart[c].to_numpy()
            commMtrx_primSpecies = {v: columns[v if v in columns else v.rsplit('.', 1)[0]]
                                    for v in centralDict["inputVariableOrchestra"]}

    

        
        
        
    
    else:
        raise ValueError(f'ORCHESTRA coupling formalism not recognized : {centralDict["couplingFormalism"]} '
                         '(tot, literbulk, diss available)')
    
    

    if isinstance(commMtrx_primSpecies, dict):
        rows = np.empty((len(commMtrxPart), len(inputVariableOrchestra)), dtype=np.float64, order="C")
        for j, col in enumerate(inputVariableOrchestra):
            rows[:, j] = np.asarray(commMtrx_primSpecies[col], dtype=np.float64)
    else:
        rows = np.empty((len(commMtrx_primSpecies), len(inputVariableOrchestra)), dtype=np.float64, order="C")
        positions = _positions(commMtrx_primSpecies, inputVariableOrchestra)
        if positions is not None:
            # same values as column by column, taken at once from the array of the frame
            rows[:] = commMtrx_primSpecies.to_numpy()[:, positions]
        else:
            for j, col in enumerate(inputVariableOrchestra):
                rows[:, j] = commMtrx_primSpecies[col].to_numpy(dtype=np.float64)

    nrCells = rows.shape[0]
    calcTime = 0

    # the solver is sized at initialisation: a grid change (e.g. COMSOL mesh nodes instead of the
    # cells of the initial conditions) needs a new solver, and the memory of the previous grid is meaningless
    label = _stepLabel(centralDict['tStep'], centralDict['dtStep'], centralDict['lStep'], centralDict['timeUnit'])
    if solver is not None and solverCells != nrCells:
        warningManager.info(f"ORCHESTRA, time-step n° {centralDict['lStep']+1} : {nrCells} cells instead of {solverCells}, solver re-initialised")
        with captureOutput(label):
            try:
                solver.stop()
            except Exception:
                pass
        solver = None
        firstCalc = True

    if solver is None:
        _cwd = os.getcwd()
        os.chdir(Path(centralDict['chemPath']).parent)

        try:
            with captureOutput(label):
                ref = time.perf_counter()
                solver = PyORCHESTRA.ORCHESTRA()
                solver.initialise(Path(centralDict['chemPath']).name, nrCells, inputVariableOrchestra, centralDict['outputVariableOrchestra'])
                init = time.perf_counter() - ref
                solverCells = nrCells
        finally:
            os.chdir(_cwd)
    else:
        init = 0

    outShape = (nrCells, len(centralDict['outputVariableOrchestra']))
    if outputArray is None or outputArray.shape != outShape:
        outputArray = np.empty(outShape, dtype=np.float64, order="C")
    outputarray = outputArray

    _runDir = os.getcwd()
    with captureOutput(label):
        if firstCalc or not centralDict['OrchestraMemory']:
            memory_option = 1
        else:
            memory_option = 0

        ref = time.perf_counter()

        # outputarray = solver.set_and_calculate_multi(rows, centralDict['speciationThreads'], memory_option)
        try:
            solver.set_and_calculate_multi_2(rows, centralDict['speciationThreads'], memory_option, outputarray)
        except RuntimeError:
            if memory_option != 1:
                raise
            # chained start estimates (each cell starts from the previous converged cell) failed, e.g. at a
            # boundary between two very different chemistries (soil -> reservoir) : new solver, each cell
            # starting from its own estimate
            warningManager.info(f"ORCHESTRA, time-step n° {centralDict['lStep']+1} : failure with chained start estimates, "
                                "new attempt with each cell starting from its own estimate (memory option 0)")
            try:
                solver.stop()
            except Exception:
                pass
            os.chdir(Path(centralDict['chemPath']).parent)
            try:
                solver = PyORCHESTRA.ORCHESTRA()
                solver.initialise(Path(centralDict['chemPath']).name, nrCells, inputVariableOrchestra, centralDict['outputVariableOrchestra'])
            finally:
                os.chdir(_runDir)
            solver.set_and_calculate_multi_2(rows, centralDict['speciationThreads'], 0, outputarray)


        calcTime = time.perf_counter() - ref
        firstCalc = False

    if False :
        refdf = commMtrx_primSpecies
        output = pd.DataFrame(outputarray, columns=(centralDict['coupledVariables']))
        output.loc[:, centralDict['colAq']] = (output.loc[:, centralDict['colAq']].mul(centralDict['waterMass'], axis=0))
    
        output = pd.DataFrame(
            output.to_numpy() @ centralDict['stoich'].to_numpy(),
            columns=centralDict['stoich'].columns)
        output.columns = refdf.columns
        tol = 1e-11
        diff = (output - refdf).abs()
        
        if (diff > tol).any().any():
            warningManager.block(f"ORCHESTRA : mass balance problem\noutput orchestra:\n{output.to_string()}\n"
                                 f"\nrefdf:\n{refdf.to_string()}\ndiff: \n{diff.to_string()}\n", 1)
            raise RuntimeError(f"ORCHESTRA : mass balance not kept (tol = {tol}, details above in warning.log)")
        
    else:
        kept = centralDict['crossDependencies']['transport']['total']
        coupled, outputVariables = centralDict['coupledVariables'], centralDict['outputVariableOrchestra']
        # the columns filled by ORCHESTRA and their check: from the settings only, built and checked once (an error is
        # raised again at every call, as before)
        outCols = frames.constant('orchestraOutCols', (coupled, kept), lambda: [s for s in coupled if s not in kept])
        frames.constant('orchestraCheck', (outputVariables, outCols),
                        lambda: checkOutputVariables(outputVariables, outCols))
        if 'failed' not in outCols and not outputManager.wanted(centralDict, 'speciation'):
            # the frame below built only if spct needs it (spct copies these values into the new commMtrx)
            outputOrchestra = _OrchestraOutput(outputarray[:, :len(outCols)], outCols)
        else:
            outputOrchestra = pd.DataFrame(outputarray[:, :len(outCols)], columns=_columnIndexOf(outCols))




    return outputOrchestra, commMtrx_primSpecies, calcTime, init

_literbulkLayouts = {}           # (id(columns), cross) -> (columns, cross known, kept rows, kept columns, {name: row})


def _literbulkLayout(columns, cross):
    """For the columns Index of the literbulk frame and the crossDependencies of the speciation: (every name of cross
    in columns, rows kept (None: no cross), the kept columns, {name: position}), what _literbulkValues computed at
    every call from them only, kept for the same Index object (immutable: the same object, the same answers) and the
    same names, as frames.indexer keeps its positions."""
    key = (id(columns), tuple(cross))
    found = _literbulkLayouts.get(key)
    if found is not None and found[0] is columns:
        return found[1:]
    known = not any(c not in columns for c in cross)
    keep, keptColumns = None, columns
    if known and cross:
        crossSet = set(cross)
        keep = [k for k, c in enumerate(columns.tolist()) if c not in crossSet]
        keptColumns = columns[keep]
    rowOf = {name: k for k, name in enumerate(keptColumns.tolist())} if known else None
    if len(_literbulkLayouts) >= 64:
        _literbulkLayouts.clear()
    _literbulkLayouts[key] = (columns, known, keep, keptColumns, rowOf)
    return known, keep, keptColumns, rowOf


class _LiterbulkNames:
    """Names of the literbulk path derived from the settings only (inputVariableOrchestraSpecies, the crossDependencies
    of the speciation, outputVariableOrchestra, inputVariableOrchestra), the same at every call: built once per run
    (frames.constant). Each one is the expression speciationOrchestra evaluated at every call, evaluated when first
    read (an error then raised where it was)."""

    def __init__(self, centralDict, cross):
        self._species = centralDict["inputVariableOrchestraSpecies"]
        self._cross = cross
        self._outputs = centralDict['outputVariableOrchestra']
        self._inputs = centralDict['inputVariableOrchestra']

    @functools.cached_property
    def species(self):
        return list(self._species)

    @functools.cached_property
    def names(self):
        return list(self._cross) + self.species

    @functools.cached_property
    def minerals(self):
        # an output '<mineral>[s].m1' makes the column '<mineral>[s]' an input variable
        return [v[:-len('.m1')] for v in self._outputs if v.endswith('[s].m1')]

    @functools.cached_property
    def unique(self):
        return len(set(self.names)) == len(self.names)

    @functools.cached_property
    def covered(self):
        return set(self._inputs + self.minerals) <= set(self.names) | set(self.minerals)

    @functools.cached_property
    def inputs(self):
        return self._inputs + self.minerals


def _literbulkNames(centralDict, cross):
    sources = (centralDict["inputVariableOrchestraSpecies"], cross, centralDict['outputVariableOrchestra'],
               centralDict['inputVariableOrchestra'])
    return frames.constant('orchestraLiterbulk', sources, lambda: _LiterbulkNames(centralDict, cross))


class _TotNames:
    """The flags of the 'tot' path derived from the components and inputVariableOrchestra only: built once per run."""

    def __init__(self, components, inputs):
        self._components, self._inputs = components, inputs

    @functools.cached_property
    def signed(self):
        return np.array([c in SIGNED_COMPONENTS for c in self._components])

    @functools.cached_property
    def componentsUnique(self):
        return len(set(self._components)) == len(self._components)

    @functools.cached_property
    def inputsUnique(self):
        return len(set(self._inputs)) == len(self._inputs)


def _totNames(components, inputs):
    return frames.constant('orchestraTot', (components, inputs), lambda: _TotNames(components, inputs))


def _columnIndexOf(names):
    """frames.columnIndex(names) for a list of names kept for the run (frames.constant): looked up once per list."""
    return frames.constant('orchestraColumnIndex', (names,), lambda: frames.columnIndex(names))


_captureWritten = [False]       # something was read from the capture file (captureOutput): to empty before next use


class _OrchestraOutput:
    """The output of ORCHESTRA for the coupled columns before it becomes a frame: values (nodes x columns, float64)
    and the column names; frame() is the frame speciationOrchestra returned before (same values, same columns)."""

    def __init__(self, values, names):
        self.values, self.names = values, names
        self.columns = _columnIndexOf(names)

    def __len__(self):
        return len(self.values)

    def to_numpy(self):
        return self.values

    def frame(self):
        return pd.DataFrame(self.values, columns=frames.columnIndex(self.names))


def _totInput(centralDict, comm, col):
    """commMtrx itself instead of commMtrx[col] for the 'tot' formalism when the speciation output is not written: the
    'tot' path takes from it arrays of columns only (systemSpecies, the crossDependencies), the same values and memory
    layout from commMtrx as from its columns col (takeArray: col keeps the order of the block, F or C, of commMtrx).
    None (commMtrx[col] built as before) unless commMtrx is one float64 block (not both C- and F-ordered: one row or
    one column) of unique columns holding every name of col, and col holds systemSpecies and the crossDependencies."""
    if centralDict.get('couplingFormalism') != 'tot' or outputManager.wanted(centralDict, 'speciation'):
        return None
    blocks = getattr(getattr(comm, '_mgr', None), 'blocks', None)
    if (blocks is None or len(blocks) != 1 or blocks[0].dtype != np.float64 or blocks[0].values.ndim != 2
            or (blocks[0].values.flags.c_contiguous and blocks[0].values.flags.f_contiguous)
            or not comm.columns.is_unique):
        return None
    if len(set(col)) != len(col) or np.any(_indexer(comm.columns, col) < 0):
        return None
    names = set(col)
    if not (set(centralDict['systemSpecies']) <= names
            and set(centralDict['crossDependencies']['speciation']['total']) <= names):
        return None
    return comm


def _crossArrays(comm, cross):
    """{name: comm[name].to_numpy()} for the cross columns (a new dict), when commMtrx has unique columns holding each
    of them once and each is a numpy integer or float column (int64 time, float64 pH ...); None otherwise (the concat of
    the frames is built, with its own errors). Their values are those of comm[cross] in pd.concat([comm[cross]
    .reset_index(drop=True), ...]), whose columns keep their dtype; to_numpy(dtype=float64) of a concat column and
    np.asarray(column, dtype=float64) make the same conversion."""
    columns = comm.columns
    if not columns.is_unique or len(set(cross)) != len(cross):
        return None
    positions = _indexer(columns, cross)
    if np.any(positions < 0):
        return None
    dtypes = _dtypes(comm)
    found = {}
    for name, p in zip(cross, positions.tolist()):
        dtype = dtypes[p]
        if not isinstance(dtype, np.dtype) or dtype.kind not in 'iuf':
            return None
        found[name] = comm.iloc[:, p].to_numpy()
    return found


_columnsSplits = {}             # (id(comm columns), id(computed columns), col) -> (both Index objects, answer)


def _columnsSplit(commColumns, spctColumns, col, computed):
    """Whether the columns of commMtrx are exactly col and the computed columns (computed: spctColumns.tolist()), the
    two lists distinct and without repeated names: computed from the names only, kept for the same two Index objects
    (immutable) and the same col, as frames.indexer keeps its positions."""
    key = (id(commColumns), id(spctColumns), tuple(col))
    found = _columnsSplits.get(key)
    if found is not None and found[0] is commColumns and found[1] is spctColumns:
        return found[2]
    answer = (not set(col) & set(computed) and set(commColumns.tolist()) == set(col) | set(computed)
              and len(set(col)) == len(col) and len(set(computed)) == len(computed))
    if len(_columnsSplits) >= 64:
        _columnsSplits.clear()
    _columnsSplits[key] = (commColumns, spctColumns, answer)
    return answer


def spct(centralDict):
    display.stepPrint("ORCHESTRA", end=" ", flush=True)
    startOrchestra = time.perf_counter()
    
    # the columns given to the speciation and those kept from commMtrx: from the settings only, built once
    systemSpecies, coupled = centralDict['systemSpecies'], centralDict['coupledVariables']
    crossSpeciation = centralDict['crossDependencies']['speciation']['total']
    col = frames.constant('orchestraInputColumns', (systemSpecies, crossSpeciation, coupled),
                          lambda: [c for c in coupled if c in set(systemSpecies + crossSpeciation)])

    direct = _totInput(centralDict, centralDict['commMtrx'], col)
    commMtrxSpct, commMtrx_primSpecies, calcWallClock, init = speciationOrchestra(
        centralDict, direct if direct is not None else _takeColumns(centralDict['commMtrx'], col))


    if 'failed' in commMtrxSpct.columns:
        if (commMtrxSpct["failed"] == 1).any():
            commMtrxSpct.to_csv("commMtrx_failed.txt", index=False, header=True, sep='\t')
            failed = np.flatnonzero(commMtrxSpct["failed"].to_numpy() == 1)
            raise RuntimeError(f"ORCHESTRA : speciation failed at time step n°{centralDict['lStep'] + 1} in "
                               f"{len(failed)} node(s), first : {failed[:10].tolist()} (output in commMtrx_failed.txt)")
        commMtrxSpct = commMtrxSpct.drop(columns="failed")
    
    
    others, crossTransport = centralDict["anythingButSpecies"], centralDict['crossDependencies']['transport']['total']
    col = frames.constant('orchestraKeptColumns', (others, crossTransport), lambda: others + crossTransport)
    
    comm = centralDict['commMtrx']
    # the labels of the Index of the computed columns (immutable): listed once for the same Index object
    computed = frames.constant('labels', (commMtrxSpct.columns,), commMtrxSpct.columns.tolist)
    light = isinstance(commMtrxSpct, _OrchestraOutput)    # float64 values: frames.allFloat64 of its frame
    if (comm.columns.is_unique and frames.allFloat64(comm)
            and (light or frames.allFloat64(commMtrxSpct)) and len(commMtrxSpct) == len(comm)
            and _columnsSplit(comm.columns, commMtrxSpct.columns, col, computed)):
        full = comm.to_numpy(dtype=float, copy=True)
        full[:, _indexer(comm.columns, computed)] = commMtrxSpct.to_numpy()
        commMtrxSpct = pd.DataFrame(full, index=pd.RangeIndex(len(comm)), columns=comm.columns)
    else:
        if light:
            commMtrxSpct = commMtrxSpct.frame()
        commMtrxSpct = pd.concat([centralDict['commMtrx'][col].reset_index(drop=True),commMtrxSpct], axis=1)
        commMtrxSpct = commMtrxSpct[centralDict['commMtrx'].columns]
    if outputManager.wanted(centralDict, 'speciation'):
       outputManager.write(centralDict, commMtrx_primSpecies, 'primarySpecies', 'PrimarySpecies')
       outputManager.write(centralDict, commMtrxSpct, 'speciation', 'ORCHESTRA')


    centralDict.update({
        "commMtrx": commMtrxSpct,
        "ORCHESTRAInterfTime_WallClock": centralDict['ORCHESTRAInterfTime_WallClock'] + time.perf_counter() - startOrchestra - calcWallClock - init,
        "ORCHESTRACalcTime_WallClock": centralDict['ORCHESTRACalcTime_WallClock'] + calcWallClock,
        "ORCHESTRACalcTime_ProcessorTime": centralDict['ORCHESTRACalcTime_ProcessorTime'] + calcWallClock,
        "ORCHESTRAInitTime" : centralDict['ORCHESTRAInitTime'] + init,
        "ORCHESTRATotalTime" : centralDict['ORCHESTRATotalTime'] + time.perf_counter() - startOrchestra
        })

    display.stepPrint(f"({writeTime((time.perf_counter() - startOrchestra))})") 

    return centralDict