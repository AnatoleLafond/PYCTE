import time
import io
import pandas as pd
import math
import sys 
import importlib.util
import os
import shutil
import difflib
import numpy as np
import re
import copy
import types
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

"""
Welcome in PYCTE ^^ 
This file is the interfacing environment.
It calls different modules and give them necesary informations (i.e. Python objects) to complete their calculations
Module need to be taylor-coded to embedded solvers.
Each solver need specific input.
I've tried to put as many comments as I could, but probably you'll need to be quite confirmed using Python
It has growns quite big since it first version (2 python file).
"""


def normalize_choice(value, mapping, field_name):
    if isinstance(value, str):
        value_norm = value.lower()
    else:
        value_norm = value
    for canonical, aliases in mapping.items():
        for alias in aliases:
            if isinstance(alias, str):
                if value_norm == alias.lower():
                    return canonical
            else:
                if value_norm == alias:
                    return canonical

    str_aliases = [
        alias.lower()
        for aliases in mapping.values()
        for alias in aliases
        if isinstance(alias, str)
    ]

    suggestion = None
    if isinstance(value, str):
        matches = difflib.get_close_matches(value_norm, str_aliases, n=1, cutoff=0.6)
        if matches:
            suggestion = matches[0]

    if suggestion:
        raise ValueError(
            f"Invalid value for '{field_name}': {value!r}. "
            f"Do you mean '{suggestion}' ? "
        )
    else:
        raise ValueError(
            f"Invalid value for '{field_name}': {value!r}. "
        )

from . import OSschemes                      # coupling schemes (operator splitting): one transport / speciation step
from . import warningManager                 # run warnings -> warning.log (all of them, at every occurrence)
from . import display                        # console display of the coupling loop: progress bar or step by step messages
from . import timeUnits                      # timeUnit -> seconds, one table for every module
from . import keywordValidation              # settings checked against their expected type, see keywordValidation.py
from . import outputManager                  # intermediate outputs (txt / vtu), written by a background thread

operator_map = OSschemes.ALIASES             # canonical names and aliases of operatorSplitting

chem_map = {
    'none': [0, '0', 'none', None, 'transportOnly'], # if you want the implementation of transport solvers
    'PhreeqC': [1, '1', 'phreeqc','PhreeqC','Phreeqc'],
    'xGEMS': [2, '2', 'xgems','xGEMS','gems'],
    'ORCHESTRA': [3, '3', 'orchestra','ORCHESTRA', 'Orchestra'],
    'nativeKinetics': [4, '4', 'nativeKinetics','kinetics', 'nativekinetics'],
    'nativeSpeciation': [5, '5', 'nativeSpeciation','speciation', 'nativespeciation'],
    'PhreeqcRM': [6, '6', 'phreeqcrm', 'PhreeqcRM', 'phreeqcRM', 'rmphreeqc'],
}

trspt_map = {
    'COMSOL': [1, '1', 'comsol','Comsol','COMSOL'],
    'nativeTransport': [2, '2', 'nativeTransport','native','trspt'],
    'PFLOTRAN': [3, '3', 'Pflotran','pflotran','PFLOTRAN'],
}

def writeTime(tps, arr=10):
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


_MISSING = object()


def _initialConditionsText(path):
    '''Text of the initial conditions file: UTF-8 (with or without BOM), UTF-16 (PowerShell 5 '>' / Out-File, some
    Notepad saves), otherwise the encoding found by charset_normalizer (latin-1, cp1252 ...).'''
    raw = Path(path).read_bytes()
    if raw.startswith((b'\xff\xfe', b'\xfe\xff')):
        return raw.decode('utf-16')
    if raw.startswith(b'\xef\xbb\xbf'):
        return raw[3:].decode('utf-8', errors='replace')
    try:
        return raw.decode('utf-8')
    except UnicodeDecodeError:
        from charset_normalizer import from_bytes
        best = from_bytes(raw).best()
        return str(best) if best is not None else raw.decode('latin-1')


def _initialConditionsError(path, text, error):
    '''Why the initial conditions are not a table of numbers: decimal comma, header line without %, or the first value
    that is not a number, with its line and column.'''
    name = Path(path).name
    number = re.compile(r'^[-+]?(\d+\.?\d*|\.\d+)([eEdD][-+]?\d+)?$|^[-+]?(nan|inf)$', re.IGNORECASE)
    decimalComma = re.compile(r'^[-+]?\d+,\d*([eE][-+]?\d+)?$')
    first = True
    for k, line in enumerate(text.splitlines(), start=1):
        line = line.split('%', 1)[0].strip()
        if not line:
            continue
        if ';' in line:
            return (f"initial conditions {name}, line {k} : the values are separated by ';' (CSV file of Excel) : "
                    "separate them by spaces or tabs (Excel : save as 'Text (tab delimited)'), with '.' as the "
                    "decimal separator")
        for j, value in enumerate(line.split(), start=1):
            if number.match(value):
                continue
            if decimalComma.match(value):
                return (f"initial conditions {name}, line {k}, column {j} : '{value}' has a decimal comma : save the "
                        "file with '.' as the decimal separator (Excel : Options > Advanced > decimal separator, or "
                        "export from a version set to English)")
            if first:
                return (f"initial conditions {name}, line {k} : '{line[:80]}' is not a line of numbers : the header "
                        "line (names of the columns) must start with %, e.g. '% x Ca+2 Cl- ...'")
            return f"initial conditions {name}, line {k}, column {j} : '{value}' is not a number"
        first = False
    ragged = _raggedLines(text)
    if ragged:
        return _raggedMessage(name, ragged)
    return f"initial conditions {name} : {error}"


def _perPhase(setting, key, phases, default):
    '''SI / mineralReversibility {phase: value}: the phases not given take the default (a missing one stopped the
    run with a bare KeyError), a name that is not a phase of the run is an error.'''
    given = setting(key, None) or {}
    unknown = [ph for ph in given if ph not in phases]
    if unknown:
        hints = {ph: difflib.get_close_matches(str(ph), [str(p) for p in phases], n=1) for ph in unknown}
        raise ValueError(f"{key} : " + ", ".join(f"{ph!r}" + (f" (did you mean {h[0]!r} ?)" if h else "")
                                                for ph, h in hints.items())
                         + f" not a phase of the run (phases : {', '.join(phases) or 'none'})")
    return {ph: given.get(ph, default) for ph in phases}


def _checkTransportNames(centralDict, mobileSpecies):
    '''nativeTransport, final columns (totals with MultiCompoundTransport): transportedSpecies and the species keys
    of firstBoundary, secondBoundary, speciesDiffCoeff and speciesPorosity. A misspelled key was ignored (zero flux
    at the boundary, the shared diffCoeff or porosity), a misspelled transported species failed at the first step.
    mobileSpecies: species the database extraction marks as mobile (None: not known).'''
    from . import nativeTransport
    columns = list(centralDict['commMtrx'].columns)
    transported = list(centralDict['transportedSpecies'])

    def hint(name, names):
        match = difflib.get_close_matches(str(name), [str(n) for n in names], n=1)
        return f" (did you mean {match[0]!r} ?)" if match else ""

    unknown = [s for s in transported if s not in columns]
    if unknown:
        raise ValueError("transportedSpecies : " + ", ".join(f"{s!r}{hint(s, columns)}" for s in unknown)
                         + " not a column of the initial conditions")
    if mobileSpecies is not None:
        fixed = [s for s in transported if s in centralDict['systemSpecies'] and s not in mobileSpecies]
        if fixed:
            warningManager.warn(f"transportedSpecies : {fixed} are exchange, surface or phase species for the "
                                "database (fixed on the solid) : they are transported with the water")
    known = set(columns) | set(centralDict['coupledVariables'])
    for key in ('firstBoundary', 'secondBoundary', 'speciesDiffCoeff', 'speciesPorosity'):
        values = centralDict.get(key) or {}
        if not hasattr(values, 'keys'):
            continue
        wrong = [k for k in values if k not in known]
        if wrong:
            raise ValueError(f"{key} : " + ", ".join(f"{k!r}{hint(k, transported)}" for k in wrong)
                             + " not a transported species (transportedSpecies : " + ", ".join(transported) + ")")
    # crossDiffusion {target: {driver: K}}: the target is a transported species ; the driver is a transported species
    # (implicit coupling) or another column, e.g. transported by another instance (explicit coupling)
    cross = centralDict.get('crossDiffusion') or {}
    wrong = sorted(str(k) for k in cross if k not in transported)
    if wrong:
        raise ValueError("crossDiffusion : " + ", ".join(f"{k!r}{hint(k, transported)}" for k in wrong)
                         + " not a transported species (transportedSpecies : " + ", ".join(transported) + ")")
    wrong = sorted({str(d) for drivers in cross.values() if hasattr(drivers, 'keys') for d in drivers
                    if d not in columns})
    if wrong:
        raise ValueError("crossDiffusion : driving species " + ", ".join(f"{k!r}{hint(k, columns)}" for k in wrong)
                         + " not a column of the initial conditions")
    # imposed flow (velocity < 0: from x = L to x = 0) entering through a 'closed' boundary: the water enters at the
    # concentration of the boundary node (zero gradient, checked: unchanged there), whatever firstBoundary / secondBoundary
    velocity = float(centralDict.get('velocity') or 0.0)
    if velocity:
        side = 0 if velocity > 0 else 1
        if nativeTransport.parseCondition(centralDict['boundaryConditions'][side])[1] == 'closed':
            warningManager.warn(f"velocity = {velocity:g} : the water enters at {'x = 0' if side == 0 else 'x = L'}, "
                                f"whose boundary is closed (boundaryConditions[{side}]) : it enters at the concentration "
                                "of the boundary node, no inflow concentration is imposed ; to impose one, give "
                                f"'constant' or 'flux' there with {('firstBoundary', 'secondBoundary')[side]}"
                                + (" (velocity < 0 : flow from x = L to x = 0)" if velocity < 0 else ""))
    # 'constant' or 'flux' boundary: a transported species without a value there has a zero flux there
    schedule = centralDict.get('boundarySchedule')
    for side, key, where in ((0, 'firstBoundary', 'x = 0'), (1, 'secondBoundary', 'x = L')):
        conditions = [c for _, c in schedule[side]] if schedule else [centralDict['boundaryConditions'][side]]
        if not any(nativeTransport.parseCondition(c)[1] in ('constant', 'flux') for c in conditions):
            continue
        values = centralDict.get(key) or {}
        absent = [s for s in transported if s not in values]
        if absent:
            warningManager.warn(f"{key} : no value for {absent[:10]}{' ...' if len(absent) > 10 else ''} at the "
                                f"{'/'.join(dict.fromkeys(conditions))} boundary {where} : zero flux for "
                                f"{'them' if len(absent) > 1 else 'it'} there (give 0 for a zero concentration)")


def _raggedLines(text):
    '''Data lines of the initial conditions whose number of values differs from the most frequent one:
    [(line number, values, expected)].'''
    counts = [(k, len(line.split('%', 1)[0].split())) for k, line in enumerate(text.splitlines(), start=1)]
    counts = [(k, c) for k, c in counts if c]
    if not counts:
        return []
    expected = max(set(c for _, c in counts), key=[c for _, c in counts].count)
    return [(k, c, expected) for k, c in counts if c != expected]


def _raggedMessage(name, ragged):
    '''Lines with a missing or extra value: an empty cell (Excel export) shifts the rest of its line.'''
    lines = ", ".join(f"line {k} : {c} values" for k, c, _ in ragged[:5]) + (", ..." if len(ragged) > 5 else "")
    return (f"initial conditions {name} : {len(ragged)} line(s) without the {ragged[0][2]} values of the other "
            f"lines ({lines}) : an empty cell would shift the following values of its line to the left ; fill "
            "every cell (0 for an absent species)")


def _settingText(value):
    '''value -> one-line Python expression for warning.log.'''
    with np.printoptions(linewidth=sys.maxsize):
        text = repr(value)
    return re.sub(r"\n\s*", " ", text)


class Settings:
    '''Settings of a run: values deep-copied at first reading and recorded for warning.log.'''

    def __init__(self, config, known=None):
        self._config = config
        self._known = known
        self._copies = {}
        self.used = {}                              # name -> (value as text, given by the user)

    @property
    def config(self):
        return self._config

    @property
    def known(self):
        return self._known

    def _source(self, name):
        '''Config attribute holding setting `name` (its name or a former one), None if not given.'''
        names = [n for n in [name] + [f for f, c in keywordValidation.ALIASES.items() if c == name]
                 if hasattr(self._config, n)]
        if len(names) > 1 and any(not _sameValue(getattr(self._config, names[0]), getattr(self._config, n))
                                  for n in names[1:]):
            raise ValueError(f"{' and '.join(names)} are the same setting (former names : "
                             f"{', '.join(names[1:])}) and are given with different values : keep {name} only")
        return names[0] if names else None

    def given(self, name):
        return name in self._copies or self._source(name) is not None

    def peek(self, name):
        '''Value given by the user, neither copied nor recorded.'''
        return self._copies[name] if name in self._copies else getattr(self._config, self._source(name) or name)

    def get(self, name, default=_MISSING):
        if name in self._copies:
            return self._copies[name]
        source = self._source(name)
        if source is not None:
            value = getattr(self._config, source)
            try:
                value = copy.deepcopy(value)
            except Exception:
                pass
            self._copies[name] = value
            self.used[name] = (_settingText(value), True)
            return value
        if default is _MISSING:
            raise AttributeError(f"pycte : setting {name!r} is missing")
        self.used[name] = (_settingText(default), False)
        return default

    def givenNames(self):
        '''Names given by the user (only the pycte settings for a module).'''
        try:
            names = [n for n in vars(self._config) if not n.startswith('_')]
        except TypeError:
            names = [n for n in dir(self._config) if not n.startswith('_')]
        if isinstance(self._config, types.ModuleType):
            if self._known is not None:
                names = [n for n in names if n in self._known]
            else:
                names = [n for n in names if not isinstance(getattr(self._config, n),
                                                            (types.ModuleType, types.FunctionType, type))]
        return names

    def header(self):
        '''warning.log block of the settings read (given, default) and of those given but unused.'''
        lines = [f"{SETTINGS_HEADER} (given, then default values) :"]
        lines += [f"    {n} = {text}" for n, (text, given) in self.used.items() if given]
        lines.append("    # default values")
        lines += [f"    {n} = {text}" for n, (text, given) in self.used.items() if not given]
        unused = [n for n in self.givenNames() if keywordValidation.ALIASES.get(n, n) not in self.used]
        if unused:
            lines.append("    # given but not used by this coupling")
            lines += [f"    {n} = {_settingText(getattr(self._config, n))}" for n in unused]
        return "\n".join(lines) + "\n\n"


def _sameValue(a, b):
    try:
        return bool(np.all(a == b)) if isinstance(a, np.ndarray) or isinstance(b, np.ndarray) else a == b
    except Exception:
        return a is b


SETTINGS_HEADER = "pycte : Settings of the run"        # start of the block, skipped by postProcess.getTimes


def saturationSpeciation(launcher):
    '''Speciation launcher keeping solute amounts when the chemistry changes the saturation column.'''
    def call(centralDict):
        column = centralDict['saturationColumn']
        sIn = centralDict['commMtrx'][column].to_numpy(dtype=float).copy()
        previous = centralDict.get('saturationPrevious')
        if (previous is not None and not np.array_equal(sIn, previous)
                and not centralDict.get('saturationWarned')):
            # two speciation calls without transport in between (operatorSplitting = 'alternative', ...): the water
            # displaced by the previous call is not expelled yet, the chemistry sees its solutes in less water
            warningManager.warn(f"saturation '{column}' : speciation called again before the transport has expelled "
                                "the water displaced by the previous call (operatorSplitting = 'alternative', ...) : "
                                "the chemistry sees these solutes concentrated in the remaining water (relative error "
                                "of the order of the saturation change of one call) ; 'SNIA' avoids it")
            centralDict['saturationWarned'] = True
        centralDict.update(launcher(centralDict))
        if centralDict.get('gasVolumeColumn'):
            from . import nativeTransport
            sGas = nativeTransport.saturationFromGas(centralDict)
            centralDict['commMtrx'] = centralDict['commMtrx'].copy()
            centralDict['commMtrx'][column] = sGas
        comm = centralDict['commMtrx']
        sOut = comm[column].to_numpy(dtype=float)
        bad = np.flatnonzero(~(np.isfinite(sOut) & (sOut > 0) & (sOut <= 1)))
        if len(bad):
            raise ValueError(f"pycte : saturation column '{column}' written by the chemistry outside ]0, 1] at node(s) "
                             f"{bad[:10].tolist()} (values {sOut[bad[:10]].tolist()})")
        if not np.array_equal(sIn, sOut):
            species = [s for s in centralDict['transportedSpecies'] if s in comm.columns]
            values = comm[species].to_numpy(dtype=float) * (sIn / sOut)[:, None]
            comm = comm.copy()
            comm[species] = values
            centralDict['commMtrx'] = comm
        return centralDict
    return call


def main(pycte_input, known=None):
    '''Runs pycte with `pycte_input` (namespace or module); `known`: names of the pycte settings.'''
    # warnings emitted during the run (including before an error) end up in warning.log
    warningManager.begin()
    try:
        return _main(Settings(pycte_input, known))
    except (Exception, KeyboardInterrupt) as error:
        # fatal error (setting, solver, ...): the error and its traceback end warning.log, then the error goes on to
        # the caller (non-zero exit code of a script). An invalid setting is already reported by keywordValidation
        if isinstance(error, BrokenProcessPool):
            # Windows (spawn): each worker imports the script again; code after pycte.run() that fails there (a
            # postProcess of a folder not written yet, a plot) kills the worker, whose error is printed above
            advice = RuntimeError("pycte : a worker process (speciationThreads / transportThreads > 1) stopped "
                                  "abruptly. Each worker runs your script again, except what is under  if __name__ == "
                                  "'__main__':  : keep there pycte.run() and every other action, before it (folder "
                                  "creation or removal, os.chdir, file copies) and after it (postProcess, plots, "
                                  "plt.show()) ; only the imports and the pycte.xxx(...) settings stay outside. The "
                                  f"error of the worker, if any, is printed above ({error})")
            warningManager.abort(advice)
            raise advice from error
        if not isinstance(error, keywordValidation.KeywordError):
            warningManager.abort(error)
        raise
    finally:
        # progress bar line ended before the error traceback (no-op after a normal end)
        display.finish()
        # end of run (normal or error), and only there: the solvers are kept alive during the whole run,
        # last step included (Strang, Symmetrical, Iterative, OSdefined call them several times per step).
        # COMSOL session closed cleanly (client disconnected, server stopped), otherwise mph cuts it when Python
        # exits, which can crash the JVM; IPhreeqc instance and ORCHESTRA solver released, so that a next
        # pycte.run() in the same Python session starts from a fresh solver (PhreeqcRM instance released as well)
        for moduleName, closer in (("comsol", "shutdownComsol"), ("phreeqc", "resetPhreeqC"),
                                   ("orchestra", "resetOrchestra"), ("rmphreeqc", "resetPhreeqcRM"),
                                   ("nativeTransport", "resetProcesses"), ("call", "resetProcesses")):
            solverModule = sys.modules.get(f"{__package__}.{moduleName}" if __package__ else moduleName)
            if solverModule is not None:
                getattr(solverModule, closer)()
        # speciationThreads / transportThreads > 1: the workers (PhreeqC, nativeSpeciation, nativeTransport) are
        # kept from one step to the next instead of being restarted at every step; they are stopped here
        warningManager.shutdown()
        # output files still being written by the output thread: completed before returning (or raising); a write
        # error is reported in warning.log, the error of the run (if any) is the one raised
        try:
            outputManager.close()
        except Exception as error:
            warningManager.warn(f"pycte : an output file could not be written : {type(error).__name__} : {error}")
        warningManager.end()


def _main(settings):
    setting = settings.get
    pyctestart = time.perf_counter()
    pathpycteInput = Path(os.getcwd())
    # every setting given is checked (type, range, unknown name) before anything is read or started
    keywordValidation.checkGiven(settings)
    maxTime = setting('maxTime', 1)
    timeStep = setting('timeStep', 1)
    dtpycte = list(setting('dtpycte', []))
    
    if dtpycte and dtpycte[0]==0: del dtpycte[0]
    if not dtpycte:
        # last step shortened when maxTime is not a multiple of timeStep: the run ends at maxTime, not after it
        # (1e-9 tolerance: a ratio that rounding puts just above an integer does not add a tiny last step)
        n = max(1, math.ceil(maxTime / timeStep - 1e-9))
        dtpycte = (np.arange(1, n + 1) * timeStep).tolist()
        if dtpycte[-1] != maxTime:
            if abs(dtpycte[-1] - maxTime) > 1e-9 * abs(maxTime):
                warningManager.info(f"pycte : maxTime = {maxTime} is not a multiple of timeStep = {timeStep} : the last "
                                    f"step lasts {maxTime - (dtpycte[-2] if n > 1 else 0):.6g}")
            dtpycte[-1] = maxTime
       
    
    def readInputFile(txtPath, coord, inputHeaders) :
        '''Initial conditions file -> (commMtrx, coupled variables), names checked against its % header.'''
        text = _initialConditionsText(txtPath)
        try:
            dataframe = pd.read_csv(io.StringIO(text), sep=r"\s+", comment="%", dtype=float, header=None)
        except ValueError as err:
            raise ValueError(_initialConditionsError(txtPath, text, err)) from None
        # a line shorter than the first one is completed with NaN by read_csv: an empty cell has shifted its values
        ragged = _raggedLines(text)
        if ragged:
            raise ValueError(_raggedMessage(Path(txtPath).name, ragged))
        values = dataframe.to_numpy()
        if not np.isfinite(values).all():
            row, col = np.argwhere(~np.isfinite(values))[0]
            dataLines = [k for k, line in enumerate(text.splitlines(), start=1) if line.split('%', 1)[0].strip()]
            raise ValueError(f"initial conditions {Path(txtPath).name} : {int((~np.isfinite(values)).sum())} value(s) "
                             f"nan or inf, the first one line {dataLines[row]}, column {col + 1} : give a number")

        header = None
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if not line.startswith('%'):
                break
            names = [w.strip().lstrip('%').strip() for w in (line.split('\t') if '\t' in line else line.split())]
            names = [w for w in names if w]
            if len(names) == len(dataframe.columns):
                header = names
        coordOf = lambda names: [c.lower() for c in names[:len(coord)]]

        if not inputHeaders:
            if header is None or coordOf(header) != coord:
                raise ValueError(f"{txtPath} : without coupledVariables, the coupled variables are read from the header "
                                 f"of the initial conditions : a line starting with %, then {' '.join(coord)} and one name "
                                 f"per column ({len(dataframe.columns)} columns)")
            inputHeaders = header[len(coord):]
        inputHeaders = list(inputHeaders)
        if len(dataframe.columns) != len(coord + inputHeaders):
            raise ValueError(f"{txtPath} : {len(dataframe.columns)} columns, {len(coord + inputHeaders)} expected "
                             f"({' '.join(coord)} and the {len(inputHeaders)} names of coupledVariables)")
        if header is not None:
            differ = [(k + 1, h, g) for k, (h, g) in enumerate(zip(header, coord + inputHeaders))
                      if h != g and not (k < len(coord) and h.lower() == g)]
            if differ:
                warningManager.warn(f"initial conditions {Path(txtPath).name} : the header differs from coupledVariables "
                                    f"in {len(differ)} column(s) ; the header is ignored, the columns of the file are "
                                    "read in the order of coupledVariables (column : header -> coupledVariables) : "
                                    + ", ".join(f"{k} : {h} -> {g}" for k, h, g in differ[:10])
                                    + (", ..." if len(differ) > 10 else ""))
        dataframe.columns = coord + inputHeaders
        return dataframe, inputHeaders

    chargeGeometryCells = []
    chargeGeometrySpecies = []
    if setting('speciesChargeGeometry', None):
        for (maille, spc) in setting('speciesChargeGeometry', None): 
            chargeGeometryCells += [maille]
            chargeGeometrySpecies += [spc]
    
    
    # intermediate outputs: output keyword (see outputManager.py); missing -> {'stepEnd': [last step]}
    from . import outputManager
    # output steps of a bundled example (not given by the user) beyond a run made shorter (maxTime): dropped
    dropped = {} if getattr(settings.config, '_exampleOutput', False) else None
    output = outputManager.normalize(setting('output', None), len(dtpycte), dropped)
    if dropped:
        warningManager.warn("pycte : output steps of the example beyond the run (" + str(len(dtpycte)) + " steps) "
                            "left out : " + ", ".join(f"{key} {steps[0]} to {steps[-1]}" for key, steps in dropped.items())
                            + " ; give pycte.output({...}) to choose the output steps")
    if setting('intermediateOutput', None) is not None:
        warningManager.warn("pycte : intermediateOutput is no longer used and is ignored : the outputs are set by "
                            "output = {'stepEnd' | 'speciation' | 'transport': [steps]}")
    comsolTags = list(setting('outputComsol', None) or [])
    # outputFormat = ['txt', 'vtu']: 'vtu' writes the same outputs as ParaView files in twin <folder>_vtu folders
    outputFormat = outputManager.normalizeFormat(setting('outputFormat', ['txt']))
    if settings.given('comsolVTU'):
        warningManager.warn("pycte : comsolVTU is no longer used and is ignored : the COMSOL VTU export follows "
                            "outputFormat = ['txt', 'vtu']")
    paths = outputManager.folders(pathpycteInput, output, comsolTags, outputFormat)   # key of centralDict['paths'] -> folder

    # coupling: named scheme (operatorSplitting) or user-defined sequence (OSdefined, or a list given
    # to operatorSplitting), see OSschemes.py. With OSdefined, missing chemistryModule / transportModule are deduced from the
    # solvers named in the sequence; when given, they must match it.
    osDefined = setting('OSdefined', None)
    osChoice = setting('operatorSplitting', None)
    if osDefined is None and isinstance(osChoice, (list, tuple)):
        osDefined, osChoice = osChoice, None
    chemChoice = (normalize_choice(setting('chemistryModule'), chem_map, 'chemistryModule')
                  if settings.given('chemistryModule') else None)
    trsptChoice = (normalize_choice(setting('transportModule'), trspt_map, 'transportModule')
                   if settings.given('transportModule') else None)
    # named solver instances (solverInstances.py): the same solver several times, each instance with its own
    # settings, species and state; a role without instance there keeps chemistryModule / transportModule
    from . import solverInstances
    declared = solverInstances.parse(setting('solverInstances')) if settings.given('solverInstances') and \
        settings.peek('solverInstances') is not None else []
    declaredRoles = {i.role for i in declared}
    for role, choice, keyword in (('speciation', chemChoice, 'chemistryModule'), ('transport', trsptChoice, 'transportModule')):
        solvers = [i.solver for i in declared if i.role == role]
        if solvers and choice is not None and choice not in solvers:
            raise ValueError(f"{keyword} = {choice} : the {role} solvers are given by solverInstances "
                             f"({', '.join(i.label + ' : ' + i.solver for i in declared if i.role == role)}) : "
                             f"remove {keyword} or declare an instance of {choice}")
    osSequence = None
    if osDefined is not None:
        osSequence = OSschemes.parseDefined(osDefined, chem_map, trspt_map, {i.label: i.role for i in declared})
        chemChoice, trsptChoice = OSschemes.resolveModules(osSequence, chemChoice, trsptChoice, declaredRoles)
        if osChoice is not None:
            warningManager.warn(f"pycte : operatorSplitting = {osChoice!r} is ignored : OSdefined sets the coupling")
        osName = 'OSdefined'
    else:
        osName = normalize_choice(1 if osChoice is None else osChoice, operator_map, 'operatorSplitting')
    instances = solverInstances.build(
        declared,
        chemChoice if chemChoice is not None else normalize_choice(1, chem_map, 'chemistryModule'),
        trsptChoice if trsptChoice is not None else normalize_choice(1, trspt_map, 'transportModule'))
    # several instances of a role: only with OSdefined, which calls each instance by its name (a named scheme couples
    # one speciation and one transport)
    multiple = solverInstances.several(instances)
    if multiple and osName != 'OSdefined':
        raise ValueError(
            "solverInstances : several " + " and ".join(f"{role} instances ("
                                                        + ", ".join(i.label for i in solverInstances.byRole(instances, role))
                                                        + ")" for role in multiple)
            + f" : only with OSdefined, which calls each instance by its name, e.g. OSdefined = ["
            + ", ".join(f"(1, '{i.label}')" for i in instances if i.solver != 'none')
            + "] ; operatorSplitting " + (f"= {osName!r} " if osChoice is not None else "(default 'SNIA') ")
            + "couples one speciation and one transport solver")
    if osSequence is not None:
        osSequence = OSschemes.bind(osSequence, [(i.label, i.role, i.solver) for i in instances])
        called = {label for _, _, label in osSequence}
        never = [i.label for i in instances if i.label not in called and i.solver != 'none']
        if multiple and never:
            raise ValueError(f"solverInstances : {', '.join(never)} never called by OSdefined : call every instance "
                             "(or remove it from solverInstances)")
    speciationInstances = solverInstances.byRole(instances, 'speciation')
    transportInstances = solverInstances.byRole(instances, 'transport')
    for instance in instances:
        instance.scoped = solverInstances.ScopedSettings(settings, instance)
    if multiple and setting('MultiCompoundTransport', False):
        raise ValueError("MultiCompoundTransport is not available with several instances of a role (solverInstances) : "
                         "the conversion to totals changes the transported species of a single transport")
    couplingInfo = [osName, speciationInstances[0].solver, transportInstances[0].solver]
    # settings of the (first) transport solver read before its set-up
    trsptSetting = transportInstances[0].scoped.get
    # rules that depend on the solvers (transportThreads = None with COMSOL only, files read by the solvers, settings
    # a solver needs): checked before any solver starts (database extraction, COMSOL licence and server, ...)
    if declared:
        for k, instance in enumerate(instances):
            keywordValidation.checkModules(instance.scoped,
                                           instance.solver if instance.role == 'speciation' else None,
                                           instance.solver if instance.role == 'transport' else None, common=k == 0)
    else:
        keywordValidation.checkModules(settings, couplingInfo[1], couplingInfo[2])
    # iterative coupling: the source term is only handled by nativeTransport, by COMSOL with a model
    # prepared for it (comsolReactionTerms = True: S_ / K_ columns of int1, I_ integrals exported by data1), or by
    # PFLOTRAN with a reaction sandbox (pflotranReactionTerms = True or 'sandbox': S_ / K_ datasets of testpycte.h5,
    # I_<species>pycte integrals written to the .tec)
    pflotranTerms = trsptSetting('pflotranReactionTerms', None)
    if pflotranTerms not in (None, False, True, 'sandbox'):
        raise ValueError(f"pflotranReactionTerms must be True or 'sandbox' (got {pflotranTerms!r})")
    if osName == 'Iterative' and not (couplingInfo[2] == 'nativeTransport'
                                      or (couplingInfo[2] == 'COMSOL' and trsptSetting('comsolReactionTerms', False))
                                      or (couplingInfo[2] == 'PFLOTRAN' and pflotranTerms)):
        raise ValueError(f"operatorSplitting = 'Iterative' needs transportModule = 'nativeTransport', 'comsol' with "
                         f"comsolReactionTerms = True and a model reading the reaction source terms, or 'pflotran' "
                         f"with pflotranReactionTerms = True and a reaction sandbox (the reaction source term is not "
                         f"passed to {couplingInfo[2]})")
    splittingLabel = osName
    if osSequence is not None:
        splittingLabel = f"OSdefined [{OSschemes.describe(osSequence, couplingInfo[1], couplingInfo[2])}]"
        for message in OSschemes.checkFractions(osSequence, couplingInfo[1],
                                                [(i.label, i.role, i.solver) for i in instances] if multiple else None):
            warningManager.warn(message)


    centralDict = { # gather keywords which do not depend on components
        "warningNbr": 0,
        "inputPath" : pathpycteInput,
        "couplingInfo" : couplingInfo,
        # (role, fraction of dt, instance) of each call
        "OSdefined" : list(osSequence) if osSequence else None,
        # iterative coupling (OSschemes.iterative): tolerances, max iterations, under-relaxation, Anderson depth
        "iterTol" : setting('iterTol', 1e-4),
        "iterAtol" : setting('iterAtol', 1e-12),
        "iterMaxIter" : setting('iterMaxIter', 50),
        "iterRelax" : setting('iterRelax', 1.0),
        "iterAnderson" : setting('iterAnderson', 3),
        # cap on k.dt of the first-order sink (moderate with COMSOL: k.I is less accurate there than in nativeTransport)
        "iterSinkMax" : setting('iterSinkMax', 1e8),
        # update of the consumption term: 'linear' (k = -Uchem/I), 'log' (k.ln(1+Uchem/X)/ln(1-k.I/X)),
        # 'exponential' (k = ln(X/Cchem)/dt) or 'constant' (constant rate U/dt, without first-order sink)
        "iterSinkMode" : setting('iterSinkMode', 'linear'),


        "paths" : paths,
        "outputFormat" : outputFormat,
        "system" : setting('system', 1),
        # console display of the coupling loop: 'progressBar' (one line redrawn, steps done, elapsed and remaining
        # time) or 'steps' (step by step messages of pycte and of the solvers), see display.py
        "consoleDisplay" : display.canonical(setting('consoleDisplay', 'progressBar')),
        # parallelism per solver: speciationThreads -> PhreeqC, ORCHESTRA, GEMS,
        # nativeSpeciation, PhreeqcRM (processes); transportThreads -> nativeTransport, COMSOL (server cores), PFLOTRAN (mpirun -n)
        "speciationThreads": setting('speciationThreads', 1),
        "transportThreads": setting('transportThreads', 1),
        "coupledVariables" : setting('coupledVariables', None),
        "timeUnit" : setting('timeUnit', 's'),       # checked below: an unknown unit stops the run
        "geometry" : setting('geometry', 1),

        "coord" : ['x', 'y', 'z'][:setting('geometry', 1)],
        "chemistryModule" : setting('chemistryModule', 1),
        "transportModule" : setting('transportModule', 1),
        "firstStepEquilibrium" : setting('firstStepEquilibrium', False),
        'nonTrivialDecomposition' : setting('nonTrivialDecomposition', None),
        "preliminaryEquilibrium" : setting('preliminaryEquilibrium', False),
        'extractDBTime' : 0,
        'MultiCompoundTransport' : setting('MultiCompoundTransport', False),
        "initialConditions" : Path(setting('initialConditions', os.path.join(pathpycteInput, 'ic.txt'))),
        "dtpycte" : dtpycte,
        "output" : output,
        }
    timeUnits.secondsPerUnit(centralDict['timeUnit'])

    # coupled variables: the coupledVariables setting, or the header of the initial conditions when it is not given
    centralDict["commMtrx"], centralDict['coupledVariables'] = readInputFile(
        centralDict['initialConditions'], centralDict['coord'], centralDict['coupledVariables'])

    # crossDependencies: always the same structure, empty blocks when the setting is not given (crossDependenciesManager)
    crossDep = setting('crossDependencies', None) or {}
    from . import crossDependenciesManager

    # coupledVariables: an alias (tk, p, ph, ...) designates the column of that name, or the column named by another
    # alias of the same quantity (crossDependencies ['tk'] with a column temperature(K))
    centralDict['crossDependencies'] = crossDependenciesManager.crossDep(centralDict, crossDep,
                                                                      centralDict['coupledVariables'])
    cd = centralDict['crossDependencies']
    # keyword unknown to a solver with a table of coupled quantities (PhreeqC, PhreeqcRM, xGEMS; PFLOTRAN)
    for kind, solver in (('Speciation', couplingInfo[1]), ('Transport', couplingInfo[2])):
        if cd[f'ignored{kind}']:
            raise ValueError(f"crossDependencies['{kind.lower()}'] : {cd[f'ignored{kind}']} unknown for {solver}, "
                             f"which couples : {', '.join(cd[f'known{kind}'])}")
    # every crossDependency is a coupled variable, i.e. a column of the initial conditions
    missing = [n for n in cd['names'] if n not in centralDict['coupledVariables']]
    if missing:
        raise ValueError(f"crossDependencies {missing} are not in coupledVariables : a crossDependency is a coupled "
                         "variable (a column of the initial conditions) that is not a species")
    # negative amount of a species: the chemistry sets it to 0, i.e. creates mass. Not checked: the coordinates, the
    # crossDependencies and the signed quantities (pe, Eh, charge balance, proton / electron totals H and E)
    signed = {'pe', 'eh', 'e', 'h', 'chargebalance', 'charge_balance', 'charge', 'e-'}
    checked = [c for c in centralDict['coupledVariables']
               if c not in cd['names'] and str(c).lower() not in signed and c in centralDict['commMtrx'].columns]
    values = centralDict['commMtrx'][checked].to_numpy(dtype=float)
    negative = values < -1e-12 * np.maximum(np.abs(values).max(axis=0, initial=0.0), 1e-300)
    if negative.any():
        worst = {checked[j]: float(values[:, j].min()) for j in np.flatnonzero(negative.any(axis=0))}
        warningManager.warn(f"initial conditions {Path(centralDict['initialConditions']).name} : negative value(s) "
                            f"in {len(worst)} column(s) (not physical ; a speciation solver sets them to 0, i.e. creates mass) : "
                            + ", ".join(f"{name} {value:.3g} ({int(negative[:, checked.index(name)].sum())} node(s))"
                                        for name, value in list(worst.items())[:10]))
    # two inputs for the same quantity (pe and Eh): the first one of each pair is used
    for kind, solver in (('speciation', couplingInfo[1]), ('transport', couplingInfo[2])):
        inputs = crossDependenciesManager.byQuantity(cd[kind], 'input')
        for first, second in (('-la("e-")', 'Eh'),):
            if first in inputs and second in inputs:
                warningManager.warn(f"crossDependencies['{kind}'] : {inputs[first]} and {inputs[second]} both given to "
                                    f"{solver}, {inputs[first]} is used and {inputs[second]} is ignored")
    # temperature column (aliases temperature, t, temp, tk ...): always in K; °C values would make PhreeqC compute at
    # 25 K and not converge ("K has not converged")
    for kind, solver in (('speciation', couplingInfo[1]), ('transport', couplingInfo[2])):
        inputs = crossDependenciesManager.byQuantity(cd[kind], 'input')
        for quantity in ('tk', 'temperature(K)'):
            column = inputs.get(quantity)
            if column is None or column not in centralDict['commMtrx'].columns:
                continue
            values = centralDict['commMtrx'][column].to_numpy(dtype=float)
            if np.nanmin(values) < keywordValidation.MIN_KELVIN:
                raise ValueError(f"crossDependencies['{kind}'] : the column {column!r} is the temperature in K for "
                                 f"{solver}, but it goes down to {np.nanmin(values):g} K : °C values ? Every "
                                 "temperature is in K (298.15 for 25 °C)")
    # PhreeqcRM carries pH and pe from one call to the next in the state of its cells: outputs only
    if couplingInfo[1] == 'PhreeqcRM':
        outputsOnly = [n for n in cd['speciation']['output'] if n not in cd['speciation']['input']
                       and cd['speciation']['quantity'][n] in ('-la("H+")', '-la("e-")', 'Eh')]
        if outputsOnly:
            warningManager.info(f"crossDependencies['speciation'] : {outputsOnly} are outputs of PhreeqcRM only (not "
                                "imposed : the charge balance and the H, O totals fix them)")

    # species: the coupled variables that are not crossDependencies. The systemSpecies setting is obsolete: accepted when
    # it gives the same names, an error otherwise
    crossNames = cd['names']
    centralDict['systemSpecies'] = [s for s in centralDict['coupledVariables'] if s not in crossNames]
    given = setting('systemSpecies', None)
    if given:
        extra = [s for s in given if s not in centralDict['systemSpecies']]
        absent = [s for s in centralDict['systemSpecies'] if s not in given]
        if extra or absent:
            raise ValueError("systemSpecies is deduced (coupledVariables without the crossDependencies) ; the setting "
                             "given differs : " + " ; ".join(
                                 ([f"{extra} are not species (crossDependencies or not in coupledVariables)"] if extra else [])
                                 + ([f"{absent} are missing"] if absent else []))
                             + ". Remove the setting, or declare the non-species columns in crossDependencies")
        warningManager.warn("systemSpecies is obsolete (deduced from coupledVariables and crossDependencies) : ignored")

    
    centralDict.update({"anythingButSpecies" : [c for c in list(centralDict["commMtrx"].columns) if c not in centralDict['coupledVariables']]})

    display.configure(centralDict['consoleDisplay'])
    chemNames = ' + '.join(i.label for i in speciationInstances)
    trsptNames = ' + '.join(i.label for i in transportInstances)
    print(f"""
####  #  #  ####  #####  ####     {splittingLabel} splitting :
#  #  #  #  #       #    #        {chemNames}--->
####  ####  #       #    ####            <---{trsptNames}
#        #  #       #    #        {maxTime}{centralDict['timeUnit']} in {len(dtpycte)} steps        
#     ####  ####    #    ####     
       """, flush=True)
    
    # periodic cell renewal: list of groups {'cells', 'times', 'concentration'} (see renewal.py); completed by
    # renewal.finalize once MultiCompoundTransport has converted commMtrx and the fresh solutions to totals
    renewalOn = False
    if setting('renewal', None):
        from . import renewal
        centralDict.update(renewal.setup(centralDict, setting('renewal')))
        renewalOn = True
        
    
    
    
    
    # solvers: set-up functions of SPECIATION_SOLVERS / TRANSPORT_SOLVERS (end of this file), which read their settings
    # into centralDict and return their launcher
    run = types.SimpleNamespace(inputPath=pathpycteInput, timeStep=timeStep, chargeGeometryCells=chargeGeometryCells,
                                comsolTags=comsolTags, defaultTransported=None)
    speciationName, transportName = centralDict['couplingInfo'][1], centralDict['couplingInfo'][2]
    # roles with several instances (solverInstances, OSdefined): each instance keeps its keys of centralDict in its own
    # store, put back only during its calls (solverInstances.setup / context)
    isolated = set(multiple)
    # worker processes (speciationThreads > 1) started now: their start-up (Python and pycte imports, ~2 s each)
    # runs during the database extraction and the solver set-up instead of delaying the first speciation
    # (PhreeqC: no process, one thread per block of nodes, each with its IPhreeqc instance, see phreeqc.spct;
    # nativeSpeciation starts in a single process and keeps its workers only if they are worth it, call._chooseProcesses)
    # (xGEMS: processes of its own, kept from one step to the next, each building its ChemicalEngine at every call)
    if any(i.solver == 'xGEMS' for i in speciationInstances) and centralDict['speciationThreads'] > 1:
        warningManager.warm(centralDict['speciationThreads'], pinned=True, group='xGEMS')
    # species transported when transportedSpecies is not given: those the database extraction marks as mobile
    # (PhreeqC, PhreeqcRM: no exchange, surface or phase species; xGEMS: aqueous species and water, no solid or gas),
    # otherwise every species (coupled variables that are not crossDependencies)
    extracted = []
    for instance in speciationInstances:
        solverInstances.setup(centralDict, instance, 'speciation' in isolated,
                              lambda: SPECIATION_SOLVERS[instance.solver](centralDict, instance.scoped, run))
        extracted += [s for s in (solverInstances.value(centralDict, instance, 'transportedSpecies') or [])
                      if s not in extracted]
    defaultTransported = ([s for s in centralDict['systemSpecies'] if s in extracted] if extracted
                          else centralDict['systemSpecies'])
    mobileSpecies = set(extracted) if extracted else None

    run.defaultTransported = defaultTransported
    if 'transport' in isolated:
        for instance in transportInstances:
            if isinstance(instance.scoped.peek('saturation') if instance.scoped.given('saturation') else None, str) \
                    or (instance.scoped.given('gasVolume') and instance.scoped.peek('gasVolume') is not None):
                raise ValueError(f"solverInstances : {instance.label} : saturation = '<column>' and gasVolume (water "
                                 "content written by the chemistry) are not available with several transport instances")
            if instance.scoped.given('areaSchedule') and instance.scoped.peek('areaSchedule') is not None:
                raise ValueError(f"solverInstances : {instance.label} : areaSchedule is not available with several "
                                 "transport instances")
    for instance in transportInstances:
        solverInstances.setup(centralDict, instance, 'transport' in isolated,
                              lambda: TRANSPORT_SOLVERS[instance.solver](centralDict, instance.scoped, run))
    if 'transport' in isolated:
        # outside the calls of the transport instances: every transported column
        centralDict['transportedSpecies'] = list(dict.fromkeys(
            s for i in transportInstances for s in solverInstances.value(centralDict, i, 'transportedSpecies')))
    speciationLauncher = {i.label: solverInstances.launcher(i) for i in speciationInstances}
    transportLauncher = {i.label: solverInstances.launcher(i) for i in transportInstances}
    # first speciation (firstStepEquilibrium) and launchers of the named schemes: the first instance of each role
    speciationKey, transportKey = speciationInstances[0].label, transportInstances[0].label
    # nativeTransport with transportThreads > 1: its first call runs in a single process and measures its duration;
    # the worker processes are started only if they are worth it (nativeTransport._chooseProcesses)

    # MultiCompoundTransport: commMtrx only holds totals; single conversion of the initial and boundary conditions
    # (species or totals) before the first step, by the solver (MULTI_COMPOUND)
    multiCompound = MULTI_COMPOUND.get(speciationName)
    if centralDict['MultiCompoundTransport'] and multiCompound is not None:
        multiCompound(centralDict, trsptSetting)

    for instance in transportInstances:
        if instance.solver != 'nativeTransport':
            continue
        with solverInstances.context(centralDict, instance):
            # Nernst-Planck: the valence of each transported species (final list, totals with MultiCompoundTransport);
            # a missing one stopped the first transport step with a bare KeyError
            if (centralDict.get('NernstPlanck')
                    and not (centralDict.get('FickDiffusion') or centralDict.get('advection') or centralDict.get('ADE'))):
                valence = centralDict.get('speciesValence') or {}
                missing = [s for s in centralDict['transportedSpecies'] if s not in valence]
                if missing:
                    raise ValueError(f"{instance.label} : NernstPlanck needs the valence of every transported "
                                     f"species : speciesValence gives none for {missing} (0 for a neutral species)")
            _checkTransportNames(centralDict, None if centralDict['MultiCompoundTransport'] else mobileSpecies)

    # renewal: initial conditions kept and fresh solutions checked on the final columns of commMtrx (totals with
    # MultiCompoundTransport), so that a wrong name stops the run here and not at the first renewal
    if renewalOn:
        centralDict.update(renewal.finalize(centralDict))

    # PhreeqcRM instance of the run (speciationThreads > 1: one instance per process, each with a block of cells; the
    # threads of the phreeqcrm package do not run at the same time), created once the columns of commMtrx are final:
    # options checked, database and kineticRates loaded, components found; an unsupported option or species stops
    # the run before the first step
    for instance in speciationInstances:
        if instance.solver == 'PhreeqcRM':
            from . import rmphreeqc
            with solverInstances.context(centralDict, instance):
                rmphreeqc.setup(centralDict)
    # settings given in the block of an instance and not read by its solver
    for label, solver, names in solverInstances.unread(instances):
        warningManager.warn(f"solverInstances : {label} : {', '.join(names)} not read by {solver} (setting of "
                            "another solver, or not used with these settings) : ignored")

    # output folders: removal of the output folders of a previous run (never a folder holding other files), creation
    # (empty) of the requested folders
    for message in outputManager.prepareFolders(pathpycteInput, centralDict['paths']):
        warningManager.warn(message)

    # warning.log: run header, then the warnings emitted during initialisation
    header = f"pycte : {centralDict['couplingInfo'][0]} {centralDict['couplingInfo'][1]}-{centralDict['couplingInfo'][2]}.\n"
    if declared:
        header += ("pycte : Solver instances : "
                   + ", ".join(f"{i.label} ({i.solver}, {i.role})" for i in instances) + "\n")
        for i in transportInstances:
            header += (f"pycte : {i.label} transports : "
                       + ", ".join(map(str, solverInstances.value(centralDict, i, 'transportedSpecies') or [])) + "\n")
    if osSequence is not None:
        header += f"pycte : OSdefined (fractions of dt) : {OSschemes.describe(osSequence, centralDict['couplingInfo'][1], centralDict['couplingInfo'][2])}\n"
    if setting("description", None): header += f"pycte : Description :\n{setting('description')}\n"
    # database of the run, read back by postProcess for the Tot!_ totals
    if centralDict.get('chemPath'): header += f"pycte : Database : {os.path.abspath(os.path.join(pathpycteInput, str(centralDict['chemPath'])))}\n"
    header += f"pycte : Time steps ({centralDict['timeUnit']}) :\n{', '.join(str(dtt) for dtt in dtpycte)}.\n"
    # every setting read by the run, given or default, and those given but not read (run again from these lines)
    header += settings.header()
    warningManager.start(header, centralDict)

    # saturation written by the chemistry: every speciation call (first equilibrium, every scheme) goes through
    # saturationSpeciation, which keeps the amounts of the transported species when the water content changes
    if centralDict.get('saturationColumn'):
        for label in speciationLauncher:
            speciationLauncher[label] = saturationSpeciation(speciationLauncher[label])

    def writeStepEnd(step=None):
        # stepEnd/StepEnd_n.txt and / or stepEnd_vtu/StepEnd_n.vtu, n = step number (0: initial state)
        if outputManager.wanted(centralDict, 'stepEnd', step):
            outputManager.write(centralDict, centralDict["commMtrx"], 'stepEnd', 'StepEnd', step)

    loopStart = time.perf_counter()
    # progress bar (consoleDisplay = 'progressBar'), redrawn at most every display.REFRESH s; in 'steps' mode the
    # step header and the remaining time are printed at every step
    display.start(len(dtpycte), dtpycte[-1], loopStart)
    # the same at every step (computed once): a 'time' column (crossDependencies, read during the set-up only), the
    # launchers of every instance given to the scheme
    timeCrossDep = 'time' in centralDict['crossDependencies']['names']
    allLaunchers = {**speciationLauncher, **transportLauncher}
    for l,t in enumerate(dtpycte):

        if l==0: dt = t
        else: dt = t-dtpycte[l-1]

        if timeCrossDep:
            centralDict['commMtrx']['time'] = dt

        centralDict.update({
            "dtStep":dt,
            "tStep":t,
            "lStep":l,
            })
        
        display.stepPrint(f"#######  step n°{l+1}/{len(dtpycte)}, dt = {dt}{centralDict['timeUnit']}  #######")
        
        if centralDict['preliminaryEquilibrium']:
            centralDict['beforeTrsptMtrx'] = centralDict["commMtrx"].copy() # not OS dependent ..
        
        if centralDict["firstStepEquilibrium"] and l==0:
            centralDict["dtStep"] = 0
            if timeCrossDep:
                centralDict['commMtrx']['time'] = 0
            centralDict.update(speciationLauncher[speciationKey](centralDict))
            if centralDict['preliminaryEquilibrium']:
                centralDict['beforeTrsptMtrx'] = centralDict["commMtrx"].copy()
            centralDict["firstStepEquilibrium"]=False
            centralDict["dtStep"]=dt
            if timeCrossDep:
                centralDict['commMtrx']['time'] = dt
        if l == 0:
            writeStepEnd(0)                        # initial state (after firstStepEquilibrium if requested)
            pd.set_option('display.max_columns', None)      # a global option: set once, kept for the whole run
        

        # one transport / speciation coupling step with the chosen scheme (OSschemes.py); the scheme sets dtStep and
        # tTransportStart (start of transport: time-dependent boundary conditions, ramp) before each call
        OSschemes.step(centralDict,
                       transportLauncher[transportKey],
                       speciationLauncher[speciationKey],
                       t, dt, allLaunchers)

        writeStepEnd()

        # renewal at the end of the step (after transport and chemistry): the next step starts from the renewed solution;
        # StepEnd_*.txt above keeps the state computed before renewal
        if renewalOn:
            centralDict.update(renewal.apply(centralDict))

        warningManager.flush(centralDict)           # new warnings of the step, in a single block

        # remaining time extrapolated from the coupling loop only (database extraction and solver set-up excluded)
        display.progress(l + 1, t)

    display.finish()
    # every output file written (a write error stops the run here, as it did when the files were written in the loop)
    outputManager.flush()
    warningManager.flush(centralDict)
    with open("warning.log", "a", encoding="utf-8") as f:
        f.write("pycte : Coupling completed\n")
        f.write(f"pycte : {centralDict['warningNbr']} warning(s) have occurred during the coupling\n\n")

    if centralDict['warningNbr']:
        print(f"Oh no, {centralDict['warningNbr']} warnings occurred :( ! See warning.log file.")

    # COMSOL licence waiting time (shared licences): left out of the total coupling time and of the COMSOL time, so
    # that the communication time (total - speciation - transport) does not depend on it
    waiting = sum(solverInstances.value(centralDict, i, 'COMSOLWaitingTime', 0) or 0
                  for i in transportInstances if i.solver == 'COMSOL')
    totalTime = time.perf_counter() - pyctestart - waiting

    def timeOf(instance, suffix):
        value = solverInstances.value(centralDict, instance, f"{instance.solver}{suffix}", 0) or 0
        if suffix == 'TotalTime' and instance.solver == 'COMSOL':
            value -= solverInstances.value(centralDict, instance, 'COMSOLWaitingTime', 0) or 0
        return value

    with open("warning.log", "a", encoding="utf-8") as warningLog:
        warningLog.write(f"pycte : Total coupling calculation time : {writeTime(totalTime)}\n")
        for i in instances:
            warningLog.write(f"{i.label} : Total {i.label} time : {writeTime(timeOf(i, 'TotalTime'))}\n")
        warningLog.write(f"pycte : Communication time between {' and '.join(i.label for i in instances)} : "
                         f"{writeTime(totalTime - sum(timeOf(i, 'TotalTime') for i in instances))}\n\n")
        for k, i in enumerate(instances):
            what = 'speciation' if i.role == 'speciation' else 'transport'
            warningLog.write(f"{i.label} : Wall clock time of total {what} calculation : {writeTime(timeOf(i, 'CalcTime_WallClock'))}\n")
            warningLog.write(f"{i.label} : Processor time of total {what} calculation : {writeTime(timeOf(i, 'CalcTime_ProcessorTime'))}\n")
            warningLog.write(f"{i.label} : Wall clock time of interfacing data : {writeTime(timeOf(i, 'InterfTime_WallClock'))}\n")
            if k == 0 and centralDict["extractDBTime"]:
                warningLog.write(f"{i.label} : Database extraction : {writeTime(centralDict['extractDBTime'])}\n")
            warningLog.write(f"{i.label} : Initialisation time : {writeTime(timeOf(i, 'InitTime'))}\n\n")
        for i in transportInstances:
            if i.solver == 'COMSOL':
                warningLog.write(f"{i.label} : Licence waiting time: {writeTime(solverInstances.value(centralDict, i, 'COMSOLWaitingTime', 0))} (this may only apply for shared licences between users)\n")




    if setting("description", None): print(f"Description = {setting('description')}")
    print(f"Total coupling time : {writeTime(totalTime, 2)}")


# ---- solver registries: canonical name (chem_map / trspt_map) -> set-up function --------------------------------

def _preloadPhreeqC(centralDict, setting):
    '''IPhreeqc instances of the set-up and of the first speciation loaded in threads from now on (the database read by
    PhreeqC while Python extracts it, phreeqc.preloadIPhreeqc): one for the kineticRates check, one for
    checkOutputNames, one for the first speciation or one per block of nodes (speciationThreads > 1). Settings read with
    peek (not recorded: warning.log lists them in the order the set-up reads them below).'''
    try:
        def peek(name):
            return setting.peek(name) if setting.given(name) else None
        knobs, rates = peek('knobs'), peek('kineticRates')
    except AttributeError:
        return                                      # setting without given / peek: nothing preloaded
    from . import phreeqc
    count = 1 + (1 if rates else 0)
    threads = centralDict.get('speciationThreads') or 1
    if threads > 1:
        nodes = len(centralDict['commMtrx']) if isinstance(centralDict.get('commMtrx'), pd.DataFrame) else 0
        if nodes:
            chunk = int(np.ceil(nodes / threads))   # blocks of phreeqc.spct
            count += len(range(0, nodes, chunk))
    elif phreeqc.phreeqc is None:
        count += 1
    phreeqc.preloadIPhreeqc(centralDict['chemPath'], knobs, rates, count)


def _speciationPhreeqC(centralDict, setting, run):
    '''PhreeqC set-up: its settings into centralDict; returns its speciation launcher'''

    centralDict['phasesAsPrimary'] = setting('phasesAsPrimary', True)
    centralDict['chemPath'] = Path(setting('chemPath', os.path.join(run.inputPath, 'phreeqc.dat')))
    from . import phreeqc
    from . import extractDB
    # database keywords (masterSpecies, solutionSpecies, phases, masterExchange, exchangeSpecies, masterSurface,
    # surfaceSpecies; fixpH adds the phase Fix_ph, fixedPhases its pseudo-phases): merged into a copy of the
    # database, <name>_pycte<ext> in the run folder, which becomes chemPath -> extractDB and every IPhreeqc
    # instance read the same definitions. Without these keywords chemPath is unchanged
    centralDict['chemPath'] = phreeqc.prepareDatabase(
        centralDict['chemPath'], run.inputPath,
        {key: setting(key, None) for key in phreeqc.DATABASE_KEYWORDS},
        setting("fixpH", None), setting("fixedPhases", None))
    _preloadPhreeqC(centralDict, setting)
    centralDict.update(extractDB.extract(centralDict))

    speciationLauncher = {'PhreeqC': phreeqc.spct}                

    centralDict.update({
    "current" : setting('current', None),
    "SI" : _perPhase(setting, "SI", centralDict['primarySpecies']['phases'], 0),
    "mineralReversibility" : _perPhase(setting, "mineralReversibility", centralDict['primarySpecies']['phases'], ""),
    "cutoffs" : {'solution' : setting('cutoffAq', 1e-20),
                'phases' : setting('cutoffPha', -1e-99),
                'exchange' : setting('cutoffExch', 1e-20),
                'surface' : setting('cutoffSurf', 1e-20)},

    "fixpH": setting("fixpH", None),
    # {name: (SI, reactant, amount)} : pseudo-phases held at every call in EQUILIBRIUM_PHASES 1, e.g.
    # {'Fix_ph': (-7.0, 'NaOH', 10), 'Fix_pe': (-12.5, 'O2', 10)} = Fix_H+ / Fix_pe of a PhreeqC TRANSPORT
    "fixedPhases": setting("fixedPhases", None),
    # content of a KNOBS block, e.g. '-convergence_tolerance 1e-12', read by every IPhreeqc instance
    "knobs": setting("knobs", None),
    "userVarBool": setting("userVarBool", {}),
    "userVarList": setting("userVarList", {}),
    "chargeGeometryCells" : run.chargeGeometryCells,
    "solMod" : setting('solMod', False),
    "step_divide" : setting('step_divide', None),
    "densityPhreeqC" : setting('densityPhreeqC', 1),
    "speciesCharge" : setting("speciesCharge", ['pH']),
    "speciesChargeGeometry" : setting('speciesChargeGeometry', None),
    "speciationCharge" : setting("speciationCharge", False),
    "pHdefault" : setting("pHdefault", 7),
    "tempDefault" : setting("tempDefault", 298.15),             # K
    "electrodeDistance" : setting("electrodeDistance", None),
    "catholyte" : setting("catholyte", None),
    "anolyte" : setting("anolyte", None),
    "solModCharge" : setting("solModCharge", 0),
    "water" : setting('water', False),
    "kinetics" : setting('kinetics', False),
    # content of a RATES block (with or without the RATES line), loaded with the database in every IPhreeqc instance
    "kineticRates" : setting('kineticRates', None),
    'supplementarySolution' : setting('supplementarySolution', None),
    "surfaceCounterIons" : setting('surfaceCounterIons', False),
    "beforeTrsptMtrx" : pd.DataFrame(),
    "kineticsSpecies" : setting('kineticsSpecies', None),
    "PhreeqCCalcTime_WallClock": 0,
    "PhreeqCCalcTime_ProcessorTime": 0,
    "PhreeqCInterfTime_WallClock": 0,
    "PhreeqCInitTime" : 0,
    "PhreeqCTotalTime" : 0,
    })

    # kineticsSpecies: each one must name a rate block of the kinetics text, which receives its -m0 (amount read
    # in commMtrx); checked here so that a wrong name stops the run before the first step
    if centralDict['kinetics'] and centralDict['kineticsSpecies']:
        phreeqc.kineticsRates(phreeqc.kineticsBlocks(centralDict['kinetics']), centralDict['kineticsSpecies'])
        # initial amounts of the nodes: -m0 (M0 of the rate laws) at every step, -m the current amount
        centralDict['kineticsM0'] = centralDict['commMtrx'][[k for k in centralDict['kineticsSpecies']
                                                             if k in centralDict['commMtrx']]].copy()
    # speciationCharge: each counter-charge species (speciesCharge, speciesChargeGeometry) must be pH or a species of
    # the solution (master species Cl, C(+4), ... or species Cl-, CO3-2, ...) ; another name used to be dropped
    if centralDict['speciationCharge']:
        charged = centralDict['speciesCharge'] or []
        candidates = [charged] if isinstance(charged, str) else list(charged)
        for _, spc in centralDict['speciesChargeGeometry'] or []:
            candidates += [spc] if isinstance(spc, str) else list(spc or [])
        unknown = [s for s in candidates if phreeqc.chargeTarget(centralDict, s) is None]
        if unknown:
            raise ValueError(f"speciesCharge / speciesChargeGeometry : {unknown} are neither pH nor a species of the "
                             f"solution in {centralDict['chemPath']} (a master species such as Cl or C(+4), or a "
                             "coupled species such as Cl- or CO3-2) : no charge balance could be put on them")
    # kineticRates: the RATES block is read once by a throw-away IPhreeqc instance, so that an input error of the
    # block (unknown option, ...) stops the run before the first step; errors in the BASIC lines only show up when
    # a rate is run (PhreeqC then aborts the speciation and its message is written in warning.log)
    if centralDict['kineticRates']:
        checker = phreeqc.takeIPhreeqc(centralDict)     # loaded during the extraction (_preloadPhreeqC)
        if checker is not None:
            checker.destroy_iphreeqc()
        else:
            checker = phreeqc.newIPhreeqc()
            try:
                phreeqc.loadDatabase(checker, centralDict)
            finally:
                checker.destroy_iphreeqc()

    # molesStorage (default True, SOLUTION_MODIFY path: solMod or MultiCompoundTransport): commMtrx holds
    # moles per cell (relative to 1 kg of initial water) instead of molalities; the water mass of each cell
    # (cellWaterMass, kg) follows the O balance instead of being reset to 1 kg at every step. nativeTransport
    # diffuses the concentrations (moles / cellWaterMass) with the water content porosity.cellWaterMass.
    centralDict['molesStorage'] = (bool(setting('molesStorage', True))
                                   and (centralDict['solMod'] or centralDict['MultiCompoundTransport']))
    if centralDict['molesStorage']:
        # initial water mass: H2O column of the IC (55.50868 mol = 1 kg), otherwise 1 kg
        centralDict['cellWaterMass'] = (centralDict['commMtrx']['H2O'].to_numpy(dtype=float) / 55.50868
                                        if 'H2O' in centralDict['commMtrx'] else np.ones(len(centralDict['commMtrx'])))
    # token of the run: the PhreeqC warnings are written once per kind and run (phreeqc.newWarnings), also by the worker
    # processes that outlive a run
    centralDict['phreeqcRun'] = time.time_ns()
    # names read back from PhreeqC checked by PhreeqC (it sets an unknown name to 0): a misspelled species stops the
    # run here, before the transport set-up; with MultiCompoundTransport once the columns are totals (_multiCompoundPhreeqC)
    if not centralDict['MultiCompoundTransport']:
        phreeqc.checkOutputNames(centralDict)
    return speciationLauncher['PhreeqC']


def _speciationPhreeqcRM(centralDict, setting, run):
    '''PhreeqcRM set-up: its settings into centralDict; returns its speciation launcher'''
    # PhreeqcRM (phreeqcrm package), see rmphreeqc.py: same keywords as PhreeqC. Species mode (SOLUTION path of
    # PhreeqC): the aqueous species of commMtrx are the solution of each reaction cell; totals mode (solMod or
    # MultiCompoundTransport, SOLUTION_MODIFY path): total H, O, charge and elements; exchanger, equilibrium
    # phases (fixpH) and kinetic reactants are rebuilt from commMtrx at every call, as with PhreeqC. Same database
    # handling as PhreeqC: database keywords merged into <name>_pycte<ext> in the run folder, same extraction
    centralDict['phasesAsPrimary'] = setting('phasesAsPrimary', True)
    centralDict['chemPath'] = Path(setting('chemPath', os.path.join(run.inputPath, 'phreeqc.dat')))
    from . import phreeqc
    from . import rmphreeqc
    centralDict['chemPath'] = phreeqc.prepareDatabase(
        centralDict['chemPath'], run.inputPath,
        {key: setting(key, None) for key in phreeqc.DATABASE_KEYWORDS},
        setting("fixpH", None), setting("fixedPhases", None))
    rmphreeqc.extract(centralDict)

    speciationLauncher = {'PhreeqcRM': rmphreeqc.spct}

    centralDict.update({
    "SI" : _perPhase(setting, "SI", centralDict['primarySpecies']['phases'], 0),
    "mineralReversibility" : _perPhase(setting, "mineralReversibility", centralDict['primarySpecies']['phases'], ""),
    "cutoffs" : {'solution' : setting('cutoffAq', 1e-20),
                'phases' : setting('cutoffPha', -1e-99),
                'exchange' : setting('cutoffExch', 1e-20),
                'surface' : setting('cutoffSurf', 1e-20)},
    "fixpH": setting("fixpH", None),
    "fixedPhases": setting("fixedPhases", None),     # same keywords as PhreeqC
    "knobs": setting("knobs", None),
    "userVarBool": setting("userVarBool", {}),
    "userVarList": setting("userVarList", {}),
    "solMod" : setting('solMod', False),
    "step_divide" : setting('step_divide', None),
    "speciationCharge" : setting("speciationCharge", False),
    "tempDefault" : setting("tempDefault", 298.15),             # K
    "pHdefault" : setting("pHdefault", 7),
    "water" : setting('water', False),
    "kinetics" : setting('kinetics', False),
    "kineticRates" : setting('kineticRates', None),
    "kineticsSpecies" : setting('kineticsSpecies', None),
    "PhreeqcRMCalcTime_WallClock": 0,
    "PhreeqcRMCalcTime_ProcessorTime": 0,
    "PhreeqcRMInterfTime_WallClock": 0,
    "PhreeqcRMInitTime" : 0,
    "PhreeqcRMTotalTime" : 0,
    })

    # kineticsSpecies: each one must name a rate block of the kinetics text (checked before the first step);
    # initial amounts of the nodes: -m0 at every step, as with PhreeqC
    if centralDict['kinetics'] and centralDict['kineticsSpecies']:
        phreeqc.kineticsRates(phreeqc.kineticsBlocks(centralDict['kinetics']), centralDict['kineticsSpecies'])
        centralDict['kineticsM0'] = centralDict['commMtrx'][[k for k in centralDict['kineticsSpecies']
                                                             if k in centralDict['commMtrx']]].copy()
    # molesStorage: as for PhreeqC (totals mode only), commMtrx in moles per cell, cellWaterMass follows the O
    # balance (mass_H2O of PhreeqcRM)
    centralDict['molesStorage'] = (bool(setting('molesStorage', True))
                                   and (centralDict['solMod'] or centralDict['MultiCompoundTransport']))
    if centralDict['molesStorage']:
        centralDict['cellWaterMass'] = (centralDict['commMtrx']['H2O'].to_numpy(dtype=float) / 55.50868
                                        if 'H2O' in centralDict['commMtrx'] else np.ones(len(centralDict['commMtrx'])))
    # the PhreeqcRM instance is created after the MultiCompoundTransport conversion (rmphreeqc.setup below)
    return speciationLauncher['PhreeqcRM']


def _speciationXGEMS(centralDict, setting, run):
    '''xGEMS set-up: its settings into centralDict; returns its speciation launcher'''


    # temperature and pressure of the nodes: coupled when crossDependencies['speciation'] names temperature(K)
    # (K) and pressure (columns of the initial conditions, Pa), otherwise these constant values
    centralDict.update({
    "tempDefault" : setting("tempDefault", 298.15),              # K
    "pressureDefault" : setting("pressureDefault", 101325.0),    # Pa (1 atm)
    "chemPath" : Path(setting('chemPath', os.path.join(run.inputPath, 'dat.lst'))).resolve(),   # relative: run folder
    "xGEMSClockTime" : 0,
    "xGEMSPrcsTime" : 0,
    "xGEMSCalcTime_WallClock": 0,
    "xGEMSCalcTime_ProcessorTime": 0,
    "xGEMSInterfTime_WallClock": 0,
    "xGEMSInitTime" : 0,
    "xGEMSTotalTime" : 0,
    "xGEMSModule" : None,})                                       # written to warning.log by gems.spct

    from . import extractDB

    centralDict.update(extractDB.extract(centralDict))

    centralDict.update({
    "transportedIC" : [s for s in centralDict['independentComponents'] if s not in ['Zz'] + centralDict['crossDependencies']['speciation']['total']]
    })

    from . import gems

    speciationLauncher = {'xGEMS': gems.spct}
    # GEMS3K writes ipmlog.txt and xGEMS.log in its current folder: it works on a copy of the GEMS files in
    # <run folder>/gemsWork, so that the folder of the database (an installed example: the pycte package) is not written
    centralDict['gemsEnginePath'] = gems.workingCopy(centralDict['chemPath'], run.inputPath)

    os.chdir(run.inputPath)
    return speciationLauncher['xGEMS']


def _speciationORCHESTRA(centralDict, setting, run):
    '''ORCHESTRA set-up: its settings into centralDict; returns its speciation launcher'''
    from . import orchestra
    speciationLauncher = {
        'ORCHESTRA': orchestra.spct}

    centralDict.update({
    "firstIC" : setting("firstIC", None),
    "chemPath" : Path(setting('chemPath', os.path.join(run.inputPath, 'chemistry1.inp'))),
    "primarySpecies" : [setting("primarySpeciesAq", []),
                       setting("primarySpeciesPha", []),
                       setting("primarySpeciesSurf", []),
                       setting("primarySpeciesExch", []),
                       setting("primarySpeciesPhantom", ['OH','H']),
                       (setting("primarySpeciesAq", []) + setting("primarySpeciesPha", []) +
                        setting("primarySpeciesSurf", [])+setting("primarySpeciesExch", []))
                       ],
    "phases" : setting("phases", []),
    "waterMass" : setting("waterMass", [1]*len(centralDict['commMtrx'])),
    "primarySpeciesExchSorbed" :  setting("primarySpeciesExchSorbed", {}),
    "couplingFormalism" : setting("couplingFormalism", 'tot'),
    "orchestraCutOff" : setting("orchestraCutOff", None),
    "OrchestraMemory" : setting("OrchestraMemory", True),
    "ORCHESTRACalcTime_WallClock": 0,
    "ORCHESTRACalcTime_ProcessorTime": 0,
    "ORCHESTRAInterfTime_WallClock": 0,
    "ORCHESTRAInitTime" : 0,
    "ORCHESTRATotalTime" : 0,
    "inputVariableOrchestra": setting("inputVariableOrchestra", None),
    'stoichUser' : setting("stoich", None),
    "inputVariableOrchestraSpecies": [
        s
        for s in (setting("inputVariableOrchestra", None) or [])
        if s not in centralDict["crossDependencies"]["speciation"]["total"]
    ],        
    'outputVariableOrchestra': setting("outputVariableOrchestra", None)
    })

    from . import extractDB
    centralDict.update(extractDB.extract(centralDict))


    from .parse_reactions import build_stoich_matrix

    # one row per species, in the order of systemSpecies (= columns of commMtrx given to ORCHESTRA, crossDependencies
    # of the speciation and of the transport left out)
    stoich = build_stoich_matrix(centralDict['reactions'], centralDict['primary_entities'], centralDict['systemSpecies'])

    # literbulk: columns multiplied by the water mass, by default every species but the phases
    cols = [s for s in centralDict['systemSpecies'] if s not in centralDict['phases']]
    notSpecies = [s for s in setting("colAq", cols) if s not in centralDict['systemSpecies']]
    if notSpecies:
        raise ValueError(f"colAq : {notSpecies} are not species (coupledVariables without the crossDependencies)")

    if centralDict['stoichUser']:
        for k in centralDict['stoichUser']:
            stoich.loc[k] = centralDict['stoichUser'][k]
    # a species ORCHESTRA does not know has a row of zeros: its mass would leave the system at the first step.
    # Not an error when the row is given in stoich (deliberate, e.g. a kinetic phase kept out of the totals) or when
    # the species is zero everywhere in the initial conditions (nothing to lose ; build_stoich_matrix already warned)
    user = set(centralDict['stoichUser'] or {})
    ic = centralDict['commMtrx']
    unknown = [s for s in centralDict['systemSpecies'] if s in stoich.index and s not in user
               and not stoich.loc[s].to_numpy().any()
               and not (s in ic.columns and not ic[s].to_numpy(dtype=float).any())]
    if unknown:
        raise ValueError(f"ORCHESTRA : {unknown} are not species of {centralDict['chemPath']} (no composition in its "
                         "primary entities) : their mass would be lost at the first step. Give the names of the .inp, "
                         "add these species to it (@species, @logKreaction, ...) or give their composition with "
                         "stoichUser = {species: {primary entity: coefficient}}")




    centralDict.update({
    # 'outputarray' : np.zeros((nrCells, len(centralDict['outputVariableOrchestra'])), dtype=np.float64) , 

    'stoich': stoich,
    "colAq": setting("colAq", cols),
    })
    return speciationLauncher['ORCHESTRA']


def _speciationNativeKinetics(centralDict, setting, run):
    '''nativeKinetics set-up: its settings into centralDict; returns its speciation launcher'''
    from . import nativeKinetics 
    speciationLauncher = {
        'nativeKinetics': nativeKinetics.spct}
    # kineticReactions: rate constants k in SI (per second, whatever timeUnit), see nativeKinetics.kineticsSolve
    centralDict.update({"kineticReactions" : setting("kineticReactions", None),
                        # solve_ivp method ('LSODA', 'BDF', 'Radau', 'RK45', ...) and tolerances
                        "kineticMethod" : setting("kineticMethod", 'LSODA'),
                        "kineticRtol" : setting("kineticRtol", 1e-9),
                        "kineticAtol" : setting("kineticAtol", 1e-14),
                        # cells integrated together per solve_ivp call (None: all the cells in one system)
                        "kineticChunkSize" : setting("kineticChunkSize", None),
                        # mass balance check (sum of the species of each cell kept within kineticTolMB)
                        "kineticCheckMB" : setting("kineticCheckMB", False),
                        "kineticTolMB" : setting("kineticTolMB", 1e-9),
                        "nativeKineticsInterfTime_WallClock": 0,
                        "nativeKineticsCalcTime_WallClock": 0,
                        "nativeKineticsCalcTime_ProcessorTime": 0,
                        "nativeKineticsTotalTime": 0,
                        "nativeKineticsInitTime": 0,
                        })
    return speciationLauncher['nativeKinetics']


def _speciationNativeSpeciation(centralDict, setting, run):
    '''nativeSpeciation set-up: its settings into centralDict; returns its speciation launcher'''
    from . import call
    speciationLauncher = {
        'nativeSpeciation': call.spct}
    centralDict.update({
        'MultiCompoundTransport' : setting("MultiCompoundTransport", False),
        "chemistry" :  setting("chemistry", None),
        "chemPath" : setting("chemPath", None),
        "nativeSpeciationCalcTime_WallClock": 0,
        "nativeSpeciationCalcTime_ProcessorTime": 0,
        "nativeSpeciationInterfTime_WallClock": 0,
        "nativeSpeciationInitTime" : 0,
        "nativeSpeciationTotalTime" : 0,})
    # molesStorage (default True, initial conditions with an H2O column): as for PhreeqC, commMtrx holds moles per
    # cell and the H2O column the moles of free water (species mode) or of the H2O component in solution, free +
    # bound (MultiCompoundTransport, see call.multiCompoundSetup); the water mass (cellWaterMass, kg) follows the
    # water balance of nativeSpeciation (bound water of species and minerals included) instead of staying at 1 kg
    centralDict['molesStorage'] = (bool(setting('molesStorage', True))
                                   and 'H2O' in centralDict['commMtrx'])
    if centralDict['molesStorage']:
        centralDict['cellWaterMass'] = centralDict['commMtrx']['H2O'].to_numpy(dtype=float) / 55.50868
    # species mode: a coupled variable without a role in the chemical system stops the run
    if centralDict['chemistry'] is not None:
        call.checkColumns(centralDict)
    return speciationLauncher['nativeSpeciation']


def _speciationNone(centralDict, setting, run):
    '''No speciation: returns a launcher that changes nothing.'''
    speciationLauncher = {'none': lambda cd: {}}
    centralDict.update({f"none{k}": 0 for k in ("CalcTime_WallClock", "CalcTime_ProcessorTime",
                                                "InterfTime_WallClock", "InitTime", "TotalTime")})
    return speciationLauncher['none']


def _transportCOMSOL(centralDict, setting, run):
    '''COMSOL set-up: its settings into centralDict; returns its transport launcher'''
    from . import comsol

    transportLauncher = {
        "COMSOL": comsol.trspt, 
    }

    centralDict.update({
    "trsptPath" : Path(setting('trsptPath', os.path.join(run.inputPath, 'comsol.mph'))),
    "comsolTags" : [setting('comsolComp', 'comp1'),
                   setting('comsolIntFunction', 'int1'),
                   setting("comsolStudy", 'std1'), 
                ],
    "outputComsol": [],
    # iterative coupling: source / sink terms written in int1, integrals read in data1 (see comsol.py)
    "comsolReactionTerms" : setting('comsolReactionTerms', False),
    "transportedSpecies" : setting('transportedSpecies', run.defaultTransported),
    "COMSOLCalcTime_WallClock": 0,
    "transportThreads" : setting('transportThreads', None),     # None: COMSOL takes the cores it needs
    "COMSOLCalcTime_ProcessorTime": 0, 
    "COMSOLInterfTime_WallClock": 0,
    "COMSOLInitTime" : 0,
    "COMSOLTotalTime" : 0,
    "COMSOLWaitingTime" : 0
    })

    # user COMSOL outputs: transport/<tag>/ and / or transport_vtu/<tag>_vtu/ (outputFormat), if output['transport']
    centralDict['outputComsol'] = list(run.comsolTags)
    return transportLauncher['COMSOL']


def _transportNativeTransport(centralDict, setting, run):
    '''nativeTransport set-up: its settings into centralDict; returns its transport launcher'''
    from . import nativeTransport

    transportLauncher = {
        "nativeTransport": nativeTransport.trspt,
    }
    bcSchedule = nativeTransport.boundarySchedule(setting('boundaryConditions', ['flux', 'flux']))

    centralDict.update({
        "velocity" : setting('velocity', 0),
        "advection" : setting('advection', None),
        "transportedSpecies" : setting('transportedSpecies', run.defaultTransported),
        "FickDiffusion" : setting('FickDiffusion', None),
        "porousTransport" : setting('porousTransport', None),
        "ADE" : setting('ADE', None),
        'adeSolver' : setting('adeSolver', 'explicit'),        # 'explicit' | 'implicit'
        'adeSubSteps' : setting('adeSubSteps', 1),            # implicit steps per transport step
        'adeTimeScheme' : setting('adeTimeScheme', 'euler'),  # implicit ADE : 'euler' | 'crank-nicolson'
        'dispersionInTransport' : True,     # dispersion (theta.D_meca = alphaL.|q|) is computed by nativeTransport
        "dispersivity" : setting('dispersivity', 0),
        "firstBoundary" : setting('firstBoundary', {cle: 0 for cle in centralDict['coupledVariables']}),
        "secondBoundary" : setting('secondBoundary', {cle: 0 for cle in centralDict['coupledVariables']}),
        # [x = 0, x = L]: 'constant': Dirichlet | 'closed': Neumann (zero flux) | 'flux': Cauchy
        # electrode boundary: 'electrode+constant' | 'electrode+closed' | 'electrode+flux' ('electrode' = 'electrode+closed')
        # time dependent: {condition: start time} or {start time: condition}, condition at time 0 required
        # e.g. [{'closed': 0, 'constant': 100}, {'flux': 0, 'closed': 50}]; boundaryConditions = conditions at time 0
        "boundaryConditions" : [entries[0][1] for entries in bcSchedule],
        "boundarySchedule" : bcSchedule if any(len(entries) > 1 for entries in bcSchedule) else None,
        # Cauchy: J_in = Q_in.c_ext + h.A.(c_ext - c_boundary), h [m/s]: number or [h_0, h_L]
        "boundaryTransferCoeff" : setting('boundaryTransferCoeff', 0.0),
        # diffusion on the activity gradient (ADE and Nernst-Planck, explicit and implicit);
        # activityDatabase: PHREEQC database ('-gamma': extended Debye-Huckel), otherwise Davies / Setchenow
        "activityGradient" : setting('activityGradient', False),
        "activityDatabase" : setting('activityDatabase', None),
        # ADE: 'upwind' (first order) | 'LUD' (linear upwind differencing, second order) | 'vanLeer' (TVD, second order); LUD and vanLeer: explicit ADE only
        "advectionScheme" : setting('advectionScheme', 'upwind'),
        "diffusionScheme" : 'fick',
        "porosity" : (
                np.array([setting("porosity",1)] * len(centralDict["commMtrx"]))
                if isinstance(setting("porosity", 1), (int, float))
                else np.array(setting("porosity", [1] * len(centralDict["commMtrx"])))
            ),
        "area" : np.array(setting('area', [1]*len(centralDict["commMtrx"]))),
        "nativeTransportCalcTime_WallClock": 0,
        "nativeTransportCalcTime_ProcessorTime": 0,
        "nativeTransportInterfTime_WallClock": 0,
        "nativeTransportTotalTime" : 0,
        'nativeTransportInitTime' : 0,
        'coordinateSystem' : setting('coordinateSystem', 'cartesian'), # coordinateSystem = cylindrical, spherical
        'cylinderHeight' : setting('cylinderHeight', 1),

        'imposedCurrent' : setting('imposedCurrent', None),
        'imposedCurrentDensity' : setting('imposedCurrentDensity', None),
        'imposedVoltage' : setting('imposedVoltage', None),
        'rampPotential' : setting('rampPotential', None),
        'temperature' : setting('temperature', 298.15),

        'electroOsmoticPermeability' : setting('electroOsmoticPermeability', 0), # k_eo [m2/(V.s)], scalar or list
        'permeability' : setting('permeability', 1e-19), # k [m2], scalar or list
        'viscosity' : setting('viscosity', 8.9e-4 ),  # mu [Pa.s],
        'hydraulicBoundary' : setting('hydraulicBoundary', 'open'),  # 'open' or 'closed'
        'boundaryPressure' : setting('boundaryPressure', [0.0, 0.0] ), # [p_0, p_L] in Pa, in 'open' mode


        'NernstPlanck' : setting('NernstPlanck', False),
        'speciesValence' : setting('speciesValence', {}),
        'speciesDiffCoeff' : setting('speciesDiffCoeff', {}),
        'speciesPorosity' : setting('speciesPorosity', {}),
        # ADE with cross-diffusion (Soret, Dufour, ...): {target: {driver: K}}, see nativeTransport.adeCrossDiffusion
        'crossDiffusion' : setting('crossDiffusion', None),
        'cflSafety' : setting('cflSafety', 0.9),
        'maxSubCycling' : setting('maxSubCycling', 100000),
        'electrodeReactions' : setting('electrodeReactions', None),
        'electrodeVolume' : setting('electrodeVolume', None),
        # electrodeNodes = [i0, iL]: interior electrode nodes (default the boundary nodes) ; zero current outside
        'electrodeNodes' : setting('electrodeNodes', None),
        'electrodeVoltageDrop' : setting('electrodeVoltageDrop', 0.0),
        # voltageDrop = True: electrode drop recomputed at the start of every transport step,
        # dV = |-1.23 - ln(10).RT/F.(pH_cathode - pH_anode)|, pH = -log10[H+],
        # or -log10(gamma.[H+]) if activityGradient (Davies / activityDatabase); replaces electrodeVoltageDrop
        'voltageDrop' : setting('voltageDrop', False),
        # electrodeResistance = R or [R_0, R_L] (ohm): resistance in series with the electrodes, outside the mesh ;
        # imposedVoltage : V = dPsi + electrode drop - R.I (lowers the current) ; imposed current : the transport does
        # not depend on it, cellVoltage_V = dPsi + electrode drop - R.I is written (voltageDrop gives the drop, sign of I)
        'electrodeResistance' : setting('electrodeResistance', None),
        # targetVoltage = V (V, imposed current only): electrode resistance recomputed at each step so that
        # cellVoltage_V = V, written in electrodeResistance_Ohm (0 if the cell voltage is already above V)
        'targetVoltage' : setting('targetVoltage', None),
        'waterEquilibrium' : setting('waterEquilibrium', None),
        'Kw' : setting('Kw', 1.0e-14),
        'electrodeKinetics' : setting('electrodeKinetics', 'faraday'),
        'electrodeArea' : setting('electrodeArea', None),
        'npSolver' : setting('npSolver', 'explicit'),
        'npImplicitMaxChange' : setting('npImplicitMaxChange', 0.1),
        'npImplicitAtol' : setting('npImplicitAtol', 1e-6),
        'npImplicitMaxIter' : setting('npImplicitMaxIter', 30),
        # coupling matrix of the transport solver, step after step, in every transport mode (nahed.py, Nahed &
        # de Lamare 2025) : True (written in couplingMatrix.txt) or off ; the solver itself is unchanged
        'nahed' : setting('nahed', None),
        # eps of the denominators J_ii.du_i + eps (eq. 9, 38-39), absolute, in mol/s like the rows of J : to be chosen
        # well below the perturbations of the rows (1e-9, value of the article, dominates them for small cells or
        # long steps) ; 0 : no eps (a zero denominator counts as 0)
        'nahedEpsilon' : setting('nahedEpsilon', 1e-9),
        # Donnan dual porosity (explicit and implicit ADE, implicit Nernst-Planck), see nativeTransport.py:
        # {'porosityDL' | 'thickness', 'CEC' + 'bulkDensity' | 'chargeDL', 'stern', 'tortuosityDL', 'activityDL', ...}
        'donnan' : setting('donnan', None),
        })

    nNodes = len(centralDict["commMtrx"])
    for key in ('speciesDiffCoeff', 'speciesPorosity'):
        perSpecies = centralDict[key] or {}
        if not isinstance(perSpecies, dict):
            raise ValueError(f"{key} must be a dict " + "{species: value}, got a "
                             + type(perSpecies).__name__ + " -> a missing ':' inside the braces makes it a set")
        resolved = {}
        for spc, val in perSpecies.items():
            val = np.asarray(val, dtype=float)
            if val.ndim == 0:
                val = np.full(nNodes, float(val))
            elif len(val) != nNodes:
                raise ValueError(f"{key}['{spc}'] has {len(val)} values, expected {nNodes} (one per node)")
            resolved[spc] = val
        centralDict[key] = resolved

    tempDiffCoeff = setting('diffCoeff', 0)
    # one value per node: a list of another length failed later (porosity at the first transport step, after the first
    # speciation; diffCoeff with a bare numpy broadcast error)
    for key, value in (('porosity', centralDict['porosity']), ('area', centralDict['area']),
                       ('diffCoeff', tempDiffCoeff)):
        if np.ndim(value) > 0 and np.size(value) != nNodes:
            raise ValueError(f"{key} has {np.size(value)} values, expected {nNodes} (one per node of the initial "
                             "conditions) or a single number")
    x = centralDict["commMtrx"][centralDict['coord']].copy().to_numpy().reshape(-1)
    if len(x) < 2:
        raise ValueError(f"nativeTransport : the initial conditions have {len(x)} node(s) : a 1D column of at least 2 "
                         "nodes is needed (one line per node, x first)")



    f_left = np.zeros(len(centralDict["commMtrx"]))
    f_right = np.zeros(len(centralDict["commMtrx"]))

    f_left[1:]   = x[1:] - x[:-1]
    f_right[:-1] = x[1:] - x[:-1]

    centralDict.update({'f_right' : f_right, 'f_left' : f_left})
    dx = np.diff(x)

    dx_flux = x[1:] - x[:-1]
    nodeSize = np.empty(len(x))
    nodeSize[1:-1] = (x[2:] - x[:-2]) / 2     
    nodeSize[0]    = (x[1]  - x[0])
    nodeSize[-1]   = (x[-1] - x[-2])
    centralDict.update({"x": x,
                        "dx" : dx,
                        "dx_flux" : dx_flux,
                        'nodeSize': nodeSize,
                        'centralDictCoord' : centralDict["commMtrx"][centralDict['coord']].copy()})


    dt = timeUnits.toSeconds(run.timeStep, centralDict['timeUnit'])





    n_adv = 1
    npMaxSubStep = np.inf

    if isinstance(tempDiffCoeff, (int,float)):
        centralDict['diffCoeff'] = np.full(len(centralDict["commMtrx"]),tempDiffCoeff)
    else:
        centralDict['diffCoeff'] = tempDiffCoeff

    # Liquid saturation S_L (0 < S_L <= 1, scalar or one value per node, default 1 = saturated).
    # nativeTransport replaces the porosity by the water content porosity * S_L (storage and diffusion,
    # speciesPorosity included), so that the pore velocity is q / (porosity * S_L). Constant in time, unless
    # saturation names a column of the initial conditions: a crossDependency of the speciation, written by the
    # chemistry (e.g. ORCHESTRA trapped gas). Then the chemistry output is converted at constant amounts to the new
    # water content (saturationSpeciation) and nativeTransport expels the water lost between two transport calls
    # (Darcy flow from the hydraulic resistances, see saturationColumn in nativeTransport.py).
    saturationSetting = setting('saturation', 1)
    centralDict['saturationColumn'] = None
    if isinstance(saturationSetting, str):
        if saturationSetting not in centralDict['commMtrx'].columns:
            raise ValueError(f"saturation = '{saturationSetting}' : no such column in the initial conditions")
        if saturationSetting not in centralDict['crossDependencies']['speciation']['total']:
            raise ValueError(f"saturation = '{saturationSetting}' : the column must be a crossDependency of the speciation "
                             "(crossDependencies = {'speciation': [..., '" + saturationSetting + "']})")
        if saturationSetting in centralDict['transportedSpecies']:
            raise ValueError(f"saturation = '{saturationSetting}' : the column cannot be a transported species")
        centralDict['saturationColumn'] = saturationSetting
        saturationSetting = centralDict['commMtrx'][saturationSetting].to_numpy(dtype=float)
    saturation = np.asarray(saturationSetting, dtype=float)
    if saturation.ndim == 0:
        saturation = np.full(len(centralDict["commMtrx"]), float(saturation))
    elif len(saturation) != len(centralDict["commMtrx"]):
        raise ValueError(f"saturation has {len(saturation)} values, expected {len(centralDict['commMtrx'])} (one per node)")
    if not np.all(np.isfinite(saturation)) or np.any(saturation <= 0) or np.any(saturation > 1):
        raise ValueError("saturation must be in ]0, 1] (liquid saturation S_L, water content = porosity * S_L)")
    centralDict['saturation'] = saturation
    centralDict['saturationPrevious'] = saturation.copy() if centralDict['saturationColumn'] else None

    # Tortuosity, PFLOTRAN convention (0 < tau <= 1): diffCoeff and speciesDiffCoeff are given in water (Dw).
    # Pore coefficient Dp = tau * Dw; nativeTransport then applies the water content: De = porosity * S_L * tau * Dw.
    # Applied before dispersion (v . alpha is not reduced by tortuosity). Default 1: unchanged behaviour.
    # tortuosity = 'millington-quirk': tau = porosity^(1/3) * S_L^(7/3), i.e. De = porosity^(4/3) * S_L^(10/3) * Dw
    # (computed once, from the initial porosity and saturation).
    tortuosity = setting('tortuosity', 1)
    centralDict['tortuosityMode'] = None
    centralDict['diffCoeffWater'] = np.asarray(centralDict['diffCoeff'], dtype=float).copy()
    centralDict['speciesDiffCoeffWater'] = {spc: val.copy() for spc, val in centralDict['speciesDiffCoeff'].items()}
    if isinstance(tortuosity, str):
        centralDict['tortuosityMode'] = 'millington-quirk'
        if tortuosity.replace('_', '-').replace(' ', '-').lower() not in ('millington-quirk', 'millingtonquirk'):
            raise ValueError(f"unknown tortuosity '{tortuosity}' : give a number in ]0, 1], one value per node, "
                             "or 'millington-quirk'")
        porosity0 = np.asarray(centralDict['porosity'], dtype=float)
        if np.any(porosity0 <= 0) or np.any(porosity0 > 1):
            raise ValueError("tortuosity = 'millington-quirk' needs a porosity in ]0, 1]")
        tortuosity = porosity0 ** (1.0 / 3.0) * saturation ** (7.0 / 3.0)
    tortuosity = np.asarray(tortuosity, dtype=float)
    if tortuosity.ndim == 0:
        tortuosity = np.full(len(centralDict["commMtrx"]), float(tortuosity))
    elif len(tortuosity) != len(centralDict["commMtrx"]):
        raise ValueError(f"tortuosity has {len(tortuosity)} values, expected {len(centralDict['commMtrx'])} (one per node)")
    if np.any(tortuosity <= 0) or np.any(tortuosity > 1):
        raise ValueError("tortuosity must be in ]0, 1] (PFLOTRAN convention : De = porosity * tortuosity * Dw)")
    centralDict['tortuosity'] = tortuosity
    centralDict['diffCoeff'] = tortuosity * np.asarray(centralDict['diffCoeff'], dtype=float)
    centralDict['speciesDiffCoeff'] = {spc: tortuosity * val for spc, val in centralDict['speciesDiffCoeff'].items()}

    # dispersion: no longer added to diffCoeff; nativeTransport applies it (theta.D_meca = alphaL.|q|,
    # local q in radial geometry). diffCoeff remains the pore coefficient Dp = tortuosity * Dw.


    meshing = setting('meshing', 0)
    if not meshing : # uniform meshing
        x = np.asarray(x)
        # without meshing, dxHalfCell = (x[1] - x[0]) / 2 everywhere: the x must be equidistant, otherwise the
        # distances between nodes (dxHalfCell) no longer match the cell volumes (nodeSize, read from x)
        if np.any(dx <= 0) or not np.allclose(dx, dx[0], rtol=1e-3, atol=0):
            k = int(np.argmax(np.abs(dx - dx[0])))
            raise ValueError(f"without meshing, x must be strictly increasing and uniform (x[1] - x[0] = {dx[0]:.6g}, "
                             f"x[{k+1}] - x[{k}] = {dx[k]:.6g}) : give meshing = [(nb1, width1), (nb2, width2), ...] "
                             "for a non-uniform mesh")
        width = x[1] - x[0]
        zones = [(len(x), width)]
        dxHalfCell = np.concatenate([np.full(nb, width / 2) for nb, width in zones])
    else:
        # meshing = [(nb1, width1), (nb2, width2), ...]: nodes at the centre of their cell, face k+1/2 is at
        # dxHalfCell_k from node k -> the x of the file must satisfy x[k+1] - x[k] = (width_k + width_k+1) / 2
        if any(int(nb) != nb or nb <= 0 or not width > 0 for nb, width in meshing):
            raise ValueError(f"meshing must be a list of (number of cells > 0, cell width > 0) (got {meshing})")
        dxHalfCell = np.concatenate([np.full(int(nb), width/2) for nb, width in meshing])
        if len(dxHalfCell) != len(x):
            raise ValueError(f"meshing has {len(dxHalfCell)} cells, expected {len(x)} (one per node)")
        gap = dxHalfCell[:-1] + dxHalfCell[1:]
        if not np.allclose(dx, gap, rtol=1e-3, atol=0):
            k = int(np.argmax(np.abs(dx - gap) / gap))
            raise ValueError(f"x does not match meshing : x[{k+1}] - x[{k}] = {dx[k]:.6g}, expected {gap[k]:.6g} "
                             "= (width_k + width_k+1) / 2 (nodes at the centre of their cell)")
        centralDict['nodeSize'] = 2 * dxHalfCell

    centralDict.update({'dxHalfCell': dxHalfCell})

    # gasVolume = column of the gas volume of each node (m3, initial conditions, then written by the chemistry
    # when it is a crossDependency of the speciation): the saturation column is computed from it with the node
    # volumes of the mesh (nodeSize x area, or the shell volumes in cylindrical / spherical coordinates),
    # S = 1 - V_gas / (porosity * V_node), at the start and after every speciation call (saturationSpeciation)
    gasVolume = setting('gasVolume', None)
    centralDict['gasVolumeColumn'] = None
    if gasVolume is not None:
        if not centralDict['saturationColumn']:
            raise ValueError("gasVolume needs saturation = '<column>' : the saturation column computed from the gas volume "
                             "(crossDependency of the speciation)")
        if gasVolume not in centralDict['commMtrx'].columns:
            raise ValueError(f"gasVolume = '{gasVolume}' : no such column in the initial conditions")
        if gasVolume in centralDict['transportedSpecies']:
            raise ValueError(f"gasVolume = '{gasVolume}' : the column cannot be a transported species")
        centralDict['gasVolumeColumn'] = gasVolume
        saturation = nativeTransport.saturationFromGas(centralDict)
        centralDict['commMtrx'][centralDict['saturationColumn']] = saturation
        centralDict['saturation'] = saturation
        centralDict['saturationPrevious'] = saturation.copy()

    # areaSchedule = [{'cells': [node indices], 'times': [t1, t2, ...], 'area': [area from t1, area from t2, ...]}]:
    # cross-section area (m2) of some nodes changed in time (timeUnit), constant in between, the 'area' setting before
    # the first time; a number per time (every cell of the group) or one value per cell. Area decrease: the removed
    # volume leaves the domain with its solution (concentrations unchanged); area increase: pure water added (amounts
    # kept, concentrations x A_before / A_after). Each transport call is split at the change times; the area of the
    # end of the call is written in the transport output (area_m2). See nativeTransport.areaSchedule.
    # e.g. areaSchedule = {'cells': [10, 11, 12], 'times': [100, 300], 'area': [0.5e-3, 1e-3]}
    areaSetting = setting('areaSchedule', None)
    centralDict['areaSchedule'] = None
    if areaSetting is not None:
        coord = str(centralDict.get('coordinateSystem') or 'cartesian').lower()
        if coord != 'cartesian':
            raise ValueError(f"areaSchedule : coordinateSystem = '{coord}' does not use the area (shell surfaces) : "
                             "areaSchedule is available in cartesian coordinates only")
        if centralDict['saturationColumn'] or centralDict['gasVolumeColumn']:
            raise ValueError("areaSchedule is not implemented with a saturation written by the chemistry "
                             "(saturation = '<column>', gasVolume)")
        donnanCfg = centralDict.get('donnan')
        if isinstance(donnanCfg, dict) and str(donnanCfg.get('coupling', 'free')).lower() == 'totals':
            raise ValueError("areaSchedule is not implemented with donnan coupling = 'totals'")
        base = np.array(np.broadcast_to(np.asarray(centralDict['area'], dtype=float), (nNodes,)))
        centralDict.update({'area': base, 'areaBase': base, 'areaActive': base,
                            'areaSchedule': nativeTransport.areaSchedule(areaSetting, base) or None})

    # ADE: nativeTransport computes the number of sub-steps itself (advection + diffusion + dispersion) in
    # explicit mode, and does not need it in implicit mode (adeSolver = 'implicit')

    useNernstPlanck = bool(centralDict['NernstPlanck']) and not (
        centralDict['FickDiffusion'] or centralDict['advection'] or centralDict['ADE'])

    if useNernstPlanck and str(centralDict.get('npSolver') or 'explicit').lower() == 'explicit':

        speciesDiffCoeff = centralDict['speciesDiffCoeff'] or {}
        speciesPorosity = centralDict['speciesPorosity'] or {}
        nNodes = len(centralDict["commMtrx"])
        dt_diff = np.inf
        limitingSpecies = None

        for spc in centralDict['transportedSpecies']:
            D_spc = np.asarray(speciesDiffCoeff.get(spc, centralDict['diffCoeff']), dtype=float)
            th_spc = np.asarray(speciesPorosity.get(spc, centralDict['porosity']), dtype=float)
            if D_spc.ndim == 0:
                D_spc = np.full(nNodes, float(D_spc))
            if th_spc.ndim == 0:
                th_spc = np.full(nNodes, float(th_spc))

            # areaSchedule: smallest stable step over every area of the run
            dt_spc = np.inf
            for areaNodes in [centralDict['area']] + [a for _, a in centralDict.get('areaSchedule') or ()]:
                cond = th_spc * areaNodes * D_spc
                with np.errstate(divide='ignore', invalid='ignore'):
                    R = dxHalfCell / np.where(cond > 0, cond, 1.0)
                R[cond <= 0] = np.inf
                R_sum = R[:-1] + R[1:]
                T_interface = np.zeros(len(cond) - 1)
                finite_R = np.isfinite(R_sum) & (R_sum > 0)
                T_interface[finite_R] = 1000.0 / R_sum[finite_R]

                T_sum = np.zeros(len(cond))
                T_sum[:-1] += T_interface
                T_sum[1:] += T_interface

                storage = 1000.0 * np.asarray(centralDict['nodeSize'], dtype=float) \
                          * np.asarray(areaNodes, dtype=float) * th_spc
                with np.errstate(divide='ignore', invalid='ignore'):
                    dt_node = np.where(T_sum > 0, storage / np.where(T_sum > 0, T_sum, 1.0), np.inf)
                dt_spc = min(dt_spc, float(np.min(dt_node)))
            if dt_spc < dt_diff:
                dt_diff = dt_spc
                limitingSpecies = spc

        dt_diff *= centralDict['cflSafety']

        if np.isfinite(dt_diff) and dt_diff > 0:
            npMaxSubStep = dt_diff
            if dt > dt_diff:
                n_diff = int(np.ceil(dt / dt_diff))
                print(f"Sub-cycling Nernst-Planck : dt_max={writeTime(dt_diff,4)} (limiting species : "
                      f"{limitingSpecies}), i.e. {n_diff} sub-steps for timeStep ; the number of sub-steps is "
                      "recomputed at every transport call from its actual duration")

    # explicit Nernst-Planck: largest stable sub-step (s); nativeTransport derives the number of sub-steps of
    # each call from it, ceil(duration of the call / npMaxSubStep), so that it follows non-uniform dtpycte,
    # Strang half steps, OSdefined fractions, ... (inf: no limit, one sub-step)
    centralDict.update({'npMaxSubStep' : npMaxSubStep,
                        # same safety factor for the migration check of nativeTransport (h.rate <= npCflSafety)
                        'npCflSafety' : centralDict['cflSafety'],
                        'subCyclingAdv' : n_adv})
    return transportLauncher['nativeTransport']


def _transportPFLOTRAN(centralDict, setting, run):
    '''PFLOTRAN set-up: its settings into centralDict; returns its transport launcher'''
    from . import pflotran

    transportLauncher = {
        "PFLOTRAN": pflotran.trspt, 
    }

    centralDict.update({
        "trsptPath" : Path(setting('trsptPath', os.path.join(run.inputPath, 'pflotran.in'))),
        "PFLOTRANCalcTime_WallClock": 0,
        "PFLOTRANCalcTime_ProcessorTime": 0,
        "PFLOTRANInterfTime_WallClock": 0,
        "PFLOTRANInitTime" : 0,
        "PFLOTRANTotalTime" : 0,})
    # iterative coupling through a reaction sandbox: S_ / K_ datasets in testpycte.h5, I_<species>pycte integrals read
    # from the .tec; transported species: systemSpecies without the IMMOBILE_SPECIES of the PFLOTRAN input
    centralDict.update({
        "pflotranReactionTerms" : 'sandbox' if setting('pflotranReactionTerms', None) else None,
        # pflotranScale = {column: factor}: column x factor in testpycte.h5, divided back on output (unit change of a
        # column for the PFLOTRAN input, e.g. moles of a mineral into a volume fraction)
        "pflotranScale" : setting('pflotranScale', None) or {},
        })
    former = [c for c in ('MX-80', 'hMX-80') if c in centralDict['commMtrx'] and c not in centralDict['pflotranScale']]
    if former:
        warningManager.warn(f"PFLOTRAN : {former} are no longer rescaled by pycte (former built-in factors : MX-80 x "
                            "0.137469997406006, hMX-80 x 0.215) : to keep these factors, give pflotranScale = "
                            "{'MX-80': 0.137469997406006, 'hMX-80': 0.215}")
    centralDict.update({
        "transportedSpecies" : setting('transportedSpecies', None)
            or [s for s in centralDict['systemSpecies'] if s not in pflotran.immobileSpecies(centralDict['trsptPath'])],
        })
    # PFLOTRAN_DIR, the executable and mpirun checked now, not after the first speciation step
    pflotran.executable()
    # PFLOTRAN runs on a copy of its input (and of the files it names) in <run folder>/pflotranWork: the input
    # folder of the user (or of an installed example) is neither rewritten nor filled with outputs
    centralDict['trsptPath'] = pflotran.workingCopy(centralDict['trsptPath'], run.inputPath)
    return transportLauncher['PFLOTRAN']


def _multiCompoundPhreeqC(centralDict, setting):
    from . import phreeqc
    phreeqc.multiCompoundSetup(centralDict)
    if centralDict['couplingInfo'][1] == 'PhreeqC':
        phreeqc.checkOutputNames(centralDict)


def _multiCompoundXGEMS(centralDict, setting):
    from . import gems
    gems.multiCompoundSetup(centralDict, setting('transportedSpecies', None) is not None)


def _multiCompoundNative(centralDict, setting):
    from . import call
    call.multiCompoundSetup(centralDict, setting('transportedSpecies', None) is not None)


SPECIATION_SOLVERS = {'PhreeqC': _speciationPhreeqC, 'PhreeqcRM': _speciationPhreeqcRM, 'xGEMS': _speciationXGEMS, 'ORCHESTRA': _speciationORCHESTRA, 'nativeKinetics': _speciationNativeKinetics, 'nativeSpeciation': _speciationNativeSpeciation, 'none': _speciationNone}
TRANSPORT_SOLVERS = {'COMSOL': _transportCOMSOL, 'nativeTransport': _transportNativeTransport, 'PFLOTRAN': _transportPFLOTRAN}
# MultiCompoundTransport conversion of each speciation solver that has one
MULTI_COMPOUND = {'PhreeqC': _multiCompoundPhreeqC, 'PhreeqcRM': _multiCompoundPhreeqC, 'xGEMS': _multiCompoundXGEMS,
                  'nativeSpeciation': _multiCompoundNative}
