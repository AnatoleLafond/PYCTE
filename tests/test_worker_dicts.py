"""Dictionaries sent to the worker processes, which are smaller than centralDict.

speciationThreads > 1: the PhreeqC blocks (threads) and the xGEMS and nativeSpeciation workers (processes) receive the
keys of <solver>.WORKER_KEYS only
(<solver>._workerDict). A key read by the worker code but missing from WORKER_KEYS would be missing in the worker: a
KeyError, or worse the default value of a .get() instead of the setting (other results).

transportThreads > 1: the nativeTransport workers receive centralDict without the keys of nativeTransport.WORKER_SKIP
(nativeTransport._workerDict). A key of WORKER_SKIP read by the worker code would be missing in the worker.

These tests scan the worker code (every function it runs with centralDict) and check the key lists against it."""
import ast
import importlib
from pathlib import Path

import pandas as pd
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "pycte" / "Source"
DICT_NAMES = {"centralDict", "cd"}

# functions a worker runs with centralDict (or a name bound to it), per solver module: the worker function and what it
# calls with centralDict
WORKER_PATHS = {
    "phreeqc": {
        "phreeqc.py": {"workerSpeciationPhreeqC", "blockSpeciationPhreeqC", "speciationPhreeqC", "resetPhreeqC",
                       "loadDatabase", "_kineticsSettings", "_equilibriumSettings", "_tailText",
                       "_loadDatabaseFile", "knobsText", "selectedOutputNames", "stoichMatrix", "chargeTarget",
                       "fixedPhaseSpecs", "kineticsBlocks", "kineticsRates", "kineticsText", "speciesToNode",
                       "chargeCandidates", "chargeSpecies", "newWarnings", "warningText", "newIPhreeqc",
                       "primaryColumns", "selectedOutputFrame", "runNode", "selectedOutputRow", "selectedOutputValues",
                       "takeIPhreeqc", "_databaseSettings"},
        "rmphreeqc.py": {"_surfaceColumns", "_surfaceMasterCharges"},
        "outputManager.py": {"wanted", "stepNumber"},
    },
    "gems": {
        "gems.py": {"speciation_xGEMS", "nodeConditions", "checkConditions", "_stoichFrame", "xgemsModule"},
    },
    "call": {                               # nativeSpeciation
        "call.py": {"_task", "speciation_native", "buildConfig"},
    },
}

TRANSPORT_PATHS = {
    "nativeTransport.py": {"basicTransport", "transportCore", "transportSegments", "areaAt"},
    "nahed.py": {"monitorFor"},
}
# uses of centralDict in the nativeTransport worker code other than a read with a key written in place, reviewed:
# writes, and keys taken from names of the worker code (an if / else of two names, the parameter of a nested helper
# always called with names, a tuple of names). Their names are string constants of the worker code, so that the test
# below (WORKER_SKIP never named) covers them. A new use: check where its key comes from, then add it here
TRANSPORT_REVIEWED = {
    "'consoleDisplay' in centralDict",
    "centralDict.get(k)",
    "centralDict.update(saved)",
    "centralDict.update({'boundaryConditions': list(bc), 'dtStep': dtSeg, 'tTransportStart': tSeg})",
    "centralDict['firstBoundary' if side == 0 else 'secondBoundary']",
    "centralDict.get(key)",
    "centralDict.get(dict_key)",
    "centralDict[fallback_key]",
    "centralDict.pop(k, None)",
    "centralDict[k]",
}


def _module(name):
    try:
        return importlib.import_module(f"pycte.Source.{name}")
    except ImportError as error:            # e.g. pycte installed without xGEMS
        pytest.skip(f"pycte.Source.{name} cannot be imported : {error}")


def _functions(paths):
    for fileName, functions in paths.items():
        tree = ast.parse((SOURCE / fileName).read_text(encoding="utf-8"))
        found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in functions]
        assert {n.name for n in found} == functions, f"{fileName} : functions not found {functions - {n.name for n in found}}"
        for function in found:
            yield fileName, function


def _scan(paths):
    """(keys used with a literal key, functions given the whole dict, other uses as (where, code)) in the worker
    functions (nested functions included)."""
    keys, calls, other = set(), set(), []
    for fileName, function in _functions(paths):
        parents = {child: parent for parent in ast.walk(function) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(function):
            if not (isinstance(node, ast.Name) and node.id in DICT_NAMES) or isinstance(node.ctx, ast.Store):
                continue
            parent = parents.get(node)
            where = f"{fileName}:{node.lineno} ({function.name})"
            if isinstance(parent, ast.Subscript) and parent.value is node:
                if isinstance(parent.slice, ast.Constant):
                    keys.add(parent.slice.value)
                else:
                    other.append((where, ast.unparse(parent)))
            elif isinstance(parent, ast.Attribute) and parent.value is node:
                call = parents.get(parent)
                if (parent.attr in ("get", "pop", "setdefault") and isinstance(call, ast.Call) and call.args
                        and isinstance(call.args[0], ast.Constant)):
                    keys.add(call.args[0].value)
                else:
                    other.append((where, ast.unparse(call if isinstance(call, ast.Call) else parent)))
            elif isinstance(parent, ast.Call) and node in parent.args:
                func = parent.func
                calls.add(func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", ast.unparse(func)))
            elif isinstance(parent, ast.Assign) and parent.value is node and all(
                    isinstance(t, ast.Name) and t.id in DICT_NAMES for t in parent.targets):
                continue                                    # cd = centralDict : cd is scanned as well
            else:
                other.append((where, ast.unparse(parent) if parent is not None else node.id))
    return keys, calls, other


@pytest.mark.parametrize("solver", sorted(WORKER_PATHS))
def test_worker_keys_hold_every_key_read(solver):
    module = _module(solver)
    keys, _, _ = _scan(WORKER_PATHS[solver])
    missing = sorted(keys - module.WORKER_KEYS)
    assert not missing, f"keys read by the {solver} workers, missing from {solver}.WORKER_KEYS : {missing}"


@pytest.mark.parametrize("solver", sorted(WORKER_PATHS) + ["nativeTransport"])
def test_worker_code_gives_the_dict_to_scanned_functions_only(solver):
    paths = TRANSPORT_PATHS if solver == "nativeTransport" else WORKER_PATHS[solver]
    _, calls, _ = _scan(paths)
    unscanned = sorted(calls - set().union(*paths.values()))
    assert not unscanned, (f"functions given centralDict in the {solver} worker code but not scanned (add them to the "
                           f"paths of the test, then check the key lists) : {unscanned}")


@pytest.mark.parametrize("solver", sorted(WORKER_PATHS))
def test_worker_code_reads_literal_keys_only(solver):
    _, _, other = _scan(WORKER_PATHS[solver])
    assert not other, (f"uses of centralDict in the {solver} worker code that the scan cannot check :\n"
                       + "\n".join(f"{where} {code}" for where, code in other))


def test_transport_skipped_keys_never_named_in_worker_code():
    nativeTransport = _module("nativeTransport")
    names = {node.value for _, function in _functions(TRANSPORT_PATHS) for node in ast.walk(function)
             if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    named = sorted(nativeTransport.WORKER_SKIP & names)
    assert not named, f"keys of nativeTransport.WORKER_SKIP named in the worker code (remove them from it) : {named}"


def test_transport_other_uses_reviewed():
    _, _, other = _scan(TRANSPORT_PATHS)
    new = [f"{where} {code}" for where, code in other if code not in TRANSPORT_REVIEWED]
    assert not new, ("uses of centralDict in the nativeTransport worker code not reviewed (a key that is not a name "
                     "written in place: check that it cannot be a key of WORKER_SKIP, then add the code to "
                     "TRANSPORT_REVIEWED) :\n" + "\n".join(new))


def _centralDict():
    comm = pd.DataFrame({"x": [0.0, 1.0, 2.0], "Ca+2": [1e-3, 2e-3, 3e-3], "pH": [7.0, 7.1, 7.2], "tk": [298.0] * 3},
                        index=pd.RangeIndex(3))
    return comm, {
        "commMtrx": comm, "dtpycte": list(range(1000)), "porosity": [0.3] * 3, "lStep": 4,
        "crossDependencies": {"speciation": {"input": ["pH"], "output": ["pH"], "total": ["pH"]},
                              "transport": {"total": ["tk"]}},
        "output": {"stepEnd": [1, 2], "speciation": {5}}, "paths": {"speciation": "speciation"},
    }


def test_phreeqc_worker_dict():
    phreeqc = _module("phreeqc")
    comm, centralDict = _centralDict()
    light = phreeqc._workerDict(centralDict)
    assert set(light) == {"commMtrx", "lStep", "crossDependencies", "output", "paths"}
    assert list(light) == [k for k in centralDict if k in light]               # order of centralDict
    assert list(light["commMtrx"].columns) == ["pH", "tk"]                      # columns of the crossDependencies
    assert light["commMtrx"].index.equals(comm.index)
    assert light["commMtrx"].equals(comm[["pH", "tk"]])
    assert light["output"] == {"speciation": {5}}
    assert light["crossDependencies"] is centralDict["crossDependencies"]
    assert centralDict["commMtrx"] is comm and list(comm.columns) == ["x", "Ca+2", "pH", "tk"]   # caller's dict unchanged


def test_gems_worker_dict():
    gems = _module("gems")
    comm, centralDict = _centralDict()
    light = gems._workerDict(centralDict)
    assert set(light) == {"commMtrx", "lStep", "crossDependencies"}
    assert list(light) == [k for k in centralDict if k in light]
    assert list(light["commMtrx"].columns) == list(comm.columns) and len(light["commMtrx"]) == 0   # column names only
    assert centralDict["commMtrx"] is comm and len(comm) == 3


def test_native_speciation_worker_dict():
    call = _module("call")
    comm, centralDict = _centralDict()
    centralDict.update(chemistry={"elements": ["Ca"]}, chemPath="db.dat", MultiCompoundTransport=False)
    light = call._workerDict(centralDict)
    assert list(light) == [k for k in centralDict if k in call.WORKER_KEYS]  # order of centralDict
    assert set(light) == {"chemistry", "chemPath", "MultiCompoundTransport"}
    assert all(light[k] is centralDict[k] for k in light)
    assert call.buildConfig(light) == call.buildConfig(centralDict)          # the same solver settings


def test_transport_worker_dict():
    nativeTransport = _module("nativeTransport")
    comm, centralDict = _centralDict()
    light = nativeTransport._workerDict(centralDict)
    assert list(light) == [k for k in centralDict if k not in {"dtpycte", "output"}]   # order of centralDict
    assert all(light[k] is centralDict[k] for k in light)                    # same objects (commMtrx pickled once)
    assert "dtpycte" in centralDict and "output" in centralDict             # caller's dict unchanged
