# pycte

PYCTE (PYthon generiC inTerfacE) couples speciation solvers (PhreeqC, ORCHESTRA, xGEMS, native speciation and
kinetics) with transport solvers (COMSOL, PFLOTRAN, and a native 1D transport code) by operator splitting.

PYCTE is distributed under the Apache 2.0 license that can be found in the LICENSE file. By using, distributing, or
contributing to this project, you agree to the terms and conditions of this license.

## Installation

Python 3.11 or newer.

```
pip install pycte
```

This installs every Python dependency (numpy, pandas, scipy, charset-normalizer, phreeqpy, h5py, mph). From a copy of
the sources, run `pip install .` in the folder containing `pyproject.toml` (`pip install -e .` for development).

Check the installation:

```
python -c "import pycte; print(pycte.examples())"
```

### Solvers

pycte only needs the solvers you use:

| Solver | Setting | What to install |
|---|---|---|
| PhreeqC | `chemModule('phreeqc')` | nothing more: IPhreeqc comes with phreeqpy |
| native speciation / kinetics | `chemModule('nativeSpeciation')`, `chemModule('nativeKinetics')` | nothing more |
| native 1D transport | `trsptModule('nativeTransport')` | nothing more |
| ORCHESTRA | `chemModule('orchestra')` | PyORCHESTRA, compiled from the ORCHESTRA sources of H. Meeussen: `pip install <PyORCHESTRA source folder>` (needs a C++ compiler, e.g. Visual Studio Build Tools on Windows). **Do not `pip install PyORCHESTRA` from PyPI: that name belongs to an unrelated project.** |
| xGEMS | `chemModule('gems')` | not on PyPI: `conda install -c conda-forge xgems` |
| COMSOL | `trsptModule('comsol')` | COMSOL Multiphysics with a valid licence (the `mph` Python package is installed with pycte) |
| PFLOTRAN | `trsptModule('pflotran')` | PFLOTRAN built from source, the environment variable `PFLOTRAN_DIR` pointing to it (pycte runs `mpirun -n 1 $PFLOTRAN_DIR/src/pflotran/pflotran`) and `mpirun` in the PATH |

## Quick start

Run first the test case matching your solvers:

```python
import pycte

pycte.load("CementClayInterface")    # PhreeqC, multi-compound transport
pycte.chemModule("orchestra")        # switch solver: the example's database and initial conditions follow
pycte.MCT(False)                     # multi-species transport (option of this example)
pycte.maxTime(100)                   # any setting can be overridden, before or after a solver switch
pycte.output({"coupling": [50, 100]})   # output steps must lie within the run (here 100 steps of 1 year)

if __name__ == "__main__":
    pycte.run()
```

The outputs are written in the current folder. When you shorten a bundled example with `pycte.maxTime(...)`, also
give output steps within the new run with `pycte.output(...)`: the example's own output steps may lie beyond it. Keep `pycte.run()` under `if __name__ == "__main__":` on Windows as
soon as several processes are used (`PIDnbr > 1`).

### Bundled test cases

| Name | Speciation | Transport | Reference |
|---|---|---|---|
| `CationExchange` | PhreeqC (default), ORCHESTRA, nativeSpeciation | nativeTransport (default), COMSOL, PFLOTRAN | ex. 11 of PhreeqC, Appelo et al. 2013 |
| `CalciteDolomite` | xGEMS | nativeTransport (default), COMSOL | Azad et al. 2016 |
| `AldicarbInfiltration` | nativeKinetics (default), PhreeqC, ORCHESTRA | COMSOL | COMSOL model of aldicarb infiltration |
| `CementClayInterface` | PhreeqC (default), ORCHESTRA | nativeTransport | cement / clay interface benchmark, 100 000 years; option `MCT(True / False)`: multi-compound (default) or multi-species transport |

`pycte.chemModule(...)` and `pycte.trsptModule(...)` switch to the files of the example for that solver. The
settings you give after `pycte.load(...)` are kept whatever the order of the calls.

## Your own case

Each setting read by pycte has its own function, e.g. `pycte.maxTime(72000)`, `pycte.timeStep(720)`,
`pycte.chemPath("my.dat")`, `pycte.initialConditions("ic.txt")`; a misspelled setting raises an AttributeError. The
older style, a script defining the settings as variables (`chemModule = 'phreeqc'`, `maxTime = 1e5`, ...) and ending
with `pycte.run()`, still works.

## Outputs

- `warning.log`: run description, every warning (solvers, Python, workers) and the computation times;
- `coupling/Coupling_<step>.txt`: the coupled variables at the steps asked with `pycte.output({'coupling': [...]})`;
- `orchestra_output.log`: ORCHESTRA solver messages.

Post-processing:

```python
df = pycte.postProcess(["Ca", "Cl"], "myRunFolder")                    # breakthrough curves (last cell)
dp = pycte.postProcess(columnProfile=(["Ca", "Cl"], "myRunFolder"))    # profiles at the last step
times = pycte.getTimes("myRunFolder")                                   # computation times, warnings count
```

PFLOTRAN runs in the folder of its input file and updates it: with a bundled example, copy the example's `.in` file
and the files it names (e.g. its `DATABASE`) next to your script, and give it with `pycte.trsptPath(...)`.

## Building a release (maintainers)

```
python -m pip install build twine
python -m build              # dist/pycte-<version>.tar.gz and .whl, built with setuptools >= 77
python -m twine upload dist/pycte-<version>*
```

Remove older files from `dist/` first, or upload only the new version. The version is set in `pyproject.toml`;
data files of a new example must be declared in `[tool.setuptools]` (`packages` and `package-data`).

Do not hesitate to contact me ! (anatole.lafond [at] cea.fr)
