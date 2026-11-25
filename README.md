# pycte

PYCTE (PYthon generiC inTerfacE) couples speciation solvers (PhreeqC, PhreeqcRM, ORCHESTRA, xGEMS, native speciation and
kinetics) with transport solvers (COMSOL, PFLOTRAN, and a native 1D transport code) by operator splitting.

PYCTE is distributed under the Apache 2.0 license that can be found in the LICENSE file. By using, distributing, or
contributing to this project, you agree to the terms and conditions of this license. The compiled solvers shipped with
pycte keep their own licenses: ORCHESTRA (Apache 2.0), GEMS3K (LGPL 3) and xGEMS (GPL 3).

## Installation

Everything is nicely detailed in the user-manual.
To make it quick, for installation:

```
pip install pycte
```

Which installs every Python dependency (numpy, pandas, scipy, charset-normalizer, phreeqpy, phreeqcrm, h5py, mph) and
the ORCHESTRA and xGEMS solvers, compiled into the pycte wheels (Windows x64, Linux x86_64 / aarch64, macOS Intel /
Apple silicon; Python 3.11 to 3.14). No C++ compiler and no conda are needed. From a copy of the sources, run
`pip install .` in the folder instead. ORCHESTRA and xGEMS are then compiled if a C++ compiler is found (C++20 for xGEMS, e.g. Visual Studio Build Tools on Windows), otherwise pycte
installs without them.

Check the installation:

```
python -c "import pycte; pycte.solvers()"
```

It lists the solvers this installation can run, and, for a missing one, how to install it.
`pycte.examples()` lists the bundled test cases.

## Quick start

Run first the test case, i.e.:

```python
import pycte

pycte.load("CationExchange")    
pycte.run()
```

You've just ran the ex. 11 from the PhreeqC's manual.

You can conveniently check for the breakthrough profile concentration:

```python
df = pycte.postProcess(["Ca+2", "Cl-"], "myRunFolder")  
```
In which myRunFolder is the folder containing the pycte output (the 'stepEnd' folder, in this case).

This is a small exemple on how PYCTE work. 5 supplementary exemples are described in the user-manual.
