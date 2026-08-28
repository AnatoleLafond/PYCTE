PYCTE (PYthon generiC inTerfacE) is distributed under the Apache 2.0 -style license that can be found in the LICENSE file. By using, distributing, or contributing to this project, you agree to the terms and conditions of this license.

Three speciation solvers are currently coupled (PhreeqC, ORCHESTRA and xGEMS) with three transport solvers (COMSOL, PFLOTRAN, and a simple 1D transport code).

Please run first the test cases (CationExchange based on ex.11 of PhreeqC from Appelo and al. 2013, calciteDolomite from Azad et al. 2016, AldicarbInfiltration from comsol batch) depending on your speciation and transport solvers.

Do not hesitate to contact me ! (anatole.lafond [at] cea.fr)

To install pycte, make : 

pip install pycte

Two examples are available :
- pycte.load("CationExchange")
- pycte.load("CalciteDolomite")

Explicit formulation of theses test cases are accessible in the PreviousVersions folder.
Implitic formulation of theses test cases are accessible in the pycte/Example folders.