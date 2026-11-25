# The project metadata are in pyproject.toml: this file only declares the compiled extensions.
import sys
from pathlib import Path

from pybind11.setup_helpers import ParallelCompile, Pybind11Extension, build_ext
from setuptools import Extension, setup

ParallelCompile("NPY_NUM_BUILD_JOBS").install()   # GEMS3K has ~50 source files

# ORCHESTRA C++ solver of H. Meeussen (Apache 2.0, see pycte/orchestra_src/NOTICE.txt), built as pycte.PyORCHESTRA.
# optional: without a C++ compiler (no wheel for the platform), pycte installs without ORCHESTRA
ORCHESTRA_SRC = Path("pycte") / "orchestra_src"
sources = sorted(p.as_posix() for p in ORCHESTRA_SRC.glob("*.cpp"))  # sorted: reproducible builds

# GEMS3K (LGPL 3) and xGEMS (GPL 3) of the GEMS team, without ThermoFun (see pycte/gems_src/NOTICE.txt), built as
# pycte.PyxGEMS. Eigen and spdlog (with its bundled fmt) are header-only: no DLL, nothing to conflict with.
# optional, as ORCHESTRA
GEMS_SRC = Path("pycte") / "gems_src"
gemsSources = sorted(p.as_posix() for d in ("GEMS3K", "xGEMS", "python") for p in (GEMS_SRC / d).glob("*.cpp"))
gemsArgs = ["/utf-8"] if sys.platform == "win32" else ["-Wno-deprecated", "-Wno-ignored-attributes"]

# reader of the PhreeqC selected output (C code of pycte, pycte/phreeqc_src), built as pycte._phreeqcRow: a row in one
# call instead of one ctypes call per column (same values). optional, as ORCHESTRA: without it, the Python loop
PHREEQC_SRC = Path("pycte") / "phreeqc_src"

# float64 values written as repr writes them (C code of pycte, pycte/text_src), built as pycte._floatText: the text
# files (outputs, inputTransport.txt of COMSOL) and the PhreeqC scripts, same characters, faster. optional, as
# ORCHESTRA: without it, the Python formatting
TEXT_SRC = Path("pycte") / "text_src"

setup(
    ext_modules=[
        Pybind11Extension(
            "pycte.PyORCHESTRA",
            sources,
            include_dirs=[ORCHESTRA_SRC.as_posix()],
            optional=True,
        ),
        Pybind11Extension(
            "pycte.PyxGEMS",
            gemsSources,
            include_dirs=[GEMS_SRC.as_posix(), (GEMS_SRC / "GEMS3K").as_posix(), (GEMS_SRC / "include").as_posix(),
                          (GEMS_SRC / "include" / "eigen3").as_posix()],
            define_macros=[("NODEARRAYLEVEL", None), ("IPMGEMPLUGIN", None), ("NO_THERMOFUN", None),
                           ("NOMINMAX", None), ("_CRT_SECURE_NO_WARNINGS", None)],
            extra_compile_args=gemsArgs,
            cxx_std=20,
            optional=True,
        ),
        Extension(
            "pycte._phreeqcRow",
            [(PHREEQC_SRC / "phreeqcRow.c").as_posix()],
            optional=True,
        ),
        Extension(
            "pycte._floatText",
            [(TEXT_SRC / "floatText.c").as_posix()],
            include_dirs=[TEXT_SRC.as_posix()],
            optional=True,
        ),
    ],
    cmdclass={"build_ext": build_ext},
)
