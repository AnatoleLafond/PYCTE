"""What the wheel ships: every data file of the examples, and no left-over file (old copies, solver outputs)."""
import fnmatch
import tomllib
from pathlib import Path

import pycte

REPO = Path(__file__).resolve().parents[1]
CONFIG = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["setuptools"]
SKIP_DIRS = {"__pycache__", "orchestra_src", "gems_src"}

# left-overs that came back in the source tree (see erreurs_potentielles_pycte_2026-11-04.md, I-9)
LEFT_OVERS = ["*_avant*", "*_new.*", "*.bak", "*.orig", "crossDepenciesManager.py", "renouvellement.py",
              "*.log", "*.out", "*.tec", "*.h5", "*.vtu"]


def _files(folder):
    return [p for p in folder.iterdir() if p.is_file()]


def _packageDir(name):
    return REPO.joinpath(*name.split("."))


def test_imports_the_repository():
    assert Path(pycte.__file__).resolve().parent == REPO / "pycte"


def test_every_example_file_is_shipped():
    """A data file of an example folder that no package-data pattern matches is missing after pip install."""
    data = CONFIG["package-data"]
    missing = []
    for package in CONFIG["packages"]:
        if not package.startswith("pycte.Examples."):
            continue
        patterns = data.get(package, [])
        for f in _files(_packageDir(package)):
            if f.suffix in (".py", ".pyc"):
                continue
            if not any(fnmatch.fnmatch(f.name, p) for p in patterns):
                missing.append(f.relative_to(REPO).as_posix())
    assert not missing, f"not in [tool.setuptools.package-data] : {missing}"


def test_every_example_folder_is_a_package():
    """A folder of pycte/Examples with files but absent from [tool.setuptools] packages is not shipped."""
    packages = set(CONFIG["packages"])
    absent = []
    for d in (REPO / "pycte" / "Examples").rglob("*"):
        if d.is_dir() and d.name not in SKIP_DIRS and _files(d):
            name = ".".join(d.relative_to(REPO).parts)
            if name not in packages:
                absent.append(name)
    assert not absent, f"not in [tool.setuptools] packages : {absent}"


def test_cement_clay_interface_transport_files():
    """CementClayInterface loads comsol_<chemistry>_<MCT|MS>.mph and pflotran_<chemistry>_<MCT|MS>.in."""
    folder = REPO / "pycte" / "Examples" / "CementClayInterface"
    for chemistry in ("phreeqc", "orchestra", "nativeSpeciation"):
        for mode in ("MCT", "MS"):
            for stem, ext in (("comsol", ".mph"), ("pflotran", ".in")):
                assert (folder / f"{stem}_{chemistry}_{mode}{ext}").is_file()


def test_no_left_over_files():
    found = []
    for f in (REPO / "pycte").rglob("*"):
        if f.is_file() and not SKIP_DIRS & set(f.relative_to(REPO).parts):
            if any(fnmatch.fnmatch(f.name, p) for p in LEFT_OVERS):
                found.append(f.relative_to(REPO).as_posix())
    # build/lib: its old files go into the next local wheel; dist/: `twine upload dist/*` sends the old builds
    for folder in ("build/lib", "dist"):
        if (REPO / folder).exists():
            found.append(folder + "/")
    assert not found, f"left-over files (move them out of the project) : {found}"
