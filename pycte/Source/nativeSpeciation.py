"""Native geochemical equilibrium solver: aqueous, mineral, gas, exchange and surface speciation (phreeqc.dat or JSON)."""

from __future__ import annotations
import json
import math
import re
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

__all__ = ["Database", "ChemicalSystem", "Speciation", "Result", "Timings",
           "ConvergenceError", "diagnose", "find_database",
           "default_phreeqc_database",
           "dh_constants", "log_gamma", "pr_fugacity", "__version__"]



R_KJ = 8.314462618e-3
T0 = 298.15
KCAL_TO_KJ = 4.184

_BLOCKS = {
    "SOLUTION_MASTER_SPECIES", "SOLUTION_SPECIES", "PHASES",
    "EXCHANGE_MASTER_SPECIES", "EXCHANGE_SPECIES",
    "SURFACE_MASTER_SPECIES", "SURFACE_SPECIES",
}

_OPTIONS = {
    "log_k", "logk", "lk", "delta_h", "deltah", "analytic",
    "analytical_expression", "gamma", "dw", "vm", "no_check", "mole_balance",
    "llnl_gamma", "co2_llnl_gamma", "millero", "add_logk", "add_log_k",
    "t_c", "p_c", "omega", "cvm", "viscosity", "erm_ddl",
}

_KEYWORD_RE = re.compile(r"^[A-Z][A-Z_]{2,}$")

_TERM_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)?\s*([A-Za-z(\[].*?)\s*$")
_CHARGE_RE = re.compile(r"([+-])(\d*)$")


def _f(x: str) -> float:
    """Float tolerant to the separators of phreeqc.dat ('1.506;')."""
    return float(x.strip().rstrip(";,").strip())


def parse_charge(name: str) -> int:
    """Charge of a species deduced from its name ('Ca+2' -> 2, 'CO3-2' -> -2)."""
    m = _CHARGE_RE.search(name)
    if not m:
        return 0
    sign = 1 if m.group(1) == "+" else -1
    return sign * (int(m.group(2)) if m.group(2) else 1)


def split_terms(side: str) -> list[tuple[float, str]]:
    """One side of a reaction -> [(coefficient, species)], split on whitespace."""
    terms = []
    sign, coef = 1.0, None
    for token in side.split():
        if token in ("+", "-"):
            sign = -1.0 if token == "-" else 1.0
            continue
        if token[0] in "+-":
            sign = -1.0 if token[0] == "-" else 1.0
            token = token[1:]
        m = _TERM_RE.match(token)
        if not m:
            try:
                coef = (coef or 1.0) * float(token)
            except ValueError:
                raise ValueError(f"terme illisible : {token!r}") from None
            continue
        value = float(m.group(1)) if m.group(1) else 1.0
        terms.append((sign * value * (coef if coef is not None else 1.0), m.group(2).strip()))
        sign, coef = 1.0, None
    if coef is not None:
        raise ValueError(f"terme illisible : {side.strip()!r}")
    return terms


def parse_reaction(line: str) -> dict[str, float]:
    """'CaCO3 = Ca+2 + CO3-2' -> {'CaCO3': -1, 'Ca+2': 1, 'CO3-2': 1} (products positive)."""
    lhs, rhs = line.split("=", 1)
    stoich: dict[str, float] = {}
    for coef, name in split_terms(lhs):
        stoich[name] = stoich.get(name, 0.0) - coef
    for coef, name in split_terms(rhs):
        stoich[name] = stoich.get(name, 0.0) + coef
    return stoich


def find_database(name: str = "phreeqc.dat") -> str:
    """Database path: as given, NATIVESPECIATION_DATABASE, current folder, then module folder."""
    import os

    candidates = [Path(name)]
    env = os.environ.get("NATIVESPECIATION_DATABASE")
    if env:
        candidates.append(Path(env))
        candidates.append(Path(env) / name)
    candidates.append(Path.cwd() / name)
    candidates.append(Path(__file__).resolve().parent / name)
    for c in candidates:
        if c.is_file():
            return str(c.resolve())
    raise FileNotFoundError(
        f"thermodynamic database not found : {name}\nLooked in :\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\nGive a full path, or set the environment variable NATIVESPECIATION_DATABASE.")


def default_phreeqc_database(name: str = "phreeqc.dat") -> str:
    """Former name of find_database."""
    warnings.warn(
        "default_phreeqc_database is replaced by find_database (current folder, module folder or "
        "NATIVESPECIATION_DATABASE).",
        DeprecationWarning, stacklevel=2)
    return find_database(name)


@dataclass
class Species:
    """Secondary species (aqueous, exchange or surface)."""
    name: str
    stoich: dict[str, float]
    log_k: float = 0.0
    delta_h: float | None = None
    analytic: list[float] | None = None
    dh_a: float | None = None
    dh_b: float | None = None
    kind: str = "aq"
    explicit_logk: bool = False

    def logk(self, tk: float = T0) -> float:
        return logk_at_T(self.log_k, self.delta_h, self.analytic, tk)


@dataclass
class Phase:
    """Mineral or gas phase; stoich holds the right-hand side of its dissolution only."""
    name: str
    solid: str
    stoich: dict[str, float]
    log_k: float = 0.0
    delta_h: float | None = None
    analytic: list[float] | None = None
    explicit_logk: bool = False
    t_c: float | None = None
    p_c: float | None = None
    omega: float | None = None

    def logk(self, tk: float = T0) -> float:
        return logk_at_T(self.log_k, self.delta_h, self.analytic, tk)


@dataclass
class Master:
    """Master species associated with an element (or with an exchange site)."""
    element: str
    species: str
    kind: str = "aq"


def logk_at_T(log_k, delta_h, analytic, tk: float) -> float:
    """log K at temperature tk (K): analytical expression, otherwise van't Hoff."""
    if analytic:
        a = list(analytic) + [0.0] * (6 - len(analytic))
        if any(a):
            return (a[0] + a[1] * tk + a[2] / tk + a[3] * math.log10(tk)
                    + a[4] / tk ** 2 + a[5] * tk ** 2)
    if delta_h is not None and abs(tk - T0) > 1e-9:
        return log_k - delta_h / (math.log(10) * R_KJ) * (1.0 / tk - 1.0 / T0)
    return log_k


@dataclass
class Database:
    master: dict[str, Master] = field(default_factory=dict)
    species: dict[str, Species] = field(default_factory=dict)
    phases: dict[str, Phase] = field(default_factory=dict)

    @classmethod
    def from_phreeqc(cls, path: str) -> "Database":
        return _parse_phreeqc(path)

    @classmethod
    def from_json(cls, path: str) -> "Database":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        db = cls()
        for d in raw.get("master", []):
            db.master[d["element"]] = Master(**d)
        for d in raw.get("species", []):
            db.species[d["name"]] = Species(**d)
        for d in raw.get("phases", []):
            db.phases[d["name"]] = Phase(**d)
        return db

    def to_json(self, path: str) -> None:
        raw = {
            "master": [asdict(m) for m in self.master.values()],
            "species": [asdict(s) for s in self.species.values()],
            "phases": [asdict(p) for p in self.phases.values()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=1)

    def subset(self, elements, phases=()) -> "Database":
        """Database reduced to the requested elements."""
        keep_master = {e: m for e, m in self.master.items() if e in elements}
        masters = {m.species for m in keep_master.values()} | {"H2O", "H+", "e-"}
        db = Database(master=keep_master)
        for name, sp in self.species.items():
            if set(sp.stoich) <= masters | {name}:
                db.species[name] = sp
        for name in phases:
            db.phases[name] = self.phases[name]
        return db


def _parse_phreeqc(path: str) -> Database:
    db = Database()
    block = None
    current = None
    pending_phase = None

    with open(path, encoding="utf-8", errors="replace") as f:
        raw_lines = f.readlines()

    for raw in raw_lines:
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        head = line.split()[0]
        if not raw[0].isspace() and _KEYWORD_RE.match(head):
            block = head if head in _BLOCKS else None
            current = None
            pending_phase = None
            continue
        if block is None:
            continue
        stripped = line.strip()

        if block.endswith("MASTER_SPECIES"):
            parts = stripped.split()
            if len(parts) >= 2:
                kind = {"SOLUTION_MASTER_SPECIES": "aq",
                        "EXCHANGE_MASTER_SPECIES": "exch",
                        "SURFACE_MASTER_SPECIES": "surf"}[block]
                db.master[parts[0]] = Master(parts[0], parts[1], kind)
            continue

        first = stripped.split()[0]
        key = first.lstrip("-").lower()
        if "=" not in stripped and key in _OPTIONS:
            vals = stripped.split()[1:]
            if current is None:
                continue
            try:
                if key in ("log_k", "logk", "lk"):
                    current.log_k = _f(vals[0])
                    current.explicit_logk = True
                elif key == "delta_h":
                    dh = _f(vals[0])
                    if len(vals) > 1 and vals[1].lower().startswith("kcal"):
                        dh *= KCAL_TO_KJ
                    current.delta_h = dh
                elif key.startswith("analytic"):
                    current.analytic = [_f(v) for v in vals[:6]]
                    current.explicit_logk = True
                elif key == "gamma" and isinstance(current, Species):
                    current.dh_a, current.dh_b = _f(vals[0]), _f(vals[1])
                elif key in ("t_c", "p_c", "omega") and isinstance(current, Phase):
                    setattr(current, key, _f(vals[0]))
            except (ValueError, IndexError):
                pass
            continue

        if "=" in stripped:
            try:
                stoich = parse_reaction(stripped)
            except ValueError:
                current = None
                continue
            if block == "PHASES":
                if pending_phase is None:
                    current = None
                    continue
                lhs, rhs = stripped.split("=", 1)
                lhs_terms = split_terms(lhs)
                solid = lhs_terms[0][1]
                stoich = {}
                for coef, name in lhs_terms[1:]:
                    stoich[name] = stoich.get(name, 0.0) - coef
                for coef, name in split_terms(rhs):
                    stoich[name] = stoich.get(name, 0.0) + coef
                current = Phase(pending_phase, solid, stoich)
                db.phases[pending_phase] = current
                pending_phase = None
            else:
                kind = {"SOLUTION_SPECIES": "aq", "EXCHANGE_SPECIES": "exch",
                        "SURFACE_SPECIES": "surf"}[block]
                name = split_terms(stripped.split("=", 1)[1])[0][1]
                current = Species(name, stoich, kind=kind)
                db.species[name] = current
            continue

        if block == "PHASES":
            pending_phase = stripped.split()[0]
            current = None

    _check_logk(db, path)
    return db


def _check_logk(db: "Database", path) -> None:
    """Warns about reactions without a readable log K (misread lines)."""
    unreadable = [s.name for s in db.species.values()
             if not s.explicit_logk
             and len({k for k, v in s.stoich.items() if abs(v) > 1e-12}) > 1]
    unreadable += [f.name for f in db.phases.values() if not f.explicit_logk]
    if unreadable:
        warnings.warn(
            f"{len(unreadable)} reactions without a readable log K in {path} "
            f"(e.g. {', '.join(unreadable[:5])}) : their log K stays 0, which would make the speciation wrong ; "
            f"check the log_k / analytic lines of the file.", stacklevel=3)


# Debye-Huckel A and B at 25 C, values of PHREEQC 3.8.6 (DH_A, DH_B with phreeqc.dat)
A25, B25 = 0.5100247894269, 0.328490633992


def water_density(tk: float) -> float:
    """Density of water (g/cm3), Kell correlation."""
    t = tk - 273.15
    return (1.0 - (t - 3.9863) ** 2 * (t + 288.9414)
            / (508929.2 * (t + 68.12963)))


def water_permittivity(tk: float) -> float:
    """Relative permittivity of water."""
    return 2727.586 + 0.6224107 * tk - 466.9151 * math.log(tk) - 52000.87 / tk


def _raw_AB(tk: float) -> tuple[float, float]:
    rho, eps = water_density(tk), water_permittivity(tk)
    a = 1.82483e6 * math.sqrt(rho) / (eps * tk) ** 1.5
    b = 50.2916 * math.sqrt(rho) / math.sqrt(eps * tk)
    return a, b


_A0, _B0 = _raw_AB(298.15)


def dh_constants(tk: float = 298.15) -> tuple[float, float]:
    """Debye-Huckel A and B constants at temperature tk (K)."""
    a, b = _raw_AB(tk)
    return a * A25 / _A0, b * B25 / _B0


def ionic_strength(conc: np.ndarray, z: np.ndarray) -> float:
    return 0.5 * float(np.dot(conc, z * z))


def log_gamma(I: float, z: np.ndarray, dh_a: np.ndarray, dh_b: np.ndarray,
              tk: float = 298.15) -> np.ndarray:
    """log10 activity coefficients, vectorised over species (and nodes when I is an array)."""
    return _log_gamma_terms(I, _log_gamma_constants(z, dh_a, dh_b, tk))


def _log_gamma_constants(z, dh_a, dh_b, tk):
    """Terms of log_gamma independent of I (computed once per system)."""
    A, B = dh_constants(tk)
    has_dh = ~np.isnan(dh_a)
    a = np.where(has_dh, np.nan_to_num(dh_a), 0.0)
    b = np.where(has_dh, np.nan_to_num(dh_b), 0.0)
    return -A * z ** 2, z == 0.0, has_dh, B * a, b


def _log_gamma_terms(I, constants):
    minus_Az2, neutral, has_dh, Ba, b = constants
    if type(I) is float:
        sq = math.sqrt(max(I, 0.0))
        davies = minus_Az2 * (sq / (1.0 + sq) - 0.3 * I)
        lg = np.where(neutral, 0.1 * I, davies)
        wateq = minus_Az2 * sq / (1.0 + Ba * sq) + b * I
        return np.where(has_dh, wateq, lg)
    I = np.asarray(I, dtype=float)
    if I.ndim:
        I = I[..., None]
    sq = np.sqrt(np.maximum(I, 0.0))

    davies = minus_Az2 * (sq / (1.0 + sq) - 0.3 * I)
    lg = np.where(neutral, 0.1 * I, davies)

    wateq = minus_Az2 * sq / (1.0 + Ba * sq) + b * I
    return np.where(has_dh, wateq, lg)


def water_activity(molalities: np.ndarray) -> float:
    """Water activity (PHREEQC osmotic approximation), bounded below."""
    return max(1.0 - 0.017 * float(np.add.reduce(molalities, axis=None)), 1e-3)


R_ATM = 0.0820573660809596
SQ2 = math.sqrt(2.0)


def pr_fugacity(y: np.ndarray, P: float, tk: float, tc: np.ndarray,
                pc: np.ndarray, omega: np.ndarray) -> tuple[np.ndarray, float]:
    """Peng-Robinson log10 fugacity coefficients and compressibility factor."""
    n = len(y)
    if n == 0 or P <= 0:
        return np.zeros(n), 1.0
    known = ~(np.isnan(tc) | np.isnan(pc) | np.isnan(omega))
    if not known.any():
        return np.zeros(n), 1.0
    tcv = np.where(known, tc, 1000.0)
    pcv = np.where(known, pc, 1e6)
    omv = np.where(known, omega, 0.0)

    kap = 0.37464 + 1.54226 * omv - 0.26992 * omv ** 2
    alpha = (1.0 + kap * (1.0 - np.sqrt(tk / tcv))) ** 2
    ai = 0.45724 * (R_ATM * tcv) ** 2 / pcv * alpha
    bi = 0.07780 * R_ATM * tcv / pcv

    aij = np.sqrt(np.outer(ai, ai))
    a = float(y @ aij @ y)
    b = float(y @ bi)
    A = a * P / (R_ATM * tk) ** 2
    B = b * P / (R_ATM * tk)

    roots = np.roots([1.0, -(1.0 - B), A - 3 * B ** 2 - 2 * B,
                      -(A * B - B ** 2 - B ** 3)])
    real = roots[np.abs(roots.imag) < 1e-10].real
    real = real[real > B]
    Z = float(real.max()) if len(real) else 1.0

    ln_phi = (bi / b * (Z - 1.0) - math.log(max(Z - B, 1e-300))
              - A / (2 * SQ2 * B) * (2 * (aij @ y) / a - bi / b)
              * math.log((Z + (1 + SQ2) * B) / (Z + (1 - SQ2) * B)))
    return np.where(known, ln_phi / math.log(10.0), 0.0), Z


class ChemicalSystem:
    def __init__(self, db: Database, elements, minerals=(), gases=(),
                 exchanger=None, surfaces=None, temperature: float = 298.15,
                 pe: float = 4.0, redox: bool = False):
        """System of `elements` with minerals, gases, exchanger and surfaces; element 'E' makes e- a component.
        temperature in K."""
        self.db = db
        self.tk = float(temperature)
        self.pe = pe
        self.redox = redox

        masters: list[str] = []
        for el in elements:
            if el in db.master:
                sp = db.master[el].species
            elif el in db.species:
                sp = el
            else:
                raise KeyError(f"element unknown in the database : {el}")
            if sp not in masters and sp != "H2O":
                masters.append(sp)
        if "H+" not in masters:
            masters.insert(0, "H+")

        self.exchanger = exchanger
        self.x_master = db.master[exchanger].species if exchanger else None
        if exchanger:
            masters.append(self.x_master)
        self.surfaces = list(surfaces or [])
        for s in self.surfaces:
            masters.append(db.master[s].species)

        self.components = masters
        self.nc = len(masters)
        self.index = {n: j for j, n in enumerate(masters)}

        self.free = {"H2O": 0, "e-": 1}
        allowed = set(masters) | set(self.free)

        self._memo: dict[str, tuple[dict[str, float], float] | None] = {}
        rows, names, kinds, logk = [], [], [], []
        free_rows = []
        for name, sp in db.species.items():
            if name in self.free:
                continue
            if name == self.x_master:
                continue
            res = self._expand(name)
            if res is None:
                continue
            stoich, lk = res
            if not set(stoich) <= allowed:
                continue
            if (not redox and "e-" not in self.index
                    and abs(stoich.get("e-", 0.0)) > 1e-12):
                continue
            names.append(name)
            kinds.append(sp.kind)
            rows.append([stoich.get(m, 0.0) for m in masters])
            free_rows.append([0.0 if f in self.index else stoich.get(f, 0.0)
                              for f in self.free])
            logk.append(lk)

        self.species = names
        self.nu = np.array(rows, dtype=float)
        self.nu_free = np.array(free_rows, dtype=float)
        self.logk = np.array(logk, dtype=float)
        self.kind = np.array(kinds)

        zc = np.array([parse_charge(m) for m in masters], dtype=float)
        self.z_components = zc
        self.z = self.nu @ zc + self.nu_free @ np.array(
            [parse_charge(f) for f in self.free], dtype=float)
        self.dh_a = np.array([_g(db.species[n].dh_a) for n in names])
        self.dh_b = np.array([_g(db.species[n].dh_b, 0.0) for n in names])

        self.is_aq = self.kind == "aq"
        self.is_exch = self.kind == "exch"
        self.is_surf = self.kind == "surf"

        if exchanger:
            jx = self.index[db.master[exchanger].species]
            self.nu_x = self.nu[:, jx].copy()
        else:
            self.nu_x = np.zeros(len(names))

        pn, prows, pfree, plogk = [], [], [], []
        for m in minerals:
            vec, vfree, lk = self.phase_data(m)
            pn.append(m)
            prows.append(vec)
            pfree.append(vfree)
            plogk.append(lk)

        self.minerals = pn
        self.nu_min = np.array(prows, dtype=float).reshape(len(pn), self.nc)
        self.nu_min_free = np.array(pfree, dtype=float).reshape(len(pn), 2)
        self.logk_min = np.array(plogk, dtype=float)

        gn, grows, gfree, glogk = [], [], [], []
        for g in gases:
            vec, vfree, lk = self.phase_data(g)
            gn.append(g)
            grows.append(vec)
            gfree.append(vfree)
            glogk.append(lk)
        self.gases = gn
        self.gas_tc = np.array([_g(db.phases[g].t_c) for g in gn])
        self.gas_pc = np.array([_g(db.phases[g].p_c) for g in gn])
        self.gas_omega = np.array([_g(db.phases[g].omega) for g in gn])
        self.nu_gas = np.array(grows, dtype=float).reshape(len(gn), self.nc)
        self.nu_gas_free = np.array(gfree, dtype=float).reshape(len(gn), 2)
        self.logk_gas = np.array(glogk, dtype=float)

    def phase_data(self, name: str):
        """Stoichiometric vector (dissolution) and effective log K of a phase."""
        ph = self.db.phases[name]
        allowed = set(self.components) | set(self.free)
        stoich, lk = {}, ph.logk(self.tk)
        for r, coef in ph.stoich.items():
            res = self._expand(r)
            if res is None or not set(res[0]) <= allowed:
                raise ValueError(f"phase {name} outside the chosen system ({r})")
            for k, v in res[0].items():
                stoich[k] = stoich.get(k, 0.0) + coef * v
            lk -= coef * res[1]
        vec = [stoich.get(x, 0.0) for x in self.components]
        vfree = [0.0 if f in self.index else stoich.get(f, 0.0)
                 for f in self.free]
        return vec, vfree, lk

    def _expand(self, name: str):
        """Species `name` on the master species -> (stoichiometry, cumulated log K) or None."""
        if name in self._memo:
            return self._memo[name]
        if name in self.components or name in self.free:
            res = ({name: 1.0}, 0.0)
            self._memo[name] = res
            return res
        sp = self.db.species.get(name)
        if sp is None:
            self._memo[name] = None
            return None
        self._memo[name] = None
        c = sp.stoich.get(name, 0.0)
        if c == 0.0:
            return None
        stoich, lk = {}, sp.logk(self.tk)
        for r, coef in sp.stoich.items():
            if r == name:
                continue
            sub = self._expand(r)
            if sub is None:
                return None
            for k, v in sub[0].items():
                stoich[k] = stoich.get(k, 0.0) - coef * v
            lk -= coef * sub[1]
        res = ({k: v / c for k, v in stoich.items() if abs(v) > 1e-14}, lk / c)
        self._memo[name] = res
        return res

    def summary(self) -> str:
        return (f"{self.nc} components : {', '.join(self.components)}\n"
                f"{len(self.species)} species "
                f"({int(self.is_aq.sum())} aq, {int(self.is_exch.sum())} exchange, "
                f"{int(self.is_surf.sum())} surface)\n"
                f"{len(self.minerals)} minerals : {', '.join(self.minerals) or '-'}\n"
                f"{len(self.gases)} gases : {', '.join(self.gases) or '-'}")


def _g(v, default=float("nan")):
    return default if v is None else v


LN10 = np.log(10.0)
R_GAS = R_ATM
H2O_PER_KG = 55.50868

__version__ = "3.1"


class ConvergenceError(RuntimeError):
    pass


@dataclass
class Timings:
    """Breakdown of the computation time of one solve (seconds)."""
    total: float = 0.0
    setup: float = 0.0
    assembly: float = 0.0
    linalg: float = 0.0
    activity: float = 0.0
    newton: int = 0
    solves: int = 0
    retries: int = 0
    outer: int = 0
    size: int = 0

    @property
    def numeric(self) -> float:
        """Pure computation time: assembly + linear algebra + activities."""
        return self.assembly + self.linalg + self.activity

    def __str__(self) -> str:
        return (f"total {1e3*self.total:7.3f} ms | numeric {1e3*self.numeric:7.3f} ms "
                f"(assembly {1e3*self.assembly:6.3f}, linear {1e3*self.linalg:6.3f}, "
                f"activities {1e3*self.activity:6.3f}) | {self.newton} iterations, "
                f"{self.solves} systems {self.size}x{self.size}")


@dataclass
class Result:
    system: ChemicalSystem = field(repr=False)
    u: np.ndarray = field(repr=False)
    c: np.ndarray = field(repr=False)
    lg: np.ndarray = field(repr=False)
    n_min: dict
    n_gas: dict
    p_gas: dict
    ionic_strength: float
    a_h2o: float
    iterations: int
    error: float = 0.0
    gas_volume: float = 0.0
    gas_z: float = 1.0
    water_mass: float = 1.0
    timings: Timings = field(default_factory=Timings, repr=False)

    @property
    def p_total(self) -> float:
        return sum(self.p_gas.values())

    @property
    def pH(self) -> float:
        return -self.u[self.system.index["H+"]]

    @property
    def pe(self) -> float:
        j = self.system.index.get("e-")
        return self.system.pe if j is None else -float(self.u[j])

    def molality(self, name: str) -> float:
        return float(self.c[self.system.species.index(name)])

    def aqueous(self, threshold: float = 0.0) -> dict:
        s = self.system
        return {n: float(cc) for n, cc, ok in zip(s.species, self.c, s.is_aq)
                if ok and cc > threshold}

    def sorbed(self, threshold: float = 0.0) -> dict:
        s = self.system
        return {n: float(cc) for n, cc, ok in
                zip(s.species, self.c, s.is_exch | s.is_surf) if ok and cc > threshold}

    def total(self, element: str) -> float:
        """Dissolved total of a component (mol/kgw), sorption excluded."""
        s = self.system
        j = s.index[s.db.master[element].species if element in s.db.master else element]
        return float(np.sum(s.nu[s.is_aq, j] * self.c[s.is_aq]))

    def activity_of(self, name: str) -> float:
        i = self.system.species.index(name)
        return float(self.c[i] * 10.0 ** self.lg[i])

    def si(self, phase: str) -> float:
        """Saturation index log10(IAP/K). For a gas, it is log10 P."""
        s = self.system
        vec, vfree, lk = s.phase_data(phase)
        la_free = np.array([np.log10(self.a_h2o), -s.pe])
        return float(np.dot(vec, self.u) + np.dot(vfree, la_free) - lk)

    def charge_balance(self) -> float:
        """Charge imbalance of the solution alone (mol/kgw)."""
        s = self.system
        return float(np.sum(s.z[s.is_aq] * self.c[s.is_aq]))


class Speciation:
    def __init__(self, system: ChemicalSystem, tol: float = 1e-10,
                 tol_si: float = 1e-10, tol_stall: float = 1e-6,
                 max_newton: int = 200, max_outer: int = 60,
                 max_step: float = 1.0, timing: bool = True, eos: str = "pr",
                 inner_gamma: bool = True):
        """Solver tolerances: mass balances (tol), saturation indices (tol_si), accepted stall (tol_stall)."""
        self.s = system
        self.tol = tol
        self.tol_si = tol_si
        self.tol_stall = tol_stall
        self.max_newton = max_newton
        self.max_outer = max_outer
        self.max_step = max_step
        self.timing = timing
        self.eos = eos
        self.inner_gamma = inner_gamma
        self._gamma_constants = None
        self._abs_nu = np.abs(system.nu)
        self._nu2 = system.nu * system.nu
        self._z2 = system.z * system.z
        self._nu_x_floor = np.maximum(system.nu_x, 1e-30)
        self._lg_x_offset = (None, None)

    def solve(self, totals: dict, pH: float | None = None,
              pH_reagent: str | None = None, cec: float = 0.0,
              sites: dict | None = None, minerals: dict | None = None,
              fugacities: dict | None = None, gas_initial: dict | None = None,
              gas_volume: float | None = None, gas_pressure: float | None = None,
              u0: np.ndarray | None = None, charge_balance: bool = True,
              lg0: np.ndarray | None = None, a_h2o0: float | None = None,
              pH0: float | None = None, pe0: float | None = None,
              water: float | None = None,
              water_mass0: float | None = None) -> Result:
        """Equilibrium of `totals` (mol/kgw, moles with `water`) with pH, exchange, surfaces, minerals, gases, warm start."""
        if water is not None:
            return self._solve_water(
                totals, water, water_mass0, minerals,
                dict(pH=pH, pH_reagent=pH_reagent, cec=cec, sites=sites,
                     u0=u0, charge_balance=charge_balance, lg0=lg0,
                     a_h2o0=a_h2o0, pH0=pH0, pe0=pe0))
        t0 = time.perf_counter() if self.timing else 0.0
        tm = Timings()
        s = self.s
        nc = s.nc
        jH = s.index["H+"]

        fug = dict(fugacities or {})
        for g in fug:
            if g not in s.gases:
                raise KeyError(f"gas '{g}' not in the system : declare it with "
                               f"ChemicalSystem(..., gases=[...])")
        if gas_volume is not None and gas_pressure is not None:
            raise ValueError("give gas_volume or gas_pressure, not both")
        mode = ("V" if gas_volume is not None else
                "P" if gas_pressure is not None else None)
        bubble = [g for g in s.gases if g not in fug] if mode else []
        ib = np.array([s.gases.index(g) for g in bubble], dtype=int)
        if_ = np.array([s.gases.index(g) for g in fug], dtype=int)
        target = np.array([fug[s.gases[i]] for i in if_], dtype=float)

        T = np.zeros(nc)
        for el, val in totals.items():
            name = s.db.master[el].species if el in s.db.master else el
            T[s.index[name]] += val
        if s.exchanger:
            if cec <= 0:
                raise ValueError("exchanger declared but cec is 0 : give "
                                 "cec=<moles of sites> to solve()")
            if not s.is_exch.any():
                raise ValueError(
                    f"exchanger '{s.exchanger}' declared but no exchange species "
                    f"found in the database : check its EXCHANGE_SPECIES block")
            T[s.index[s.db.master[s.exchanger].species]] = cec
        for site, moles in (sites or {}).items():
            T[s.index[s.db.master[site].species]] = moles

        T_sol = T.copy()
        n0 = np.array([(minerals or {}).get(m, 0.0) for m in s.minerals])
        if len(n0):
            T = T + s.nu_min.T @ n0
        for g, moles in (gas_initial or {}).items():
            T = T + moles * s.nu_gas[s.gases.index(g)]

        je = s.index.get("e-")
        bad = [s.components[j] for j in range(nc)
               if j != jH and j != je and T[j] <= 0]
        if bad:
            raise ValueError(f"zero or negative total for : {bad}")

        fixed_pH = pH is not None
        uidx = np.array([j for j in range(nc) if not (fixed_pH and j == jH)])
        eqidx = np.array([j for j in range(nc) if not (fixed_pH and j == jH)])
        if not fixed_pH:
            charge_row = jH if charge_balance else None
        elif pH_reagent is not None:
            name = (s.db.master[pH_reagent].species
                    if pH_reagent in s.db.master else pH_reagent)
            if name not in s.index:
                raise KeyError(f"pH_reagent '{pH_reagent}' is not a component")
            charge_row = s.index[name]
        else:
            charge_row = None

        if u0 is not None:
            u = u0.copy()
        else:
            u = np.clip(np.log10(np.maximum(T_sol, 1e-9)) - 0.5, -9.0, 1.0)
            u[jH] = -7.0 if pH0 is None else -pH0
            if je is not None:
                u[je] = -s.pe if pe0 is None else -pe0
            if s.exchanger:
                jx = s.index[s.db.master[s.exchanger].species]
                u[jx] = np.log10(max(cec, 1e-12)) - 6.0
        if fixed_pH:
            u[jH] = -pH

        start = None
        if (u0 is not None and lg0 is not None and a_h2o0 is not None
                and len(lg0) == len(s.species) and np.all(np.isfinite(lg0))
                and np.isfinite(a_h2o0) and a_h2o0 > 0):
            start = (np.asarray(lg0, dtype=float), float(a_h2o0))
        u_init = u.copy()
        active = [i for i in range(len(s.minerals)) if n0[i] > 0]
        n_act = n0.copy()
        n_fug = np.zeros(len(if_))
        n_tot = 0.0
        bubble_on = mode == "P" and len(ib) > 0
        if self.timing:
            tm.setup = time.perf_counter() - t0

        steps = (self.max_step, 0.3 * self.max_step, 0.1 * self.max_step)
        for essai, step in enumerate(steps):
            try:
                (u, n_act, n_fug, n_tot, active, bubble_on, lg, lphi, Zgas,
                 a_h2o, I, it_total, err) = self._outer_loop(
                    u_init.copy(), n0.copy(),
                    [i for i in range(len(s.minerals)) if n0[i] > 0],
                    np.zeros(len(if_)), 0.0, mode == "P" and len(ib) > 0,
                    T, uidx, eqidx, charge_row, ib, if_, target, mode,
                    gas_volume, gas_pressure, cec, step, tm, start)
                break
            except ConvergenceError as exc:
                if essai == len(steps) - 1:
                    if pH_reagent is not None and hasattr(exc, "u"):
                        jr = charge_row
                        cr = float(np.sum(np.abs(s.nu[:, jr])
                                          * 10.0 ** (s.logk + s.nu @ exc.u)))
                        if cr < 1e-10 * float(np.max(np.abs(T))):
                            zr = s.z_components[jr]
                            opposite = [n for n, z in zip(s.components,
                                                        s.z_components)
                                      if z * zr < 0 and n != "H+"]
                            raise ValueError(
                                f"pH_reagent '{pH_reagent}' (charge {zr:+.0f}) "
                                f"cannot restore electroneutrality at pH "
                                f"{pH:.2f} : it would need a negative concentration. "
                                f"Choose a reagent of opposite charge"
                                + (f" (e.g. {opposite[0]})" if opposite
                                   else "") + ".") from exc
                    raise
                tm.retries += 1

        q = float(np.dot(s.z_components, T))
        pH_final = -u[jH]
        if (pH is None and charge_balance and (pH_final < 2.5 or pH_final > 11.5)
                and abs(q) > 1e-3 * float(np.max(np.abs(T)))):
            warnings.warn(
                f"pH {pH_final:.2f} reached with a charge imbalance of the totals of {q:+.3e} eq/kgw, carried by H+ / OH- : "
                f"check that the totals include the sorbed fraction (total = dissolved + sorbed) and that the CEC "
                f"is balanced by cations.", stacklevel=2)

        if len(s.minerals):
            la_free = np.array([np.log10(a_h2o), -s.pe])
            si = s.nu_min @ u + s.nu_min_free @ la_free - s.logk_min
            blocked = [s.minerals[i] for i in range(len(s.minerals))
                        if i not in active and si[i] > 1e-6]
            if blocked:
                warnings.warn(
                    f"supersaturated phases left out of the active set : {', '.join(blocked)} (the active set cycled on "
                    f"phases sharing elements) : the result is not an exact equilibrium for them.", stacklevel=2)

        c = self._concentrations(u, lg, a_h2o)
        p_all = self._pressures(u, a_h2o, np.arange(len(s.gases)), lphi)
        n_gas = {s.gases[i]: -float(n_fug[k]) for k, i in enumerate(if_)}
        vol = gas_volume if mode == "V" else 0.0
        if mode == "V":
            for i in ib:
                n_gas[s.gases[i]] = float(p_all[i] * gas_volume
                                          / (Zgas * R_GAS * s.tk))
        elif mode == "P" and len(ib):
            psum = float(p_all[ib].sum())
            for i in ib:
                n_gas[s.gases[i]] = (float(n_tot * p_all[i] / psum)
                                     if bubble_on else 0.0)
            vol = (n_tot * Zgas * R_GAS * s.tk / gas_pressure) if bubble_on else 0.0
        tm.size = len(uidx) + len(active) + len(if_) + (1 if bubble_on else 0)
        if self.timing:
            tm.total = time.perf_counter() - t0
        return Result(system=s, u=u, c=c, lg=lg,
                      n_min={m: float(n_act[i]) for i, m in enumerate(s.minerals)},
                      n_gas=n_gas,
                      p_gas={s.gases[i]: float(p_all[i])
                             for i in list(if_) + list(ib)},
                      ionic_strength=I, a_h2o=a_h2o, iterations=it_total,
                      error=err, gas_volume=vol, gas_z=Zgas, timings=tm)

    def _solve_water(self, totals, water, water_mass0, minerals, options,
                     tol: float = 1e-12, max_iter: int = 50) -> Result:
        """Water balance: fixed point on the water mass around solve."""
        s = self.s
        if len(s.gases):
            raise ValueError("water balance not available with gases")
        minerals = dict(minerals or {})
        W = float(water) + sum(s.nu_min_free[i, 0] * minerals.get(m, 0.0)
                               for i, m in enumerate(s.minerals))
        w = (float(water_mass0) if water_mass0 and water_mass0 > 0
             else max(float(water) / H2O_PER_KG, 1e-6))
        options = dict(options)
        newton = 0
        t0 = time.perf_counter()
        for _ in range(max_iter):
            sites = options.get("sites")
            r = self.solve({k: v / w for k, v in totals.items()},
                           minerals={m: n / w for m, n in minerals.items()},
                           **dict(options, cec=options["cec"] / w,
                                  sites=None if sites is None else
                                  {k: v / w for k, v in sites.items()}))
            newton += r.timings.newton
            n_min = np.array([r.n_min[m] for m in s.minerals])
            bound = w * (float(s.nu_free[:, 0] @ r.c)
                         + float(s.nu_min_free[:, 0] @ n_min))
            w_new = (W - bound) / H2O_PER_KG
            if not w_new > 0:
                raise ConvergenceError(
                    f"bilan d'eau : masse d'eau negative ({w_new:.3e} kg)")
            if abs(w_new - w) <= tol * w:
                break
            w = w_new
            options.update(u0=r.u, lg0=r.lg, a_h2o0=r.a_h2o)
        else:
            raise ConvergenceError("bilan d'eau non converge")
        r.n_min = {m: n * w for m, n in r.n_min.items()}
        r.water_mass = w
        r.timings.newton = newton
        if self.timing:
            r.timings.total = time.perf_counter() - t0
        return r

    def _outer_loop(self, u, n_act, active, n_fug, n_tot, bubble_on, T, uidx,
                    eqidx, charge_row, ib, if_, target, mode, gas_volume,
                    gas_pressure, cec, step_max, tm, start=None):
        """Fixed point on the activities, wrapping the active set and Newton."""
        s = self.s
        lphi, Zgas = np.zeros(len(s.gases)), 1.0
        it_total, err = 0, 0.0
        if start is None:
            c0 = self._concentrations(u, np.zeros(len(s.species)), 1.0)
        else:
            c0 = self._concentrations(u, start[0], start[1])
        aq0 = c0 * s.is_aq
        I = 0.5 * float(np.dot(aq0, self._z2))
        lg = self._log_gamma(I, aq0, cec)
        a_h2o = water_activity(aq0)
        essais_phase = {}
        for outer in range(self.max_outer):
            tm.outer += 1
            lphi_prev = lphi.copy()

            for _ in range(40):
                u, n_act, n_fug, n_tot, it, err, lg, a_h2o = self._newton(
                    u, n_act, active, n_fug, n_tot, lg, a_h2o, T, uidx, eqidx,
                    charge_row, ib, if_, target, mode, gas_volume, gas_pressure,
                    bubble_on, lphi, Zgas, cec, step_max, tm)
                it_total += it
                changed = False
                neg = [i for i in active if n_act[i] < 0.0]
                if neg:
                    worst = min(neg, key=lambda i: n_act[i])
                    active.remove(worst)
                    n_act[worst] = 0.0
                    changed = True
                else:
                    la_free = np.array([np.log10(a_h2o), -s.pe])
                    si = s.nu_min @ u + s.nu_min_free @ la_free - s.logk_min
                    cand = [i for i in range(len(s.minerals))
                            if i not in active and si[i] > 1e-8
                            and essais_phase.get(i, 0) < 2]
                    if cand:
                        best = max(cand, key=lambda i: si[i])
                        active.append(best)
                        essais_phase[best] = essais_phase.get(best, 0) + 1
                        changed = True

                if mode == "P" and len(ib):
                    psum = float(self._pressures(u, a_h2o, ib, lphi).sum())
                    if bubble_on and n_tot < 0.0:
                        bubble_on, n_tot, changed = False, 0.0, True
                    elif not bubble_on and psum > gas_pressure * (1 + 1e-10):
                        bubble_on, n_tot, changed = True, 1e-12, True
                if not changed:
                    break

            ta = time.perf_counter() if self.timing else 0.0
            lg_prev = lg
            c = self._concentrations(u, lg, a_h2o)
            aq = c * s.is_aq
            I = 0.5 * float(np.dot(aq, self._z2))
            lg = self._log_gamma(I, aq, cec)
            a_h2o = water_activity(aq)
            lphi, Zgas = self._update_eos(u, a_h2o, ib, if_, target, mode,
                                          n_tot, gas_volume, lphi, Zgas)
            if self.timing:
                tm.activity += time.perf_counter() - ta
            if (np.maximum.reduce(np.abs(lg - lg_prev), axis=None) < 1e-10
                    and (not len(lphi) or np.maximum.reduce(np.abs(lphi - lphi_prev), axis=None) < 1e-12)):
                break
        else:
            raise ConvergenceError("coefficients d'activite non converges")


        return (u, n_act, n_fug, n_tot, active, bubble_on, lg, lphi, Zgas,
                a_h2o, I, it_total, err)

    def _pressures(self, u, a_h2o, idx, lphi=None) -> np.ndarray:
        """Partial pressures (atm) of the gases `idx`."""
        s = self.s
        if not len(idx):
            return np.zeros(0)
        la_free = np.array([np.log10(a_h2o), -s.pe])
        lp = s.nu_gas[idx] @ u + s.nu_gas_free[idx] @ la_free - s.logk_gas[idx]
        if lphi is not None:
            lp = lp - lphi[idx]
        np.maximum(lp, -300.0, out=lp)
        np.minimum(lp, 30.0, out=lp)
        return 10.0 ** lp

    def _update_eos(self, u, a_h2o, ib, if_, target, mode, n_tot, gas_volume,
                    lphi, Z):
        """Updates the fugacity coefficients (outer loop)."""
        s = self.s
        if self.eos != "pr" or not len(s.gases):
            return lphi, Z
        new = lphi.copy()
        for k, i in enumerate(if_):
            y = np.array([1.0])
            lp, _ = pr_fugacity(y, 10.0 ** target[k], s.tk,
                                         s.gas_tc[[i]], s.gas_pc[[i]],
                                         s.gas_omega[[i]])
            new[i] = lp[0]
        if mode and len(ib):
            p_b = self._pressures(u, a_h2o, ib, lphi)
            ptot = float(p_b.sum())
            if ptot > 0:
                lp, Z = pr_fugacity(p_b / ptot, ptot, s.tk,
                                             s.gas_tc[ib], s.gas_pc[ib],
                                             s.gas_omega[ib])
                new[ib] = lp
        return new, Z

    def _gamma_terms(self):
        """Constant terms of the activity coefficients (aqueous species, exchange species), computed once."""
        s = self.s
        if self._gamma_constants is None:
            self._gamma_constants = (
                _log_gamma_constants(s.z, s.dh_a, s.dh_b, s.tk),
                _log_gamma_constants(s.nu_x, s.dh_a, s.dh_b, s.tk))
        return self._gamma_constants

    def _log_gamma(self, I, aq, cec) -> np.ndarray:
        """log10 activity coefficients of all species (aqueous, exchange, surface)."""
        s = self.s
        self._gamma_terms()
        lg = _log_gamma_terms(I, self._gamma_constants[0])
        lg = np.where(s.is_aq, lg, 0.0)
        if s.exchanger and cec > 0:
            if self._lg_x_offset[0] != cec:
                self._lg_x_offset = (cec, np.log10(self._nu_x_floor / cec))
            lg_x = (_log_gamma_terms(I, self._gamma_constants[1])
                    + self._lg_x_offset[1])
            lg = np.where(s.is_exch, lg_x, lg)
        return lg

    def _revise_guess(self, u, lg, a_h2o, T_eff, rows, max_pass: int = 60,
                      marge: float = 20.0):
        """Diagonal log-Newton preconditioning of the unknowns."""
        s = self.s
        nu2 = self._nu2
        for _ in range(max_pass):
            c = self._concentrations(u, lg, a_h2o)
            S = s.nu.T @ c
            ok = True
            du = np.zeros(s.nc)
            for j in rows:
                if T_eff[j] <= 0 or S[j] <= 0:
                    continue
                ratio = S[j] / T_eff[j]
                if 1.0 / marge < ratio < marge:
                    continue
                ok = False
                nu_eff = float(nu2[:, j] @ c) / S[j]
                du[j] = -np.log10(ratio) / max(nu_eff, 1e-3)
            if ok:
                break
            u = u + np.clip(du, -3.0, 3.0)
        return u

    def _concentrations(self, u, lg, a_h2o) -> np.ndarray:
        s = self.s
        la_free = np.array([np.log10(a_h2o), -s.pe])
        lc = s.logk + s.nu @ u + s.nu_free @ la_free - lg
        np.maximum(lc, -300.0, out=lc)
        np.minimum(lc, 30.0, out=lc)
        return 10.0 ** lc

    @staticmethod
    def _rows(M, X):
        """M @ x for every row x of X, rounded exactly as one M @ x per row (rows made contiguous like 1-D vectors)."""
        return np.matmul(M, np.ascontiguousarray(X)[..., None])[..., 0]

    @staticmethod
    def _dots(X, v):
        """np.dot(x, v) for every row x of X, rounded exactly as one dot per row (rows made contiguous)."""
        return np.matmul(np.ascontiguousarray(X)[:, None, :], v)[:, 0]

    def _concentrations_block(self, U, LG, A):
        """_concentrations of every row of U, LG, A."""
        s = self.s
        la_free = np.empty((len(A), 2))
        la_free[:, 0] = np.log10(A)
        la_free[:, 1] = -s.pe
        lc = s.logk + self._rows(s.nu, U) + self._rows(s.nu_free, la_free) - LG
        np.maximum(lc, -300.0, out=lc)
        np.minimum(lc, 30.0, out=lc)
        return 10.0 ** lc

    def _activities_block(self, C, lg_x):
        """Ionic strength, log10 activity coefficients and water activity of every row (lg_x: exchange offsets)."""
        s = self.s
        aq = C * s.is_aq
        I = 0.5 * self._dots(aq, self._z2)
        constants = self._gamma_terms()
        lg = np.where(s.is_aq, _log_gamma_terms(I, constants[0]), 0.0)
        if lg_x is not None:
            lg = np.where(s.is_exch, _log_gamma_terms(I, constants[1]) + lg_x, lg)
        a_h2o = np.maximum(1.0 - 0.017 * np.add.reduce(aq, axis=1), 1e-3)
        return I, lg, a_h2o

    def _revise_block(self, U, LG, A, T, rows, max_pass: int = 60, marge: float = 20.0):
        """_revise_guess of every row."""
        s = self.s
        U = U.copy()
        rows = np.asarray(rows, dtype=int)
        live = np.arange(len(U))
        for _ in range(max_pass):
            if not len(live):
                break
            c = self._concentrations_block(U[live], LG[live], A[live])
            S = self._rows(s.nu.T, c)
            t, sr = T[live][:, rows], S[:, rows]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = sr / t
                out = (t > 0) & (sr > 0) & ~((1.0 / marge < ratio) & (ratio < marge))
            moving = out.any(axis=1)
            du = np.zeros((len(live), s.nc))
            for k, i in zip(*np.nonzero(out)):
                j = rows[i]
                nu_eff = float(self._nu2[:, j] @ c[k]) / S[k, j]
                du[k, j] = -np.log10(ratio[k, i]) / max(nu_eff, 1e-3)
            live = live[moving]
            U[live] = U[live] + np.clip(du[moving], -3.0, 3.0)
        return U

    def _newton_block(self, U, LG, A, NA, T, lg_x, charge_row, step_max, act=()):
        """_newton of every row, all with the active minerals `act` (no gas nor imposed pH)
        -> (u, lg, a_h2o, n_min, iterations, error, converged)."""
        s = self.s
        n, nc = U.shape
        act = list(act)
        na = len(act)
        NA = NA.copy()
        if na:
            nu_a = s.nu_min[act]
            abs_nu_a = np.abs(nu_a)
            free_a = s.nu_min_free[act]
            logk_a = s.logk_min[act]
            charge_a = nu_a @ s.z_components if charge_row is not None else None
            T_eff = T - self._rows(nu_a.T, NA[:, act])
        else:
            T_eff = T
        U = self._revise_block(U, LG, A, T_eff, [j for j in range(nc) if j != charge_row])
        LG, A = LG.copy(), A.copy()
        iterations = np.full(n, self.max_newton, dtype=int)
        error = np.zeros(n)
        converged = np.zeros(n, dtype=bool)
        best = np.full(n, np.inf)
        stall = np.zeros(n, dtype=int)
        small = np.zeros(n, dtype=bool)
        abs_z = np.abs(s.z)
        live = np.arange(n)
        for it in range(1, self.max_newton + 1):
            if not len(live):
                break
            u, lg, a = U[live], LG[live], A[live]
            c = self._concentrations_block(u, lg, a)
            k = np.flatnonzero(small[live]) if self.inner_gamma else []
            if len(k):
                _, lg[k], a[k] = self._activities_block(c[k], None if lg_x is None else lg_x[live[k]])
                c[k] = self._concentrations_block(u[k], lg[k], a[k])
                LG[live[k]], A[live[k]] = lg[k], a[k]
            t = T[live]
            R = self._rows(s.nu.T, c) - t
            scale = np.maximum(np.abs(t), self._rows(self._abs_nu.T, c))
            if na:
                n_a = NA[live][:, act]
                R = R + self._rows(nu_a.T, n_a)
                scale = np.maximum(scale, self._rows(abs_nu_a.T, np.abs(n_a)))
            scale = np.maximum(scale, 1e-30)
            if charge_row is not None:
                R[:, charge_row] = self._dots(c, s.z)
                scale[:, charge_row] = np.maximum(self._dots(c, abs_z), 1e-30)
            err_mb = np.maximum.reduce(np.abs(R) / scale, axis=1)
            if na:
                la_free = np.empty((len(a), 2))
                la_free[:, 0] = np.log10(a)
                la_free[:, 1] = -s.pe
                Rmin = self._rows(nu_a, u) + self._rows(free_a, la_free) - logk_a
                worst = np.maximum.reduce(np.abs(Rmin), axis=1)
                err_si = np.where(worst > 0.0, worst, 0.0)
            else:
                err_si = np.zeros(len(live))
            relative = err_si * self.tol / self.tol_si
            err = np.where(relative > err_mb, relative, err_mb)
            done = (err_mb < self.tol) & (err_si < self.tol_si)
            improved = err < best[live] * (1.0 - 1e-3)
            best[live] = np.where(improved, err, best[live])
            stall[live] = np.where(improved, 0, stall[live] + 1)
            done |= (stall[live] >= 10) & (err < self.tol_stall)
            iterations[live[done]] = it
            error[live] = err_mb
            converged[live[done]] = True
            keep = ~done
            live, u, c, R = live[keep], u[keep], c[keep], R[keep]
            if not len(live):
                break
            J = LN10 * np.matmul(s.nu.T, s.nu[None, :, :] * c[:, :, None])
            if charge_row is not None:
                J[:, charge_row] = LN10 * np.matmul((s.z * c)[:, None, :], s.nu)[:, 0, :]
            if na:
                n_a = n_a[keep]
                full = np.zeros((len(live), nc + na, nc + na))
                full[:, :nc, :nc] = J
                full[:, :nc, nc:] = nu_a.T
                if charge_row is not None:
                    full[:, charge_row, nc:] = charge_a
                full[:, nc:, :nc] = nu_a
                J = full
                R = np.concatenate((R, Rmin[keep]), axis=1)
            rs = np.maximum(np.abs(J).max(axis=2), 1e-300)
            Js = J / rs[:, :, None]
            cs = np.maximum(np.abs(Js).max(axis=1), 1e-300)
            Js /= cs[:, None, :]
            rhs = -R / rs
            try:
                dx = np.linalg.solve(Js, rhs[..., None])[..., 0] / cs
            except np.linalg.LinAlgError:
                dx = np.empty_like(rhs)
                for k in range(len(live)):
                    try:
                        dx[k] = np.linalg.solve(Js[k], rhs[k]) / cs[k]
                    except np.linalg.LinAlgError:
                        dx[k] = np.linalg.lstsq(Js[k], rhs[k], rcond=None)[0] / cs[k]
            big = np.maximum.reduce(np.abs(dx[:, :nc]), axis=1)
            small[live] = big < 0.5
            over = big > step_max
            if over.any():
                dx[over] = dx[over] * (step_max / big[over])[:, None]
            U[live] = u + dx[:, :nc]
            if na:
                n_a = n_a + dx[:, nc:]
                NA[live[:, None], act] = n_a
                if it >= 2:
                    negative = np.any(n_a < 0.0, axis=1)
                    iterations[live[negative]] = it
                    converged[live[negative]] = True
                    live = live[~negative]
        return U, LG, A, NA, iterations, error, converged

    def _update_active(self, active, n_act, si, tries):
        """Active set step of _outer_loop for one row (active, n_act, tries changed in place) -> whether it changed."""
        neg = [i for i in active if n_act[i] < 0.0]
        if neg:
            worst = min(neg, key=lambda i: n_act[i])
            active.remove(worst)
            n_act[worst] = 0.0
            return True
        cand = [i for i in range(len(self.s.minerals))
                if i not in active and si[i] > 1e-8 and tries.get(i, 0) < 2]
        if cand:
            best = max(cand, key=lambda i: si[i])
            active.append(best)
            tries[best] = tries.get(best, 0) + 1
            return True
        return False

    def _saturation_block(self, U, A):
        """Saturation indices of the minerals for every row."""
        s = self.s
        la_free = np.empty((len(A), 2))
        la_free[:, 0] = np.log10(A)
        la_free[:, 1] = -s.pe
        return self._rows(s.nu_min, U) + self._rows(s.nu_min_free, la_free) - s.logk_min

    def _outer_block(self, U, lg0, a0, NA, active, T, lg_x, charge_row, step_max):
        """_outer_loop of every row from its warm start (lg0, a0), active minerals in `active` (lists changed in place)
        -> (u, lg, a_h2o, I, n_min, iterations, error, converged)."""
        n = len(U)
        U, NA = U.copy(), NA.copy()
        I, LG, A = self._activities_block(self._concentrations_block(U, lg0, a0), lg_x)
        iterations = np.zeros(n, dtype=int)
        error = np.zeros(n)
        converged = np.zeros(n, dtype=bool)
        failed = np.zeros(n, dtype=bool)
        tries = [{} for _ in range(n)]
        live = np.arange(n)
        for _ in range(self.max_outer):
            if not len(live):
                break
            pending = live
            for _ in range(40):
                if not len(pending):
                    break
                groups = {}
                for i in pending:
                    groups.setdefault(tuple(active[i]), []).append(i)
                changed = []
                for act, rows in groups.items():
                    rows = np.array(rows, dtype=int)
                    u, lg, a, nm, it, err, ok = self._newton_block(
                        U[rows], LG[rows], A[rows], NA[rows], T[rows], None if lg_x is None else lg_x[rows],
                        charge_row, step_max, act)
                    iterations[rows] += it
                    error[rows] = err
                    failed[rows[~ok]] = True
                    rows, u, lg, a, nm = rows[ok], u[ok], lg[ok], a[ok], nm[ok]
                    U[rows], LG[rows], A[rows], NA[rows] = u, lg, a, nm
                    if not len(self.s.minerals):
                        continue
                    si = self._saturation_block(u, a)
                    for k, i in enumerate(rows):
                        if self._update_active(active[i], NA[i], si[k], tries[i]):
                            changed.append(i)
                pending = np.array(sorted(changed), dtype=int)
            live = live[~failed[live]]
            if not len(live):
                break
            offsets = None if lg_x is None else lg_x[live]
            lg = LG[live]
            I_new, lg_new, a_new = self._activities_block(self._concentrations_block(U[live], lg, A[live]), offsets)
            stable = np.maximum.reduce(np.abs(lg_new - lg), axis=1) < 1e-10
            LG[live], A[live], I[live] = lg_new, a_new, I_new
            converged[live[stable]] = True
            live = live[~stable]
        return U, LG, A, I, NA, iterations, error, converged

    def _attempt_block(self, T, cec, u0, lg0, a0, N0, charge_row):
        """First attempt of solve() (step max_step) of every row, minerals n0 = rows of N0
        -> (u, lg, a_h2o, I, n_min, iterations, error, converged, c, active)."""
        s = self.s
        lg_x = np.log10(self._nu_x_floor[None, :] / cec[:, None]) if s.exchanger else None
        active = [[i for i in range(len(s.minerals)) if N0[k, i] > 0] for k in range(len(N0))]
        U, LG, A, I, NA, iterations, error, converged = self._outer_block(u0, lg0, a0, N0, active, T, lg_x,
                                                                          charge_row, self.max_step)
        return U, LG, A, I, NA, iterations, error, converged, self._concentrations_block(U, LG, A), active

    def _charge_warning(self, T, u, charge_balance):
        """Whether solve() would warn about the charge imbalance of the totals (the node then goes through solve())."""
        if not charge_balance:
            return False
        q = float(np.dot(self.s.z_components, T))
        pH = -u[self.s.index["H+"]]
        return (pH < 2.5 or pH > 11.5) and abs(q) > 1e-3 * float(np.max(np.abs(T)))

    def _blocked(self, u, a_h2o, active):
        """Whether solve() would warn about supersaturated minerals left out of the active set."""
        s = self.s
        if not len(s.minerals):
            return False
        la_free = np.array([np.log10(a_h2o), -s.pe])
        si = s.nu_min @ u + s.nu_min_free @ la_free - s.logk_min
        return any(i not in active and si[i] > 1e-6 for i in range(len(s.minerals)))

    def _block_result(self, u, c, lg, I, a_h2o, n_min, iterations, error, newton, water_mass=1.0):
        """Result of one row of a block, as solve() builds it without gas."""
        s = self.s
        return Result(system=s, u=u.copy(), c=c.copy(), lg=lg.copy(),
                      n_min={m: float(n_min[i]) for i, m in enumerate(s.minerals)}, n_gas={}, p_gas={},
                      ionic_strength=float(I), a_h2o=float(a_h2o), iterations=int(iterations), error=float(error),
                      gas_volume=0.0, gas_z=1.0, water_mass=water_mass, timings=Timings(newton=int(newton)))

    def solve_many(self, totals: list, cec, u0, lg0, a_h2o0, minerals: list | None = None,
                   charge_balance: bool = True, water=None, water_mass0=None) -> list:
        """solve() of many nodes at once, each from its warm start (rows of u0, lg0, a_h2o0), without gas, imposed pH
        nor surface -> list of Result (same values as solve), None for a node that must go through solve() (invalid
        input, not converged at the first attempt, warning to emit)."""
        s = self.s
        n = len(totals)
        results = [None] * n
        if not n or len(s.gases) or s.surfaces or (s.exchanger and not s.is_exch.any()):
            return results
        keys = list(totals[0])
        if any(list(t) != keys for t in totals):
            return results
        try:
            cols = [s.index[s.db.master[k].species if k in s.db.master else k] for k in keys]
        except KeyError:
            return results
        if len(set(cols)) != len(cols):
            return results
        values = np.array([[t[k] for k in keys] for t in totals], dtype=float).reshape(n, len(keys))
        nm = len(s.minerals)
        N0 = np.array([[(m or {}).get(name, 0.0) for name in s.minerals] for m in (minerals or [None] * n)],
                      dtype=float).reshape(n, nm)
        cec = np.asarray(cec, dtype=float)
        u0, lg0, a_h2o0 = (np.asarray(x, dtype=float) for x in (u0, lg0, a_h2o0))
        live = (np.all(np.isfinite(u0), axis=1) & np.all(np.isfinite(lg0), axis=1) & np.isfinite(a_h2o0)
                & (a_h2o0 > 0) & (u0.shape[1] == s.nc) & (lg0.shape[1] == len(s.species)))
        jH, je = s.index["H+"], s.index.get("e-")
        jx = s.index[s.db.master[s.exchanger].species] if s.exchanger else None
        if jx is not None:
            live &= cec > 0
        checked = np.array([j for j in range(s.nc) if j != jH and j != je], dtype=int)
        charge_row = jH if charge_balance else None

        def totalsBlock(rows, w=None):
            T = np.zeros((len(rows), s.nc))
            T[:, cols] = values[rows] if w is None else values[rows] / w[:, None]
            if jx is not None:
                T[:, jx] = cec[rows] if w is None else cec[rows] / w
            if nm:
                T = T + self._rows(s.nu_min.T, N0[rows] if w is None else N0[rows] / w[:, None])
            return T

        def accepted(T, U, A, active):
            return np.array([not self._charge_warning(T[k], U[k], charge_balance)
                             and not self._blocked(U[k], A[k], active[k]) for k in range(len(T))], dtype=bool)

        live = np.flatnonzero(live)
        if water is None:
            T = totalsBlock(live)
            keep = ~np.any(T[:, checked] <= 0, axis=1)
            live, T = live[keep], T[keep]
            U, LG, A, I, NA, it, err, ok, C, active = self._attempt_block(T, cec[live], u0[live], lg0[live],
                                                                         a_h2o0[live], N0[live], charge_row)
            ok &= accepted(T, U, A, active)
            for k, i in enumerate(live):
                if ok[k]:
                    results[i] = self._block_result(U[k], C[k], LG[k], I[k], A[k], NA[k], it[k], err[k], it[k])
            return results

        W = np.array([float(water[k]) + sum(s.nu_min_free[i, 0] * (minerals[k] or {}).get(m, 0.0)
                                            for i, m in enumerate(s.minerals)) if minerals else float(water[k]) + 0
                      for k in range(n)])
        m0 = np.zeros(n) if water_mass0 is None else np.asarray(water_mass0, dtype=float)
        w = np.where(m0 > 0, m0, np.maximum(np.asarray(water, dtype=float) / H2O_PER_KG, 1e-6))
        U, LG, A = u0.copy(), lg0.copy(), a_h2o0.copy()
        newton = np.zeros(n, dtype=int)
        for _ in range(50):
            if not len(live):
                break
            T = totalsBlock(live, w[live])
            keep = ~np.any(T[:, checked] <= 0, axis=1)
            live, T = live[keep], T[keep]
            Ub, LGb, Ab, Ib, NAb, it, err, ok, C, active = self._attempt_block(
                T, cec[live] / w[live], U[live], LG[live], A[live], N0[live] / w[live][:, None], charge_row)
            ok &= accepted(T, Ub, Ab, active)
            nxt = []
            for k, i in enumerate(live):
                if not ok[k]:
                    continue
                newton[i] += it[k]
                n_min = np.array([float(NAb[k, j]) for j in range(nm)])
                bound = w[i] * (float(s.nu_free[:, 0] @ C[k]) + float(s.nu_min_free[:, 0] @ n_min))
                w_new = (W[i] - bound) / H2O_PER_KG
                if not w_new > 0:
                    continue
                if abs(w_new - w[i]) <= 1e-12 * w[i]:
                    results[i] = self._block_result(Ub[k], C[k], LGb[k], Ib[k], Ab[k], n_min * w[i], it[k], err[k],
                                                    newton[i], water_mass=float(w[i]))
                    continue
                w[i] = w_new
                U[i], LG[i], A[i] = Ub[k], LGb[k], Ab[k]
                nxt.append(i)
            live = np.array(nxt, dtype=int)
        return results

    def _newton(self, u, n_act, active, n_fug, n_tot, lg, a_h2o, T, uidx, eqidx,
                charge_row, ib, if_, target, mode, gas_volume, gas_pressure,
                bubble_on, lphi, Zgas, cec, step_max, tm):
        """Damped Newton with frozen activity coefficients and active set."""
        s = self.s
        act = np.array(active, dtype=int)
        nu_a = s.nu_min[act] if len(act) else np.zeros((0, s.nc))
        nu_f = s.nu_gas[if_] if len(if_) else np.zeros((0, s.nc))
        nu_b = s.nu_gas[ib] if len(ib) else np.zeros((0, s.nc))
        abs_nu = self._abs_nu
        vrt = gas_volume / (Zgas * R_GAS * s.tk) if mode == "V" else 0.0
        bubP = mode == "P" and bubble_on and len(ib) > 0
        n_extra = len(act) + len(if_) + (1 if bubP else 0)
        krow = list(eqidx).index(charge_row) if charge_row is not None else None
        best, stall, w = np.inf, 0, np.zeros(len(ib))
        small_step = False

        T_eff = T.copy()
        if len(act):
            T_eff = T_eff - nu_a.T @ n_act[act]
        if len(if_):
            T_eff = T_eff - nu_f.T @ n_fug
        rows = [j for j in eqidx if j != charge_row]
        u = self._revise_guess(u, lg, a_h2o, T_eff, rows)

        ixgrid = np.ix_(eqidx, uidx)
        nu_a_u = nu_a[:, uidx] if len(act) else np.zeros((0, len(uidx)))
        nu_f_u = nu_f[:, uidx] if len(if_) else np.zeros((0, len(uidx)))
        abs_nu_a = np.abs(nu_a) if len(act) else None
        abs_nu_f = np.abs(nu_f) if len(if_) else None
        abs_nu_b = np.abs(nu_b) if len(ib) else None
        nu_c = np.empty_like(s.nu)
        nfull = len(uidx) + n_extra
        Jf = np.zeros((nfull, nfull))
        Rf = np.zeros(nfull)
        if len(act):
            k0 = len(uidx)
            Jf[k0:k0 + len(act), :len(uidx)] = nu_a_u
        if len(if_):
            k0 = len(uidx) + len(act)
            Jf[k0:k0 + len(if_), :len(uidx)] = nu_f_u

        for it in range(1, self.max_newton + 1):
            ta = time.perf_counter() if self.timing else 0.0
            c = self._concentrations(u, lg, a_h2o)
            if self.inner_gamma and small_step:
                aq = c * s.is_aq
                lg = self._log_gamma(0.5 * float(np.dot(aq, self._z2)), aq, cec)
                a_h2o = water_activity(aq)
                c = self._concentrations(u, lg, a_h2o)
            la_free = np.array([np.log10(a_h2o), -s.pe])

            p_b = self._pressures(u, a_h2o, ib, lphi) if len(ib) else np.zeros(0)
            if mode == "V" and len(ib):
                n_b = p_b * vrt
            elif bubP:
                psum = max(float(p_b.sum()), 1e-300)
                w = p_b / psum
                n_b = n_tot * w
            else:
                n_b = np.zeros(len(ib))

            R = s.nu.T @ c - T
            scale = np.maximum(np.abs(T), abs_nu.T @ c)
            if len(act):
                R = R + nu_a.T @ n_act[act]
                scale = np.maximum(scale, abs_nu_a.T @ np.abs(n_act[act]))
            if len(if_):
                R = R + nu_f.T @ n_fug
                scale = np.maximum(scale, abs_nu_f.T @ np.abs(n_fug))
            if len(ib):
                R = R + nu_b.T @ n_b
                scale = np.maximum(scale, abs_nu_b.T @ n_b)
            R, scale = R[eqidx].copy(), np.maximum(scale[eqidx], 1e-30)
            if krow is not None:
                R[krow] = float(np.dot(s.z, c))
                scale[krow] = max(float(np.dot(np.abs(s.z), c)), 1e-30)

            Rmin = (nu_a @ u + s.nu_min_free[act] @ la_free - s.logk_min[act]
                    if len(act) else np.zeros(0))
            Rfug = ((nu_f @ u + s.nu_gas_free[if_] @ la_free - s.logk_gas[if_]
                     - lphi[if_]) - target if len(if_) else np.zeros(0))
            Rgas = (np.array([float(p_b.sum()) / gas_pressure - 1.0]) if bubP
                    else np.zeros(0))

            err_mb = float(np.maximum.reduce(np.abs(R) / scale, axis=None)) if len(R) else 0.0
            err_si = max([0.0]
                         + [float(np.maximum.reduce(np.abs(x), axis=None)) for x in (Rmin, Rfug, Rgas)
                            if len(x)])
            err = max(err_mb, err_si * self.tol / self.tol_si)
            if self.timing:
                tm.assembly += time.perf_counter() - ta
            if err_mb < self.tol and err_si < self.tol_si:
                tm.newton += it
                return u, n_act, n_fug, n_tot, it, err_mb, lg, a_h2o

            if err < best * (1.0 - 1e-3):
                best, stall = err, 0
            else:
                stall += 1
            if stall >= 10 and err < self.tol_stall:
                tm.newton += it
                return u, n_act, n_fug, n_tot, it, err_mb, lg, a_h2o

            ta = time.perf_counter() if self.timing else 0.0
            np.multiply(s.nu, c[:, None], out=nu_c)
            J = LN10 * (s.nu.T @ nu_c)
            if mode == "V" and len(ib):
                J = J + LN10 * (nu_b.T @ (nu_b * n_b[:, None]))
            elif bubP:
                J = J + nu_b.T @ (LN10 * n_b[:, None] * (nu_b - w @ nu_b))
            if krow is not None:
                J[charge_row] = LN10 * ((s.z * c) @ s.nu)
            J = J[ixgrid]

            if n_extra:
                nu_ = len(uidx)
                Jf[:nu_, :nu_] = J
                Rf[:nu_] = R
                k0 = nu_
                if len(act):
                    Jf[:nu_, k0:k0 + len(act)] = nu_a_u.T
                    if krow is not None:
                        Jf[krow, k0:k0 + len(act)] = nu_a @ s.z_components
                    Rf[k0:k0 + len(act)] = Rmin
                    k0 += len(act)
                if len(if_):
                    Jf[:nu_, k0:k0 + len(if_)] = nu_f_u.T
                    if krow is not None:
                        Jf[krow, k0:k0 + len(if_)] = nu_f @ s.z_components
                    Rf[k0:k0 + len(if_)] = Rfug
                    k0 += len(if_)
                if bubP:
                    Jf[:nu_, k0] = (nu_b.T @ w)[uidx]
                    if krow is not None:
                        Jf[krow, k0] = float((nu_b.T @ w) @ s.z_components)
                    Jf[k0, :nu_] = LN10 * (p_b @ nu_b)[uidx] / gas_pressure
                    Rf[k0] = Rgas[0]
                J, R = Jf, Rf
            if self.timing:
                tm.assembly += time.perf_counter() - ta

            tl = time.perf_counter() if self.timing else 0.0
            rs = np.maximum(np.abs(J).max(axis=1), 1e-300)
            Js = J / rs[:, None]
            cs = np.maximum(np.abs(Js).max(axis=0), 1e-300)
            Js /= cs
            try:
                dx = np.linalg.solve(Js, -R / rs) / cs
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(Js, -R / rs, rcond=None)[0] / cs
            if self.timing:
                tm.linalg += time.perf_counter() - tl
            tm.solves += 1

            big = float(np.maximum.reduce(np.abs(dx[:len(uidx)]), axis=None))
            small_step = big < 0.5
            if big > step_max:
                dx = dx * (step_max / big)
            u = u.copy()
            u[uidx] += dx[:len(uidx)]
            k = len(uidx)
            if len(act):
                n_act = n_act.copy()
                n_act[act] += dx[k:k + len(act)]
                k += len(act)
                if it >= 2 and np.any(n_act[act] < 0.0):
                    tm.newton += it
                    return u, n_act, n_fug, n_tot, it, err_mb, lg, a_h2o
            if len(if_):
                n_fug = n_fug + dx[k:k + len(if_)]
                k += len(if_)
            if bubP:
                n_tot = float(n_tot + dx[k])

        tm.newton += it
        if err_mb >= err_si:
            j = int(np.argmax(np.abs(R[:len(eqidx)]) / scale))
            where = f"mass balance of {s.components[eqidx[j]]}"
        elif len(act) and np.max(np.abs(Rmin)) >= err_si - 1e-30:
            where = f"saturation of {s.minerals[act[int(np.argmax(np.abs(Rmin)))]]}"
        else:
            where = "imposed partial pressure"
        exc = ConvergenceError(
            f"Newton not converged after {it} iterations : residual {err:.2e} "
            f"({where}) ; reduce max_step, check that every total is positive, or relax tol.")
        exc.u = u
        raise exc


def diagnose(system: ChemicalSystem, totals: dict, cec: float = 0.0,
             sites: dict | None = None, minerals: dict | None = None,
             pH: float | None = None) -> None:
    """Prints a full report of a case (changes nothing)."""
    s = system
    print(f"nativespeciation {__version__}   file {__file__}")
    print(f"database : {len(s.db.master)} master species, {len(s.db.species)} "
          f"species, {len(s.db.phases)} phases")
    print(f"temperature {s.tk:.2f} K, pe {s.pe}, redox {s.redox}")

    print("\ncomponents (charge) :",
          ", ".join(f"{n}({z:+.0f})" for n, z in zip(s.components, s.z_components)))
    for kind, label in (("aq", "aqueous"), ("exch", "exchange"), ("surf", "surface")):
        sel = [i for i, k in enumerate(s.kind) if k == kind]
        if not sel:
            continue
        print(f"\n{label} species ({len(sel)}) :")
        for i in sel:
            stoich = " ".join(f"{s.nu[i, j]:+g} {c}" for j, c in
                              enumerate(s.components) if s.nu[i, j])
            print(f"   {s.species[i]:14s} z={s.z[i]:+.0f} logK={s.logk[i]:+9.4f}"
                  f"  nu_X={s.nu_x[i]:g}   {stoich}")

    T = np.zeros(s.nc)
    for el, val in totals.items():
        name = s.db.master[el].species if el in s.db.master else el
        T[s.index[name]] += val
    if s.exchanger:
        T[s.index[s.db.master[s.exchanger].species]] = cec
    for site, moles in (sites or {}).items():
        T[s.index[s.db.master[site].species]] = moles
    n0 = np.array([(minerals or {}).get(m, 0.0) for m in s.minerals])
    if len(n0):
        T = T + s.nu_min.T @ n0

    print("\nsystem totals (mol/kgw) :")
    for j, n in enumerate(s.components):
        print(f"   {n:14s} T={T[j]:12.5e}  z*T={s.z_components[j] * T[j]:+12.5e}")
    q = float(np.dot(s.z_components, T))
    print(f"   {'sum z*T':14s}                {q:+12.5e}"
          f"   <- must be 0 when the pH is free")

    r = Speciation(s).solve(totals, pH=pH, cec=cec, sites=sites, minerals=minerals)
    print(f"\nresult : pH {r.pH:.4f}, I {r.ionic_strength:.4e}, "
          f"residual {r.error:.2e}, {r.iterations} iterations")
    print("   dissolved :", {e: f"{r.total(e):.4e}" for e in totals})
    if r.sorbed(1e-14):
        print("   sorbed    :", {k: f"{v:.4e}" for k, v in r.sorbed(1e-14).items()})
    if r.n_min:
        print("   minerals  :", {k: f"{v:.4e}" for k, v in r.n_min.items()})