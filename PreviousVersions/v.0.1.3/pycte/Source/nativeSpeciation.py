"""
nativespeciation — solveur de speciation geochimique a l'equilibre (fichier unique).

Speciation aqueuse, phases minerales, phases gazeuses (fugacite imposee, bulle
a volume fixe, bulle a pression fixe, equation d'etat de Peng-Robinson),
echange d'ions (Gaines-Thomas) et complexation de surface non electrostatique.
Lit directement phreeqc.dat (generation 2 ou 3) ou une base JSON maison.

    from nativespeciation import Database, ChemicalSystem, Speciation

En cas de resultat inattendu :
    from nativespeciation import diagnose
    diagnose(system, totaux, ...)          # rapport complet a copier-coller

Chaque resultat porte le detail du temps de calcul dans r.timings.

Seule dependance : numpy. Le fichier de base thermodynamique doit etre fourni
par l'utilisateur ; find_database() le cherche dans le repertoire courant, a
cote du module, ou sous la variable NATIVESPECIATION_DATABASE.
"""

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
    """Flottant tolerant aux separateurs de phreeqc.dat ('1.506;')."""
    return float(x.strip().rstrip(";,").strip())


def parse_charge(name: str) -> int:
    """Charge d'une espece deduite de son nom ('Ca+2' -> 2, 'CO3-2' -> -2)."""
    m = _CHARGE_RE.search(name)
    if not m:
        return 0
    sign = 1 if m.group(1) == "+" else -1
    return sign * (int(m.group(2)) if m.group(2) else 1)


def split_terms(side: str) -> list[tuple[float, str]]:
    """Decoupe un membre de reaction : '2 H+ + CO3-2' -> [(2,'H+'), (1,'CO3-2')].

    On decoupe sur les '+' entoures d'espaces, pour ne pas casser les noms
    d'especes qui contiennent eux-memes un '+' (Ca+2, NaHCO3+, ...).
    """
    terms = []
    for chunk in re.split(r"\s+\+\s+", side.strip()):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = _TERM_RE.match(chunk)
        if not m:
            raise ValueError(f"terme illisible : {chunk!r}")
        coef = float(m.group(1)) if m.group(1) else 1.0
        terms.append((coef, m.group(2).strip()))
    return terms


def parse_reaction(line: str) -> dict[str, float]:
    """'CaCO3 = Ca+2 + CO3-2' -> {'CaCO3': -1, 'Ca+2': +1, 'CO3-2': +1}.

    Convention : produits positifs, reactifs negatifs, de sorte que
    log K = somme(coef * log a).
    """
    lhs, rhs = line.split("=", 1)
    stoich: dict[str, float] = {}
    for coef, name in split_terms(lhs):
        stoich[name] = stoich.get(name, 0.0) - coef
    for coef, name in split_terms(rhs):
        stoich[name] = stoich.get(name, 0.0) + coef
    return stoich


def find_database(name: str = "phreeqc.dat") -> str:
    """Localise un fichier de base thermodynamique.

    Cherche, dans l'ordre : le chemin tel quel, la variable d'environnement
    NATIVESPECIATION_DATABASE, le repertoire courant, celui de ce module.
    Aucune dependance exterieure : le fichier doit etre fourni par vous.
    """
    import os

    candidats = [Path(name)]
    env = os.environ.get("NATIVESPECIATION_DATABASE")
    if env:
        candidats.append(Path(env))
        candidats.append(Path(env) / name)
    candidats.append(Path.cwd() / name)
    candidats.append(Path(__file__).resolve().parent / name)
    for c in candidats:
        if c.is_file():
            return str(c.resolve())
    raise FileNotFoundError(
        f"base thermodynamique introuvable : {name}\nCherchee dans :\n  "
        + "\n  ".join(str(c) for c in candidats)
        + "\nIndiquer un chemin complet, ou definir la variable "
          "d'environnement NATIVESPECIATION_DATABASE.")


def default_phreeqc_database(name: str = "phreeqc.dat") -> str:
    """Ancien nom de find_database, conserve pour les scripts existants.

    Jusqu'a la version 2 ce nom allait chercher le fichier dans l'installation
    de phreeqpython. Cette dependance a ete retiree : la fonction cherche
    desormais le fichier comme find_database.
    """
    warnings.warn(
        "default_phreeqc_database est remplace par find_database ; il ne va "
        "plus chercher le fichier dans phreeqpython mais dans le repertoire "
        "courant, a cote du module, ou sous NATIVESPECIATION_DATABASE.",
        DeprecationWarning, stacklevel=2)
    return find_database(name)


@dataclass
class Species:
    """Espece secondaire (aqueuse, d'echange ou de surface)."""
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
    """Phase minerale ou gazeuse (reaction de dissolution).

    ATTENTION : stoich ne contient PAS le solide/gaz lui-meme, seulement le
    second membre. C'est indispensable pour les gaz, dont la formule porte le
    meme nom que l'espece aqueuse correspondante ('CO2 = CO2') : les deux
    s'annuleraient dans un dictionnaire commun.
    """
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
    """Espece maitresse associee a un element (ou a un site d'echange)."""
    element: str
    species: str
    kind: str = "aq"


def logk_at_T(log_k, delta_h, analytic, tk: float) -> float:
    """log K a la temperature tk (K) : expression analytique sinon van't Hoff."""
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
        """Extrait une base reduite aux elements demandes (utile pour figer
        une petite base maison a partir de phreeqc.dat)."""
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
    """Detecte un format d'option non reconnu : une reaction non triviale sans
    log K explicite est presque toujours le signe d'une ligne mal lue."""
    muets = [s.name for s in db.species.values()
             if not s.explicit_logk
             and len({k for k, v in s.stoich.items() if abs(v) > 1e-12}) > 1]
    muets += [f.name for f in db.phases.values() if not f.explicit_logk]
    if muets:
        warnings.warn(
            f"{len(muets)} reactions sans log K lisible dans {path} "
            f"(ex. : {', '.join(muets[:5])}). Leur log K est reste a 0, ce qui "
            f"fausserait completement la speciation. Format de mot-cle non "
            f"reconnu : ouvrir le fichier et verifier l'ecriture des lignes "
            f"log_k / analytic.", stacklevel=3)


A25, B25 = 0.51002, 0.32849


def water_density(tk: float) -> float:
    """Masse volumique de l'eau (g/cm3), correlation de Kell."""
    t = tk - 273.15
    return (1.0 - (t - 3.9863) ** 2 * (t + 288.9414)
            / (508929.2 * (t + 68.12963)))


def water_permittivity(tk: float) -> float:
    """Permittivite relative de l'eau."""
    return 2727.586 + 0.6224107 * tk - 466.9151 * math.log(tk) - 52000.87 / tk


def _raw_AB(tk: float) -> tuple[float, float]:
    rho, eps = water_density(tk), water_permittivity(tk)
    a = 1.82483e6 * math.sqrt(rho) / (eps * tk) ** 1.5
    b = 50.2916 * math.sqrt(rho) / math.sqrt(eps * tk)
    return a, b


_A0, _B0 = _raw_AB(298.15)


def dh_constants(tk: float = 298.15) -> tuple[float, float]:
    """Constantes A et B de Debye-Huckel a la temperature tk (K)."""
    a, b = _raw_AB(tk)
    return a * A25 / _A0, b * B25 / _B0


def ionic_strength(conc: np.ndarray, z: np.ndarray) -> float:
    return 0.5 * float(np.dot(conc, z * z))


def log_gamma(I: float, z: np.ndarray, dh_a: np.ndarray, dh_b: np.ndarray,
              tk: float = 298.15) -> np.ndarray:
    """log10 des coefficients d'activite (vectorise sur les especes).

    dh_a vaut NaN pour les especes sans parametres '-gamma'.
    I peut etre un tableau (une force ionique par noeud, forme (N,)) : le
    resultat est alors de forme (N, nombre d'especes).
    """
    A, B = dh_constants(tk)
    I = np.asarray(I, dtype=float)
    if I.ndim:
        I = I[..., None]
    sq = np.sqrt(np.maximum(I, 0.0))
    has_dh = ~np.isnan(dh_a)

    davies = -A * z ** 2 * (sq / (1.0 + sq) - 0.3 * I)
    lg = np.where(z == 0.0, 0.1 * I, davies)

    a = np.where(has_dh, np.nan_to_num(dh_a), 0.0)
    b = np.where(has_dh, np.nan_to_num(dh_b), 0.0)
    wateq = -A * z ** 2 * sq / (1.0 + B * a * sq) + b * I
    return np.where(has_dh, wateq, lg)


def water_activity(molalities: np.ndarray) -> float:
    """Activite de l'eau, approximation osmotique utilisee par PHREEQC.

    Bornee par le bas : sur un itere transitoire aberrant la somme des
    molalites peut depasser 1/0.017 et rendre l'activite negative, ce qui
    propagerait un NaN par son logarithme.
    """
    return max(1.0 - 0.017 * float(np.sum(molalities)), 1e-3)


R_ATM = 0.0820573660809596
SQ2 = math.sqrt(2.0)


def pr_fugacity(y: np.ndarray, P: float, tk: float, tc: np.ndarray,
                pc: np.ndarray, omega: np.ndarray) -> tuple[np.ndarray, float]:
    """Coefficients de fugacite (log10) et facteur de compressibilite.

    y  : fractions molaires, P : pression totale (atm), tk : temperature (K).
    Les gaz sans constantes critiques sont traites comme parfaits.
    """
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
                 exchanger=None, surfaces=None, temperature: float = 25.0,
                 pe: float = 4.0, redox: bool = False):
        """
        elements   : liste d'elements ('Ca', 'C', 'Na', 'Cl', ...)
        minerals   : liste de phases de la base ('Calcite', 'Gypsum', ...)
        gases      : liste de gaz de la base ('CO2(g)', 'O2(g)', ...) ; le mode
                     (fugacite imposee, volume fixe, pression fixe) se choisit
                     dans solve()
        exchanger  : nom de l'echangeur ('X') ou None
        surfaces   : liste de sites de surface ('Hfo_w', 'Hfo_s') ou None
        redox      : si False, toutes les especes faisant intervenir e- sont
                     ecartees (pas de couples redox)
        """
        self.db = db
        self.tk = temperature + 273.15
        self.pe = pe
        self.redox = redox

        masters: list[str] = []
        for el in elements:
            if el in db.master:
                sp = db.master[el].species
            elif el in db.species:
                sp = el
            else:
                raise KeyError(f"element inconnu dans la base : {el}")
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
            if not redox and abs(stoich.get("e-", 0.0)) > 1e-12:
                continue
            names.append(name)
            kinds.append(sp.kind)
            rows.append([stoich.get(m, 0.0) for m in masters])
            free_rows.append([stoich.get(f, 0.0) for f in self.free])
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
        """Vecteur stoechiometrique et logK effectif d'une phase.

        La reaction est ecrite en dissolution (le solide est a gauche) : le
        vecteur retourne donne aussi bien la composition du solide en
        composants que la stoechiometrie du produit ionique.
        """
        ph = self.db.phases[name]
        allowed = set(self.components) | set(self.free)
        stoich, lk = {}, ph.logk(self.tk)
        for r, coef in ph.stoich.items():
            res = self._expand(r)
            if res is None or not set(res[0]) <= allowed:
                raise ValueError(f"phase {name} hors du systeme choisi ({r})")
            for k, v in res[0].items():
                stoich[k] = stoich.get(k, 0.0) + coef * v
            lk -= coef * res[1]
        vec = [stoich.get(x, 0.0) for x in self.components]
        vfree = [stoich.get(f, 0.0) for f in self.free]
        return vec, vfree, lk

    def _expand(self, name: str):
        """Exprime l'espece 'name' dans la base des especes maitresses.

        Retourne (stoichiometrie sur les maitresses, logK cumule) ou None.
        """
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
        return (f"{self.nc} composants : {', '.join(self.components)}\n"
                f"{len(self.species)} especes "
                f"({int(self.is_aq.sum())} aq, {int(self.is_exch.sum())} echange, "
                f"{int(self.is_surf.sum())} surface)\n"
                f"{len(self.minerals)} mineraux : {', '.join(self.minerals) or '-'}\n"
                f"{len(self.gases)} gaz : {', '.join(self.gases) or '-'}")


def _g(v, default=float("nan")):
    return default if v is None else v


LN10 = np.log(10.0)
R_GAS = R_ATM

__version__ = "3.1"


class ConvergenceError(RuntimeError):
    pass


@dataclass
class Timings:
    """Decomposition du temps de calcul d'une resolution (secondes)."""
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
        """Temps de calcul pur : assemblage + algebre lineaire + activites."""
        return self.assembly + self.linalg + self.activity

    def __str__(self) -> str:
        return (f"total {1e3*self.total:7.3f} ms | pur {1e3*self.numeric:7.3f} ms "
                f"(assemblage {1e3*self.assembly:6.3f}, lineaire {1e3*self.linalg:6.3f}, "
                f"activites {1e3*self.activity:6.3f}) | {self.newton} iterations, "
                f"{self.solves} systemes {self.size}x{self.size}")


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
    timings: Timings = field(default_factory=Timings, repr=False)

    @property
    def p_total(self) -> float:
        return sum(self.p_gas.values())

    @property
    def pH(self) -> float:
        return -self.u[self.system.index["H+"]]

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
        """Total dissous d'un composant (mol/kgw), sorption exclue."""
        s = self.system
        j = s.index[s.db.master[element].species if element in s.db.master else element]
        return float(np.sum(s.nu[s.is_aq, j] * self.c[s.is_aq]))

    def activity_of(self, name: str) -> float:
        i = self.system.species.index(name)
        return float(self.c[i] * 10.0 ** self.lg[i])

    def si(self, phase: str) -> float:
        """Indice de saturation log10(IAP/K). Pour un gaz, c'est log10 P."""
        s = self.system
        vec, vfree, lk = s.phase_data(phase)
        la_free = np.array([np.log10(self.a_h2o), -s.pe])
        return float(np.dot(vec, self.u) + np.dot(vfree, la_free) - lk)

    def charge_balance(self) -> float:
        """Desequilibre de charge de la *solution* seule (mol/kgw).

        Il est nul en l'absence d'echangeur ou de surface ; sinon il est
        compense par la charge portee par le solide (couche diffuse ignoree
        dans le modele non electrostatique)."""
        s = self.system
        return float(np.sum(s.z[s.is_aq] * self.c[s.is_aq]))


class Speciation:
    def __init__(self, system: ChemicalSystem, tol: float = 1e-10,
                 tol_si: float = 1e-10, tol_stall: float = 1e-6,
                 max_newton: int = 200, max_outer: int = 60,
                 max_step: float = 1.0, timing: bool = True, eos: str = "pr",
                 inner_gamma: bool = True):
        """
        tol       : residu relatif vise sur les bilans de matiere
        tol_si    : residu vise sur les indices de saturation (unites log10)
        tol_stall : residu au-dela duquel une stagnation est une vraie erreur.
                    Entre tol et tol_stall, la stagnation est due a la
                    compensation entre termes de signes opposes (la precision
                    machine est atteinte) et la solution est acceptee.
        """
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

    def solve(self, totals: dict, pH: float | None = None,
              pH_reagent: str | None = None, cec: float = 0.0,
              sites: dict | None = None, minerals: dict | None = None,
              fugacities: dict | None = None, gas_initial: dict | None = None,
              gas_volume: float | None = None, gas_pressure: float | None = None,
              u0: np.ndarray | None = None) -> Result:
        """
        totals      : totaux du systeme par element (mol/kgw), fractions sorbee
                      et solide comprises
        pH          : impose le pH ; sinon il decoule de l'electroneutralite
        pH_reagent  : composant qui accompagne l'acide ou la base servant a
                      imposer le pH ('Cl' pour HCl, 'Na' pour NaOH). Son total
                      devient libre et l'electroneutralite est retablie, ce qui
                      reproduit exactement le Fix_H+ de PHREEQC. Sans lui, le
                      pH est impose sans contre-ion : la solution porte alors
                      un desequilibre de charge et la force ionique differe.
        cec         : capacite d'echange (mol de sites)
        sites       : moles de sites de surface ({'Hfo_w': 2e-3, ...})
        minerals    : moles de mineral initialement present ({'Calcite': 10.})
        fugacities  : log10 des pressions partielles imposees, reservoir infini
                      ({'CO2(g)': -3.5}) : equivalent d'un gaz place dans
                      EQUILIBRIUM_PHASES
        gas_volume  : volume (L) d'une bulle a volume fixe (GAS_PHASE
                      -fixed_volume) ; la pression totale est un resultat
        gas_pressure: pression totale (bar) d'une bulle a pression fixe
                      (GAS_PHASE -fixed_pressure) ; le volume est un resultat
        gas_initial : moles de gaz deja dans la bulle, ajoutees aux totaux
        u0          : initialisation (demarrage a chaud pour le transport)

        Les gaz declares dans le systeme et absents de 'fugacities' forment la
        bulle ; sans gas_volume ni gas_pressure ils sont ignores.
        """
        t0 = time.perf_counter() if self.timing else 0.0
        tm = Timings()
        s = self.s
        nc = s.nc
        jH = s.index["H+"]

        fug = dict(fugacities or {})
        for g in fug:
            if g not in s.gases:
                raise KeyError(f"gaz '{g}' absent du systeme : le declarer via "
                               f"ChemicalSystem(..., gases=[...])")
        if gas_volume is not None and gas_pressure is not None:
            raise ValueError("choisir gas_volume OU gas_pressure, pas les deux")
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
                raise ValueError("echangeur declare mais cec nulle : passer "
                                 "cec=<moles de sites> a solve()")
            if not s.is_exch.any():
                raise ValueError(
                    f"echangeur '{s.exchanger}' declare mais aucune espece "
                    f"d'echange trouvee dans la base : verifier que le bloc "
                    f"EXCHANGE_SPECIES est bien lu")
            T[s.index[s.db.master[s.exchanger].species]] = cec
        for site, moles in (sites or {}).items():
            T[s.index[s.db.master[site].species]] = moles

        T_sol = T.copy()
        n0 = np.array([(minerals or {}).get(m, 0.0) for m in s.minerals])
        if len(n0):
            T = T + s.nu_min.T @ n0
        for g, moles in (gas_initial or {}).items():
            T = T + moles * s.nu_gas[s.gases.index(g)]

        bad = [s.components[j] for j in range(nc) if j != jH and T[j] <= 0]
        if bad:
            raise ValueError(f"total nul ou negatif pour : {bad}")

        fixed_pH = pH is not None
        uidx = np.array([j for j in range(nc) if not (fixed_pH and j == jH)])
        eqidx = np.array([j for j in range(nc) if not (fixed_pH and j == jH)])
        if not fixed_pH:
            charge_row = jH
        elif pH_reagent is not None:
            name = (s.db.master[pH_reagent].species
                    if pH_reagent in s.db.master else pH_reagent)
            if name not in s.index:
                raise KeyError(f"pH_reagent '{pH_reagent}' absent des composants")
            charge_row = s.index[name]
        else:
            charge_row = None

        if u0 is not None:
            u = u0.copy()
        else:
            u = np.clip(np.log10(np.maximum(T_sol, 1e-9)) - 0.5, -9.0, 1.0)
            u[jH] = -7.0
            if s.exchanger:
                jx = s.index[s.db.master[s.exchanger].species]
                u[jx] = np.log10(max(cec, 1e-12)) - 6.0
        if fixed_pH:
            u[jH] = -pH

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
                    gas_volume, gas_pressure, cec, step, tm)
                break
            except ConvergenceError as exc:
                if essai == len(steps) - 1:
                    if pH_reagent is not None and hasattr(exc, "u"):
                        jr = charge_row
                        cr = float(np.sum(np.abs(s.nu[:, jr])
                                          * 10.0 ** (s.logk + s.nu @ exc.u)))
                        if cr < 1e-10 * float(np.max(np.abs(T))):
                            zr = s.z_components[jr]
                            oppose = [n for n, z in zip(s.components,
                                                        s.z_components)
                                      if z * zr < 0 and n != "H+"]
                            raise ValueError(
                                f"pH_reagent '{pH_reagent}' (charge {zr:+.0f}) "
                                f"ne peut pas retablir l'electroneutralite a pH "
                                f"{pH:.2f} : sa concentration s'annule, il en "
                                f"faudrait une quantite negative. Choisir un "
                                f"reactif de charge opposee"
                                + (f" (par exemple {oppose[0]})" if oppose
                                   else "") + ".") from exc
                    raise
                tm.retries += 1

        q = float(np.dot(s.z_components, T))
        pH_final = -u[jH]
        if (pH is None and (pH_final < 2.5 or pH_final > 11.5)
                and abs(q) > 1e-3 * float(np.max(np.abs(T)))):
            warnings.warn(
                f"pH {pH_final:.2f} obtenu avec un desequilibre de charge des "
                f"totaux de {q:+.3e} eq/kgw, entierement porte par H+/OH-. "
                f"Verifier que les totaux incluent la fraction sorbee "
                f"(convention : total = dissous + sorbe) et que la CEC est "
                f"compensee par des cations.", stacklevel=2)

        if len(s.minerals):
            la_free = np.array([np.log10(a_h2o), -s.pe])
            si = s.nu_min @ u + s.nu_min_free @ la_free - s.logk_min
            bloquees = [s.minerals[i] for i in range(len(s.minerals))
                        if i not in active and si[i] > 1e-6]
            if bloquees:
                warnings.warn(
                    f"phases sursaturees mais absentes de l'ensemble actif : "
                    f"{', '.join(bloquees)}. L'algorithme d'ensemble actif a "
                    f"cycle sur ces phases (competition sur les memes elements) "
                    f"et s'est arrete sur un compromis : le resultat n'est pas "
                    f"un equilibre exact pour elles.", stacklevel=2)

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

    def _outer_loop(self, u, n_act, active, n_fug, n_tot, bubble_on, T, uidx,
                    eqidx, charge_row, ib, if_, target, mode, gas_volume,
                    gas_pressure, cec, step_max, tm):
        """Point fixe sur les activites, englobant l'ensemble actif et Newton."""
        s = self.s
        lphi, Zgas = np.zeros(len(s.gases)), 1.0
        it_total, err = 0, 0.0
        c0 = self._concentrations(u, np.zeros(len(s.species)), 1.0)
        aq0 = c0 * s.is_aq
        I = ionic_strength(aq0, s.z)
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
            I = ionic_strength(aq, s.z)
            lg = self._log_gamma(I, aq, cec)
            a_h2o = water_activity(aq)
            lphi, Zgas = self._update_eos(u, a_h2o, ib, if_, target, mode,
                                          n_tot, gas_volume, lphi, Zgas)
            if self.timing:
                tm.activity += time.perf_counter() - ta
            if (np.max(np.abs(lg - lg_prev)) < 1e-10
                    and (not len(lphi) or np.max(np.abs(lphi - lphi_prev)) < 1e-12)):
                break
        else:
            raise ConvergenceError("coefficients d'activite non converges")


        return (u, n_act, n_fug, n_tot, active, bubble_on, lg, lphi, Zgas,
                a_h2o, I, it_total, err)

    def _pressures(self, u, a_h2o, idx, lphi=None) -> np.ndarray:
        """Pressions partielles (atm) des gaz d'indices idx.

        La loi d'action de masse porte sur la fugacite f = phi * p ; on retire
        donc log10(phi) pour obtenir la pression partielle.
        """
        s = self.s
        if not len(idx):
            return np.zeros(0)
        la_free = np.array([np.log10(a_h2o), -s.pe])
        lp = s.nu_gas[idx] @ u + s.nu_gas_free[idx] @ la_free - s.logk_gas[idx]
        if lphi is not None:
            lp = lp - lphi[idx]
        return 10.0 ** np.clip(lp, -300.0, 30.0)

    def _update_eos(self, u, a_h2o, ib, if_, target, mode, n_tot, gas_volume,
                    lphi, Z):
        """Met a jour les coefficients de fugacite (boucle externe, comme les
        coefficients d'activite). Les gaz a fugacite imposee sont traites purs
        a leur propre pression partielle, la bulle comme un melange a sa
        pression totale."""
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

    def _log_gamma(self, I, aq, cec) -> np.ndarray:
        """log10 des coefficients d'activite de toutes les especes.

        Aqueux   : Davies / Debye-Huckel etendu.
        Echange  : convention de Gaines-Thomas, a_i = gamma_i nu_X,i n_i / CEC,
                   ou gamma_i est calcule avec la charge du cation echange
                   (egale au nombre de sites nu_X,i), comme dans PHREEQC.
        Surface  : modele non electrostatique, a_i = n_i (gamma = 1).
        """
        s = self.s
        lg = log_gamma(I, s.z, s.dh_a, s.dh_b, s.tk)
        lg = np.where(s.is_aq, lg, 0.0)
        if s.exchanger and cec > 0:
            lg_x = (log_gamma(I, s.nu_x, s.dh_a, s.dh_b, s.tk)
                    + np.log10(np.maximum(s.nu_x, 1e-30) / cec))
            lg = np.where(s.is_exch, lg_x, lg)
        return lg

    def _revise_guess(self, u, lg, a_h2o, T_eff, rows, max_pass: int = 60,
                      marge: float = 20.0):
        """Pre-conditionnement des inconnues avant Newton.

        Le pas de Newton en logarithme ne corrige une surestimation que de
        1/ln10 = 0.43 decade par iteration : partir d'une espece 1e18 fois trop
        concentree demanderait 42 iterations rien que pour elle, et les
        composants couples derivent pendant ce temps. On applique donc d'abord
        un Newton diagonal en log, exact pour un composant porte par une seule
        espece, jusqu'a ramener chaque bilan dans un facteur 'marge'.

        La marge est large a dessein. T_eff ignore les transferts de phase
        encore inconnus (ce qu'un gaz va prendre a la solution, par exemple) :
        viser un facteur 2 ferait donc deriver une initialisation deja bonne,
        et ruinerait le demarrage a chaud d'un couplage.
        """
        s = self.s
        nu2 = s.nu * s.nu
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
        return 10.0 ** np.clip(lc, -300.0, 30.0)

    def _newton(self, u, n_act, active, n_fug, n_tot, lg, a_h2o, T, uidx, eqidx,
                charge_row, ib, if_, target, mode, gas_volume, gas_pressure,
                bubble_on, lphi, Zgas, cec, step_max, tm):
        """Newton amorti a coefficients d'activite et ensemble actif geles.

        Inconnues : u, moles des mineraux actifs, moles echangees avec les gaz
        a fugacite imposee, moles totales de la bulle a pression fixe.
        """
        s = self.s
        act = np.array(active, dtype=int)
        nu_a = s.nu_min[act] if len(act) else np.zeros((0, s.nc))
        nu_f = s.nu_gas[if_] if len(if_) else np.zeros((0, s.nc))
        nu_b = s.nu_gas[ib] if len(ib) else np.zeros((0, s.nc))
        abs_nu = np.abs(s.nu)
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
                lg = self._log_gamma(ionic_strength(aq, s.z), aq, cec)
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

            err_mb = float(np.max(np.abs(R) / scale)) if len(R) else 0.0
            err_si = max([0.0]
                         + [float(np.max(np.abs(x))) for x in (Rmin, Rfug, Rgas)
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

            big = float(np.max(np.abs(dx[:len(uidx)])))
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
            where = f"bilan sur {s.components[eqidx[j]]}"
        elif len(act) and np.max(np.abs(Rmin)) >= err_si - 1e-30:
            where = f"saturation de {s.minerals[act[int(np.argmax(np.abs(Rmin)))]]}"
        else:
            where = "pression partielle imposee"
        exc = ConvergenceError(
            f"Newton non converge apres {it} iterations : residu {err:.2e} "
            f"({where}). Pistes : reduire max_step, verifier que tous les "
            f"totaux sont strictement positifs, ou relacher tol.")
        exc.u = u
        raise exc


def diagnose(system: ChemicalSystem, totals: dict, cec: float = 0.0,
             sites: dict | None = None, minerals: dict | None = None,
             pH: float | None = None) -> None:
    """Rapport complet sur un cas : a lancer et a copier-coller en cas de
    resultat inattendu. N'imprime que des donnees, ne modifie rien."""
    s = system
    print(f"nativespeciation {__version__}   fichier {__file__}")
    print(f"base : {len(s.db.master)} especes maitresses, {len(s.db.species)} "
          f"especes, {len(s.db.phases)} phases")
    print(f"temperature {s.tk - 273.15:.1f} C, pe {s.pe}, redox {s.redox}")

    print("\ncomposants (charge) :",
          ", ".join(f"{n}({z:+.0f})" for n, z in zip(s.components, s.z_components)))
    for kind, label in (("aq", "aqueuses"), ("exch", "echange"), ("surf", "surface")):
        sel = [i for i, k in enumerate(s.kind) if k == kind]
        if not sel:
            continue
        print(f"\nespeces {label} ({len(sel)}) :")
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

    print("\ntotaux du systeme (mol/kgw) :")
    for j, n in enumerate(s.components):
        print(f"   {n:14s} T={T[j]:12.5e}  z*T={s.z_components[j] * T[j]:+12.5e}")
    q = float(np.dot(s.z_components, T))
    print(f"   {'somme z*T':14s}                {q:+12.5e}"
          f"   <- doit etre nul si le pH est libre")

    r = Speciation(s).solve(totals, pH=pH, cec=cec, sites=sites, minerals=minerals)
    print(f"\nresultat : pH {r.pH:.4f}, I {r.ionic_strength:.4e}, "
          f"residu {r.error:.2e}, {r.iterations} iterations")
    print("   dissous :", {e: f"{r.total(e):.4e}" for e in totals})
    if r.sorbed(1e-14):
        print("   sorbe   :", {k: f"{v:.4e}" for k, v in r.sorbed(1e-14).items()})
    if r.n_min:
        print("   mineraux:", {k: f"{v:.4e}" for k, v in r.n_min.items()})