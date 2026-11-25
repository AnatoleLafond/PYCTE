"""
Chemical equilibrium of a DataFrame of nodes, in parallel.

One node per row. The returned DataFrame has exactly the same columns, in the
same order: every column the configuration recognises is updated, the rest
(coordinates, pressure, flags) is copied through untouched.

Two ways of encoding the chemical state, selected by cfg.comp
-------------------------------------------------------------
* comp=True  : columns hold ELEMENT totals ('Na', 'Ca', 'C'). They are used as
               they are on input, and receive the dissolved total on output
               (or the whole system total, see cfg.totals_output).
* comp=False : columns hold SPECIES ('Na+', 'HCO3-', 'CaX2'). They are
               decomposed onto the components through the stoichiometric
               matrix to build the totals, then each column receives its own
               species concentration at equilibrium.
pycte sets comp = MultiCompoundTransport (True : totals transported, as in
xGEMS and PhreeqC ; False : species transported).

Conventions
-----------
* Element columns hold SYSTEM totals on input (dissolved + sorbed + solid).
  A fractional-step coupling then transports the dissolved part only, and reads
  the immobile inventory back from the mineral and sorbed-species columns.
* Mineral columns hold moles present, before and after.
* Gases are held at fixed fugacity (reservoir); the column carries log10 P.

Parallelism
-----------
The DataFrame is split into as many chunks as processes. Each worker builds the
chemical system once, then walks its nodes with a warm start: the solution of a
node initialises the next one, which cuts the cost by roughly three on
neighbouring cells.

On Windows processes are created by 'spawn': this module must be importable and
the call guarded by if __name__ == "__main__".
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import concurrent.futures
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .nativeSpeciation import (Database, ChemicalSystem, Speciation,
                              ConvergenceError, find_database)


@dataclass
class Config:
    """Chemical system description and role of each DataFrame column.

    Fields marked 'constant or column' accept either a number applied to every
    node, or the name of a DataFrame column for a per-node value.
    """
    database: object = "phreeqc.dat"

    elements: tuple = ()
    minerals: tuple = ()
    gases: dict = field(default_factory=dict)
    sorbed_species: tuple = ()
    cec: object = 0.0
    exchanger: str = "X"
    temperature: object = 25.0
    ph: object = None
    ph_reagent: str | None = None
    diagnostics: dict = field(default_factory=dict)
    comp: bool = True
    species: tuple = ()
    totals_output: str = "dissolved"
    redox: bool = False
    pe: float = 4.0
    tol: float = 1e-10

    def uses_exchanger(self) -> bool:
        """Whether the chemical system must include the exchanger.

        In element mode the CEC has to be stated, as a number or a column. In
        species mode it may be left at zero: it is then read back from the
        sorbed-species columns, but the exchange species must still be part of
        the system. Set exchanger=None to leave it out for good.
        """
        if not self.exchanger:
            return False
        if isinstance(self.cec, str):
            return True
        return (not self.comp) or float(self.cec) > 0.0

    def database_source(self):
        """Either a Database instance, or an absolute path to load."""
        if isinstance(self.database, Database):
            return self.database
        path = Path(self.database)
        if path.is_file():
            return str(path.resolve())
        return find_database(str(self.database))


def _value(field_, row, columns):
    """Resolve a 'constant or column' field for one node."""
    if isinstance(field_, str) and field_ in columns:
        return row[field_]
    return field_


def _load_database(path: str) -> Database:
    """PHREEQC text database, or JSON database written by Database.to_json."""
    if str(path).lower().endswith(".json"):
        return Database.from_json(path)
    return Database.from_phreeqc(path)


_CACHE: dict = {}


def _system(cfg: Config, source, temperature: float):
    """Chemical system, built once per process and per temperature.

    'source' is either an already built Database or a path to load. Equilibrium
    constants depend on temperature, so one system per distinct temperature,
    rounded to a hundredth of a degree so the cache is useful.
    """
    tag = ("db_object", id(source)) if isinstance(source, Database) else source
    key = (tag, cfg.elements, cfg.minerals, tuple(sorted(cfg.gases)),
           cfg.exchanger, round(float(temperature), 2), cfg.redox, cfg.pe)
    if key not in _CACHE:
        db = (source if isinstance(source, Database)
              else _CACHE.setdefault(("db", source), _load_database(source)))
        system = ChemicalSystem(db, elements=list(cfg.elements),
                                minerals=list(cfg.minerals),
                                gases=list(cfg.gases),
                                exchanger=(cfg.exchanger
                                           if cfg.uses_exchanger() else None),
                                temperature=float(temperature),
                                pe=cfg.pe, redox=cfg.redox)
        _CACHE[key] = (system, Speciation(system, tol=cfg.tol))
    return _CACHE[key]


def _species_columns(cfg: Config, system, columns):
    """DataFrame columns that name a species of the chemical system."""
    if cfg.species:
        missing = [c for c in cfg.species if c not in system.species]
        if missing:
            raise KeyError(f"species absent from the chemical system: {missing}")
        return list(cfg.species)
    return [c for c in columns if c in system.species]


def _totals_from_species(system, names, values):
    """Decompose species onto components: T_j = sum_i n_i nu_ij.

    Same operation as the primToSecSpecies matrix of xGEMS, here straight from
    the stoichiometric matrix the system already holds.
    """
    rows = [system.species.index(n) for n in names]
    return np.asarray(values, dtype=float) @ system.nu[rows]


def _check_columns(cfg: Config, system, columns):
    """Fail early, with the list of what is missing and what is available."""
    missing, role = [], {}

    def want(name, what):
        if name not in columns:
            missing.append(name)
            role[name] = what

    if cfg.comp:
        for e in cfg.elements:
            want(e, "element total")
    for m in cfg.minerals:
        want(m, "moles of mineral")
    for gas, value in cfg.gases.items():
        if isinstance(value, str):
            want(value, f"log10 P of {gas}")
    for field_, what in ((cfg.cec, "exchange capacity"),
                         (cfg.temperature, "temperature"),
                         (cfg.ph, "imposed pH")):
        if isinstance(field_, str):
            want(field_, what)

    if missing:
        detail = "\n  ".join(f"{n!r} ({role[n]})" for n in missing)
        hint = ""
        if cfg.comp and all(e not in columns for e in cfg.elements):
            hint = (
                "\nNot one of the element columns is there. Two ways out:\n"
                "  * the matrix carries species, not element totals: set\n"
                "    MultiCompoundTransport=False (comp=False) so the columns\n"
                "    are decomposed onto the components;\n"
                "  * the matrix carries totals under the master-species names:\n"
                "    declare them as they are, elements=('Ca+2','Cl-','Na+'...),\n"
                "    which the solver accepts just as well as element symbols.")
        raise KeyError(
            f"columns missing from the matrix:\n  {detail}\n"
            f"available columns:\n  {list(columns)}{hint}")

    if cfg.comp and cfg.exchanger and not cfg.uses_exchanger():
        warnings.warn(
            f"exchanger {cfg.exchanger!r} declared but no capacity given: it is "
            f"left out of the chemical system. Set cec to a value or to a "
            f"column name, or exchanger=None to silence this.", stacklevel=3)


def _equilibrate_chunk(cfg: Config, source, chunk: pd.DataFrame):
    """Equilibrate the nodes of one chunk. Returns the updated chunk."""
    out = chunk.copy()
    columns = list(chunk.columns)
    statuses, iterations, totals_rows = [], [], []

    written: dict = {}

    def slot(name):
        if name not in written:
            try:
                written[name] = out[name].to_numpy(dtype=float, copy=True)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"column {name!r} must be numeric to receive a result, "
                    f"its dtype is {out[name].dtype}") from exc
        return written[name]
    t_solve = t_numeric = 0.0
    u_previous = None
    species = None
    jH = jX = None
    checked = False

    for position, idx in enumerate(chunk.index):
        row = chunk.loc[idx]
        temperature = float(_value(cfg.temperature, row, columns))
        system, solver = _system(cfg, source, temperature)
        if not checked:
            _check_columns(cfg, system, columns)
            checked = True
        if species is None and not cfg.comp:
            species = _species_columns(cfg, system, columns)
            jH = system.index["H+"]
            jX = (system.index[system.db.master[cfg.exchanger].species]
                  if system.exchanger else None)

        minerals = {m: float(row[m]) for m in cfg.minerals}
        fugacities = {g: float(_value(v, row, columns))
                      for g, v in cfg.gases.items()}
        cec = float(_value(cfg.cec, row, columns) or 0.0)
        ph = _value(cfg.ph, row, columns)
        ph = None if ph is None or (isinstance(ph, float) and np.isnan(ph)) \
            else float(ph)
        reagent = cfg.ph_reagent

        if cfg.comp:
            totals = {e: float(row[e]) for e in cfg.elements}
        else:
            vector = _totals_from_species(system, species,
                                          row[species].to_numpy())
            totals = {name: max(float(vector[j]), 1e-20)
                      for j, name in enumerate(system.components)
                      if j != jH and j != jX}
            if jX is not None and cec <= 0.0:
                cec = float(vector[jX])
        totals_rows.append(dict(totals))

        ref = time.perf_counter()
        try:
            r = solver.solve(totals, pH=ph, pH_reagent=reagent, cec=cec,
                             minerals=minerals, fugacities=fugacities,
                             u0=u_previous)
        except (ConvergenceError, ValueError):
            try:
                r = solver.solve(totals, pH=ph, pH_reagent=reagent, cec=cec,
                                 minerals=minerals, fugacities=fugacities)
            except (ConvergenceError, ValueError) as exc:
                t_solve += time.perf_counter() - ref
                statuses.append(str(exc).split('.')[0])
                iterations.append(0)
                u_previous = None
                continue
        t_solve += time.perf_counter() - ref
        t_numeric += r.timings.numeric
        u_previous = r.u
        statuses.append("")
        iterations.append(r.timings.newton)

        if cfg.comp:
            for e in cfg.elements:
                slot(e)[position] = (totals[e] if cfg.totals_output == "system"
                                     else r.total(e))
            if cfg.sorbed_species:
                sorbed = r.sorbed()
                for s in cfg.sorbed_species:
                    if s in columns:
                        slot(s)[position] = sorbed.get(s, 0.0)
        else:
            for name in species:
                slot(name)[position] = r.c[system.species.index(name)]
        for m in cfg.minerals:
            slot(m)[position] = r.n_min[m]
        for quantity, column in cfg.diagnostics.items():
            if column in columns:
                slot(column)[position] = _diagnostic(r, quantity)

    for name, values in written.items():
        out[name] = values

    return (out, pd.DataFrame(totals_rows, index=chunk.index),
            statuses, iterations, t_solve, t_numeric)


def _diagnostic(r, quantity: str) -> float:
    """Scalar quantity to report in an output column."""
    if quantity == "pH":
        return r.pH
    if quantity in ("I", "ionic_strength"):
        return r.ionic_strength
    if quantity == "a_H2O":
        return r.a_h2o
    if quantity.startswith("SI:"):
        return r.si(quantity[3:])
    if quantity.startswith("P:"):
        return r.p_gas.get(quantity[2:], 0.0)
    if quantity.startswith("m:"):
        return r.molality(quantity[2:])
    if quantity == "iterations":
        return float(r.timings.newton)
    raise KeyError(f"unknown diagnostic: {quantity}")


def _task(args):
    """Worker entry point (must live at module level to be picklable)."""
    return _equilibrate_chunk(*args)


def equilibrate_dataframe(df: pd.DataFrame, cfg: Config, n_procs: int = 1,
                          logfile: str | None = None):
    """Equilibrate every row of df. Returns (DataFrame, info).

    The returned DataFrame has the same columns in the same order. Nodes that
    failed to converge keep their input row and are reported in info.
    """
    start = time.perf_counter()
    n_procs = max(1, int(n_procs))
    source = cfg.database_source()

    if n_procs == 1:
        chunks = [_equilibrate_chunk(cfg, source, df)]
    else:
        size = int(np.ceil(len(df) / n_procs))
        pieces = [df.iloc[i:i + size] for i in range(0, len(df), size)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_procs) as ex:
            chunks = list(ex.map(_task, [(cfg, source, p) for p in pieces]))

    outs, totals, statuses, iterations, solves, numerics = zip(*chunks)
    result = pd.concat(outs)[df.columns]
    component_totals = pd.concat(totals)
    statuses = [s for block in statuses for s in block]
    iterations = [n for block in iterations for n in block]

    failures = [(i, s) for i, s in enumerate(statuses) if s]
    if logfile and failures:
        with open(logfile, "a") as f:
            for i, s in failures:
                f.write(f"node {i}, PID {os.getpid()}: {s}\n")

    info = {
        "nodes": len(df),
        "failures": len(failures),
        "failure_details": failures[:20],
        "mean_iterations": float(np.mean(iterations)) if iterations else 0.0,
        "iterations": iterations,
        "statuses": statuses,
        "total_time": time.perf_counter() - start,
        "solve_time_cpu": float(sum(solves)),
        "solve_time_wall": float(max(solves)) if solves else 0.0,
        "numeric_time": float(sum(numerics)),
        "component_totals": component_totals,
    }
    info["interface_time"] = info["total_time"] - info["solve_time_wall"]
    return result, info


if __name__ == "__main__":
    N = 400
    x = np.linspace(0.0, 1.0, N)

    DATABASE = "phreeqc.dat"
    db = Database.from_phreeqc(find_database(DATABASE))

    df_elements = pd.DataFrame({
        "x": x, "y": 0.0,
        "Na": 1e-2 + 5e-3 * x, "Ca": 1e-3 * (1.0 + x),
        "Cl": 1.2e-2 + 5e-3 * x, "C": 2e-3 * np.ones(N), "S": 1e-3 * np.ones(N),
        "Calcite": 1e-2 * np.ones(N), "CO2(g)": -3.5 + 0.5 * x,
        "CEC": 1e-3 * np.ones(N), "NaX": 0.0, "CaX2": 0.0, "pH": 0.0, "I": 0.0,
    })
    cfg_elements = Config(
        database=db,
        comp=True,
        elements=("Na", "Ca", "Cl", "C", "S"),
        minerals=("Calcite",),
        gases={"CO2(g)": "CO2(g)"},
        sorbed_species=("NaX", "CaX2"),
        cec="CEC",
        diagnostics={"pH": "pH", "I": "I"},
    )

    for n_procs in (1, 4):
        out_e, info = equilibrate_dataframe(df_elements, cfg_elements,
                                            n_procs=n_procs,
                                            logfile="chemistry.log")
        print(f"\ncomp=True, {n_procs} process(es): {info['nodes']} nodes, "
              f"{info['failures']} failure(s), "
              f"{info['mean_iterations']:.1f} iterations on average")
        print(f"  total {1e3 * info['total_time']:8.1f} ms | "
              f"solve wall {1e3 * info['solve_time_wall']:7.1f} ms | "
              f"numeric {1e3 * info['numeric_time']:7.1f} ms | "
              f"interface {1e3 * info['interface_time']:6.1f} ms")
        print(f"  same columns, same order: "
              f"{list(out_e.columns) == list(df_elements.columns)}")

    species = ["H+", "OH-", "Na+", "Ca+2", "Cl-", "CO2", "HCO3-", "CO3-2",
               "SO4-2", "CaSO4", "NaSO4-", "CaHCO3+", "CaCO3", "NaHCO3",
               "NaCO3-", "CaOH+", "NaOH", "NaX", "CaX2"]
    cfg_species = Config(
        database=DATABASE,
        comp=False,
        elements=("Na", "Ca", "Cl", "C", "S"),
        minerals=("Calcite",),
        gases={"CO2(g)": "CO2(g)"},
        cec=0.0,
        diagnostics={"pH": "pH", "I": "I"},
    )

    system, solver = _system(cfg_species, cfg_species.database_source(), 25.0)
    rows = []
    for i in range(N):
        r = solver.solve(
            {e: df_elements.at[i, e] for e in cfg_elements.elements},
            cec=df_elements.at[i, "CEC"],
            minerals={"Calcite": df_elements.at[i, "Calcite"]},
            fugacities={"CO2(g)": df_elements.at[i, "CO2(g)"]})
        rows.append([r.c[system.species.index(n)] for n in species])
    df_species = pd.DataFrame(rows, columns=species)
    df_species.insert(0, "y", 0.0)
    df_species.insert(0, "x", x)
    df_species["Calcite"] = 1e-2
    df_species["CO2(g)"] = -3.5 + 0.5 * x
    df_species["pH"] = 0.0
    df_species["I"] = 0.0

    out_s, info = equilibrate_dataframe(df_species, cfg_species, n_procs=1)
    print(f"\ncomp=False, 1 process: {info['nodes']} nodes, "
          f"{info['failures']} failure(s), "
          f"{info['mean_iterations']:.1f} iterations on average, "
          f"{1e3 * info['total_time']:.0f} ms")
    print(f"  same columns, same order: "
          f"{list(out_s.columns) == list(df_species.columns)}")

    gap_ph = float(np.max(np.abs(out_s["pH"] - out_e["pH"])))
    gap_na = float(np.max(np.abs(out_s["Na+"] - df_species["Na+"])
                          / df_species["Na+"]))
    print(f"  pH gap between the two encodings: {gap_ph:.2e}")
    print(f"  relative drift of Na+ (already equilibrated state): {gap_na:.2e}")

    print("\nExcerpt, species mode")
    print(out_s.iloc[::133, :10].to_string(
        float_format=lambda v: f"{v:.4e}"))