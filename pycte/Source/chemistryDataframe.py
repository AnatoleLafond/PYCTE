"""Parallel chemical equilibrium of a DataFrame of nodes (species or element-total columns, others copied)."""

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
                              ConvergenceError, find_database, H2O_PER_KG)
from . import frames


@dataclass
class Config:
    """Chemical system and role of each DataFrame column (a field is a constant or a column name)."""
    database: object = "phreeqc.dat"

    elements: tuple = ()
    minerals: tuple = ()
    gases: dict = field(default_factory=dict)
    sorbed_species: tuple = ()
    cec: object = 0.0
    exchanger: str = "X"
    temperature: object = 298.15                # K, or the name of a column (K)
    ph: object = None
    ph_reagent: str | None = None
    diagnostics: dict = field(default_factory=dict)
    comp: bool = True
    species: tuple = ()
    totals_output: str = "dissolved"
    redox: bool = False
    pe: float = 4.0
    tol: float = 1e-10
    charge_balance: bool = False
    water: bool = False

    def uses_exchanger(self) -> bool:
        """Whether the chemical system includes the exchanger."""
        if not self.exchanger:
            return False
        if isinstance(self.cec, str):
            return True
        return (not self.comp) or float(self.cec) > 0.0

    def database_source(self):
        """Either a Database instance, or an absolute path to load (found once per name and working folder: the
        resolution, file system calls, took ~1 ms at every speciation call)."""
        if isinstance(self.database, Database):
            return self.database
        key = (type(self.database), str(self.database), os.getcwd())
        source = _SOURCES.get(key)
        if source is None:
            path = Path(self.database)
            if path.is_file():
                source = str(path.resolve())
            else:
                source = find_database(str(self.database))
            _SOURCES[key] = source
        return source


_SOURCES: dict = {}


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
    """Chemical system, cached per process and per temperature (0.01 K)."""
    tag = ("db_object", id(source)) if isinstance(source, Database) else source
    # everything the system and the solver are built from: exchanger as actually used (it depends on cec and on the
    # matrix encoding, not only on its name) and tolerance included
    key = (tag, cfg.elements, cfg.minerals, tuple(sorted(cfg.gases)),
           cfg.exchanger if cfg.uses_exchanger() else None, round(float(temperature), 2), cfg.redox, cfg.pe, cfg.tol)
    if key not in _CACHE:
        if isinstance(source, Database):
            db = source
        else:
            # database read once per process (setdefault(..., _load_database(source)) read it at every new key)
            if ("db", source) not in _CACHE:
                _CACHE[("db", source)] = _load_database(source)
            db = _CACHE[("db", source)]
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


_speciesCache: dict = {}


def _speciesOf(cfg: Config, system, columns):
    """_species_columns(cfg, system, columns), kept per chemical system (systems live in _CACHE for the whole process,
    the object is kept with the result) and per columns: the same list, without scanning system.species at every
    call."""
    key = (id(system), tuple(columns), cfg.species)
    found = _speciesCache.get(key)
    if found is None or found[0] is not system:
        found = _speciesCache[key] = (system, _species_columns(cfg, system, columns))
    return list(found[1])


_rowsCache: dict = {}


def _speciesRows(system, species):
    """(system.nu[[system.species.index(n) for n in species]], {n: system.species.index(n)}), kept per system and
    species: the same array (values and layout, only read) and the same positions, built once."""
    key = (id(system), tuple(species))
    found = _rowsCache.get(key)
    if found is None or found[0] is not system:
        found = _rowsCache[key] = (system, system.nu[[system.species.index(n) for n in species]],
                                   {n: system.species.index(n) for n in species})
    return found[1], found[2]


def _totals_from_species(system, names, values):
    """Species -> component totals: T_j = sum_i n_i nu_ij."""
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
    if cfg.water:
        want("H2O", "moles of the H2O component" if cfg.comp
             else "moles of free water")
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


def _equilibrate_chunk(cfg: Config, source, chunk: pd.DataFrame, u_start=None, want_totals=True):
    """Equilibrates the nodes of one chunk (warm start from u_start or the previous node). want_totals: build the frame
    of the component totals (only written to the speciation output), otherwise None in its place."""
    columns = chunk.columns.tolist()
    # unique float64 columns: the values read from one array (the columns of chunk, no Series built per column and per
    # call), the same values as chunk[name]
    fast = chunk.columns.is_unique and frames.allFloat64(chunk)
    base = chunk.to_numpy() if fast else None
    statuses, iterations, totals_rows = [], [], []
    u_rows = [None] * len(chunk)
    water_masses = np.full(len(chunk), np.nan)
    if u_start is not None:
        u_start = np.asarray(u_start, dtype=float)
        if u_start.ndim != 2 or len(u_start) != len(chunk):
            u_start = None

    written: dict = {}

    def slot(name):
        if name not in written:
            if fast:
                written[name] = base[:, chunk.columns.get_loc(name)].copy()
                return written[name]
            try:
                written[name] = chunk[name].to_numpy(dtype=float, copy=True)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"column {name!r} must be numeric to receive a result, "
                    f"its dtype is {chunk[name].dtype}") from exc
        return written[name]
    t_solve = t_numeric = t_init = 0.0
    u_previous = None
    species = None
    jH = jX = jE = None
    checked = False
    arrays: dict = {}
    nu_rows: dict = {}
    free_rows: dict = {}
    water: dict = {}
    position_of: dict = {}

    def cell(name, position):
        # fast: the column as Python floats (each value is float(chunk[name].to_numpy()[position]), the float every
        # caller takes of it)
        if name not in arrays:
            arrays[name] = base[:, chunk.columns.get_loc(name)].tolist() if fast else chunk[name].to_numpy()
        return arrays[name][position]

    def value(field_, position):
        if isinstance(field_, str) and field_ in columns:
            return cell(field_, position)
        return field_

    species_values = None
    finite_rows = None
    h_balance = (cfg.comp and not cfg.charge_balance and "H" in columns
                 and "H" not in cfg.elements)
    nodes = []
    mineral_cells = None
    # chemical system of each temperature of the chunk: _system (its cache key built at every call) once per value
    systems: dict = {}

    for position, idx in enumerate(chunk.index):
        temperature = float(value(cfg.temperature, position))
        if not temperature >= 200.0:
            raise ValueError(f"nativeSpeciation : temperature {temperature:g} K at node {idx} : the temperature is in "
                             f"K (column {cfg.temperature!r} or chemistry['temperature']), give "
                             f"{temperature + 273.15:g} for {temperature:g} °C")
        found = systems.get(temperature)
        if found is None:
            ref = time.perf_counter()
            found = systems[temperature] = _system(cfg, source, temperature)
            t_init += time.perf_counter() - ref
        system, solver = found
        if not checked:
            _check_columns(cfg, system, columns)
            checked = True
        if species is None and not cfg.comp:
            species = _speciesOf(cfg, system, columns)
            jH = system.index["H+"]
            jE = system.index.get("e-")
            jX = (system.index[system.db.master[cfg.exchanger].species]
                  if system.exchanger else None)
            positions = frames.indexer(chunk.columns, species) if fast and len(set(species)) == len(species) else None
            # the same C-ordered copy whatever the route (ascontiguousarray)
            species_values = np.ascontiguousarray(
                base[:, positions] if positions is not None else chunk[species].to_numpy(dtype=float))

        if mineral_cells is None:
            # the column of each mineral that cell() reads (arrays[m], made by its first call: the same error for a
            # missing column, at the same node), found once instead of one cell() call per mineral and per node
            for m in cfg.minerals:
                cell(m, position)
            mineral_cells = [(m, arrays[m]) for m in cfg.minerals]
        minerals = {m: float(column[position]) for m, column in mineral_cells}
        fugacities = {g: float(value(v, position))
                      for g, v in cfg.gases.items()}
        cec = float(value(cfg.cec, position) or 0.0)
        ph = value(cfg.ph, position)
        ph = None if ph is None or (isinstance(ph, float) and np.isnan(ph)) \
            else float(ph)
        reagent = cfg.ph_reagent

        if cfg.comp:
            totals = {e: float(cell(e, position)) for e in cfg.elements}
            if h_balance:
                totals["H"] = float(cell("H", position))
            if cfg.water:
                W = float(cell("H2O", position))
                water = dict(water=W, water_mass0=W / H2O_PER_KG)
        else:
            if id(system) not in nu_rows:
                nu_rows[id(system)], position_of[id(system)] = _speciesRows(system, species)
            vector = species_values[position].copy() @ nu_rows[id(system)]
            # the values as Python floats at once (tolist: float(vector[j]) of each)
            listed = vector.tolist()
            totals = {name: max(listed[j], 1e-20)
                      for j, name in enumerate(system.components)
                      if j != jH and j != jX and j != jE}
            if jE is not None:
                totals[system.components[jE]] = float(vector[jE])
            if not cfg.charge_balance:
                totals[system.components[jH]] = float(vector[jH])
            if jX is not None and cec <= 0.0:
                cec = float(vector[jX])
            if cfg.water:
                if id(system) not in free_rows:
                    free_rows[id(system)] = system.nu_free[
                        [system.species.index(n) for n in species], 0]
                free = float(cell("H2O", position))
                water = dict(water=free + float(species_values[position]
                                                @ free_rows[id(system)]),
                             water_mass0=free / H2O_PER_KG)
        if want_totals:
            totals_rows.append(dict(totals))    # read only for the frame of the totals (speciation output)
        start = None
        if u_start is not None and finite_rows is None:
            finite_rows = np.isfinite(u_start).all(axis=1).tolist()     # np.all(np.isfinite(row)) of each row
        if (u_start is not None and u_start.shape[1] == system.nc + 1 + len(system.species)
                and finite_rows[position]):
            start = (u_start[position, :system.nc], u_start[position, system.nc + 1:],
                     float(u_start[position, system.nc]))
        nodes.append(dict(system=system, solver=solver, totals=totals, cec=cec, ph=ph, reagent=reagent,
                          minerals=minerals, fugacities=fugacities, water=dict(water), start=start,
                          electroneutral=cfg.charge_balance or (cfg.comp and not h_balance)))

    ref = time.perf_counter()
    batched = _solve_block(cfg, nodes)
    t_solve += time.perf_counter() - ref

    species_targets = None
    species_rows: dict = {}
    mineral_targets = None
    aqueous: dict = {}
    # values written column by column once the nodes are done instead of value by value in the loop: the species
    # amounts r.c[rows] * r.water_mass of each node (not cfg.comp) and the mineral amounts r.n_min[m], the same values
    # in the same columns. The writes of a node go species, H2O, minerals, diagnostics (a later write of the same column
    # wins): a set is deferred only when no later write of the node shares one of its columns
    later = {c for c in cfg.diagnostics.values() if c in columns}
    defer_minerals = not (set(cfg.minerals) & later)
    defer_species = (not cfg.comp and species is not None
                     and not (set(species) & (later | set(cfg.minerals) | ({"H2O"} if cfg.water else set()))))
    done_positions, species_amounts, mineral_amounts = [], [], []
    for position, node in enumerate(nodes):
        system, totals = node['system'], node['totals']
        r = batched[position]
        if r is None:
            u_guess, lg_guess, a_guess = node['start'] or (u_previous, None, None)
            attempts = ([dict(u0=u_guess, lg0=lg_guess, a_h2o0=a_guess)]
                        if u_guess is not None else [])
            attempts += [dict(pe0=pe) for pe in _cold_pe(cfg, system, value, position)]

            ref = time.perf_counter()
            for attempt in attempts:
                try:
                    r = node['solver'].solve(totals, pH=node['ph'], pH_reagent=node['reagent'], cec=node['cec'],
                                             minerals=node['minerals'], fugacities=node['fugacities'],
                                             charge_balance=node['electroneutral'], **node['water'], **attempt)
                    break
                except (ConvergenceError, ValueError) as exc:
                    error = exc
            else:
                t_solve += time.perf_counter() - ref
                statuses.append(str(error).split('.')[0])
                iterations.append(0)
                u_previous = None
                continue
            t_solve += time.perf_counter() - ref
        t_numeric += r.timings.numeric
        u_previous = r.u
        u_rows[position] = np.concatenate((r.u, [r.a_h2o], r.lg))
        if cfg.water:
            water_masses[position] = r.water_mass
        statuses.append("")
        iterations.append(r.timings.newton)

        if cfg.comp:
            scale = r.water_mass if cfg.water else 1.0
            # r.total(e), its terms taken once per node (_aqueousTotals: the same products, the same sum)
            total = _aqueousTotals(r, aqueous) if cfg.totals_output != "system" or h_balance else None
            for e in cfg.elements:
                slot(e)[position] = (totals[e] if cfg.totals_output == "system"
                                     else total(e) * scale)
            if h_balance:
                slot("H")[position] = total("H") * scale
            if cfg.water:
                slot("H2O")[position] = scale * (
                    H2O_PER_KG + float(system.nu_free[:, 0] @ r.c))
            if cfg.sorbed_species:
                sorbed = r.sorbed()
                for s in cfg.sorbed_species:
                    if s in columns:
                        slot(s)[position] = sorbed.get(s, 0.0)
        else:
            # species amounts of the node: one product per row of the system (same values as species by species)
            if species_targets is None:
                species_targets = [slot(name) for name in species]
            rows = species_rows.get(id(system))
            if rows is None:
                where = position_of[id(system)]
                rows = species_rows[id(system)] = np.array([where[name] for name in species], dtype=np.intp)
            if defer_species:
                species_amounts.append(r.c[rows] * r.water_mass)
            else:
                for target, amount in zip(species_targets, (r.c[rows] * r.water_mass).tolist()):
                    target[position] = amount
            if cfg.water:
                slot("H2O")[position] = H2O_PER_KG * r.water_mass
        if mineral_targets is None:
            mineral_targets = [(m, slot(m)) for m in cfg.minerals]
        if defer_minerals:
            n_min = r.n_min
            mineral_amounts.append([n_min[m] for m in cfg.minerals])
        else:
            for m, target in mineral_targets:
                target[position] = r.n_min[m]
        done_positions.append(position)
        for quantity, column in cfg.diagnostics.items():
            if column in columns:
                slot(column)[position] = _diagnostic(r, quantity)

    # deferred values: each column receives its values at the nodes done, in the order of the columns (a name given
    # twice: its last column wins, as in the loop)
    if species_amounts or (mineral_amounts and mineral_targets):
        done = np.array(done_positions, dtype=np.intp)
        if species_amounts:
            amounts = np.array(species_amounts)
            for k, target in enumerate(species_targets):
                target[done] = amounts[:, k]
        if mineral_amounts and mineral_targets:
            # the amounts as float64 (each value converted as an assignment into the float64 column converts it)
            amounts = np.array(mineral_amounts, dtype=np.float64)
            for k, (m, target) in enumerate(mineral_targets):
                target[done] = amounts[:, k]

    if chunk.columns.is_unique and frames.allFloat64(chunk):
        full = chunk.to_numpy(dtype=float, copy=True)
        if written:
            # the columns written, at their positions (unique columns: the get_loc of each name), copied at once
            full[:, frames.indexer(chunk.columns, list(written))] = np.array(list(written.values())).T
        out = pd.DataFrame(full, index=chunk.index, columns=chunk.columns)
    else:
        out = chunk.copy()
        for name, values in written.items():
            out[name] = values

    width = next((len(u) for u in u_rows if u is not None), 0)
    u_out = np.full((len(chunk), width), np.nan)
    for position, u in enumerate(u_rows):
        if u is not None and len(u) == width:
            u_out[position] = u

    return (out, _totalsFrame(totals_rows, chunk.index) if want_totals else None,
            statuses, iterations, t_solve, t_numeric, t_init, u_out, water_masses)


def _aqueousTotals(r, cache):
    """r.total of a node as a function of the component: the dissolved total float(np.sum(nu[is_aq, j] * c[is_aq])) of
    Result.total, with the column nu[is_aq, j] found once per system (cache: {id(system): {component: column}}) and
    c[is_aq] once per node: the same products, summed by the same reduction (np.sum of an array is np.add.reduce)."""
    s = r.system
    mask = s.is_aq
    columns = cache.setdefault(id(s), {})
    c = r.c[mask]

    def total(element):
        column = columns.get(element)
        if column is None:
            j = s.index[s.db.master[element].species if element in s.db.master else element]
            column = columns[element] = s.nu[mask, j]
        return float(np.add.reduce(column * c, None))
    return total


def _totalsFrame(rows, index):
    """pd.DataFrame(rows, index=index) for a list of dicts; built from one float array when every row has the same keys
    in the same order and float values only (the DataFrame of a list of dicts converts each column from objects)."""
    if rows:
        keys = list(rows[0])
        if all(list(row) == keys for row in rows) and all(type(v) is float for row in rows for v in row.values()):
            return pd.DataFrame(np.array([list(row.values()) for row in rows], dtype=np.float64),
                                index=index, columns=keys)
    return pd.DataFrame(rows, index=index)


def _solve_block(cfg: Config, nodes):
    """Results of Speciation.solve_many for the nodes with a warm start (same system, no gas nor imposed pH); None for
    the other nodes, solved one by one."""
    found = [None] * len(nodes)
    rows = [k for k, node in enumerate(nodes) if node['start'] is not None]
    if (not rows or cfg.gases
            or any(node['ph'] is not None or node['reagent'] is not None for node in nodes)
            or any(node['solver'] is not nodes[0]['solver'] for node in nodes)):
        return found
    selected = [nodes[k] for k in rows]
    results = nodes[0]['solver'].solve_many(
        [node['totals'] for node in selected], [node['cec'] for node in selected],
        np.array([node['start'][0] for node in selected]), np.array([node['start'][1] for node in selected]),
        np.array([node['start'][2] for node in selected]), minerals=[node['minerals'] for node in selected],
        charge_balance=selected[0]['electroneutral'],
        water=[node['water']['water'] for node in selected] if cfg.water else None,
        water_mass0=[node['water']['water_mass0'] for node in selected] if cfg.water else None)
    for k, r in zip(rows, results):
        found[k] = r
    return found


def _cold_pe(cfg: Config, system, value, position):
    """Initial pe guesses of a node without previous solution."""
    if "e-" not in system.index:
        return [None]
    guesses = []
    try:
        pe = float(value(cfg.diagnostics.get("pe"), position))
        ph = float(value(cfg.diagnostics.get("pH"), position))
        if np.isfinite(pe) and np.isfinite(ph) and ph > 0:
            guesses.append(pe + ph - 7.0)
    except (TypeError, ValueError):
        pass
    for pe in (0.0, 3.0, -3.0, float(cfg.pe)):
        if all(abs(pe - g) > 0.5 for g in guesses):
            guesses.append(pe)
    return guesses


def _diagnostic(r, quantity: str) -> float:
    """Scalar quantity to report in an output column."""
    if quantity == "pH":
        return r.pH
    if quantity == "pe":
        return r.pe
    if quantity in ("I", "ionic_strength"):
        return r.ionic_strength
    if quantity == "a_H2O":
        return r.a_h2o
    if quantity == "water_mass":
        return r.water_mass
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
                          logfile: str | None = None, u_start=None, want_totals=True):
    """Equilibrates every row of df -> (DataFrame, info); a failed node keeps its row. want_totals: info
    ['component_totals'] built (only the speciation output writes it), otherwise None."""
    start = time.perf_counter()
    n_procs = max(1, int(n_procs))
    source = cfg.database_source()
    if u_start is not None:
        u_start = np.asarray(u_start, dtype=float)
        if u_start.ndim != 2 or len(u_start) != len(df):
            u_start = None

    if n_procs == 1:
        chunks = [_equilibrate_chunk(cfg, source, df, u_start, want_totals)]
    else:
        size = int(np.ceil(len(df) / n_procs))
        pieces = [(df.iloc[i:i + size], None if u_start is None else u_start[i:i + size])
                  for i in range(0, len(df), size)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_procs) as ex:
            chunks = list(ex.map(_task, [(cfg, source, p, u, want_totals) for p, u in pieces]))

    outs, totals, statuses, iterations, solves, numerics, inits, us, masses = zip(*chunks)
    if len(outs) == 1 and outs[0].columns is df.columns:
        # the frame of the chunk, built on its own array with the columns of df in their order: df.columns would take
        # a copy of it (same values, same layout)
        result = outs[0]
    else:
        result = (outs[0] if len(outs) == 1 else pd.concat(outs))[df.columns]
    component_totals = (totals[0] if len(totals) == 1 else pd.concat(totals)) if want_totals else None
    statuses = [s for block in statuses for s in block]
    iterations = [n for block in iterations for n in block]

    failures = [(i, s) for i, s in enumerate(statuses) if s]
    if logfile and failures:
        with open(logfile, "a", encoding="utf-8") as f:
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
        "init_time": float(max(inits)) if inits else 0.0,
        "component_totals": component_totals,
        "u": _stack_u(us),
        "water_mass": np.concatenate(masses),
    }
    info["interface_time"] = info["total_time"] - info["solve_time_wall"]
    return result, info


def _stack_u(blocks):
    """Solutions of the chunks -> one array (NaN rows for failed nodes): the array of a single chunk itself (its own,
    C-ordered as the copy np.vstack would make)."""
    width = max((b.shape[1] for b in blocks), default=0)
    if len(blocks) == 1 and blocks[0].flags.c_contiguous:
        return blocks[0]
    return np.vstack([b if b.shape[1] == width else np.full((len(b), width), np.nan)
                      for b in blocks]) if blocks else np.zeros((0, 0))


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