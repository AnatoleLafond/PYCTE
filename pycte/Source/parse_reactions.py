"""Parser of ORCHESTRA .inp databases: reactions, components (elements, sorption sites), stoichiometric matrix."""
import os
import re
import sys
import json

import pandas as pd

try:
    from . import warningManager
except ImportError:
    import warningManager


def split_top_level(s):
    """Splits on commas outside quotes and parentheses."""
    parts = []
    buf = []
    depth = 0
    in_quotes = False
    for ch in s:
        if ch == '"':
            in_quotes = not in_quotes
            buf.append(ch)
        elif in_quotes:
            buf.append(ch)
        elif ch == '(':
            depth += 1
            buf.append(ch)
        elif ch == ')':
            depth -= 1
            buf.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append(''.join(buf))
    return [p.strip() for p in parts]


def strip_comments(text):
    """Removes '//' comments outside quoted strings."""
    out, in_quotes, i, n = [], False, 0, len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            in_quotes = not in_quotes
            out.append(ch)
            i += 1
        elif not in_quotes and ch == '/' and i + 1 < n and text[i + 1] == '/':
            while i < n and text[i] != '\n':
                i += 1
        else:
            out.append(ch)
            i += 1
    return ''.join(out)


def parse_tagged_calls(text, tag):
    """Argument strings of every @<tag>(...) call."""
    calls = []
    for m in re.finditer(r'@' + re.escape(tag) + r'\s*\(', text):
        start = m.end()
        depth = 1
        i = start
        in_quotes = False
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == '"':
                in_quotes = not in_quotes
            elif not in_quotes:
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
            i += 1
        calls.append(text[start:i - 1])
    return calls


def parse_logkreaction_calls(text):
    """Backward-compatible alias."""
    return parse_tagged_calls(text, 'logKreaction')


def _to_number_if_possible(s):
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        return s


_texts = {}


def _read(filepath):
    """Text of the .inp file without its comments, read once per file version (path, modification time, size) instead
    of once per builder."""
    path = os.path.abspath(str(filepath))
    try:
        key = (path, os.path.getmtime(path), os.path.getsize(path))
    except OSError:
        key = None
    text = _texts.get(key) if key is not None else None
    if text is None:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            text = strip_comments(f.read())
        if key is not None:
            if len(_texts) >= 8:
                _texts.clear()
            _texts[key] = text
    return text


def build_reaction_dict(filepath):
    """{product: {reactant: coefficient, ...}} for every @logKreaction."""
    text = _read(filepath)
    reactions = {}
    for call in parse_logkreaction_calls(text):
        args = split_top_level(call)
        if len(args) < 2:
            continue
        name = args[0].strip()
        rest = args[2:]
        if len(rest) % 2 != 0:
            continue
        reactants = {}
        for i in range(0, len(rest), 2):
            coef_str = rest[i].strip()
            reactant = rest[i + 1].strip()
            try:
                coef = float(coef_str)
            except ValueError:
                coef = coef_str
            reactants[reactant] = coef
        reactions[name] = reactants
    return reactions


def build_primary_entities(filepath):
    """@primary_entity calls -> {element: arguments} (sites excluded)."""
    text = _read(filepath)
    primary_entities = {}
    for call in parse_tagged_calls(text, 'primary_entity'):
        args = split_top_level(call)
        if not args:
            continue
        name = args[0].strip()
        primary_entities[name] = [_to_number_if_possible(a) for a in args[1:]]
    return primary_entities


def build_adsorption_models(filepath):
    """@adsmodel calls -> {model: {phase, concentration, type}}."""
    text = _read(filepath)
    models = {}
    for call in parse_tagged_calls(text, 'adsmodel'):
        args = [a.strip() for a in split_top_level(call)]
        if not args:
            continue
        models[args[0]] = {
            'phase':         args[1] if len(args) > 1 else None,
            'concentration': _to_number_if_possible(args[2]) if len(args) > 2 else None,
            'type':          args[3] if len(args) > 3 else None,
        }
    return models


def build_surface_sites(filepath):
    """@surfsite calls -> {<model>_<site>: properties, with the 'total' site density}."""
    text = _read(filepath)
    models = build_adsorption_models(filepath)
    sites = {}
    for call in parse_tagged_calls(text, 'surfsite'):
        args = [a.strip() for a in split_top_level(call)]
        if len(args) < 2:
            continue
        model, site = args[0], args[1]
        nsites = _to_number_if_possible(args[2]) if len(args) > 2 else 1.0
        charge = _to_number_if_possible(args[3]) if len(args) > 3 else 0.0
        conc = models.get(model, {}).get('concentration')
        total = conc * nsites if isinstance(conc, float) and isinstance(nsites, float) \
            else None
        sites[f"{model}_{site}"] = {
            'model': model, 'site': site, 'nsites': nsites, 'charge': charge,
            'phase': models.get(model, {}).get('phase'),
            'concentration': conc, 'total': total,
        }
    return sites


def build_surface_species(filepath):
    """@surfspecies calls -> {sorbed species: properties}."""
    text = _read(filepath)
    out = {}
    for call in parse_tagged_calls(text, 'surfspecies'):
        args = [a.strip() for a in split_top_level(call)]
        if len(args) < 3:
            continue
        out[f"{args[0]}_{args[2]}"] = {
            'model': args[0], 'site': args[1], 'raw': args[2],
            'nsites': _to_number_if_possible(args[3]) if len(args) > 3 else 1.0,
        }
    return out


def build_components(filepath):
    """Every master unknown: @primary_entity elements and @surfsite sites."""
    comps = {}
    for name, args in build_primary_entities(filepath).items():
        comps[name] = {'kind': 'element', 'args': args}
    for name, info in build_surface_sites(filepath).items():
        comps[name] = dict(info, kind='site')
    for site, (col, factor) in site_column_map(comps).items():
        comps[site]['column'] = col
        comps[site]['factor'] = factor
    return comps


def site_column_map(components):
    """{site: (column, factor)}: a single-site model uses its model name and coef / nsites."""
    sites = {n: v for n, v in components.items() if v.get('kind') == 'site'}
    per_model = {}
    for name, info in sites.items():
        per_model.setdefault(info.get('model'), []).append(name)

    out = {}
    for model, names in per_model.items():
        if model and len(names) == 1 and model not in components:
            site = names[0]
            n = sites[site].get('nsites', 1.0)
            factor = 1.0 / n if isinstance(n, float) and n != 0 else 1.0
            out[site] = (model, factor)
        else:
            for site in names:
                out[site] = (site, 1.0)
    return out


def build_alias_map(filepath, extra=None):
    """PHREEQC -> ORCHESTRA names of the sorbed species (e.g. CaX2 -> Exch_X2-Ca)."""
    aliases = {}
    for orch, info in build_surface_species(filepath).items():
        site, raw = info['site'], info['raw']
        m = re.match(r'^(' + re.escape(site) + r'\d*)-(.+)$', raw)
        if m:
            aliases[m.group(2) + m.group(1)] = orch
        aliases.setdefault(raw, orch)
    for orch, info in build_surface_sites(filepath).items():
        s = info['site']
        for variant in (s, s + '-', s.lower(), s.lower() + '-'):
            aliases.setdefault(variant, orch)
    if extra:
        aliases.update(extra)
    return aliases


def _as_set(components):
    return set(components.keys()) if isinstance(components, dict) else set(components)


def find_implicit_primaries(reactions, components):
    """Reactants neither defined as products nor declared as components."""
    known = _as_set(components)
    return {r for reactants in reactions.values() for r in reactants
            if r not in known and r not in reactions}


def resolve_to_primary(species, reactions, primary_set, _cache=None, _stack=None,
                       on_unknown='primary', unknown_seen=None):
    """Species -> {component: coefficient}, substituting reactions recursively."""
    if _cache is None:
        _cache = {}
    if _stack is None:
        _stack = set()
    if unknown_seen is None:
        unknown_seen = set()

    if species in _cache:
        return _cache[species]

    if species in primary_set:
        return {species: 1.0}

    if species not in reactions:
        unknown_seen.add(species)
        if on_unknown == 'raise':
            raise KeyError(
                f"{species} is not a component (@primary_entity / @surfsite) "
                f"nor defined by @logKreaction."
            )
        result = {} if on_unknown == 'ignore' else {species: 1.0}
        _cache[species] = result
        return result

    if species in _stack:
        raise ValueError(f"Cyclic reaction definition detected at '{species}'")

    _stack.add(species)
    result = {}
    for reactant, coef in reactions[species].items():
        if not isinstance(coef, (int, float)):
            raise ValueError(
                f"Non-numeric coefficient for {species} -> {reactant}: {coef!r}"
            )
        sub = resolve_to_primary(reactant, reactions, primary_set, _cache, _stack,
                                 on_unknown=on_unknown, unknown_seen=unknown_seen)
        for prim, subcoef in sub.items():
            result[prim] = result.get(prim, 0.0) + coef * subcoef
    _stack.discard(species)

    _cache[species] = result
    return result


def resolve_reactions_to_primary(reactions, components, on_unknown='primary'):
    """resolve_to_primary for every species of `reactions`."""
    primary_set = _as_set(components)
    if on_unknown == 'primary':
        primary_set |= find_implicit_primaries(reactions, components)
    cache = {}
    return {s: resolve_to_primary(s, reactions, primary_set, cache,
                                  on_unknown=on_unknown)
            for s in reactions}


def component_list(reactions, components, on_unknown='primary'):
    """Matrix columns: declared components, then leftover roots."""
    cols = list(components)
    if on_unknown == 'primary':
        cols += sorted(find_implicit_primaries(reactions, components))
    return cols


def build_stoich_matrix(reactions, components, species_list, filepath=None,
                        aliases=None, on_unknown='primary', verbose=True,
                        site_columns='model'):
    """(species x components) stoichiometric matrix, rows in the order of `species_list`."""
    components = dict(components) if isinstance(components, dict) \
        else {c: {'kind': 'element'} for c in components}
    if filepath:
        for name, info in build_surface_sites(filepath).items():
            components.setdefault(name, dict(info, kind='site'))

    cols = component_list(reactions, components, on_unknown)
    resolved = resolve_reactions_to_primary(reactions, components, on_unknown)

    table = build_alias_map(filepath) if filepath else {}
    if aliases:
        table.update(aliases)

    smap = site_column_map(components) if site_columns == 'model' else {}
    model_cols = {col for col, _ in smap.values()} - set(cols)

    def to_columns(row):
        out = {}
        for prim, coef in row.items():
            col, f = smap.get(prim, (prim, 1.0))
            out[col] = out.get(col, 0.0) + coef * f
        return out

    rows, used_alias, missing = {}, {}, []
    for name in species_list:
        target = name if (name in cols or name in resolved or name in model_cols) \
            else table.get(name, name)
        if target != name:
            used_alias[name] = target
        if target in model_cols:
            rows[name] = {target: 1.0}
        elif target in cols:
            rows[name] = to_columns({target: 1.0})
        elif target in resolved:
            rows[name] = to_columns(
                {p: c for p, c in resolved[target].items() if p in cols})
        else:
            rows[name] = {}
            missing.append(name)

    final_cols = list(dict.fromkeys(smap.get(c, (c, 1.0))[0] for c in cols))
    stoich = pd.DataFrame(0.0, index=list(species_list), columns=final_cols)
    for name, row in rows.items():
        for prim, coef in row.items():
            stoich.loc[name, prim] = coef

    if verbose and used_alias:
        print("Alias PHREEQC -> ORCHESTRA : "
              + ", ".join(f"{k} -> {v}" for k, v in used_alias.items()))
    if verbose and missing:
        warningManager.warn("ORCHESTRA : not in ORCHESTRA database : "
                            + ", ".join(sorted(missing))
                            + "\n  -> Consider adding them in the '.inp' file (e.g., @species/@logKreaction, etc.) ?")
    stoich = stoich.loc[:, (stoich != 0).any(axis=0)]
    return stoich


build_stoich_matrixD = build_stoich_matrix



def align_stoich(stoich, species_list, component_list_, strict=True, verbose=True):
    """Stoichiometric matrix restricted to `species` and `components`, in that order."""
    species_list = list(species_list)
    component_list_ = list(component_list_)
 
    missing_rows = [s for s in species_list if s not in stoich.index]
    missing_cols = [c for c in component_list_ if c not in stoich.columns]
    if strict and (missing_rows or missing_cols):
        raise KeyError(
            f"Not in the species matrix : {missing_rows}, "
            f"components: {missing_cols}. Re-build this matriw with these species in species_list, or give strict=False for zeros values.")
 
    out = stoich.reindex(index=species_list, columns=component_list_,
                         fill_value=0.0)
 
    if verbose:
        dropped = [c for c in stoich.columns if c not in component_list_]
        lost = {c: float(stoich.loc[species_list, c].abs().sum())
                for c in dropped if c in stoich.columns}
        lost = {c: v for c, v in lost.items() if v > 0}
    return out
 

def sorbed_species(filepath):
    """Names of the sorbed species."""
    return list(build_surface_species(filepath))


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'chemistry1.inp'

    reactions = build_reaction_dict(path)
    components = build_components(path)

    print(json.dumps(reactions, indent=2, ensure_ascii=False))
    print(f"\n# {len(reactions)} reactions parsed", file=sys.stderr)

    elements = [c for c, v in components.items() if v['kind'] == 'element']
    sites = [c for c, v in components.items() if v['kind'] == 'site']
    print(f"# {len(elements)} elements : {elements}", file=sys.stderr)
    print(f"# {len(sites)} site(s)    : {sites}", file=sys.stderr)
    for s in sites:
        print(f"#   {s}: total = {components[s]['total']} mol/L "
              f"(phase {components[s]['phase']}, charge {components[s]['charge']})",
              file=sys.stderr)
    print(f"# sorbed species : {sorbed_species(path)}", file=sys.stderr)
    print(f"# racines non declarees : "
          f"{sorted(find_implicit_primaries(reactions, components))}", file=sys.stderr)

    resolved = resolve_reactions_to_primary(reactions, components)
    print(json.dumps(resolved, indent=2, ensure_ascii=False))

    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', None)
    print(build_stoich_matrix(reactions, components, list(reactions), filepath=path),
          file=sys.stderr)