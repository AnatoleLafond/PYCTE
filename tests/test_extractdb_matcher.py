"""extractDB._PrimaryMatcher reads the master species of a formula as the regular expression it replaced did:
'(?:' + '|'.join(re.escape(ps) for ps in sorted(primarySpecies, key=len, reverse=True)) + ')(?![a-z])'."""
import random
import re
from pathlib import Path

from pycte.Source import extractDB

EXAMPLES = Path(__file__).resolve().parents[1] / "pycte" / "Examples"


def _check(primary, formulas):
    old = re.compile('(?:' + '|'.join(re.escape(ps) for ps in sorted(primary, key=len, reverse=True)) + ')(?![a-z])')
    new = extractDB._PrimaryMatcher(primary)
    for formula in formulas:
        for i in range(len(formula)):
            m = old.match(formula, i)
            assert new(formula, i) == (m.group(0) if m else None), (primary, formula, i)


def test_databases_of_the_examples():
    rng = random.Random(7)
    paths = [EXAMPLES / "CationExchange" / "phreeqc.dat", *sorted((EXAMPLES / "CementClayInterface").glob("*.dat"))]
    for path in paths:
        db = extractDB.extract_master_species(path)
        primary = tuple(db['SOLUTION_MASTER_SPECIES'] + db['SURFACE_MASTER_SPECIES'] + db['EXCHANGE_SPECIES']
                        + db['EXCHANGE_MASTER_SPECIES'] + db['EXCHANGE_MASTER_ELEMENT'] + db['PHASES'])
        formulas = list(primary) + sorted(extractDB.extract_solution_species(path))
        alphabet = sorted(set("".join(formulas)))
        formulas += ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 10))) for _ in range(3000)]
        _check(primary, formulas)


def test_prefixes_duplicates_and_empty_names():
    rng = random.Random(11)
    alphabet = list("ABCSabcrin()+-0123_.")
    for k in range(200):
        primary = tuple("".join(rng.choice(alphabet) for _ in range(rng.randint(0 if k % 40 == 0 else 1, 4)))
                        for _ in range(rng.randint(0, 25)))
        formulas = ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 10))) for _ in range(200)]
        _check(primary, formulas)


def test_master_species_read_once_per_file_version():
    path = EXAMPLES / "CationExchange" / "phreeqc.dat"
    first, second = extractDB.extract_master_species(path), extractDB.extract_master_species(path)
    assert first == second
    first['PHASES'].append('changed by a caller')
    assert extractDB.extract_master_species(path) == second
