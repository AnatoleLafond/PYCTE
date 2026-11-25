"""pycte._floatText writes each float64 as repr() writes it (and numpy astype(str), and f"{x}"), character for
character: the text files (outputs, inputTransport.txt of COMSOL) and the PhreeqC scripts are the same with it."""
import math
import os

import numpy as np
import pytest

try:
    from pycte import _floatText
except ImportError:                     # pycte installed without a C compiler: the Python formatting is used
    _floatText = None

pytestmark = pytest.mark.skipif(_floatText is None, reason="pycte._floatText not compiled")


def _values():
    rng = np.random.default_rng(2026)
    special = [0.0, -0.0, math.inf, -math.inf, math.nan, 5e-324, -5e-324, 2.2250738585072014e-308,
               2.225073858507201e-308, 1.7976931348623157e308, 1e16, 1e15, 9999999999999998.0, 1e-4, 1e-5,
               9.999999999999999e-5, 1e22, 1e23, 2.0 ** 53, 2.0 ** 53 + 2, 2.0 ** 63, 0.1, 0.2, 0.3, 1 / 3, 2 / 3,
               123456789012345678.0, 1234567890123456.7, 12345678901234567.0, 4.35, 0.1 + 0.2, 1e-7, 1e-300, 1e300]
    powers = []
    for e in range(-1074, 1024):
        x = math.ldexp(1.0, e)
        powers += [x, math.nextafter(x, 0), math.nextafter(x, math.inf)]
    for e in range(-323, 309):
        x = float(f"1e{e}")
        powers += [x, math.nextafter(x, 0), math.nextafter(x, math.inf)]
    bits = rng.integers(0, 2 ** 63, 300_000, dtype=np.int64).astype(np.uint64)
    bits |= rng.integers(0, 2, 300_000, dtype=np.uint64) << np.uint64(63)
    decimals = rng.integers(1, 10 ** 15, 100_000) * 10.0 ** rng.integers(-25, 25, 100_000)
    return np.concatenate([special, powers, -np.array(powers), bits.view(np.float64), decimals,
                           np.exp(rng.normal(0, 20, 100_000)), rng.integers(0, 2 ** 53, 50_000).astype(float)])


def test_texts_are_repr_and_numpy():
    values = _values()
    texts = _floatText.texts(values)
    assert texts == [repr(v) for v in values.tolist()]
    assert texts == values.astype(str).tolist()
    assert texts == [f"{v}" for v in values.tolist()]


def test_texts_of_a_strided_column():
    values = _values()[:30_000].reshape(-1, 3)
    for j in range(3):
        assert _floatText.texts(values[:, j]) == [repr(v) for v in values[:, j].tolist()]


def test_rows_are_the_joined_reprs():
    values = _values()
    values = values[np.isfinite(values)][:70_000].reshape(-1, 7)
    for array in (values, np.asfortranarray(values), values[:, ::2], values[:0]):
        expected = "".join("\t".join(map(repr, row)) + os.linesep for row in array.tolist())
        assert _floatText.rows(array, "\t", os.linesep) == expected


def test_only_float64_arrays():
    with pytest.raises(TypeError):
        _floatText.texts(np.arange(3, dtype=np.float32))
    with pytest.raises(TypeError):
        _floatText.rows(np.arange(3.0), "\t", "\n")
