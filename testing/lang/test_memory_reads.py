"""Runtime-index reads through the memory family (layer-memory).

Reads of a known-valued array at runtime indices are element placeholders the
layer lowers per target: a select chain, an inner product with the index's
equality bits, a multiplexer over the index bits, the backend's memory read,
or a table read (halo2-base: a dynamic lookup; Noir: a native array read).
These tests compile kernels with each construction forced
(`ZINNIA_FORCE_CONSTRUCTION`, a measurement override) and with the layer's
own choice, and check:

* the mock's outputs equal NumPy's, position by position, for every index in
  range (negative ones included where NumPy accepts them);
* an index NumPy rejects has no witness;
* halo2-base and Noir accept NumPy's result and reject a claim one ulp off.
"""
import os

import numpy as np
import pytest

from testing.lang.mock_outputs import compile_kernel, numpy_kernel, run_mock
from testing.lang.test_dtype_conversions import halo2_accepts, noir_accepts

CONSTRUCTIONS = [
    None,
    "gather: element by a select chain",
    "gather: element by indicators",
    "gather: element through memory",
    "memory: element by index bits",
    "memory: element by table read",
]

ELEMENT_1D = ('def kernel(x: Float[Array, "9"], i: Integer):\n'
              '    return x[i]\n')

ELEMENT_2D = ('def kernel(x: Float[Array, "4 5"], i: Integer, j: Integer):\n'
              '    return x[i, j]\n')

# 0450's shape, smaller: two index arrays broadcast over a full first axis.
PLANE_GATHER = ('def kernel(a: Float[Array, "3 4 5"], second: Integer[Array, "2"], third: Integer[Array, "2"]):\n'
                '    return a[:, np.array(second).reshape(-1, 1), third]\n')

FANCY_1D = ('def kernel(x: Float[Array, "12"], idx: Integer[Array, "6"]):\n'
            '    return x[idx]\n')

TAKE_ALONG = ('def kernel(a: Float[Array, "3 3 5"], b: Integer[Array, "3 3"]):\n'
              '    return np.take_along_axis(a, b[..., np.newaxis], axis=-1)[..., 0]\n')


def compiled(source, construction, backend="halo2"):
    """`source` compiled with `construction` forced (None: the layer's choice)."""
    from zinnia import ZKCircuit
    from zinnia.config.zinnia_config import ZinniaConfig

    saved = os.environ.get("ZINNIA_FORCE_CONSTRUCTION")
    os.environ["ZINNIA_FORCE_CONSTRUCTION"] = construction or ""
    try:
        return ZKCircuit.from_source("kernel", source, config=ZinniaConfig(backend=backend)).compile()
    finally:
        if saved is None:
            del os.environ["ZINNIA_FORCE_CONSTRUCTION"]
        else:
            os.environ["ZINNIA_FORCE_CONSTRUCTION"] = saved


def numpy_result(source, *args):
    try:
        return np.asarray(numpy_kernel(source)(*[np.copy(a) for a in args]), dtype=float).ravel()
    except IndexError:
        return None


def check(program, source, *args):
    expected = numpy_result(source, *args)
    satisfied, outputs = run_mock(program, *args)
    if expected is None:
        assert not satisfied, f"accepted where NumPy raises: {args[1:]}"
    else:
        assert satisfied, f"rejected: {args[1:]}"
        np.testing.assert_allclose(np.asarray(outputs, dtype=float), expected, atol=1e-9, rtol=0)


RNG = np.random.default_rng(5)


@pytest.mark.parametrize("construction", CONSTRUCTIONS)
def test_element_reads_match_numpy(construction):
    x1 = RNG.normal(size=9).round(3)
    program = compiled(ELEMENT_1D, construction)
    for i in range(-11, 11):
        check(program, ELEMENT_1D, x1, i)
    x2 = RNG.normal(size=(4, 5)).round(3)
    program = compiled(ELEMENT_2D, construction)
    for i in range(-5, 5):
        for j in (-6, -1, 0, 2, 4, 5):
            check(program, ELEMENT_2D, x2, i, j)


@pytest.mark.parametrize("construction", CONSTRUCTIONS)
def test_gathers_match_numpy(construction):
    a = RNG.normal(size=(3, 4, 5)).round(3)
    program = compiled(PLANE_GATHER, construction)
    for second, third in [([0, 3], [4, 0]), ([-1, 2], [-5, 1]), ([1, 4], [0, 0]), ([0, 0], [5, 1])]:
        check(program, PLANE_GATHER, a, np.array(second), np.array(third))
    x = RNG.normal(size=12).round(3)
    program = compiled(FANCY_1D, construction)
    for idx in ([0, 11, 5, 5, -1, -12], [3, 3, 3, 3, 3, 3], [0, 1, 2, 12, 4, 5]):
        check(program, FANCY_1D, x, np.array(idx))


@pytest.mark.parametrize("construction", CONSTRUCTIONS)
def test_take_along_axis_matches_numpy(construction):
    a = RNG.normal(size=(3, 3, 5)).round(3)
    program = compiled(TAKE_ALONG, construction)
    for b in (RNG.integers(0, 5, size=(3, 3)), np.full((3, 3), 4), np.zeros((3, 3), dtype=int)):
        check(program, TAKE_ALONG, a, b)
    bad = np.zeros((3, 3), dtype=int)
    bad[1, 2] = 5
    satisfied, _ = run_mock(program, a, bad)
    assert not satisfied


CLAIM = ('def kernel(x: Float[Array, "40"], idx: Integer[Array, "24"], expected: Float[Array, "24"]):\n'
         '    assert np.all(x[idx] == expected)\n')


@pytest.mark.parametrize("construction", ["memory: element by table read", "memory: element by index bits", None])
def test_backends_accept_numpy_and_reject_a_wrong_claim(construction):
    x = RNG.normal(size=40).round(3)
    idx = RNG.integers(-40, 40, size=24)
    good = x[idx]
    wrong = good.copy()
    wrong[7] += 2.0**-32
    halo2 = compiled(CLAIM, construction)
    assert halo2_accepts(halo2, x, idx, good)
    assert not halo2_accepts(halo2, x, idx, wrong)
    out_of_range = idx.copy()
    out_of_range[3] = 40
    assert not halo2_accepts(halo2, x, out_of_range, good)
    noir = compiled(CLAIM, construction, backend="noir")
    assert noir_accepts(noir, x, idx, good)
    assert not noir_accepts(noir, x, idx, wrong)
    assert not noir_accepts(noir, x, out_of_range, good)


# The NumPy index wrap (a negative index counts from the end; anything outside
# [-n, n) raises) is itself a layer choice: a pinned sign hint, or a pinned
# position one-hot whose hits double as the read's equality indicators.
WRAPS = ["memory: wrap by pinned sign", "memory: wrap by position one-hot"]

WRITE_THEN_READ = ('def kernel(x: Integer[Array, "5"], i: Integer, j: Integer, expected: Integer):\n'
                   '    y = np.copy(x)\n'
                   '    y[i] = 7\n'
                   '    assert y[j] == expected\n')


@pytest.mark.parametrize("construction", WRAPS)
def test_wraps_match_numpy(construction):
    x1 = RNG.normal(size=9).round(3)
    program = compiled(ELEMENT_1D, construction)
    for i in range(-11, 11):
        check(program, ELEMENT_1D, x1, i)
    a = RNG.normal(size=(3, 4, 5)).round(3)
    program = compiled(PLANE_GATHER, construction)
    for second, third in [([0, 3], [4, 0]), ([-1, -4], [-5, 1]), ([0, 0], [5, 1]), ([-5, 0], [0, 0])]:
        check(program, PLANE_GATHER, a, np.array(second), np.array(third))


@pytest.mark.parametrize("construction", WRAPS)
def test_wraps_pin_writes_and_reads_on_every_backend(construction):
    x = np.array([3, 1, 4, 1, 5])
    halo2 = compiled(WRITE_THEN_READ, construction)
    noir = compiled(WRITE_THEN_READ, construction, backend="noir")
    for i, j in [(0, 0), (-1, 4), (-5, 2), (2, -3), (4, -1)]:
        y = x.copy()
        y[i] = 7
        good, wrong = int(y[j]), int(y[j]) + 1
        assert halo2_accepts(halo2, x, i, j, good)
        assert not halo2_accepts(halo2, x, i, j, wrong)
        assert noir_accepts(noir, x, i, j, good)
        assert not noir_accepts(noir, x, i, j, wrong)
    for i, j in [(5, 0), (-6, 0), (0, 5), (0, -6)]:  # NumPy raises
        for claimed in (0, 3, 7):
            assert not halo2_accepts(halo2, x, i, j, claimed)
            assert not noir_accepts(noir, x, i, j, claimed)
