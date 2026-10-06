"""Runtime slice bounds follow Python's slice semantics.

`arr[i:j]` accepts any integer bounds: a negative bound counts from the end
of the axis, and every bound is clamped into `[0, len]` (`slice.indices`).
The circuit therefore computes NumPy's result for every pair of bounds, and
a proof binds that result: a public value other than NumPy's is rejected.

(This file used to pin the opposite policy: runtime bounds outside
`[0, len]` had no proof, although Python accepts them.)

The scalar baseline is unchanged: `arr[i]` with `i` outside `[-len, len)`
raises in NumPy and has no proof.
"""
import numpy as np
import pytest

from zinnia import ZKCircuit, requires, zk_circuit
from zinnia.lang.operator import Integer, NDArray

from testing.lang.mock_outputs import assert_matches_numpy

BN254 = 0x30644E72E131A029B85045B68181585D2833E84879B9709143E1F593F0000001
HALO2 = {"k": 12, "precision_bits": 32, "lookup_bits": 8}

X8 = np.array([-1, 10, 8, 4, -1, 2, 1, -10])

SUM_1D = 'def kernel(x: Integer[Array, "8"], i: Integer, j: Integer):\n    return np.sum(x[i:j])\n'
ROWS_2D = ('def kernel(x: Integer[Array, "4 3"], i: Integer, j: Integer):\n'
           '    return x[i:j, :].sum(axis=0)\n')
BOX_2D = ('def kernel(x: Integer[Array, "4 3"], i: Integer, j: Integer, k: Integer, l: Integer):\n'
          '    return x[i:j, k:l].sum()\n')
STEPPED = 'def kernel(x: Integer[Array, "8"], i: Integer, j: Integer):\n    return x[i:j:2]\n'

# Bounds inside [0, len], negative (counting from the end), and beyond
# either end (clamped), including empty and reversed ranges.
BOUNDS_1D = [(2, 5), (0, 8), (8, 8), (-1, 1), (-3, 8), (-3, -1), (2, 100), (-100, 3),
             (7, 2), (9, 12), (-2, -1), (3, -2), (-9, -8)]


def test_scalar_index_out_of_range_has_no_proof():
    """Scalar `arr[i]`: `-1` is NumPy's last element; `8` raises."""
    @zk_circuit
    def g(x: NDArray[Integer, 8], i: int):
        out = x[i]
        assert out >= -1000000

    assert g(X8, -1).satisfied is True
    assert g(X8, 8).satisfied is False


@pytest.mark.parametrize("i, j", BOUNDS_1D)
def test_runtime_slice_bounds_follow_python(i, j):
    assert_matches_numpy(SUM_1D, X8, i, j)


@pytest.mark.parametrize("i, j", [(1, 3), (-2, 4), (-9, 2), (3, 10), (2, 1)])
def test_runtime_row_slice_follows_python(i, j):
    assert_matches_numpy(ROWS_2D, np.arange(12).reshape(4, 3), i, j)


@pytest.mark.parametrize("i, j, k, l", [(1, 3, 0, 2), (-3, 9, -2, 3), (-9, -1, 1, -1), (2, 2, 0, 3)])
def test_runtime_box_slice_follows_python(i, j, k, l):
    assert_matches_numpy(BOX_2D, np.arange(12).reshape(4, 3) - 5, i, j, k, l)


@pytest.mark.parametrize("i, j", [(1, 8), (-5, 100), (-100, -2), (6, 3)])
def test_runtime_stepped_slice_follows_python(i, j):
    # A stepped slice has a runtime length: compare the leading elements.
    expected = X8[i:j:2]
    from testing.lang.mock_outputs import compile_kernel, run_mock_layout
    satisfied, values, shapes = run_mock_layout(compile_kernel(STEPPED), X8, i, j)
    assert satisfied
    assert list(shapes[0]) == [len(expected)]
    assert list(values[: len(expected)]) == list(expected)


@pytest.mark.parametrize("i, j", [(-1, 1), (2, 100), (-3, -1), (7, 2)])
def test_proof_binds_the_python_result(i, j):
    """halo2-base proves NumPy's sum for bounds outside `[0, len]`, and a
    proof whose public value is changed does not verify."""
    program = ZKCircuit.from_source("kernel", SUM_1D).compile()
    keys = program.setup(params=HALO2)
    proof = program.prove_with_keys(keys, X8, i, j)
    assert program.verify_with_keys(keys, proof)
    value = int.from_bytes(bytes.fromhex(proof["public_values"][0]), "little")
    signed = value - BN254 if value > BN254 // 2 else value
    assert signed == int(np.sum(X8[i:j]))
    tampered = dict(proof, public_values=list(proof["public_values"]))
    tampered["public_values"][0] = ((value + 1) % BN254).to_bytes(32, "little").hex()
    assert not program.verify_with_keys(keys, tampered)


def test_slice_in_range_with_requires_satisfied():
    """Annotated slice with a satisfying `@requires` discharges Proved;
    no witness check fires, program compiles and the assertion holds.
    """
    @zk_circuit
    @requires(lambda x, i, j: 0 <= i)
    @requires(lambda x, i, j: i < 8)
    @requires(lambda x, i, j: 0 < j)
    @requires(lambda x, i, j: j <= 8)
    @requires(lambda x, i, j: i < j)
    def f(x: NDArray[Integer, 8], i: int, j: int):
        sub = x[i:j]
        out = np.sum(sub)
        assert out >= -1000000

    r = f(np.array([1, 2, 3, 4, 5, 6, 7, 8]), 2, 5)
    assert r.satisfied is True
