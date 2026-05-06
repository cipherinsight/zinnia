"""Regression tests for `compiler.ufunc-methods-add-outer`.

numpy ufunc methods like np.add.outer / np.subtract.outer are rewritten
in the Python transformer to a flat (np, "<op>_outer") dispatch.
"""
from zinnia import *


def test_np_add_outer_1d():
    @zk_circuit
    def foo(a: NDArray[Integer, 3], b: NDArray[Integer, 4]):
        out = np.add.outer(a, b)
        # out[i, j] = a[i] + b[j]
        assert out[0, 0] == 1 + 10
        assert out[2, 3] == 3 + 40

    assert foo(np.asarray([1, 2, 3]), np.asarray([10, 20, 30, 40]))


def test_np_subtract_outer_1d():
    @zk_circuit
    def foo(a: NDArray[Integer, 3], b: NDArray[Integer, 3]):
        out = np.subtract.outer(a, b)
        assert out[0, 0] == 1 - 10
        assert out[2, 2] == 3 - 30

    assert foo(np.asarray([1, 2, 3]), np.asarray([10, 20, 30]))


def test_np_multiply_outer_matches_np_outer():
    @zk_circuit
    def foo(a: NDArray[Integer, 3], b: NDArray[Integer, 3]):
        out_mul = np.multiply.outer(a, b)
        out_basic = np.outer(a, b)
        assert out_mul[1, 2] == out_basic[1, 2]

    assert foo(np.asarray([1, 2, 3]), np.asarray([4, 5, 6]))
