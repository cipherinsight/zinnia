"""Regression tests for `compiler.dynamic-shape-reshape-and-repeat`.

Reshape with runtime-int shape elements is supported when the source has
a statically-known total element count (the "implied bound" case): the
result is a DynamicNDArray with total_bound = source total, plus a
runtime assertion that prod(runtime_shape) == total_bound.
"""
import pytest

from zinnia import *


def test_reshape_static_to_runtime_dim():
    # Source has statically-known total (16). Target uses one runtime dim.
    @zk_circuit
    def foo(x: NDArray[Integer, 4, 4], rows: int):
        # rows must equal 4 for prod to match: 16 = rows * 4 → rows = 4.
        out = x.reshape(rows, 4)
        # The compile path must succeed; we don't index `out` further
        # because slicing with runtime-shape dyn-ndarrays is a follow-up
        # (see compiler.dyn-ndarray-slice-uses-runtime-shape).
        assert True

    arr = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8],
                      [9, 10, 11, 12], [13, 14, 15, 16]])
    assert foo(arr, 4)


def test_reshape_runtime_dim_inconsistent_fails():
    # Same shape source, but provide a runtime dim that doesn't match the
    # static total. The proof should be unsatisfiable.
    @zk_circuit
    def foo(x: NDArray[Integer, 4, 4], rows: int):
        out = x.reshape(rows, 4)
        assert True

    arr = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8],
                      [9, 10, 11, 12], [13, 14, 15, 16]])
    # rows=3 → prod = 12 != 16 → assert in circuit fails.
    result = foo(arr, 3)
    assert not result


def test_reshape_two_runtime_dims():
    @zk_circuit
    def foo(x: NDArray[Integer, 2, 6], a: int, b: int):
        out = x.reshape(a, b)
        assert True

    arr = np.asarray([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    assert foo(arr, 3, 4)
    assert foo(arr, 4, 3)
    # 5 * 2 = 10 != 12 → unsat
    assert not foo(arr, 5, 2)


def test_reshape_runtime_followed_by_full_slice():
    # Regression for compiler.dyn-ndarray-slice-uses-runtime-shape: after
    # reshape with runtime dims, a full-range slice + setitem must not
    # over-count target elements (was: panic with 4096 vs 16).
    @zk_circuit
    def foo(x: NDArray[Integer, 2, 4], a: int, b: int):
        out = x.reshape(a, b)
        out[:, :] = out[:, :]  # round-trip slice; compile path is the test
        assert True

    arr = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]])
    assert foo(arr, 2, 4)
    assert foo(arr, 4, 2)
    assert foo(arr, 8, 1)


def test_reshape_runtime_followed_by_partial_slice():
    @zk_circuit
    def foo(x: NDArray[Integer, 2, 4], a: int, b: int, k: int):
        out = x.reshape(a, b)
        # Partial slice along the second axis with a runtime stop.
        out[:, :k] = out[:, :k]
        assert True

    arr = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]])
    assert foo(arr, 2, 4, 2)
    assert foo(arr, 4, 2, 1)
