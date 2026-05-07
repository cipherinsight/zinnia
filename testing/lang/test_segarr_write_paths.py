"""Regression tests for `compiler.segarr-write-paths` (P3 of the
segment-native-static-arrays epic).

These exercise native `Value::StaticArray` write paths: element setitem at
static and dynamic indices, multi-dim subscripts (`arr[i, j]` and
`arr[i][j]`), slice setitem (static and runtime bounds), augmented
assignment, and the headline tight-loop benchmark that the legacy O(N) mux
chain used to time out on.
"""

import time

import pytest
from zinnia import *


# ───────────────────────── 1-D element setitem ───────────────────────────

def test_static_array_1d_setitem_static_index():
    @zk_circuit
    def foo():
        a = np.asarray([10, 20, 30, 40])
        a[2] = 99
        assert a[0] == 10
        assert a[1] == 20
        assert a[2] == 99
        assert a[3] == 40
    assert foo()


def test_static_array_1d_setitem_dynamic_index():
    @zk_circuit
    def foo(j: Integer, v: Integer):
        a = np.asarray([10, 20, 30, 40])
        a[j] = v
        # Read the updated cell back at the same dynamic index.
        assert a[j] == v
    assert foo(2, 999)
    assert foo(0, -7)


def test_static_array_1d_setitem_negative_index():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4, 5])
        a[-1] = 100
        a[-2] = 50
        assert a[4] == 100
        assert a[3] == 50
    assert foo()


# ───────────────────────── 2-D element setitem ───────────────────────────

def test_static_array_2d_tuple_setitem():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        a[0, 0] = 100
        a[1, 2] = 200
        assert a[0, 0] == 100
        assert a[0, 1] == 2
        assert a[1, 2] == 200
        assert a[0, 2] == 3
    assert foo()


def test_static_array_2d_chained_setitem():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        a[0][0] = 100
        a[1][2] = 200
        assert a[0, 0] == 100
        assert a[1, 2] == 200
        assert a[0, 1] == 2
    assert foo()


def test_static_array_2d_dynamic_inner_setitem():
    @zk_circuit
    def foo(j: Integer):
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        a[1, j] = 99
        assert a[1, j] == 99
    assert foo(0)
    assert foo(2)


# ───────────────────────── slice setitem ─────────────────────────────────

def test_static_array_2d_slice_static_bounds():
    @zk_circuit
    def foo():
        a = np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        a[1:3, :] = np.asarray([[10, 20, 30], [40, 50, 60]])
        assert a[0, 0] == 0
        assert a[1, 0] == 10
        assert a[1, 2] == 30
        assert a[2, 1] == 50
        assert a[3, 0] == 0
    assert foo()


def test_static_array_2d_slice_runtime_bound():
    @zk_circuit
    def foo(k: Integer):
        a = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]])
        a[:, :k] = np.asarray([[0, 0, 0, 0], [0, 0, 0, 0]])
        # Cells past index k should remain unchanged.
        if k == 2:
            assert a[0, 0] == 0
            assert a[0, 1] == 0
            assert a[0, 2] == 3
            assert a[1, 3] == 8
    assert foo(2)


def test_static_array_1d_slice_setitem_scalar():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4, 5])
        a[1:4] = 0
        assert a[0] == 1
        assert a[1] == 0
        assert a[2] == 0
        assert a[3] == 0
        assert a[4] == 5
    assert foo()


# ───────────────────────── augmented assignment ──────────────────────────

def test_static_array_1d_aug_static_index():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4])
        a[2] = a[2] + 100
        assert a[0] == 1
        assert a[2] == 103
    assert foo()


def test_static_array_1d_aug_dynamic_index():
    @zk_circuit
    def foo(j: Integer):
        a = np.asarray([10, 20, 30, 40])
        a[j] = a[j] + 5
        if j == 1:
            assert a[j] == 25
    assert foo(1)
    assert foo(3)


# ───────────────────────── cache invalidation ────────────────────────────

def test_static_array_dynamic_write_then_static_read():
    """After a dynamic-index write, a static-index read at a *different*
    offset must still see the original value (cache invalidation routes
    through `ir_read_memory`, which sees the unchanged zkRAM cell). A
    static-index read at the *written* offset must see the new value."""
    @zk_circuit
    def foo(j: Integer, v: Integer):
        a = np.asarray([10, 20, 30, 40])
        a[j] = v
        # After dynamic write, static-index reads return the right value.
        if j == 2:
            assert a[0] == 10
            assert a[1] == 20
            assert a[2] == v
            assert a[3] == 40
    assert foo(2, 99)


# ───────────────────────── headline performance test ─────────────────────

def test_static_array_dynamic_write_loop_64_compiles_quickly():
    """The headline test: a tight in-place loop doing dynamic-index writes
    over a `StaticArray[Integer, 64]`. The legacy O(N) mux chain used to
    time out (>60s); the segment write should compile in well under that."""
    @zk_circuit
    def foo():
        a = np.zeros((64,), dtype=int)
        for i in range(64):
            a[(i * 7) % 64] = i
        # Sanity check: a[0] should be 0 (i=0 writes index 0).
        assert a[0] == 0
    t0 = time.time()
    assert foo()
    elapsed = time.time() - t0
    # Should be well under 60 seconds — the legacy path timed out at 60s.
    assert elapsed < 30, f"compile-and-run took {elapsed:.1f}s, expected < 30s"
