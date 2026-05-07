"""Regression tests for `compiler.segarr-elementwise-ops` (P4a of the
segment-native-static-arrays epic).

These exercise native `Value::StaticArray` elementwise paths: binary
arithmetic on equal-shape arrays, scalar-array broadcasts, multi-axis
broadcast (e.g. (3,1) + (1,4)), 2-D arithmetic, comparison ops returning
Boolean StaticArrays, unary ops (`-arr`, `~arr`), constant folding through
the cached payload, and the heterogeneous-list mixed case routed through
the legacy boundary shim.
"""

import pytest
from zinnia import *


# ───────────────────────── Equal-shape binary ────────────────────────────

def test_static_array_1d_add_equal_shape():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4])
        b = np.asarray([10, 20, 30, 40])
        c = a + b
        assert c[0] == 11
        assert c[1] == 22
        assert c[2] == 33
        assert c[3] == 44
    assert foo()


def test_static_array_1d_sub_equal_shape():
    @zk_circuit
    def foo():
        a = np.asarray([10, 20, 30])
        b = np.asarray([1, 2, 3])
        c = a - b
        assert c[0] == 9
        assert c[1] == 18
        assert c[2] == 27
    assert foo()


def test_static_array_1d_mul_equal_shape():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4])
        b = np.asarray([5, 6, 7, 8])
        c = a * b
        assert c[0] == 5
        assert c[2] == 21
        assert c[3] == 32
    assert foo()


def test_static_array_2d_add_equal_shape():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        b = np.asarray([[10, 20, 30], [40, 50, 60]])
        c = a + b
        assert c[0, 0] == 11
        assert c[0, 2] == 33
        assert c[1, 0] == 44
        assert c[1, 2] == 66
    assert foo()


# ───────────────────────── Scalar broadcast ──────────────────────────────

def test_static_array_scalar_on_right():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4])
        c = a + 10
        assert c[0] == 11
        assert c[3] == 14
    assert foo()


def test_static_array_scalar_on_left():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4])
        c = 9 * a
        assert c[0] == 9
        assert c[1] == 18
        assert c[3] == 36
    assert foo()


def test_static_array_scalar_sub_noncommutative():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        c = 10 - a
        d = a - 1
        assert c[0] == 9
        assert c[1] == 8
        assert d[0] == 0
        assert d[1] == 1
    assert foo()


# ───────────────────────── Broadcasting ──────────────────────────────────

def test_static_array_broadcast_3x1_plus_1x4():
    @zk_circuit
    def foo():
        a = np.asarray([[1], [2], [3]])
        b = np.asarray([[10, 20, 30, 40]])
        c = a + b
        # (3,1) + (1,4) → (3,4) outer-add table
        assert c[0, 0] == 11
        assert c[0, 3] == 41
        assert c[2, 0] == 13
        assert c[2, 3] == 43
        assert c[1, 2] == 32
    assert foo()


def test_static_array_broadcast_row_vector():
    @zk_circuit
    def foo():
        # (2,3) + (3,) → (2,3): the 1-D row broadcasts down rows.
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        b = np.asarray([100, 200, 300])
        c = a + b
        assert c[0, 0] == 101
        assert c[0, 2] == 303
        assert c[1, 1] == 205
    assert foo()


# ───────────────────────── Comparisons ───────────────────────────────────

def test_static_array_gt_scalar_returns_boolean():
    @zk_circuit
    def foo():
        a = np.asarray([1, 5, 3, 8, 2])
        mask = a > 4
        # Iterating the boolean mask gives booleans the user can use.
        assert mask[0] == False
        assert mask[1] == True
        assert mask[2] == False
        assert mask[3] == True
        assert mask[4] == False
    assert foo()


def test_static_array_eq_arrays():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4])
        b = np.asarray([1, 0, 3, 0])
        mask = a == b
        assert mask[0] == True
        assert mask[1] == False
        assert mask[2] == True
        assert mask[3] == False
    assert foo()


# ───────────────────────── Unary ─────────────────────────────────────────

def test_static_array_unary_neg():
    @zk_circuit
    def foo():
        a = np.asarray([1, -2, 3, -4])
        c = -a
        assert c[0] == -1
        assert c[1] == 2
        assert c[2] == -3
        assert c[3] == 4
    assert foo()


def test_static_array_unary_invert_int():
    @zk_circuit
    def foo():
        a = np.asarray([0, 1, 2, 3])
        c = ~a
        # Bitwise NOT on int: ~x = -x - 1
        assert c[0] == -1
        assert c[1] == -2
        assert c[2] == -3
        assert c[3] == -4
    assert foo()


# ───────────────────────── Constant folding ──────────────────────────────

def test_static_array_constant_folding_add_scalar():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        c = a + 1
        # If cells fold at compile time the assertion below is statically
        # provable; the test would hang or error if the values weren't
        # forwarded as compile-time constants. Functionally identical to
        # the runtime check, but exercises the folded-cache path.
        assert c[0] == 2
        assert c[1] == 3
        assert c[2] == 4
    assert foo()


# ───────────────────────── Mixed: StaticArray + Python list ─────────────

def test_static_array_plus_python_list():
    # Currently `np.asarray([1,2,3])` is a StaticArray; a Python list
    # `[10, 20, 30]` is a Value::List of Integer leaves. The boundary shim
    # should still produce a correct elementwise sum via the legacy path.
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        b = [10, 20, 30]
        c = a + b
        assert c[0] == 11
        assert c[1] == 22
        assert c[2] == 33
    assert foo()
