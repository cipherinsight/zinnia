"""Regression tests for `compiler.segarr-boolean-mask` (P5b of the
segment-native-static-arrays epic).

Cover Boolean StaticArray construction, boolean-mask read (static + dynamic
mask), boolean-mask write (scalar + array RHS), boolean ops on Boolean
StaticArrays, and reductions on Boolean StaticArrays.
"""

from zinnia import *
from zinnia import ZKCircuit


# ───────────────────────── Boolean construction ───────────────────────────


def test_boolean_array_literal_construction():
    @zk_circuit
    def foo():
        m = np.array([True, False, True])
        assert m[0] == True
        assert m[1] == False
        assert m[2] == True
    assert foo()


def test_comparison_produces_boolean_mask():
    @zk_circuit
    def foo():
        a = np.array([1, 5, 3, 7])
        m = a > 2
        assert m[0] == False
        assert m[1] == True
        assert m[2] == True
        assert m[3] == True
    assert foo()


# ───────────────────────── Static mask read ──────────────────────────────


def test_static_mask_read_returns_static_shape():
    @zk_circuit
    def foo():
        a = np.array([10, 20, 30, 40])
        m = np.array([True, False, True, False])
        b = a[m]
        # Two surviving cells, in original order.
        assert b[0] == 10
        assert b[1] == 30
    assert foo()


def test_static_mask_read_via_comparison_constants():
    @zk_circuit
    def foo():
        a = np.array([1, 5, 3, 7])
        b = a[a > 2]
        assert b[0] == 5
        assert b[1] == 3
        assert b[2] == 7
    assert foo()


# ───────────────────────── Dynamic mask read ─────────────────────────────


def test_dynamic_mask_read_returns_dynamic_ndarray():
    """Mask comes from a comparison whose value is runtime — output is a
    DynamicNDArray with total_bound == arr.size."""
    @zk_circuit
    def foo(threshold: Integer):
        a = np.array([1, 5, 3, 7])
        # Comparison with runtime scalar yields a Boolean mask whose cells
        # are not all compile-time-known, so the read goes through the
        # dynamic-mask path.
        m = a > threshold
        b = a[m]
        # threshold = 2 → keep [5, 3, 7] → sum 15.
        assert np.sum(b) == 15

    assert foo(2)


# ───────────────────────── Mask write — scalar RHS ────────────────────────


def test_static_mask_write_scalar():
    @zk_circuit
    def foo():
        a = np.array([10, 20, 30, 40])
        m = np.array([True, False, True, False])
        a[m] = 0
        assert a[0] == 0
        assert a[1] == 20
        assert a[2] == 0
        assert a[3] == 40
    assert foo()


def test_static_mask_write_full_shape_rhs():
    @zk_circuit
    def foo():
        a = np.array([10, 20, 30, 40])
        m = np.array([True, False, True, False])
        rhs = np.array([100, 200, 300, 400])
        a[m] = rhs
        assert a[0] == 100
        assert a[1] == 20
        assert a[2] == 300
        assert a[3] == 40
    assert foo()


def test_dynamic_mask_write_scalar():
    """Set-to-zero write where the mask depends on runtime data — exercises
    the per-cell select+write path on a dynamic mask."""
    @zk_circuit
    def foo(t: Integer):
        a = np.array([1, 2, 3, 4])
        m = a > t
        a[m] = 0
        # When t = 2: mask is [F, F, T, T] → a == [1, 2, 0, 0] → sum 3.
        assert np.sum(a) == 3
    assert foo(2)


# ───────────────────────── 2-D mask write ────────────────────────────────


def test_2d_mask_write_negatives_to_zero():
    """The classic `arr[arr > 0] = 0`-shaped operation but for negatives:
    set all positive entries to zero."""
    @zk_circuit
    def foo():
        a = np.array([[1, -2], [-3, 4]])
        a[a > 0] = 0
        # All originally-positive cells become 0.
        assert a[0, 0] == 0
        assert a[0, 1] == -2
        assert a[1, 0] == -3
        assert a[1, 1] == 0
    assert foo()


# ───────────────────────── Boolean ops ──────────────────────────────────


def test_boolean_and_or_not():
    @zk_circuit
    def foo():
        m1 = np.array([True, True, False, False])
        m2 = np.array([True, False, True, False])
        a = m1 & m2
        o = m1 | m2
        n = ~m1
        assert a[0] == True and a[1] == False and a[2] == False and a[3] == False
        assert o[0] == True and o[1] == True and o[2] == True and o[3] == False
        assert n[0] == False and n[1] == False and n[2] == True and n[3] == True
    assert foo()


# ───────────────────────── Boolean reductions ────────────────────────────


def test_boolean_any_all_sum():
    @zk_circuit
    def foo():
        m1 = np.array([True, False, True])
        m2 = np.array([True, True, True])
        m3 = np.array([False, False, False])
        assert m1.any() == True
        assert m1.all() == False
        assert m2.all() == True
        assert m3.any() == False
        # sum counts true cells.
        assert m1.sum() == 2
        assert m2.sum() == 3
        assert m3.sum() == 0
    assert foo()


# ───────────────────── Correlation-style mask write ──────────────────────


def test_correlation_style_mask_write_compiles():
    """The `correlation` benchmark sets cells of a stddev array that fall
    below a threshold to 1.0. This is the canonical numpy idiom that P5b
    aims to support natively. Just check it compiles end-to-end (the benchmark
    itself was compile_error before this card)."""
    @zk_circuit
    def foo():
        stddev = np.array([0.5, 0.05, 0.2, 0.01])
        stddev[stddev <= 0.1] = 1.0
        # 0.05 and 0.01 should have been set to 1.0; others stay.
        assert stddev[0] == 0.5
        assert stddev[1] == 1.0
        assert stddev[2] == 0.2
        assert stddev[3] == 1.0
    assert foo()
