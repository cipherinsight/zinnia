"""Regression tests for `C01` — negative-step slice *setitem* (write side).

The read-side negative-step defaults were fixed on 2026-06-10
(binding-audit case 0419), but three sibling write-side copies still used
sign-agnostic defaults (`start=0, stop=len`) and were wrong for negative
steps:

  - `static_array_slice_setitem` (`a[::-1] = b` on a Value::StaticArray):
    `s=0, e=len, st=-1` → `while 0 > len` is immediately false → zero
    coordinates. Scalar RHS silently no-ops; array RHS falls through to a
    legacy path that writes only `a[0]`.
  - `dyn_setitem_slice` (same on a DynamicNDArray): scalar RHS writes
    nothing; array RHS panics with a bogus shape-mismatch.
  - `list_slice_range` (`lst[3::-1]` Python list slice read): returns an
    empty list instead of `[lst[3], lst[2], lst[1], lst[0]]`.

For step<0 the Pythonic defaults are `start=len-1, stop=-1` (i.e. "before
index 0").
"""

from zinnia import *
import numpy as np


# ─────────────── static-array negative-step setitem ────────────────────

def test_static_setitem_full_reverse_array_rhs():
    """`a[::-1] = b` reverses b into a (numpy: a == b[::-1])."""
    @zk_circuit
    def foo():
        a = np.asarray([0, 0, 0, 0, 0])
        b = np.asarray([1, 2, 3, 4, 5])
        a[::-1] = b
        # numpy: a == [5, 4, 3, 2, 1]
        assert a[0] == 5
        assert a[1] == 4
        assert a[2] == 3
        assert a[3] == 2
        assert a[4] == 1
    assert foo()


def test_static_setitem_step_minus_two_array_rhs():
    """`a[::-2] = b` writes b to indices [6, 4, 2, 0]."""
    @zk_circuit
    def foo():
        a = np.asarray([0, 0, 0, 0, 0, 0, 0])
        b = np.asarray([10, 20, 30, 40])
        a[::-2] = b
        # numpy targets indices 6,4,2,0 in that order → a = [40,0,30,0,20,0,10]
        assert a[6] == 10
        assert a[4] == 20
        assert a[2] == 30
        assert a[0] == 40
        assert a[1] == 0
        assert a[3] == 0
        assert a[5] == 0
    assert foo()


def test_static_setitem_start_only_scalar_rhs():
    """`a[3::-1] = scalar` writes the scalar to indices [3, 2, 1, 0]."""
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4, 5])
        a[3::-1] = 0
        # numpy: indices 3,2,1,0 set to 0; index 4 untouched
        assert a[0] == 0
        assert a[1] == 0
        assert a[2] == 0
        assert a[3] == 0
        assert a[4] == 5
    assert foo()


# ─────────────── dyn (bounded) array negative-step setitem ──────────────

def test_dyn_setitem_full_reverse_array_rhs():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([0, 0, 0, 0, 0]))
        b = np.promote_to_dynamic(np.asarray([1, 2, 3, 4, 5]))
        a[::-1] = b
        # a == [5, 4, 3, 2, 1] → sum unchanged but order reversed
        assert a[0] == 5
        assert a[1] == 4
        assert a[2] == 3
        assert a[3] == 2
        assert a[4] == 1
    assert foo()


def test_dyn_setitem_step_minus_two_array_rhs():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([0, 0, 0, 0, 0, 0, 0]))
        b = np.promote_to_dynamic(np.asarray([10, 20, 30, 40]))
        a[::-2] = b
        assert a[6] == 10
        assert a[4] == 20
        assert a[2] == 30
        assert a[0] == 40
        assert a[1] == 0
        assert a[3] == 0
        assert a[5] == 0
    assert foo()


def test_dyn_setitem_start_only_scalar_rhs():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3, 4, 5]))
        a[3::-1] = 0
        assert a[0] == 0
        assert a[1] == 0
        assert a[2] == 0
        assert a[3] == 0
        assert a[4] == 5
    assert foo()


# ─────────────── Python list negative-step slice read ───────────────────

def test_list_slice_start_only_negative_step():
    """`lst[3::-1]` → [lst[3], lst[2], lst[1], lst[0]]."""
    @zk_circuit
    def foo():
        lst = [10, 20, 30, 40, 50]
        rev = lst[3::-1]
        assert len(rev) == 4
        assert rev[0] == 40
        assert rev[1] == 30
        assert rev[2] == 20
        assert rev[3] == 10
    assert foo()
