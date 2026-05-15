"""Regression tests for `fuzz-finding-v3-implicit-reverse-slice`.

Before the fix, implicit-bound reverse slices (`a[::-1]`, `a[::-2]`)
returned an array whose contents did not match numpy. The static
slice-bound normalisation defaulted `start=0, stop=len` regardless of
step sign; for step<0 the Pythonic defaults are `start=len-1, stop=-1`
(i.e. "before index 0"). The explicit-bound form `a[hi:lo:-1]` was
unaffected and is covered by `test_static_array_1d_slice_reverse` —
included here too as a baseline.
"""

from zinnia import *
import numpy as np


# ─────────────── implicit-bound reverse slices (the bug) ────────────────

def test_reverse_slice_implicit_input_array():
    @zk_circuit
    def foo(a: NDArray[Integer, 5]):
        b = a[::-1]
        assert b[0] == a[4]
        assert b[1] == a[3]
        assert b[2] == a[2]
        assert b[3] == a[1]
        assert b[4] == a[0]
    assert foo(np.asarray([1, 2, 3, 4, 5]))


def test_reverse_slice_implicit_local_array():
    @zk_circuit
    def foo():
        a = np.asarray([10, 20, 30, 40, 50])
        b = a[::-1]
        assert b[0] == 50
        assert b[1] == 40
        assert b[2] == 30
        assert b[3] == 20
        assert b[4] == 10
    assert foo()


def test_reverse_slice_implicit_step_minus_two():
    @zk_circuit
    def foo():
        a = np.asarray([0, 1, 2, 3, 4, 5, 6])
        b = a[::-2]   # numpy: [6, 4, 2, 0]
        assert b[0] == 6
        assert b[1] == 4
        assert b[2] == 2
        assert b[3] == 0
    assert foo()


def test_reverse_slice_implicit_start_only():
    """`a[3::-1]` → start=3, stop default = -1 → indices [3, 2, 1, 0]."""
    @zk_circuit
    def foo():
        a = np.asarray([0, 10, 20, 30, 40])
        b = a[3::-1]
        assert b[0] == 30
        assert b[1] == 20
        assert b[2] == 10
        assert b[3] == 0
    assert foo()


def test_reverse_slice_implicit_stop_only():
    """`a[:1:-1]` → start default = len-1, stop=1 → indices [4, 3, 2]."""
    @zk_circuit
    def foo():
        a = np.asarray([0, 10, 20, 30, 40])
        b = a[:1:-1]
        assert b[0] == 40
        assert b[1] == 30
        assert b[2] == 20
    assert foo()


# ─────────────── explicit-bound reverse slice (baseline) ────────────────

def test_reverse_slice_explicit_baseline():
    """The pre-existing explicit form must still work after the fix."""
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4, 5])
        b = a[4:0:-1]
        assert b[0] == 5
        assert b[1] == 4
        assert b[2] == 3
        assert b[3] == 2
    assert foo()
