"""Regression tests for `fuzz-finding-v3-aug-assign-subscript`.

Before the fix, `a[i] += v` on a subscript target emitted a distinct
`ASTAugAssignStatement` whose lowering produced stale-neighbour reads:
after `a[2] += 100`, asserting `a[0] == 1` was reported provably
unsatisfiable. The semantically-equivalent explicit form
`a[2] = a[2] + 100` worked correctly.

The fix desugars `a[i] op= v` to `a[i] = a[i] op v` at the Python AST
transformer, so the augmented form routes through the same code path
as the explicit form. Scalar `+=` on a Name target (`x += 1`) is
untouched — its baseline is included.
"""

from zinnia import *
import numpy as np


# ─────────────── element-wise `a[i] += v` (the core bug) ────────────────

def test_aug_assign_element_local_array():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4, 5, 6])
        a[2] += 100
        assert a[0] == 1
        assert a[1] == 2
        assert a[2] == 103
        assert a[3] == 4
        assert a[4] == 5
        assert a[5] == 6
    assert foo()


def test_aug_assign_element_input_array():
    @zk_circuit
    def foo(a: NDArray[Integer, 4]):
        a[1] += 50
        assert a[0] == 10
        assert a[1] == 70
        assert a[2] == 30
        assert a[3] == 40
    assert foo(np.asarray([10, 20, 30, 40]))


def test_aug_assign_element_mul():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4])
        a[2] *= 5
        assert a[0] == 1
        assert a[1] == 2
        assert a[2] == 15
        assert a[3] == 4
    assert foo()


def test_aug_assign_element_sub():
    @zk_circuit
    def foo():
        a = np.asarray([10, 20, 30, 40])
        a[1] -= 5
        assert a[0] == 10
        assert a[1] == 15
        assert a[2] == 30
        assert a[3] == 40
    assert foo()


# ─────────────────── slice `a[i:j] += arr` aug-assign ───────────────────

def test_aug_assign_slice_local_array():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4, 5, 6])
        a[1:4] += np.asarray([10, 20, 30])
        assert a[0] == 1
        assert a[1] == 12
        assert a[2] == 23
        assert a[3] == 34
        assert a[4] == 5
        assert a[5] == 6
    assert foo()


# ─────────────────── baseline: explicit-form equivalence ────────────────

def test_aug_assign_baseline_explicit_form_still_works():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4, 5, 6])
        a[2] = a[2] + 100
        assert a[0] == 1
        assert a[2] == 103
    assert foo()


# ─────────────────── baseline: scalar `x += 1` unchanged ────────────────

def test_aug_assign_scalar_name_target_unchanged():
    @zk_circuit
    def foo():
        x = 5
        x += 3
        assert x == 8
        y = 10
        y *= 2
        assert y == 20
    assert foo()
