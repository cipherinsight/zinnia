"""Regression tests for C08 — Dyn boolean-mask write maps a compacted RHS
positionally.

`a[mask] += v` on a DynamicNDArray desugars (frontend) to
`a[mask] = a[mask] + v`. The RHS `a[mask] + v` is the output of `dyn_filter`
— a COMPACTED array (true-position values packed to the front, padding
behind). The old `dyn_setitem_mask` required RHS.logical_shape == target
shape and wrote `rhs[i]` at position `i`, so for a non-contiguous mask it
wrote PADDING into late true positions. numpy's contract is that cell `i`
takes `rhs[#trues before i]`.

Also exercises the numpy-true contract directly: `a[mask] = packed_rhs`
where `packed_rhs` is sized to the true count.

These tests FAIL at commit 4b69c08 (and at the C06 HEAD) and pass after the
fix. Masks are made runtime-dependent (compared against a witness input) so
the dynamic-mask path is exercised rather than constant-folding away.
"""

from zinnia import *


# ── a[mask] += v with a non-contiguous mask (the 0490 bug class) ──────────


def test_dyn_mask_iadd_noncontiguous():
    """mask = [T,F,T,F,T] (true positions 0,2,4). numpy:
    a = [10,20,30,40,50]; a[mask] += 5 -> [15,20,35,40,55]."""

    @zk_circuit
    def foo(threshold: Integer):
        a = np.promote_to_dynamic(np.asarray([10, 20, 30, 40, 50]))
        sel = np.promote_to_dynamic(np.asarray([1, 0, 1, 0, 1]))
        # Runtime-dependent mask: sel > threshold with threshold = 0 keeps
        # exactly the 1-positions, i.e. [T,F,T,F,T] — non-contiguous.
        mask = sel > threshold
        a[mask] += 5
        # Expected post: [15, 20, 35, 40, 55], sum 165.
        assert a[0] == 15
        assert a[1] == 20
        assert a[2] == 35
        assert a[3] == 40
        assert a[4] == 55
        assert np.sum(a) == 165

    assert foo(0)


def test_dyn_mask_iadd_trailing_true():
    """A mask whose LAST position is true is the position the positional
    bug clobbers with padding. mask = [F,F,T] -> a[2] += 100."""

    @zk_circuit
    def foo(threshold: Integer):
        a = np.promote_to_dynamic(np.asarray([1, 2, 3]))
        sel = np.promote_to_dynamic(np.asarray([0, 0, 1]))
        mask = sel > threshold
        a[mask] += 100
        assert a[0] == 1
        assert a[1] == 2
        assert a[2] == 103

    assert foo(0)


def test_dyn_mask_imul_noncontiguous():
    """Same shape with `*=` to exercise the integer-mul MaskReadView
    propagation path. mask = [T,F,T,F]; a*=10 at trues."""

    @zk_circuit
    def foo(threshold: Integer):
        a = np.promote_to_dynamic(np.asarray([2, 3, 4, 5]))
        sel = np.promote_to_dynamic(np.asarray([1, 0, 1, 0]))
        mask = sel > threshold
        a[mask] *= 10
        assert a[0] == 20
        assert a[1] == 3
        assert a[2] == 40
        assert a[3] == 5

    assert foo(0)


# ── a[mask] = packed_rhs sized to the true count (numpy true contract) ────


def test_dyn_mask_assign_packed_rhs_true_count():
    """numpy: a[mask] = rhs where len(rhs) == true_count. The RHS values
    are consumed sequentially, one per true mask cell.

    a = [10,20,30,40,50]; mask = [T,F,T,F,T] (3 trues);
    a[mask] = [7, 8, 9] -> [7,20,8,40,9]."""

    @zk_circuit
    def foo(threshold: Integer):
        a = np.promote_to_dynamic(np.asarray([10, 20, 30, 40, 50]))
        sel = np.promote_to_dynamic(np.asarray([1, 0, 1, 0, 1]))
        mask = sel > threshold
        rhs = np.promote_to_dynamic(np.asarray([7, 8, 9]))
        a[mask] = rhs
        assert a[0] == 7
        assert a[1] == 20
        assert a[2] == 8
        assert a[3] == 40
        assert a[4] == 9

    assert foo(0)


# ── Sanity: a contiguous (prefix) mask was already correct ───────────────


def test_dyn_mask_iadd_contiguous_prefix():
    """mask = [T,T,F,F,F]: the OLD positional path happened to be correct
    here (trues are a contiguous prefix). Guards against regressing it."""

    @zk_circuit
    def foo(threshold: Integer):
        a = np.promote_to_dynamic(np.asarray([10, 20, 30, 40, 50]))
        sel = np.promote_to_dynamic(np.asarray([1, 1, 0, 0, 0]))
        mask = sel > threshold
        a[mask] += 1
        assert a[0] == 11
        assert a[1] == 21
        assert a[2] == 30
        assert a[3] == 40
        assert a[4] == 50

    assert foo(0)
