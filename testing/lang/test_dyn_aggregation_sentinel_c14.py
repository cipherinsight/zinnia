"""Regression tests for C14 — bounded-dynamic aggregation sentinel audit.

Covers two confirmed bugs in `src/ops/dyn_ndarray/aggregation.rs`:

1. The axis-reduction path (`dyn_aggregate_axis`) seeded integer
   max/argmax with sentinel `0` and compared with `gt_i_bounded`. For an
   all-negative active region `max(axis=...)` returned `0` (wrong) — no
   negative element is `> 0`, so the accumulator never updated. The fix
   routes integer min/max/argmin/argmax through the first-active pattern
   (already used by the full-reduction path) with the signed-correct
   `gt_i`/`lt_i` comparisons.

2. `keepdims=True` was silently ignored on the bounded-dynamic reduction
   dispatch (`dispatch_dyn_ndarray_method`), mis-shaping the result. It
   now rejects loudly.

All numeric expectations are cross-checked against numpy in the comments.
These fail at 4b69c08 (pre-fix): the max/argmax cases return 0, and the
keepdims case silently produces a rank-reduced array.
"""

import numpy as np
import pytest

from zinnia.api.zk_circuit import ZKCircuit
from zinnia.debug.exception.base import ZinniaException


def _prove(src, *args):
    return ZKCircuit.from_source("foo", src).prove(*args, backend="mock").proof_bytes_hex


# ─────────── axis-reduction max/argmax over all-negative active region ───────────


def test_dyn_axis_max_all_negative_1d():
    # numpy: max([-5,-3,-7,-2]) == -2 (NOT 0)
    src = """
def foo(x: Int[Array, "N<=6"]):
    assert x.max(axis=0) == -2
"""
    assert _prove(src, np.asarray([-5, -3, -7, -2])) == "mock_satisfied"


def test_dyn_axis_max_all_negative_2d_axis0():
    # data (2,2) = [[-4,-5],[-1,-2]]; numpy max(axis=0) == [-1,-2]
    src = """
def foo(x: Int[Array, "2 N<=3"]):
    m = x.max(axis=0)
    assert m[0] == -1
    assert m[1] == -2
"""
    assert _prove(src, np.asarray([[-4, -5], [-1, -2]])) == "mock_satisfied"


def test_dyn_axis_max_all_negative_2d_axis1():
    # numpy max(axis=1) of [[-4,-5],[-1,-2]] == [-4,-1]
    src = """
def foo(x: Int[Array, "2 N<=3"]):
    m = x.max(axis=1)
    assert m[0] == -4
    assert m[1] == -1
"""
    assert _prove(src, np.asarray([[-4, -5], [-1, -2]])) == "mock_satisfied"


def test_dyn_axis_max_mixed_sign():
    # numpy max([-3,5,-7,2]) == 5
    src = """
def foo(x: Int[Array, "N<=6"]):
    assert x.max(axis=0) == 5
"""
    assert _prove(src, np.asarray([-3, 5, -7, 2])) == "mock_satisfied"


def test_dyn_axis_argmax_all_negative():
    # numpy argmax([-5,-3,-7,-2]) == 3
    src = """
def foo(x: Int[Array, "N<=6"]):
    assert x.argmax(axis=0) == 3
"""
    assert _prove(src, np.asarray([-5, -3, -7, -2])) == "mock_satisfied"


def test_dyn_axis_argmin_all_negative():
    # numpy argmin([-5,-3,-7,-2]) == 2
    src = """
def foo(x: Int[Array, "N<=6"]):
    assert x.argmin(axis=0) == 2
"""
    assert _prove(src, np.asarray([-5, -3, -7, -2])) == "mock_satisfied"


# ─────────── min sentinel smoke (i64::MAX path stays correct) ───────────


def test_dyn_axis_min_all_positive():
    # numpy min([5,3,7,2]) == 2
    src = """
def foo(x: Int[Array, "N<=6"]):
    assert x.min(axis=0) == 2
"""
    assert _prove(src, np.asarray([5, 3, 7, 2])) == "mock_satisfied"


# ─────────── full reduction (axis=None) was already correct; lock it in ───────────


def test_dyn_full_max_all_negative():
    # numpy max([-5,-3,-7,-2]) == -2 (full reduction uses first-active pattern)
    src = """
def foo(x: Int[Array, "N<=6"]):
    assert x.max() == -2
"""
    assert _prove(src, np.asarray([-5, -3, -7, -2])) == "mock_satisfied"


# ─────────── keepdims must reject loudly on the dyn path ───────────


@pytest.mark.parametrize("method", ["sum", "max", "min", "argmax", "mean"])
def test_dyn_keepdims_rejects_loudly(method):
    src = f"""
def foo(x: Int[Array, "2 N<=3"]):
    s = x.{method}(axis=0, keepdims=True)
"""
    with pytest.raises(ZinniaException, match="keepdims"):
        ZKCircuit.from_source("foo", src).prove(
            np.asarray([[1, 2], [3, 4]]), backend="mock"
        )


def test_dyn_keepdims_false_still_works():
    # keepdims=False (the default) is unaffected; numpy sum(axis=0) = [4,6]
    src = """
def foo(x: Int[Array, "2 N<=3"]):
    s = x.sum(axis=0, keepdims=False)
    assert s[0] == 4
    assert s[1] == 6
"""
    assert _prove(src, np.asarray([[1, 2], [3, 4]])) == "mock_satisfied"


# ─────────── G01: static runtime-length fast path (compact full reduction) ───────────
#
# Slicing a bounded dynamic array (`x[:k]` with constant `k`) yields a
# compact dyn array whose `runtime_length.static_val == Some(k)`, with an
# envelope strictly larger than `k`. The full-reduction lowering now iterates
# the static prefix `0..k` with the unmasked reduction step (no per-slot
# `LtI + LogicalAnd` active machinery), which must NOT change which elements
# are active: only the first `k` slots participate, regardless of trailing
# buffer contents. Each expectation is numpy-checked against `arr[:k]`.


def test_dyn_static_len_sum():
    # numpy: sum([1,2,3,4]) == 10  (trailing 5,6,7,8 must be excluded)
    src = """
def foo(x: Int[Array, "N<=8"]):
    y = x[:4]
    assert y.sum() == 10
"""
    assert _prove(src, np.asarray([1, 2, 3, 4, 5, 6, 7, 8])) == "mock_satisfied"


def test_dyn_static_len_prod():
    # numpy: prod([2,3,4]) == 24
    src = """
def foo(x: Int[Array, "N<=8"]):
    y = x[:3]
    assert y.prod() == 24
"""
    assert _prove(src, np.asarray([2, 3, 4, 5, 6])) == "mock_satisfied"


def test_dyn_static_len_max_all_negative():
    # numpy: max([-5,-3,-7,-2]) == -2 (trailing positives excluded)
    src = """
def foo(x: Int[Array, "N<=8"]):
    y = x[:4]
    assert y.max() == -2
"""
    assert _prove(src, np.asarray([-5, -3, -7, -2, 9, 9, 9, 9])) == "mock_satisfied"


def test_dyn_static_len_min():
    # numpy: min([-5,-3,-7,-2,1]) == -7
    src = """
def foo(x: Int[Array, "N<=8"]):
    y = x[:5]
    assert y.min() == -7
"""
    assert _prove(src, np.asarray([-5, -3, -7, -2, 1, 99])) == "mock_satisfied"


def test_dyn_static_len_argmax_all_negative():
    # numpy: argmax([-5,-3,-7,-2]) == 3
    src = """
def foo(x: Int[Array, "N<=8"]):
    y = x[:4]
    assert y.argmax() == 3
"""
    assert _prove(src, np.asarray([-5, -3, -7, -2, 9])) == "mock_satisfied"


def test_dyn_static_len_float_sum():
    # numpy: sum([1.5,2.5,3.0]) == 7.0
    src = """
def foo(x: Float[Array, "N<=8"]):
    y = x[:3]
    assert y.sum() == 7.0
"""
    assert _prove(src, np.asarray([1.5, 2.5, 3.0, 9.0, 9.0])) == "mock_satisfied"


# ─────────── G01: factored per-axis-position mask (axis reduction) ───────────
#
# The 2D bounded-axis reductions exercise the hoisted-mask path: the active
# indicator now factors into a precomputed per-axis-position part AND a
# per-output part, combined with a single LogicalAnd. The active region (and
# thus every result) must be identical to the unfactored emission.


def test_dyn_axis_sum_bounded_2d():
    # numpy sum(axis=1) of [[1,2,3],[4,5,6],[7,8,9]] == [6,15,24]
    src = """
def foo(x: Int[Array, "3 N<=4"]):
    m = x.sum(axis=1)
    assert m[0] == 6
    assert m[1] == 15
    assert m[2] == 24
"""
    assert (
        _prove(src, np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])) == "mock_satisfied"
    )


def test_dyn_axis0_max_bounded_2d():
    # numpy max(axis=0) of [[1,2,3],[4,5,6],[7,8,9]] == [7,8,9]
    src = """
def foo(x: Int[Array, "M<=4 3"]):
    m = x.max(axis=0)
    assert m[0] == 7
    assert m[1] == 8
    assert m[2] == 9
"""
    assert (
        _prove(src, np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])) == "mock_satisfied"
    )
