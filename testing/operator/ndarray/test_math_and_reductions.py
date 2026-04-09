"""
Coverage for the element-wise math, reductions, and element-wise comparison
landed in the static-ndarray "math + reductions" PR:

  - vectorized np.* unary math (np.sqrt, np.exp, np.sin, np.abs, np.sign,
    np.negative, np.minimum/maximum, np.equal, ...)
  - new element-wise: np.floor, np.ceil, np.trunc, np.round, np.reciprocal,
    np.where, np.clip
  - element-wise comparison for ndarray-shaped composites (eq/ne/lt/...)
  - new reductions: np.mean, np.var, np.std, np.cumsum, np.cumprod
"""

import math
from zinnia import *


# ───────────────────────────────────────────────────────────────────────
# Vectorized np.* unary math (these used to error on ndarray inputs)
# ───────────────────────────────────────────────────────────────────────

def test_np_sqrt_vectorized():
    @zk_circuit
    def foo():
        a = np.asarray([1.0, 4.0, 9.0, 16.0])
        out = np.sqrt(a)
        assert (out == np.asarray([1.0, 2.0, 3.0, 4.0])).all()

    assert foo()


def test_np_abs_vectorized_2d():
    @zk_circuit
    def foo():
        a = np.asarray([[-1.0, 2.0], [-3.0, 4.0]])
        out = np.abs(a)
        assert (out == np.asarray([[1.0, 2.0], [3.0, 4.0]])).all()

    assert foo()


def test_np_sign_vectorized():
    @zk_circuit
    def foo():
        a = np.asarray([-3.0, 0.0, 5.0])
        out = np.sign(a)
        assert (out == np.asarray([-1.0, 0.0, 1.0])).all()

    assert foo()


def test_np_negative_vectorized():
    @zk_circuit
    def foo():
        a = np.asarray([1.0, -2.0, 3.0])
        out = np.negative(a)
        assert (out == np.asarray([-1.0, 2.0, -3.0])).all()

    assert foo()


def test_np_minimum_two_arrays():
    @zk_circuit
    def foo():
        a = np.asarray([1, 5, 3])
        b = np.asarray([4, 2, 6])
        out = np.minimum(a, b)
        assert (out == np.asarray([1, 2, 3])).all()

    assert foo()


def test_np_maximum_with_broadcast():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 5], [3, 2]])
        out = np.maximum(a, 4)
        assert (out == np.asarray([[4, 5], [4, 4]])).all()

    assert foo()


def test_np_equal_vectorized():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        b = np.asarray([1, 0, 3])
        out = np.equal(a, b)
        assert (out == np.asarray([True, False, True])).all()

    assert foo()


def test_np_less_vectorized():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4])
        out = np.less(a, 3)
        assert (out == np.asarray([True, True, False, False])).all()

    assert foo()


# ───────────────────────────────────────────────────────────────────────
# New element-wise ops: floor / ceil / trunc / round / reciprocal /
# where / clip
# ───────────────────────────────────────────────────────────────────────

def test_floor_basic():
    @zk_circuit
    def foo():
        a = np.asarray([1.7, -1.2, 0.0, 3.5])
        out = np.floor(a)
        assert (out == np.asarray([1.0, -2.0, 0.0, 3.0])).all()

    assert foo()


def test_ceil_basic():
    @zk_circuit
    def foo():
        a = np.asarray([1.2, -1.7, 0.0, 3.5])
        out = np.ceil(a)
        assert (out == np.asarray([2.0, -1.0, 0.0, 4.0])).all()

    assert foo()


def test_trunc_basic():
    @zk_circuit
    def foo():
        a = np.asarray([1.7, -1.7, 0.5, -0.5])
        out = np.trunc(a)
        assert (out == np.asarray([1.0, -1.0, 0.0, 0.0])).all()

    assert foo()


def test_round_basic():
    @zk_circuit
    def foo():
        a = np.asarray([1.4, 1.6, -1.4, -1.6])
        out = np.round(a)
        # half-away-from-zero: -1.6 → -2, 1.6 → 2
        assert (out == np.asarray([1.0, 2.0, -1.0, -2.0])).all()

    assert foo()


def test_reciprocal_basic():
    @zk_circuit
    def foo():
        a = np.asarray([1.0, 2.0, 4.0, 8.0])
        out = np.reciprocal(a)
        assert (out == np.asarray([1.0, 0.5, 0.25, 0.125])).all()

    assert foo()


def test_where_scalar_branches():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4, 5])
        out = np.where(a > 2, 100, 0)
        assert (out == np.asarray([0, 0, 100, 100, 100])).all()

    assert foo()


def test_where_array_branches():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        b = np.asarray([10, 20, 30])
        cond = np.asarray([True, False, True])
        out = np.where(cond, a, b)
        assert (out == np.asarray([1, 20, 3])).all()

    assert foo()


def test_clip_scalar_bounds():
    @zk_circuit
    def foo():
        a = np.asarray([-2, 0, 2, 4, 6])
        out = np.clip(a, 0, 4)
        assert (out == np.asarray([0, 0, 2, 4, 4])).all()

    assert foo()


def test_clip_2d():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 5, 9], [-3, 0, 12]])
        out = np.clip(a, 0, 8)
        assert (out == np.asarray([[1, 5, 8], [0, 0, 8]])).all()

    assert foo()


# ───────────────────────────────────────────────────────────────────────
# Element-wise comparison on numeric composites
# ───────────────────────────────────────────────────────────────────────

def test_equality_returns_array_of_bools():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        b = np.asarray([1, 0, 3])
        out = a == b
        # New behaviour: ndarray of bools, not a single scalar.
        assert (out == np.asarray([True, False, True])).all()

    assert foo()


def test_lt_returns_array_of_bools_with_broadcast():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        out = a < 3
        assert (out == np.asarray([[True, True, False], [False, False, False]])).all()

    assert foo()


def test_compound_boolean_via_elementwise():
    @zk_circuit
    def foo():
        a = np.asarray([1, 5, 8, 12, 4])
        in_range = np.logical_and(a >= 3, a <= 10)
        assert (in_range == np.asarray([False, True, True, False, True])).all()

    assert foo()


def test_comparison_with_broadcast_shapes():
    @zk_circuit
    def foo():
        col = np.asarray([[1], [2], [3]])
        row = np.asarray([[1, 2, 3]])
        out = col == row
        expected = np.asarray([
            [True,  False, False],
            [False, True,  False],
            [False, False, True],
        ])
        assert (out == expected).all()

    assert foo()


# ───────────────────────────────────────────────────────────────────────
# Reductions: mean / var / std / cumsum / cumprod
# ───────────────────────────────────────────────────────────────────────

def test_mean_no_axis():
    @zk_circuit
    def foo():
        a = np.asarray([1.0, 2.0, 3.0, 4.0])
        m = np.mean(a)
        assert m == 2.5

    assert foo()


def test_mean_axis_0():
    @zk_circuit
    def foo():
        a = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        m = np.mean(a, axis=0)
        assert (m == np.asarray([2.5, 3.5, 4.5])).all()

    assert foo()


def test_mean_axis_1():
    @zk_circuit
    def foo():
        a = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        m = np.mean(a, axis=1)
        assert (m == np.asarray([2.0, 5.0])).all()

    assert foo()


def test_var_no_axis():
    @zk_circuit
    def foo():
        a = np.asarray([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        # mean = 5, deviations = [-3,-1,-1,-1,0,0,2,4],
        # squared = [9,1,1,1,0,0,4,16], sum = 32, var = 32/8 = 4.0
        v = np.var(a)
        assert v == 4.0

    assert foo()


def test_std_no_axis():
    @zk_circuit
    def foo():
        a = np.asarray([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        # var = 4.0, so std = 2.0
        s = np.std(a)
        assert s == 2.0

    assert foo()


def test_var_axis_0():
    @zk_circuit
    def foo():
        a = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        # mean per col = [3, 4]
        # deviations per col: col0=[-2,0,2], col1=[-2,0,2]
        # sq mean per col: 8/3 each
        v = np.var(a, axis=0)
        # 8/3 ≈ 2.6667
        diff0 = v[0] - (8.0 / 3.0)
        diff1 = v[1] - (8.0 / 3.0)
        assert -1e-6 < diff0 < 1e-6
        assert -1e-6 < diff1 < 1e-6

    assert foo()


def test_cumsum_1d():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4])
        out = np.cumsum(a)
        assert (out == np.asarray([1, 3, 6, 10])).all()

    assert foo()


def test_cumsum_axis_0_2d():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        out = np.cumsum(a, axis=0)
        assert (out == np.asarray([[1, 2, 3], [5, 7, 9]])).all()

    assert foo()


def test_cumsum_axis_1_2d():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        out = np.cumsum(a, axis=1)
        assert (out == np.asarray([[1, 3, 6], [4, 9, 15]])).all()

    assert foo()


def test_cumprod_1d():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4])
        out = np.cumprod(a)
        assert (out == np.asarray([1, 2, 6, 24])).all()

    assert foo()


def test_cumsum_no_axis_flattens_2d():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2], [3, 4]])
        out = np.cumsum(a)
        assert (out == np.asarray([1, 3, 6, 10])).all()

    assert foo()


# ───────────────────────────────────────────────────────────────────────
# Method-form sanity (arr.mean(), arr.std(), arr.cumsum(), ...)
# ───────────────────────────────────────────────────────────────────────

def test_method_form_mean_std_cumsum():
    @zk_circuit
    def foo():
        a = np.asarray([1.0, 2.0, 3.0, 4.0])
        assert a.mean() == 2.5
        assert a.cumsum()[3] == 10
        assert a.cumprod()[3] == 24

    assert foo()
