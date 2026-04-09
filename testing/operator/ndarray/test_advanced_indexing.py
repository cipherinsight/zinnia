"""
Advanced indexing on static NDArrays:
  - boolean masking with a compile-time-known mask
  - fancy indexing with a compile-time-known integer index array

When the mask/index values are not compile-time constants, we hard-error
with a hint that lowering to dynamic ndarrays is the planned path. That
hard-error case lives in the future bounded-dynamic envelope work and is
out of scope for this PR.
"""

import pytest
from zinnia import *


# ---------------------------------------------------------------------------
# Boolean masking
# ---------------------------------------------------------------------------

def test_boolean_mask_1d():
    @zk_circuit
    def foo():
        a = np.asarray([10, 20, 30, 40, 50])
        mask = np.asarray([True, False, True, False, True])
        out = a[mask]
        assert (out == np.asarray([10, 30, 50])).all()

    assert foo()


def test_boolean_mask_2d_full_shape():
    @zk_circuit
    def foo():
        a = np.asarray([
            [1, 2, 3],
            [4, 5, 6],
        ])
        mask = np.asarray([
            [True, False, True],
            [False, True, False],
        ])
        out = a[mask]
        assert (out == np.asarray([1, 3, 5])).all()

    assert foo()


def test_boolean_mask_all_false():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        mask = np.asarray([False, False, False])
        out = a[mask]
        assert out.shape == (0,) or len(out) == 0

    # Empty-result case — just make sure it compiles and doesn't crash.
    foo()


# ---------------------------------------------------------------------------
# Fancy indexing
# ---------------------------------------------------------------------------

def test_fancy_index_1d():
    @zk_circuit
    def foo():
        a = np.asarray([10, 20, 30, 40, 50])
        idx = np.asarray([0, 2, 4])
        out = a[idx]
        assert (out == np.asarray([10, 30, 50])).all()

    assert foo()


def test_fancy_index_with_repeats():
    @zk_circuit
    def foo():
        a = np.asarray([100, 200, 300])
        idx = np.asarray([0, 0, 1, 2, 2, 2])
        out = a[idx]
        assert (out == np.asarray([100, 100, 200, 300, 300, 300])).all()

    assert foo()


def test_fancy_index_picks_rows():
    """Fancy indexing along axis 0 selects whole rows from a 2-D array."""
    @zk_circuit
    def foo():
        a = np.asarray([
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
        ])
        idx = np.asarray([0, 2])
        out = a[idx]
        assert (out == np.asarray([[1, 2], [5, 6]])).all()

    assert foo()


def test_fancy_index_negative():
    @zk_circuit
    def foo():
        a = np.asarray([10, 20, 30, 40])
        idx = np.asarray([-1, -2])
        out = a[idx]
        assert (out == np.asarray([40, 30])).all()

    assert foo()


def test_fancy_index_2d_index_shape_preserved():
    """A 2-D index array applied to a 1-D source preserves the index shape."""
    @zk_circuit
    def foo():
        a = np.asarray([10, 20, 30, 40])
        idx = np.asarray([[0, 1], [2, 3]])
        out = a[idx]
        assert (out == np.asarray([[10, 20], [30, 40]])).all()

    assert foo()
