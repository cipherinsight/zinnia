"""
Tests for the splitting family (np.split / array_split / hsplit / vsplit /
dsplit) and np.block on static NDArrays.
"""

import pytest
from zinnia import *


# ───────────────────────────────────────────────────────────────────────
# np.split — equal sections, list of indices
# ───────────────────────────────────────────────────────────────────────

def test_split_equal_sections_1d():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4, 5, 6])
        parts = np.split(a, 3)
        assert (parts[0] == np.asarray([1, 2])).all()
        assert (parts[1] == np.asarray([3, 4])).all()
        assert (parts[2] == np.asarray([5, 6])).all()

    assert foo()


def test_split_indices_list_1d():
    @zk_circuit
    def foo():
        a = np.asarray([10, 20, 30, 40, 50])
        parts = np.split(a, [2, 4])
        assert (parts[0] == np.asarray([10, 20])).all()
        assert (parts[1] == np.asarray([30, 40])).all()
        assert (parts[2] == np.asarray([50])).all()

    assert foo()


def test_split_axis_1():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]])
        parts = np.split(a, 2, axis=1)
        assert (parts[0] == np.asarray([[1, 2], [5, 6]])).all()
        assert (parts[1] == np.asarray([[3, 4], [7, 8]])).all()

    assert foo()


# ───────────────────────────────────────────────────────────────────────
# np.array_split — uneven sections allowed
# ───────────────────────────────────────────────────────────────────────

def test_array_split_uneven():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        parts = np.array_split(a, 3)
        # NumPy: extras go to the front -> [4, 3, 3]
        assert (parts[0] == np.asarray([1, 2, 3, 4])).all()
        assert (parts[1] == np.asarray([5, 6, 7])).all()
        assert (parts[2] == np.asarray([8, 9, 10])).all()

    assert foo()


def test_array_split_more_sections_than_elems():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        parts = np.array_split(a, 5)
        # 3 / 5 -> base 0, extras 3 -> sizes [1,1,1,0,0]
        assert (parts[0] == np.asarray([1])).all()
        assert (parts[1] == np.asarray([2])).all()
        assert (parts[2] == np.asarray([3])).all()
        assert len(parts[3]) == 0
        assert len(parts[4]) == 0

    foo()


# ───────────────────────────────────────────────────────────────────────
# h/v/dsplit
# ───────────────────────────────────────────────────────────────────────

def test_hsplit_2d():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]])
        parts = np.hsplit(a, 2)
        assert (parts[0] == np.asarray([[1, 2], [5, 6]])).all()
        assert (parts[1] == np.asarray([[3, 4], [7, 8]])).all()

    assert foo()


def test_hsplit_1d_falls_back_to_axis_0():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4])
        parts = np.hsplit(a, 2)
        assert (parts[0] == np.asarray([1, 2])).all()
        assert (parts[1] == np.asarray([3, 4])).all()

    assert foo()


def test_vsplit_2d():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]])
        parts = np.vsplit(a, 2)
        assert (parts[0] == np.asarray([[1, 2], [3, 4]])).all()
        assert (parts[1] == np.asarray([[5, 6], [7, 8]])).all()

    assert foo()


def test_dsplit_3d():
    @zk_circuit
    def foo():
        a = np.asarray([
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            [[9, 10, 11, 12], [13, 14, 15, 16]],
        ])  # (2, 2, 4)
        parts = np.dsplit(a, 2)
        assert (parts[0] == np.asarray([
            [[1, 2], [5, 6]],
            [[9, 10], [13, 14]],
        ])).all()
        assert (parts[1] == np.asarray([
            [[3, 4], [7, 8]],
            [[11, 12], [15, 16]],
        ])).all()

    assert foo()


# ───────────────────────────────────────────────────────────────────────
# np.block
# ───────────────────────────────────────────────────────────────────────

def test_block_2x2_2d_leaves():
    @zk_circuit
    def foo():
        A = np.asarray([[1, 2], [3, 4]])
        B = np.asarray([[5, 6], [7, 8]])
        C = np.asarray([[9, 10], [11, 12]])
        D = np.asarray([[13, 14], [15, 16]])
        out = np.block([[A, B], [C, D]])
        expected = np.asarray([
            [1, 2, 5, 6],
            [3, 4, 7, 8],
            [9, 10, 13, 14],
            [11, 12, 15, 16],
        ])
        assert (out == expected).all()

    assert foo()


def test_block_1x2_hstack_equivalent():
    @zk_circuit
    def foo():
        A = np.asarray([[1, 2], [3, 4]])
        B = np.asarray([[5, 6], [7, 8]])
        out = np.block([[A, B]])
        # one outer row, two columns -> hstack
        assert (out == np.asarray([[1, 2, 5, 6], [3, 4, 7, 8]])).all()

    assert foo()


def test_block_2x1_vstack_equivalent():
    @zk_circuit
    def foo():
        A = np.asarray([[1, 2], [3, 4]])
        B = np.asarray([[5, 6], [7, 8]])
        out = np.block([[A], [B]])
        assert (out == np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]])).all()

    assert foo()


def test_block_three_deep_3d():
    @zk_circuit
    def foo():
        # Each leaf is 3-D shape (1,1,2). Block depth 3 means we concat
        # along axes 0, 1, 2 successively from outermost to innermost.
        A = np.asarray([[[1, 2]]])
        B = np.asarray([[[3, 4]]])
        C = np.asarray([[[5, 6]]])
        D = np.asarray([[[7, 8]]])
        out = np.block([[[A, B]], [[C, D]]])
        # Innermost concat (axis 2): [A,B] -> [[[1,2,3,4]]], [C,D] -> [[[5,6,7,8]]]
        # Middle concat (axis 1): each is [[[…]]] with one row, so still single-row
        # Outermost concat (axis 0): two rows
        expected = np.asarray([
            [[1, 2, 3, 4]],
            [[5, 6, 7, 8]],
        ])
        assert (out == expected).all()

    assert foo()
