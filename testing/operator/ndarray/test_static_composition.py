"""
Operator composition smoke tests for static NDArrays. Roadmap §1.4: as the
operator set grows, broadcasting / slicing / indexing / reduction need to
compose cleanly. These tests pin down the combinations.
"""

from zinnia import *


def test_broadcast_then_sum():
    """Outer-product via broadcast then reduce-sum."""
    @zk_circuit
    def foo():
        col = np.asarray([[1], [2], [3]])
        row = np.asarray([[10, 20, 30, 40]])
        out = (col + row).sum()
        # sum of [[11..41],[12..42],[13..43]] = (11+21+31+41) + (12+22+32+42) + (13+23+33+43)
        # = 104 + 108 + 112 = 324
        assert out == 324

    assert foo()


def test_slice_then_broadcast():
    """Slice a row and a column, broadcast them into an outer product."""
    @zk_circuit
    def foo():
        a = np.asarray([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ])
        row = a[0, :]            # [1,2,3,4]
        col = a[:, 0]            # [1,5,9]
        out = col[:, None] * row[None, :]
        expected = np.asarray([
            [1, 2, 3, 4],
            [5, 10, 15, 20],
            [9, 18, 27, 36],
        ])
        assert (out == expected).all()

    assert foo()


def test_newaxis_then_broadcast_then_sum():
    """`a[:, None] - a[None, :]` is the pairwise-difference matrix idiom."""
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 4, 7])
        diffs = a[:, None] - a[None, :]
        # The diagonal is zero; sum of all entries should be zero too
        # (every diff cancels with its transpose).
        assert diffs.sum() == 0

    assert foo()


def test_transpose_then_broadcast():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3], [4, 5, 6]])  # (2, 3)
        b = a.T                                  # (3, 2)
        scaled = b * np.asarray([10, 100])       # broadcast (2,) over (3, 2)
        expected = np.asarray([
            [10, 400],
            [20, 500],
            [30, 600],
        ])
        assert (scaled == expected).all()

    assert foo()


def test_ellipsis_then_arithmetic():
    @zk_circuit
    def foo():
        a = np.asarray([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ])
        # Take last column of last 2-D slab, then add a constant.
        v = a[1, ..., 1] + 100
        # a[1] = [[5,6],[7,8]];  a[1, :, 1] = [6, 8];  + 100 = [106, 108]
        assert (v == np.asarray([106, 108])).all()

    assert foo()


def test_fancy_index_then_broadcast():
    @zk_circuit
    def foo():
        a = np.asarray([10, 20, 30, 40])
        idx = np.asarray([0, 2])
        picked = a[idx]              # [10, 30]
        out = picked + np.asarray([1, 2])
        assert (out == np.asarray([11, 32])).all()

    assert foo()


def test_boolean_mask_then_sum():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4, 5])
        mask = np.asarray([True, False, True, False, True])
        assert a[mask].sum() == 9

    assert foo()
