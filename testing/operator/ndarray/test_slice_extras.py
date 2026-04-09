"""
Tests for ellipsis (`...`) and `np.newaxis` / `None` in static-NDArray
subscripts. These exercise the @zk_circuit AST -> IR slicing path; the pure
Python NDArray runtime path is intentionally out of scope for this PR.
"""

from zinnia import *


# ---------------------------------------------------------------------------
# Ellipsis
# ---------------------------------------------------------------------------

def test_ellipsis_picks_last_column():
    @zk_circuit
    def foo():
        a = np.asarray([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ])
        # Equivalent to a[:, -1]
        col = a[..., -1]
        assert (col == np.asarray([3, 6, 9])).all()

    assert foo()


def test_ellipsis_picks_first_row():
    @zk_circuit
    def foo():
        a = np.asarray([
            [1, 2, 3],
            [4, 5, 6],
        ])
        # Equivalent to a[0, :]
        row = a[0, ...]
        assert (row == np.asarray([1, 2, 3])).all()

    assert foo()


def test_ellipsis_alone_is_full_array():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2], [3, 4]])
        b = a[...]
        assert (b == a).all()

    assert foo()


def test_ellipsis_three_dim_middle():
    @zk_circuit
    def foo():
        a = np.asarray([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ])  # shape (2, 2, 2)
        # a[1, ..., 0] -> a[1, :, 0] -> [5, 7]
        v = a[1, ..., 0]
        assert (v == np.asarray([5, 7])).all()

    assert foo()


# ---------------------------------------------------------------------------
# np.newaxis / None
# ---------------------------------------------------------------------------

def test_newaxis_at_front():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])         # shape (3,)
        b = a[None, :]                     # shape (1, 3)
        assert (b == np.asarray([[1, 2, 3]])).all()

    assert foo()


def test_newaxis_at_back():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])         # shape (3,)
        b = a[:, None]                     # shape (3, 1)
        assert (b == np.asarray([[1], [2], [3]])).all()

    assert foo()


def test_newaxis_via_np_newaxis_attribute():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        b = a[np.newaxis, :]
        assert (b == np.asarray([[1, 2, 3]])).all()

    assert foo()


def test_newaxis_two_dim():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2], [3, 4]])   # shape (2, 2)
        b = a[:, None, :]                   # shape (2, 1, 2)
        assert (b == np.asarray([[[1, 2]], [[3, 4]]])).all()

    assert foo()


# ---------------------------------------------------------------------------
# Combinations: newaxis + ellipsis, newaxis + broadcast.
# ---------------------------------------------------------------------------

def test_newaxis_then_ellipsis():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2], [3, 4]])   # shape (2, 2)
        b = a[None, ...]                    # shape (1, 2, 2)
        assert (b == np.asarray([[[1, 2], [3, 4]]])).all()

    assert foo()


def test_newaxis_enables_outer_product_via_broadcast():
    """The classic NumPy idiom: a[:, None] * b[None, :] -> outer product."""
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        b = np.asarray([10, 20, 30, 40])
        out = a[:, None] * b[None, :]   # shape (3, 4)
        expected = np.asarray([
            [10, 20, 30, 40],
            [20, 40, 60, 80],
            [30, 60, 90, 120],
        ])
        assert (out == expected).all()

    assert foo()
