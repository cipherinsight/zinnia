"""
Broadcasting tests for static NDArrays.

These cover NumPy-style shape broadcasting on fully-static (compile-time
inferrable) ndarrays — e.g. `(3, 1) + (1, 4) -> (3, 4)`. Scalar↔array
broadcasting was already supported; these tests pin down the new
shape-level behaviour and guard against regressions.
"""

from zinnia import *


# ---------------------------------------------------------------------------
# Scalar ↔ array (already supported; kept here so the broadcast story is
# self-contained in one file).
# ---------------------------------------------------------------------------

def test_scalar_plus_array():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        b = a + 10
        assert (b == np.asarray([[11, 12, 13], [14, 15, 16]])).all()

    assert foo()


def test_array_minus_scalar():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4])
        b = a - 1
        assert (b == np.asarray([0, 1, 2, 3])).all()

    assert foo()


# ---------------------------------------------------------------------------
# Shape-level broadcasting — the main thing this PR adds.
# ---------------------------------------------------------------------------

def test_column_plus_row_outer():
    """The canonical (3, 1) + (1, 4) -> (3, 4) outer-sum case."""
    @zk_circuit
    def foo():
        col = np.asarray([[1], [2], [3]])         # shape (3, 1)
        row = np.asarray([[10, 20, 30, 40]])      # shape (1, 4)
        out = col + row                            # shape (3, 4)
        expected = np.asarray([
            [11, 21, 31, 41],
            [12, 22, 32, 42],
            [13, 23, 33, 43],
        ])
        assert (out == expected).all()

    assert foo()


def test_lower_rank_left_pads_with_ones():
    """A 1-D operand should be left-padded to match the rank of the 2-D one."""
    @zk_circuit
    def foo():
        mat = np.asarray([[1, 2, 3], [4, 5, 6]])   # shape (2, 3)
        vec = np.asarray([10, 20, 30])             # shape (3,) -> broadcast to (2, 3)
        out = mat + vec
        expected = np.asarray([[11, 22, 33], [14, 25, 36]])
        assert (out == expected).all()

    assert foo()


def test_broadcast_three_dimensions():
    """A higher-rank case to make sure the stride math is right beyond 2-D."""
    @zk_circuit
    def foo():
        # shape (2, 1, 3)
        a = np.asarray([
            [[1, 2, 3]],
            [[4, 5, 6]],
        ])
        # shape (1, 4, 1)
        b = np.asarray([
            [[10], [20], [30], [40]],
        ])
        out = a + b   # shape (2, 4, 3)
        # First slab (i=0): rows of a=[1,2,3] tiled, plus column [10,20,30,40]
        expected = np.asarray([
            [[11, 12, 13], [21, 22, 23], [31, 32, 33], [41, 42, 43]],
            [[14, 15, 16], [24, 25, 26], [34, 35, 36], [44, 45, 46]],
        ])
        assert (out == expected).all()

    assert foo()


def test_broadcast_multiplication():
    """Make sure broadcasting kicks in for ops other than +."""
    @zk_circuit
    def foo():
        col = np.asarray([[2], [3], [5]])
        row = np.asarray([[1, 10, 100]])
        out = col * row
        expected = np.asarray([
            [2, 20, 200],
            [3, 30, 300],
            [5, 50, 500],
        ])
        assert (out == expected).all()

    assert foo()


def test_broadcast_int_float_promotion():
    """Mixed dtypes still get promoted correctly through the broadcast path."""
    @zk_circuit
    def foo():
        col = np.asarray([[1], [2]])
        row = np.asarray([[0.5, 1.5, 2.5]])
        out = col * row
        # 2x3 floats — compare element-wise via .all on a comparison.
        expected = np.asarray([[0.5, 1.5, 2.5], [1.0, 3.0, 5.0]])
        assert (out == expected).all()

    assert foo()


def test_broadcast_then_reduce():
    """Composition smoke test: broadcast outer-product, then reduce-sum."""
    @zk_circuit
    def foo():
        col = np.asarray([[1], [2], [3]])
        row = np.asarray([[1, 1, 1, 1]])
        out = (col * row).sum()
        # = (1+2+3) * 4
        assert out == 24

    assert foo()
