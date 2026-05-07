"""Regression tests for matmul shape rules — specifically 1D @ 2D, which
was previously misrouted through the elementwise dispatch and produced
the wrong shape (`compiler.chained-assignment-dynamic-slice` triage).
"""
from zinnia import *


def test_matmul_1d_at_2d():
    # numpy: (8,) @ (8, 7) → (7,)
    @zk_circuit
    def foo(data: NDArray[Float, 8, 8]):
        a = data[:, 0]
        b = data[:, 1:8]
        r = a @ b
        # If r is 1-D of length 7, len == 7; if it leaked back to 2-D, len would differ.
        assert len(r) == 7
        # And the values should be the dot product of a with each col of b.
        # With data zeroed, all results are 0.
        assert r[0] == 0.0
        assert r[6] == 0.0

    import numpy as np
    assert foo(np.zeros((8, 8)))


def test_matmul_1d_at_2d_nontrivial_values():
    @zk_circuit
    def foo():
        a = np.asarray([1.0, 2.0, 3.0])         # (3,)
        b = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)
        r = a @ b
        # r[0] = 1*1 + 2*3 + 3*5 = 22
        # r[1] = 1*2 + 2*4 + 3*6 = 28
        assert len(r) == 2
        assert r[0] == 22.0
        assert r[1] == 28.0

    assert foo()


def test_matmul_2d_at_1d_unchanged():
    # Sanity: existing 2D @ 1D case still works.
    @zk_circuit
    def foo():
        a = np.asarray([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
        b = np.asarray([5.0, 6.0])                # (2,)
        r = a @ b
        # r[0] = 1*5 + 2*6 = 17; r[1] = 3*5 + 4*6 = 39
        assert len(r) == 2
        assert r[0] == 17.0
        assert r[1] == 39.0

    assert foo()


def test_chained_slice_assign_with_matmul():
    # The covariance / correlation pattern: chained `a[X] = a[Y] = expr`
    # where `expr = vec @ matrix_slice`. After the matmul fix, both
    # targets have shape (7,) matching the value (7,).
    @zk_circuit
    def foo(data: NDArray[Float, 8, 8]):
        out = np.zeros((8, 8), dtype=data.dtype)
        i = 0
        out[i + 1:8, i] = out[i, i + 1:8] = data[:, i] @ data[:, i + 1:8]
        # Sanity: out[0, 1..7] and out[1..7, 0] should equal each other.
        assert out[0, 7] == out[7, 0]

    import numpy as np
    assert foo(np.ones((8, 8)))
