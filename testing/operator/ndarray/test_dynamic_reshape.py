"""
End-to-end tests for dynamic ndarray reshape and materializing transpose.

Exercises:
  - reshape: 1D→2D, 2D→1D, 2D→3D, -1 inference, multi-arg form
  - transpose then binary op (validates materializing transpose correctness)
  - transpose then aggregation
  - reshape then transpose
  - chained reshape

Dynamic ndarrays are NOT used as circuit inputs. All arrays are constructed
inside the circuit and promoted via np.promote_to_dynamic.
"""

from zinnia import *


# ── Reshape ──────────────────────────────────────────────────────────────

def test_reshape_1d_to_2d():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3, 4, 5, 6]))
        b = a.reshape((2, 3))
        assert b[0, 0] == 1
        assert b[0, 2] == 3
        assert b[1, 0] == 4
        assert b[1, 2] == 6

    assert foo()


def test_reshape_2d_to_1d():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[1, 2, 3], [4, 5, 6]]))
        b = a.reshape((6,))
        assert b[0] == 1
        assert b[3] == 4
        assert b[5] == 6
        assert b.sum() == 21

    assert foo()


def test_reshape_2d_to_3d():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]]))
        b = a.reshape((2, 2, 2))
        assert b[0, 0, 0] == 1
        assert b[0, 0, 1] == 2
        assert b[0, 1, 0] == 3
        assert b[1, 0, 0] == 5
        assert b[1, 1, 1] == 8

    assert foo()


def test_reshape_infer_minus_one():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3, 4, 5, 6]))
        b = a.reshape((2, -1))  # infer 3
        assert b[0, 0] == 1
        assert b[0, 2] == 3
        assert b[1, 0] == 4
        assert b.sum() == 21

    assert foo()


def test_reshape_multi_arg():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3, 4, 5, 6]))
        b = a.reshape(3, 2)
        assert b[0, 0] == 1
        assert b[0, 1] == 2
        assert b[2, 1] == 6

    assert foo()


def test_reshape_identity():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[1, 2], [3, 4]]))
        b = a.reshape((2, 2))
        assert b[0, 0] == 1
        assert b[1, 1] == 4
        assert b.sum() == 10

    assert foo()


# ── Transpose then other ops (validates materialization) ─────────────────

def test_transpose_then_sum_axis():
    """After transpose, axis reduction must see correct element order."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[1, 2, 3], [4, 5, 6]]))
        # a is (2,3): rows [1,2,3] and [4,5,6]
        t = a.T  # (3,2): rows [1,4], [2,5], [3,6]
        s = t.sum(axis=1)  # sum each row of transposed: [5, 7, 9]
        assert s[0] == 5
        assert s[1] == 7
        assert s[2] == 9

    assert foo()


def test_transpose_then_binary_op():
    """Binary op on transposed array must use correct element positions."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[1, 2], [3, 4]]))
        t = a.T  # [[1, 3], [2, 4]]
        b = np.promote_to_dynamic(np.asarray([[10, 20], [30, 40]]))
        c = t + b  # [[11, 23], [32, 44]]
        assert c[0, 0] == 11
        assert c[0, 1] == 23
        assert c[1, 0] == 32
        assert c[1, 1] == 44

    assert foo()


def test_transpose_element_access():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[1, 2, 3], [4, 5, 6]]))
        t = a.T  # (3,2)
        assert t[0, 0] == 1
        assert t[0, 1] == 4
        assert t[1, 0] == 2
        assert t[2, 1] == 6

    assert foo()


def test_transpose_then_reshape():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[1, 2, 3], [4, 5, 6]]))
        t = a.T  # (3,2): [[1,4],[2,5],[3,6]]
        r = t.reshape((6,))  # [1,4,2,5,3,6]
        assert r[0] == 1
        assert r[1] == 4
        assert r[2] == 2
        assert r[5] == 6

    assert foo()


def test_reshape_then_transpose():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3, 4, 5, 6]))
        b = a.reshape((2, 3))  # [[1,2,3],[4,5,6]]
        c = b.T  # (3,2): [[1,4],[2,5],[3,6]]
        assert c[0, 0] == 1
        assert c[0, 1] == 4
        assert c[2, 0] == 3
        assert c[2, 1] == 6

    assert foo()


def test_chained_reshape():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3, 4, 5, 6]))
        b = a.reshape((2, 3))
        c = b.reshape((3, 2))
        assert c[0, 0] == 1
        assert c[0, 1] == 2
        assert c[1, 0] == 3
        assert c[2, 1] == 6

    assert foo()


# ── Reshape + arithmetic ────────────────────────────────────────────────

def test_reshape_then_add():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3, 4]))
        b = a.reshape((2, 2))
        c = np.promote_to_dynamic(np.asarray([[10, 20], [30, 40]]))
        d = b + c
        assert d.sum() == 110  # 11+22+33+44

    assert foo()


def test_reshape_preserves_sum():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3, 4, 5, 6]))
        b = a.reshape((2, 3))
        assert a.sum() == b.sum()
        assert b.sum() == 21

    assert foo()
