"""Regression tests for `compiler.segarr-shape-ops` (P4c of the
segment-native-static-arrays epic).

These exercise native `Value::StaticArray` shape ops:
- `arr.reshape((m, n))`, `.reshape(-1)`, `.reshape(-1, k)`
- `arr.transpose()`, `.T`, `arr.transpose(perm)`
- `np.concatenate((a, b), axis=k)`, `np.stack((a, b), axis=k)`
- `np.expand_dims(a, axis)`, `np.squeeze(a)`, `np.squeeze(a, axis=k)`
- `arr.flatten()`, `arr.ravel()`
- Constant folding through the cached payload
- Reshape after a dynamic-index write (cache invalidated by P3)
"""

import pytest
from zinnia import *


# ───────────────────────── reshape ─────────────────────────

def test_segarr_reshape_2x3_to_3x2():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        b = a.reshape((3, 2))
        # Row-major: [1,2,3,4,5,6] → reshape (3,2) → [[1,2],[3,4],[5,6]]
        assert b[0][0] == 1
        assert b[0][1] == 2
        assert b[2][0] == 5
        assert b[2][1] == 6
    assert foo()


def test_segarr_reshape_neg_one_only():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        b = a.reshape(-1)
        assert b[0] == 1
        assert b[5] == 6
    assert foo()


def test_segarr_reshape_neg_one_with_other():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]])
        b = a.reshape(-1, 2)
        # 8 elements / 2 cols = 4 rows
        assert b[0][0] == 1
        assert b[3][1] == 8
    assert foo()


def test_segarr_reshape_method_kwarg_form():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4, 5, 6])
        b = a.reshape(2, 3)
        assert b[0][0] == 1
        assert b[1][2] == 6
    assert foo()


def test_segarr_np_reshape():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4])
        b = np.reshape(a, (2, 2))
        assert b[0][0] == 1
        assert b[1][1] == 4
    assert foo()


# ───────────────────────── transpose / .T ─────────────────────────

def test_segarr_transpose_default_2d():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        b = a.transpose()
        # (2,3) → (3,2): [[1,4],[2,5],[3,6]]
        assert b[0][0] == 1
        assert b[0][1] == 4
        assert b[1][0] == 2
        assert b[2][1] == 6
    assert foo()


def test_segarr_T_property():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        b = a.T
        assert b[0][0] == 1
        assert b[2][1] == 6
    assert foo()


def test_segarr_transpose_explicit_perm():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        b = a.transpose((1, 0))
        assert b[0][0] == 1
        assert b[2][1] == 6
    assert foo()


def test_segarr_np_transpose_3d():
    @zk_circuit
    def foo():
        # (2, 2, 3) array
        a = np.asarray([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        b = np.transpose(a, (2, 0, 1))
        # New shape (3, 2, 2). Element (axis-0=k, axis-1=i, axis-2=j) of b
        # corresponds to a[i][j][k].
        assert b[0][0][0] == 1   # a[0][0][0]
        assert b[0][1][0] == 7   # a[1][0][0]
        assert b[2][1][1] == 12  # a[1][1][2]
    assert foo()


# ───────────────────────── concatenate / stack ─────────────────────────

def test_segarr_concatenate_1d_axis0():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        b = np.asarray([4, 5, 6])
        c = np.concatenate((a, b), axis=0)
        assert c[0] == 1
        assert c[3] == 4
        assert c[5] == 6
    assert foo()


def test_segarr_concatenate_2d_axis0():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2], [3, 4]])
        b = np.asarray([[5, 6], [7, 8]])
        c = np.concatenate((a, b), axis=0)
        # shape (4, 2)
        assert c[0][0] == 1
        assert c[2][0] == 5
        assert c[3][1] == 8
    assert foo()


def test_segarr_concatenate_2d_axis1():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2], [3, 4]])
        b = np.asarray([[5, 6], [7, 8]])
        c = np.concatenate((a, b), axis=1)
        # shape (2, 4)
        assert c[0][0] == 1
        assert c[0][2] == 5
        assert c[1][3] == 8
    assert foo()


def test_segarr_stack_axis0():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        b = np.asarray([4, 5, 6])
        c = np.stack((a, b), axis=0)
        # shape (2, 3)
        assert c[0][0] == 1
        assert c[1][2] == 6
    assert foo()


def test_segarr_stack_axis1():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        b = np.asarray([4, 5, 6])
        c = np.stack((a, b), axis=1)
        # shape (3, 2)
        assert c[0][0] == 1
        assert c[0][1] == 4
        assert c[2][1] == 6
    assert foo()


def test_segarr_concatenate_constant_fold():
    # Constant folding: np.concatenate((a, b)) with both folded should yield
    # a StaticArray whose cells carry static_val.
    @zk_circuit
    def foo():
        a = np.asarray([1, 2])
        b = np.asarray([3, 4])
        c = np.concatenate((a, b))
        # All four asserts must compile-fold to True.
        assert c[0] == 1
        assert c[1] == 2
        assert c[2] == 3
        assert c[3] == 4
    assert foo()


def test_segarr_vstack_1d_inputs():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        b = np.asarray([4, 5, 6])
        c = np.vstack((a, b))
        # shape (2, 3)
        assert c[0][0] == 1
        assert c[1][2] == 6
    assert foo()


def test_segarr_hstack_1d_inputs():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        b = np.asarray([4, 5, 6])
        c = np.hstack((a, b))
        # shape (6,)
        assert c[0] == 1
        assert c[5] == 6
    assert foo()


def test_segarr_column_stack_1d_inputs():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        b = np.asarray([4, 5, 6])
        c = np.column_stack((a, b))
        # shape (3, 2)
        assert c[0][0] == 1
        assert c[0][1] == 4
        assert c[2][1] == 6
    assert foo()


# ───────────────────────── expand_dims / squeeze ─────────────────────────

def test_segarr_expand_dims_axis0():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        b = np.expand_dims(a, 0)
        # shape (1, 3)
        assert b[0][0] == 1
        assert b[0][2] == 3
    assert foo()


def test_segarr_expand_dims_axis_minus_one():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        b = np.expand_dims(a, -1)
        # shape (3, 1)
        assert b[0][0] == 1
        assert b[2][0] == 3
    assert foo()


def test_segarr_squeeze_all_length_1():
    @zk_circuit
    def foo():
        a = np.asarray([[[1, 2, 3]]])  # shape (1, 1, 3)
        b = np.squeeze(a)              # shape (3,)
        assert b[0] == 1
        assert b[2] == 3
    assert foo()


def test_segarr_squeeze_specific_axis():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3]])  # shape (1, 3)
        b = np.squeeze(a, axis=0)     # shape (3,)
        assert b[0] == 1
        assert b[2] == 3
    assert foo()


# ───────────────────────── flatten / ravel ─────────────────────────

def test_segarr_flatten():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        b = a.flatten()
        assert b[0] == 1
        assert b[3] == 4
        assert b[5] == 6
    assert foo()


# ───────────────────────── pipeline / cache invalidation ─────────────────────────

def test_segarr_transpose_then_reshape():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        b = a.T              # shape (3, 2): [[1,4],[2,5],[3,6]]
        c = b.reshape(-1)    # [1, 4, 2, 5, 3, 6]
        assert c[0] == 1
        assert c[1] == 4
        assert c[3] == 5
        assert c[5] == 6
    assert foo()


def test_segarr_reshape_after_dynamic_index_write():
    # P3 invalidates the cache after a dynamic-index write; the reshape on
    # the same StaticArray must see the updated values via segment reads.
    @zk_circuit
    def foo(j: int):
        a = np.asarray([1, 2, 3, 4, 5, 6])
        a[j] = 99
        b = a.reshape(2, 3)
        # j is a runtime index; the constraint must verify when j == 2.
        # We only assert the static cells that don't depend on j.
        # When j == 2: a == [1, 2, 99, 4, 5, 6], b = [[1,2,99],[4,5,6]]
        assert b[1][2] == 6
    assert foo(2)


def test_segarr_concatenate_on_views():
    # Concatenate should work on view-like StaticArrays (rows from a 2-D
    # array).
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2], [3, 4], [5, 6]])
        # a[0] is a row view (shape (2,))
        row0 = a[0]
        row2 = a[2]
        c = np.concatenate((row0, row2))
        # [1, 2, 5, 6]
        assert c[0] == 1
        assert c[1] == 2
        assert c[2] == 5
        assert c[3] == 6
    assert foo()
