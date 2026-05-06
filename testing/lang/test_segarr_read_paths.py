"""Regression tests for `compiler.segarr-read-paths` (P2 of the
segment-native-static-arrays epic).

These exercise native `Value::StaticArray` read paths: element access at
static / dynamic indices, multi-dim subscripts (`arr[i, j]` and `arr[i][j]`),
slice reads (static contiguous → view; static non-contiguous → materialise;
dynamic bound → materialise), and for-loop / comprehension iteration.
"""

from zinnia import *


# ───────────────────────── 1-D element reads ─────────────────────────────

def test_static_array_1d_static_index():
    @zk_circuit
    def foo():
        a = np.asarray([10, 20, 30, 40])
        assert a[0] == 10
        assert a[2] == 30
        assert a[3] == 40
    assert foo()


def test_static_array_1d_dynamic_index():
    @zk_circuit
    def foo(j: Integer):
        a = np.asarray([10, 20, 30, 40])
        assert a[j] == a[j]  # tautology — exercises the dynamic read path.
        # Spot-check: index 0 yields 10.
        if j == 0:
            assert a[j] == 10
    assert foo(0)
    assert foo(2)


def test_static_array_1d_negative_index():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4, 5])
        assert a[-1] == 5
        assert a[-2] == 4
    assert foo()


# ───────────────────────── 2-D element reads ─────────────────────────────

def test_static_array_2d_tuple_index():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        assert a[0, 0] == 1
        assert a[0, 2] == 3
        assert a[1, 1] == 5
    assert foo()


def test_static_array_2d_chained_index():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        assert a[0][0] == 1
        assert a[0][2] == 3
        assert a[1][1] == 5
    assert foo()


def test_static_array_2d_dynamic_inner():
    @zk_circuit
    def foo(j: Integer):
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        if j == 0:
            assert a[1, j] == 4
        if j == 2:
            assert a[1, j] == 6
    assert foo(0)
    assert foo(2)


def test_static_array_2d_dynamic_outer_then_static_inner():
    @zk_circuit
    def foo(i: Integer):
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        if i == 0:
            assert a[i, 1] == 2
        if i == 1:
            assert a[i, 1] == 5
    assert foo(0)
    assert foo(1)


def test_static_array_2d_chained_dynamic():
    @zk_circuit
    def foo(i: Integer, j: Integer):
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        # Chained dynamic subscript exercises view+native read.
        if i == 0 and j == 0:
            assert a[i][j] == 1
        if i == 1 and j == 2:
            assert a[i][j] == 6
    assert foo(0, 0)
    assert foo(1, 2)


# ───────────────────────────── Slice reads ────────────────────────────────

def test_static_array_1d_slice_view():
    """step=1, contiguous → view path."""
    @zk_circuit
    def foo():
        a = np.asarray([10, 20, 30, 40, 50])
        b = a[1:4]
        assert b[0] == 20
        assert b[2] == 40
    assert foo()


def test_static_array_1d_slice_step():
    """step=2 → materialise path."""
    @zk_circuit
    def foo():
        a = np.asarray([0, 1, 2, 3, 4, 5, 6, 7])
        b = a[1:7:2]   # [1, 3, 5]
        assert b[0] == 1
        assert b[1] == 3
        assert b[2] == 5
    assert foo()


def test_static_array_1d_slice_reverse():
    """Explicit reverse slice (step=-1, explicit start/stop) → materialise."""
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4, 5])
        b = a[4:0:-1]   # [5, 4, 3, 2]
        assert b[0] == 5
        assert b[3] == 2
    assert foo()


def test_static_array_2d_slice_runtime_bound():
    """Dynamic stop on inner axis → materialise path."""
    @zk_circuit
    def foo(k: Integer):
        a = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]])
        b = a[:, :k]
        # b is shape (2, max_axis=4); first cell (in-bounds for k>=1) is 1.
        if k >= 1:
            assert b[0, 0] == 1
    assert foo(2)
    assert foo(4)


# ─────────────────────────── Iteration ────────────────────────────────────

def test_static_array_1d_for_loop_sum():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4, 5])
        s = 0
        for x in a:
            s = s + x
        assert s == 15
    assert foo()


def test_static_array_1d_comprehension_sum():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3, 4])
        b = [x + 10 for x in a]
        assert b[0] + b[1] + b[2] + b[3] == 50
    assert foo()


def test_static_array_2d_for_loop_row_sum():
    """for x in arr: x is a row view; we sum its scalar elements."""
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        total = 0
        for row in a:
            for v in row:
                total = total + v
        assert total == 21
    assert foo()


# ────────────── Mixed read + legacy-write (P1 shim) ──────────────────────

def test_static_array_dynamic_read_in_loop_with_scalar_write():
    """Dynamic-index read of a StaticArray inside a loop, accumulating into
    a scalar. The scalar update goes through the legacy path; the array
    reads use the native P2 dispatch."""
    @zk_circuit
    def foo(j: Integer):
        a = np.asarray([10, 20, 30, 40])
        s = 0
        for _ in range(3):
            s = s + a[j]
        # When j==0, sum is 30. When j==3, sum is 120.
        if j == 0:
            assert s == 30
        if j == 3:
            assert s == 120
    assert foo(0)
    assert foo(3)
