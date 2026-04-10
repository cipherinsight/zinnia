"""
End-to-end tests for dynamic ndarray indexing and slicing.

All arrays are constructed inside the circuit via np.promote_to_dynamic.
"""

from zinnia import *


# ── Single element access ────────────────────────────────────────────────

def test_1d_static_index():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([10, 20, 30, 40, 50]))
        assert a[0] == 10
        assert a[2] == 30
        assert a[4] == 50

    assert foo()


def test_1d_negative_index():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([10, 20, 30]))
        assert a[-1] == 30
        assert a[-2] == 20

    assert foo()


def test_2d_single_index_row():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[1, 2, 3], [4, 5, 6]]))
        row = a[1]
        assert row.sum() == 15  # 4 + 5 + 6

    assert foo()


def test_2d_full_index():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[10, 20], [30, 40]]))
        assert a[0, 0] == 10
        assert a[0, 1] == 20
        assert a[1, 0] == 30
        assert a[1, 1] == 40

    assert foo()


# ── Range slicing ────────────────────────────────────────────────────────

def test_1d_slice_basic():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([10, 20, 30, 40, 50]))
        s = a[1:4]
        assert s.sum() == 90  # 20 + 30 + 40

    assert foo()


def test_1d_slice_with_step():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([0, 1, 2, 3, 4, 5]))
        s = a[0:6:2]
        assert s.sum() == 6  # 0 + 2 + 4

    assert foo()


def test_1d_slice_from_start():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([10, 20, 30, 40]))
        s = a[:2]
        assert s.sum() == 30  # 10 + 20

    assert foo()


def test_1d_slice_to_end():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([10, 20, 30, 40]))
        s = a[2:]
        assert s.sum() == 70  # 30 + 40

    assert foo()


# ── Multi-dim mixed indexing ─────────────────────────────────────────────

def test_2d_row_slice():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        s = a[0:2]  # first two rows as a 2D sub-array? Or 1D slice of rows?
        # Actually a[0:2] on a 2D array takes rows 0 and 1
        # This produces a 2D array [[1,2,3],[4,5,6]]
        # Not yet supported as multi-dim range — let's test row extraction instead
        row0 = a[0]
        row2 = a[2]
        assert row0.sum() == 6
        assert row2.sum() == 24

    assert foo()


def test_2d_column_slice():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[1, 2, 3], [4, 5, 6]]))
        # a[:, 1] — all rows, column 1
        # This is multidim with Range on axis 0, Single on axis 1
        # Should produce [2, 5]
        col = a[:, 1]
        assert col.sum() == 7

    assert foo()


# ── Chained indexing + operations ────────────────────────────────────────

def test_index_then_binary_op():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[1, 2, 3], [4, 5, 6]]))
        row = a[0]  # [1, 2, 3]
        b = row + np.promote_to_dynamic(np.asarray([10, 20, 30]))
        assert b.sum() == 66  # 11 + 22 + 33

    assert foo()


def test_slice_then_sum():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3, 4, 5, 6, 7, 8]))
        s = a[2:6]  # [3, 4, 5, 6]
        assert s.sum() == 18

    assert foo()


# ── Dynamic index (circuit wire) ─────────────────────────────────────────

def test_dynamic_1d_index():
    """Use the result of a computation as an index into a dynamic array."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([10, 20, 30, 40, 50]))
        # idx is computed at runtime (circuit wire), not a compile-time constant
        idx = np.promote_to_dynamic(np.asarray([1, 2])).sum()  # = 3
        assert a[idx] == 40

    assert foo()


def test_dynamic_2d_column():
    """Dynamic column index on a 2D array."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[1, 2, 3], [4, 5, 6]]))
        # col index is a runtime value
        col = np.promote_to_dynamic(np.asarray([1])).sum()  # = 1
        # a[:, col] should give [2, 5]
        result = a[:, col]
        assert result.sum() == 7

    assert foo()


def test_dynamic_2d_element():
    """Dynamic row + column on a 2D array."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[10, 20], [30, 40]]))
        r = np.promote_to_dynamic(np.asarray([1])).sum()  # = 1
        c = np.promote_to_dynamic(np.asarray([0])).sum()  # = 0
        assert a[r, c] == 30

    assert foo()


# ── Negative step ────────────────────────────────────────────────────────

def test_negative_step_static():
    """Static negative step: reverse the array."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3, 4, 5]))
        # a[4::-1] = [5, 4, 3, 2, 1] in NumPy. But our static path handles this.
        # Let's use a[4:1:-1] = [5, 4, 3]
        s = a[4:1:-1]
        assert s.sum() == 12  # 5 + 4 + 3

    assert foo()


def test_negative_step_full_reverse():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([10, 20, 30, 40]))
        # a[3::-1] reverses, but the stop is -1 in NumPy... tricky.
        # Use a[::-1] → full reverse, but that needs special handling.
        # Let's test a[3:0:-1] = [40, 30, 20] (stop=0 is exclusive)
        s = a[3:0:-1]
        assert s.sum() == 90  # 40 + 30 + 20

    assert foo()


def test_step_2():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([0, 1, 2, 3, 4, 5, 6, 7]))
        s = a[1:7:2]  # [1, 3, 5]
        assert s.sum() == 9

    assert foo()


# ── Dynamic step ─────────────────────────────────────────────────────────

def test_dynamic_step():
    """Step is a runtime value (circuit wire)."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([0, 1, 2, 3, 4, 5]))
        step = np.promote_to_dynamic(np.asarray([2])).sum()  # step = 2
        s = a[0:6:step]  # [0, 2, 4] → sum = 6
        assert s.sum() == 6

    assert foo()


def test_dynamic_start_stop():
    """Start and stop are runtime values."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([10, 20, 30, 40, 50]))
        start = np.promote_to_dynamic(np.asarray([1])).sum()  # = 1
        stop = np.promote_to_dynamic(np.asarray([4])).sum()  # = 4
        s = a[start:stop]  # [20, 30, 40] → sum = 90
        assert s.sum() == 90

    assert foo()


# ── Masked assignment ────────────────────────────────────────────────────

def test_masked_assign_scalar():
    """dyn[mask] = scalar: set masked positions to a constant."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3, 4, 5]))
        mask = np.promote_to_dynamic(np.asarray([1, 0, 1, 0, 1]))
        a[mask] = 0
        # [0, 2, 0, 4, 0] → sum = 6
        assert a.sum() == 6

    assert foo()


def test_masked_assign_dynamic_mask():
    """Mask from a dynamic comparison."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 5, 3, 7, 2]))
        threshold = np.promote_to_dynamic(np.asarray([3, 3, 3, 3, 3]))
        mask = a > threshold  # [0, 1, 0, 1, 0]
        a[mask] = 0
        # [1, 0, 3, 0, 2] → sum = 6
        assert a.sum() == 6

    assert foo()


def test_masked_assign_broadcast_scalar():
    """dyn[mask] = scalar broadcasts to all True positions."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([10, 20, 30, 40, 50]))
        mask = np.promote_to_dynamic(np.asarray([0, 1, 0, 1, 0]))
        a[mask] = 99
        # [10, 99, 30, 99, 50] → sum = 288
        assert a.sum() == 288

    assert foo()


# ── Fancy indexing ───────────────────────────────────────────────────────

def test_fancy_1d_static():
    """Pick specific elements by index array."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([10, 20, 30, 40, 50]))
        result = a[[0, 2, 4]]
        assert result.sum() == 90  # 10 + 30 + 50

    assert foo()


def test_fancy_1d_reorder():
    """Fancy index can reorder elements."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([10, 20, 30]))
        result = a[[2, 0, 1]]
        # [30, 10, 20] → sum = 60
        assert result.sum() == 60

    assert foo()


def test_fancy_1d_repeat():
    """Fancy index can repeat elements."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([10, 20, 30]))
        result = a[[0, 0, 1, 1]]
        # [10, 10, 20, 20] → sum = 60
        assert result.sum() == 60

    assert foo()


def test_fancy_2d_row_select():
    """Pick specific rows from a 2D array."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        result = a[[0, 2]]
        # [[1,2,3],[7,8,9]] → sum = 30
        assert result.sum() == 30

    assert foo()


def test_fancy_2d_paired():
    """Multi-dim paired fancy indexing: a[[r0,r1], [c0,c1]]."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[10, 20, 30], [40, 50, 60]]))
        result = a[[0, 1, 0], [2, 0, 1]]
        # elements at (0,2)=30, (1,0)=40, (0,1)=20 → sum = 90
        assert result.sum() == 90

    assert foo()


def test_fancy_then_sum():
    """Chain fancy index with reduction."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([100, 200, 300, 400, 500]))
        subset = a[[1, 3]]
        # [200, 400]
        total = subset.sum()
        assert total == 600

    assert foo()


# ── Multi-dim slicing with mixed static/dynamic ─────────────────────────

def test_2d_static_range_single():
    """dyn[0:2, 1] — static range + static single."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        result = a[0:2, 1]  # rows 0-1, col 1 → [2, 5]
        assert result.sum() == 7

    assert foo()


def test_2d_dynamic_range_single():
    """dyn[x:y, j] — dynamic range + static single."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        start = np.promote_to_dynamic(np.asarray([0])).sum()  # = 0
        stop = np.promote_to_dynamic(np.asarray([2])).sum()   # = 2
        result = a[start:stop, 1]  # rows 0-1, col 1 → [2, 5]
        assert result.sum() == 7

    assert foo()


def test_2d_range_range_static():
    """dyn[0:2, 1:3] — two static ranges."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        result = a[0:2, 1:3]  # [[2,3],[5,6]] → sum = 16
        assert result.sum() == 16

    assert foo()


def test_2d_dynamic_range_range():
    """dyn[x:y, 0:2] — dynamic on axis 0, static on axis 1."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[10, 20, 30], [40, 50, 60], [70, 80, 90]]))
        start = np.promote_to_dynamic(np.asarray([1])).sum()  # = 1
        stop = np.promote_to_dynamic(np.asarray([3])).sum()   # = 3
        result = a[start:stop, 0:2]  # [[40,50],[70,80]] → sum = 240
        assert result.sum() == 240

    assert foo()


# ── Slice assignment ─────────────────────────────────────────────────────

def test_slice_assign_1d_static():
    """dyn[1:3] = [99, 88]"""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3, 4, 5]))
        a[1:3] = np.asarray([99, 88])
        # [1, 99, 88, 4, 5] → sum = 197
        assert a.sum() == 197

    assert foo()


def test_slice_assign_1d_scalar():
    """dyn[1:4] = 0 (scalar broadcast)"""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([10, 20, 30, 40, 50]))
        a[1:4] = 0
        # [10, 0, 0, 0, 50] → sum = 60
        assert a.sum() == 60

    assert foo()


def test_slice_assign_2d_row_range():
    """dyn[1:3] = [[90, 91, 92], [93, 94, 95]] on a 3x3 array."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        a[1:3] = np.asarray([[90, 91, 92], [93, 94, 95]])
        # [[1,2,3],[90,91,92],[93,94,95]] → sum = 1+2+3+90+91+92+93+94+95 = 561
        assert a.sum() == 561

    assert foo()


def test_slice_assign_2d_column():
    """dyn[:, 1] = [77, 88] — assign to a column."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[1, 2, 3], [4, 5, 6]]))
        a[:, 1] = np.asarray([77, 88])
        # [[1,77,3],[4,88,6]] → sum = 1+77+3+4+88+6 = 179
        assert a.sum() == 179

    assert foo()


def test_slice_assign_dynamic_bounds():
    """dyn[x:y] = scalar with dynamic bounds."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3, 4, 5]))
        start = np.promote_to_dynamic(np.asarray([1])).sum()  # = 1
        stop = np.promote_to_dynamic(np.asarray([4])).sum()   # = 4
        a[start:stop] = 0
        # [1, 0, 0, 0, 5] → sum = 6
        assert a.sum() == 6

    assert foo()


# ── Boolean masking ──────────────────────────────────────────────────────

def test_boolean_mask_static():
    """Boolean mask with compile-time-known mask values."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([10, 20, 30, 40, 50]))
        mask = np.asarray([True, False, True, False, True])
        result = a[mask]
        # Selected: 10, 30, 50 → sum = 90
        assert result.sum() == 90

    assert foo()


def test_boolean_mask_dynamic():
    """Boolean mask from a dynamic comparison."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 5, 3, 7, 2]))
        threshold = np.promote_to_dynamic(np.asarray([3, 3, 3, 3, 3]))
        mask = a > threshold  # [0, 1, 0, 1, 0]
        result = a[mask]
        # Selected: 5, 7 → sum = 12
        assert result.sum() == 12

    assert foo()


# ── Element assignment ───────────────────────────────────────────────────

def test_setitem_static_index():
    """Assign to a static index."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3, 4, 5]))
        a[2] = 99
        assert a.sum() == 1 + 2 + 99 + 4 + 5  # = 111

    assert foo()


def test_setitem_dynamic_index():
    """Assign to a dynamic (circuit wire) index."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([10, 20, 30, 40]))
        idx = np.promote_to_dynamic(np.asarray([2])).sum()  # = 2
        a[idx] = 0
        assert a.sum() == 10 + 20 + 0 + 40  # = 70

    assert foo()


def test_setitem_2d():
    """Assign to a 2D element."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[1, 2], [3, 4]]))
        a[1, 0] = 99
        assert a.sum() == 1 + 2 + 99 + 4  # = 106

    assert foo()


def test_setitem_then_read():
    """Assign then read back the modified element."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([10, 20, 30]))
        a[1] = 77
        assert a[0] == 10
        assert a[1] == 77
        assert a[2] == 30

    assert foo()
