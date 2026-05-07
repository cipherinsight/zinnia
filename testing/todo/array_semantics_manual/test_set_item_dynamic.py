"""
Tests for dynamic-index slice assignment and read-back. These cover the
combinations that were silently broken before the set_nested_value /
multidim_subscript "row mux" fix:

  - arr[x, :] = vec   (write a row at a runtime-known row index)
  - arr[x, :] = scalar
  - arr[x, 1:3] = vec
  - arr[x, j] = v     (existing test_set_single_item_by_variable still works)
  - reading arr[x, :] back as a row
"""

from zinnia import *


# ───────────────────────────────────────────────────────────────────────
# Write side: arr[x, ...] = ...
# ───────────────────────────────────────────────────────────────────────

def test_dynamic_row_full_slice_write():
    @zk_circuit
    def foo(x: int):
        array = np.zeros((4, 4), dtype=int)
        array[x, :] = [1, 2, 3, 4]
        # The targeted row should be the new vector.
        assert array[x, 0] == 1
        assert array[x, 1] == 2
        assert array[x, 2] == 3
        assert array[x, 3] == 4
        # Other rows must remain zero (checked at static row indices).
        for i in range(4):
            if i != x:
                assert array[i, 0] == 0
                assert array[i, 3] == 0

    for i in range(4):
        assert foo(i)


def test_dynamic_row_full_slice_scalar_broadcast():
    @zk_circuit
    def foo(x: int):
        array = np.zeros((4, 4), dtype=int)
        array[x, :] = 7
        for j in range(4):
            assert array[x, j] == 7
        for i in range(4):
            if i != x:
                for j in range(4):
                    assert array[i, j] == 0

    for i in range(4):
        assert foo(i)


def test_dynamic_row_partial_slice_write():
    @zk_circuit
    def foo(x: int):
        array = np.zeros((4, 4), dtype=int)
        array[x, 1:3] = [9, 8]
        # Targeted row: cols 1 and 2 written, cols 0 and 3 unchanged
        assert array[x, 0] == 0
        assert array[x, 1] == 9
        assert array[x, 2] == 8
        assert array[x, 3] == 0

    for i in range(4):
        assert foo(i)


def test_dynamic_row_dynamic_col_still_works():
    """Original `arr[x, y] = v` path must keep working."""
    @zk_circuit
    def foo(x: int, y: int):
        array = np.zeros((4, 4), dtype=int)
        array[x, y] = 42
        assert array[x, y] == 42
        # A different cell should still be zero.
        other_x = (x + 1) % 4
        assert array[other_x, y] == 0

    for i in range(4):
        for j in range(4):
            assert foo(i, j)


# ───────────────────────────────────────────────────────────────────────
# Read side: arr[x, :] returns the right row, including when subsequently
# combined with broadcast / comparison.
# ───────────────────────────────────────────────────────────────────────

def test_dynamic_row_full_slice_read():
    @zk_circuit
    def foo(x: int):
        array = np.asarray([
            [10, 20, 30, 40],
            [50, 60, 70, 80],
            [90, 100, 110, 120],
            [130, 140, 150, 160],
        ])
        row = array[x, :]
        # Element-wise check via static indices into the read-back row.
        assert row[0] == 10 + x * 40
        assert row[3] == 40 + x * 40

    for i in range(4):
        assert foo(i)


def test_dynamic_row_full_slice_read_then_compare():
    """The original baseline failure was actually two bugs in one: write,
    then read-and-compare. Pin the round-trip down."""
    @zk_circuit
    def foo(x: int):
        array = np.zeros((4, 4), dtype=int)
        array[x, :] = [1, 2, 3, 4]
        assert (array[x, :] == np.asarray([1, 2, 3, 4])).all()
        for i in range(4):
            if i != x:
                assert (array[i, :] == np.asarray([0, 0, 0, 0])).all()

    for i in range(4):
        assert foo(i)
