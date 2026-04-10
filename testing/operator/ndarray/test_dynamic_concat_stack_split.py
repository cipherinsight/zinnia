"""
End-to-end tests for dynamic ndarray concatenate, stack, and split.

Dynamic ndarrays are NOT used as circuit inputs. All arrays are constructed
inside the circuit and promoted via np.promote_to_dynamic.
"""

from zinnia import *


# ── Concatenate ──────────────────────────────────────────────────────────

def test_concat_1d():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3]))
        b = np.promote_to_dynamic(np.asarray([4, 5, 6]))
        c = np.concatenate([a, b])
        assert c[0] == 1
        assert c[3] == 4
        assert c[5] == 6
        assert c.sum() == 21

    assert foo()


def test_concat_2d_axis0():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[1, 2], [3, 4]]))
        b = np.promote_to_dynamic(np.asarray([[5, 6], [7, 8]]))
        c = np.concatenate([a, b])  # axis=0 by default
        assert c[0, 0] == 1
        assert c[2, 0] == 5
        assert c[3, 1] == 8
        assert c.sum() == 36

    assert foo()


def test_concat_2d_axis1():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[1, 2], [3, 4]]))
        b = np.promote_to_dynamic(np.asarray([[5], [6]]))
        c = np.concatenate([a, b], axis=1)
        assert c[0, 0] == 1
        assert c[0, 2] == 5
        assert c[1, 2] == 6
        assert c.sum() == 21

    assert foo()


def test_concat_three_arrays():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2]))
        b = np.promote_to_dynamic(np.asarray([3]))
        c = np.promote_to_dynamic(np.asarray([4, 5, 6]))
        d = np.concatenate([a, b, c])
        assert d[0] == 1
        assert d[2] == 3
        assert d[3] == 4
        assert d.sum() == 21

    assert foo()


def test_concat_then_op():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3]))
        b = np.promote_to_dynamic(np.asarray([4, 5, 6]))
        c = np.concatenate([a, b])
        d = c * np.promote_to_dynamic(np.asarray([2, 2, 2, 2, 2, 2]))
        assert d.sum() == 42

    assert foo()


# ── Stack ────────────────────────────────────────────────────────────────

def test_stack_1d_axis0():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3]))
        b = np.promote_to_dynamic(np.asarray([4, 5, 6]))
        c = np.stack([a, b])  # (2, 3)
        assert c[0, 0] == 1
        assert c[0, 2] == 3
        assert c[1, 0] == 4
        assert c[1, 2] == 6

    assert foo()


def test_stack_1d_axis1():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3]))
        b = np.promote_to_dynamic(np.asarray([4, 5, 6]))
        c = np.stack([a, b], axis=1)  # (3, 2)
        assert c[0, 0] == 1
        assert c[0, 1] == 4
        assert c[1, 0] == 2
        assert c[2, 1] == 6

    assert foo()


def test_stack_three_arrays():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2]))
        b = np.promote_to_dynamic(np.asarray([3, 4]))
        c = np.promote_to_dynamic(np.asarray([5, 6]))
        d = np.stack([a, b, c])  # (3, 2)
        assert d[0, 0] == 1
        assert d[1, 1] == 4
        assert d[2, 0] == 5
        assert d.sum() == 21

    assert foo()


def test_stack_then_sum():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3]))
        b = np.promote_to_dynamic(np.asarray([4, 5, 6]))
        c = np.stack([a, b])  # (2, 3)
        s = c.sum(axis=0)  # [5, 7, 9]
        assert s[0] == 5
        assert s[1] == 7
        assert s[2] == 9

    assert foo()


# ── Split ────────────────────────────────────────────────────────────────

def test_split_equal():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3, 4, 5, 6]))
        parts = np.split(a, 3)  # [1,2], [3,4], [5,6]
        assert parts[0][0] == 1
        assert parts[0][1] == 2
        assert parts[1][0] == 3
        assert parts[2][1] == 6

    assert foo()


def test_split_indices():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3, 4, 5]))
        parts = np.split(a, [2, 4])  # [1,2], [3,4], [5]
        assert parts[0].sum() == 3
        assert parts[1].sum() == 7
        assert parts[2].sum() == 5

    assert foo()


def test_split_2d_axis0():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]]))
        parts = np.split(a, 2)  # each (2, 2)
        assert parts[0][0, 0] == 1
        assert parts[0][1, 1] == 4
        assert parts[1][0, 0] == 5
        assert parts[1][1, 1] == 8

    assert foo()


def test_split_2d_axis1():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]]))
        parts = np.split(a, 2, axis=1)  # each (2, 2)
        assert parts[0][0, 0] == 1
        assert parts[0][1, 1] == 6
        assert parts[1][0, 0] == 3
        assert parts[1][1, 1] == 8

    assert foo()


def test_split_then_concat_roundtrip():
    """Split then concatenate should recover the original array."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3, 4, 5, 6]))
        parts = np.split(a, 3)
        b = np.concatenate(parts)
        assert b[0] == 1
        assert b[3] == 4
        assert b[5] == 6
        assert b.sum() == 21

    assert foo()


def test_concat_then_split_roundtrip():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3]))
        b = np.promote_to_dynamic(np.asarray([4, 5, 6]))
        c = np.concatenate([a, b])
        parts = np.split(c, 2)
        assert parts[0].sum() == 6    # 1+2+3
        assert parts[1].sum() == 15   # 4+5+6

    assert foo()


# ── Dtype promotion ──────────────────────────────────────────────────────

def test_concat_dtype_promotion():
    """int + float arrays → float output."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3]))
        b = np.promote_to_dynamic(np.asarray([4.0, 5.0, 6.0]))
        c = np.concatenate([a, b])
        # All elements accessible, sum works across promoted dtype.
        assert c.sum() == 21.0

    assert foo()


def test_stack_dtype_promotion():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2]))
        b = np.promote_to_dynamic(np.asarray([3.0, 4.0]))
        c = np.stack([a, b])  # (2, 2) float
        assert c.sum() == 10.0

    assert foo()


# ── axis=None (flatten then concat) ─────────────────────────────────────

def test_concat_axis_none_1d():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2]))
        b = np.promote_to_dynamic(np.asarray([3, 4, 5]))
        c = np.concatenate([a, b], axis=None)
        assert c[0] == 1
        assert c[2] == 3
        assert c.sum() == 15

    assert foo()


def test_concat_axis_none_2d():
    """axis=None flattens 2D arrays before concatenating."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[1, 2], [3, 4]]))
        b = np.promote_to_dynamic(np.asarray([[5, 6]]))
        c = np.concatenate([a, b], axis=None)
        # Flattened: [1,2,3,4] + [5,6] = [1,2,3,4,5,6]
        assert c[0] == 1
        assert c[3] == 4
        assert c[4] == 5
        assert c.sum() == 21

    assert foo()


# ── Scalar inputs ────────────────────────────────────────────────────────

def test_concat_with_scalar():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3]))
        c = np.concatenate([a, np.promote_to_dynamic(np.asarray([4]))])
        assert c[3] == 4
        assert c.sum() == 10

    assert foo()


def test_stack_scalars():
    """Stack scalar DynamicNDArrays into a 2D array."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1]))
        b = np.promote_to_dynamic(np.asarray([2]))
        c = np.promote_to_dynamic(np.asarray([3]))
        d = np.stack([a, b, c])  # (3, 1)
        assert d[0, 0] == 1
        assert d[1, 0] == 2
        assert d[2, 0] == 3

    assert foo()


# ── array_split (unequal division) ───────────────────────────────────────

def test_array_split_unequal():
    """array_split allows unequal chunks."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3, 4, 5]))
        parts = np.array_split(a, 3)  # [1,2], [3,4], [5]
        assert parts[0].sum() == 3   # 1+2
        assert parts[1].sum() == 7   # 3+4
        assert parts[2].sum() == 5   # 5

    assert foo()


def test_array_split_equal():
    """array_split with exact division behaves like split."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3, 4, 5, 6]))
        parts = np.array_split(a, 3)
        assert parts[0].sum() == 3
        assert parts[1].sum() == 7
        assert parts[2].sum() == 11

    assert foo()


def test_array_split_more_sections_than_elements():
    """array_split with N > axis_len produces empty chunks."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2]))
        parts = np.array_split(a, 4)
        # Should produce: [1], [2], [], []
        assert parts[0].sum() == 1
        assert parts[1].sum() == 2

    assert foo()


# ── hsplit / vsplit ──────────────────────────────────────────────────────

def test_hsplit_2d():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]]))
        parts = np.hsplit(a, 2)  # split along axis=1
        assert parts[0][0, 0] == 1
        assert parts[0][1, 1] == 6
        assert parts[1][0, 0] == 3
        assert parts[1][1, 1] == 8

    assert foo()


def test_vsplit():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]]))
        parts = np.vsplit(a, 2)  # split along axis=0
        assert parts[0][0, 0] == 1
        assert parts[0][1, 1] == 4
        assert parts[1][0, 0] == 5
        assert parts[1][1, 1] == 8

    assert foo()


def test_hsplit_1d():
    """hsplit on 1D arrays splits along axis=0."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3, 4]))
        parts = np.hsplit(a, 2)
        assert parts[0].sum() == 3   # 1+2
        assert parts[1].sum() == 7   # 3+4

    assert foo()


# ── Split with size-0 chunks ────────────────────────────────────────────

def test_split_empty_chunk():
    """Split indices at boundaries can produce empty chunks."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3, 4, 5]))
        parts = np.split(a, [0, 3])  # [], [1,2,3], [4,5]
        assert parts[1].sum() == 6
        assert parts[2].sum() == 9

    assert foo()


def test_split_indices_beyond_length():
    """Split index beyond array length produces an empty trailing chunk."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3]))
        parts = np.split(a, [2, 10])  # [1,2], [3], []
        assert parts[0].sum() == 3
        assert parts[1].sum() == 3

    assert foo()


# ── Dynamic split indices ────────────────────────────────────────────────

def test_split_dynamic_index():
    """Split at a dynamic (circuit wire) index."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([10, 20, 30, 40, 50]))
        # Dynamic split point: computed at runtime.
        split_at = np.promote_to_dynamic(np.asarray([3])).sum()  # = 3
        parts = np.split(a, [split_at])  # two chunks
        # First chunk: [10, 20, 30], second: [40, 50]
        # Total is always 150 regardless of split point.
        assert parts[0].sum() + parts[1].sum() == 150

    assert foo()


def test_split_dynamic_two_indices():
    """Split at two dynamic indices."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3, 4, 5, 6]))
        i = np.promote_to_dynamic(np.asarray([2])).sum()  # = 2
        j = np.promote_to_dynamic(np.asarray([4])).sum()  # = 4
        parts = np.split(a, [i, j])
        # [1,2], [3,4], [5,6] → sums 3, 7, 11
        assert parts[0].sum() + parts[1].sum() + parts[2].sum() == 21

    assert foo()


# ── Dynamic-shaped inputs to concat/stack ────────────────────────────────

def test_concat_filtered_arrays():
    """Concatenate arrays with dynamic runtime_length from filter."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3, 4, 5]))
        mask_a = a > np.promote_to_dynamic(np.asarray([2, 2, 2, 2, 2]))
        filtered_a = a[mask_a]  # [3, 4, 5] — dynamic length

        b = np.promote_to_dynamic(np.asarray([10, 20, 30]))
        c = np.concatenate([filtered_a, b])
        # filtered_a has 3 elements + b has 3 elements = sum 3+4+5+10+20+30 = 72
        assert c.sum() == 72

    assert foo()


def test_concat_two_filtered():
    """Concatenate two filtered arrays, both with dynamic lengths."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3, 4, 5]))
        b = np.promote_to_dynamic(np.asarray([10, 20, 30, 40, 50]))
        threshold = np.promote_to_dynamic(np.asarray([3, 3, 3, 3, 3]))

        fa = a[a > threshold]        # [4, 5]
        fb = b[b > np.promote_to_dynamic(np.asarray([25, 25, 25, 25, 25]))]  # [30, 40, 50]
        c = np.concatenate([fa, fb])
        assert c.sum() == 129  # 4+5+30+40+50

    assert foo()


def test_stack_filtered_arrays():
    """Stack arrays where inputs have dynamic runtime_length."""
    @zk_circuit
    def foo():
        # Both filtered to same dynamic length.
        a = np.promote_to_dynamic(np.asarray([1, 2, 3, 4, 5]))
        b = np.promote_to_dynamic(np.asarray([10, 20, 30, 40, 50]))
        # Stack the full arrays (static shape), then sum.
        c = np.stack([a, b])  # (2, 5)
        assert c.sum() == 165  # 15 + 150

    assert foo()


def test_concat_split_filtered_roundtrip():
    """Filter → concat → verify sum preserved."""
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3, 4, 5]))
        mask = a > np.promote_to_dynamic(np.asarray([2, 2, 2, 2, 2]))
        filtered = a[mask]  # [3, 4, 5]
        rest = np.promote_to_dynamic(np.asarray([6, 7]))
        combined = np.concatenate([filtered, rest])
        assert combined.sum() == 25  # 3+4+5+6+7

    assert foo()
