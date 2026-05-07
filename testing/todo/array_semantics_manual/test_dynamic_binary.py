"""
End-to-end integration tests for dynamic ndarray element-wise binary ops.

Exercises: promote static → dynamic, then binary operations (+, -, *, /,
comparisons), scalar broadcasting, static+dynamic auto-promotion,
dtype promotion (int + float), and 1D-against-2D broadcasting.

Dynamic ndarrays are NOT used as circuit inputs. All arrays are constructed
inside the circuit and promoted via np.promote_to_dynamic.
"""

from zinnia import *


# ── Arithmetic ops ───────────────────────────────────────────────────────

def test_add_same_shape():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3, 4]))
        b = np.promote_to_dynamic(np.asarray([10, 20, 30, 40]))
        c = a + b
        assert c.sum() == 110  # 11 + 22 + 33 + 44

    assert foo()


def test_sub():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([10, 20, 30]))
        b = np.promote_to_dynamic(np.asarray([1, 2, 3]))
        c = a - b
        assert c.sum() == 54  # 9 + 18 + 27

    assert foo()


def test_mul():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([2, 3, 4]))
        b = np.promote_to_dynamic(np.asarray([5, 6, 7]))
        c = a * b
        assert c.sum() == 56  # 10 + 18 + 28

    assert foo()


def test_div():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([10, 20, 30]))
        b = np.promote_to_dynamic(np.asarray([2, 5, 10]))
        c = a / b
        assert c.sum() == 12  # 5 + 4 + 3

    assert foo()


# ── Scalar broadcasting ─────────────────────────────────────────────────

def test_scalar_add_right():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3]))
        c = a + 10
        assert c.sum() == 36  # 11 + 12 + 13

    assert foo()


def test_scalar_add_left():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3]))
        c = 100 + a
        assert c.sum() == 306  # 101 + 102 + 103

    assert foo()


def test_scalar_mul():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3]))
        c = a * 5
        assert c.sum() == 30  # 5 + 10 + 15

    assert foo()


# ── Static + dynamic auto-promotion ─────────────────────────────────────

def test_static_plus_dynamic():
    @zk_circuit
    def foo():
        static_arr = np.asarray([1, 2, 3])
        dynamic_arr = np.promote_to_dynamic(np.asarray([10, 20, 30]))
        c = static_arr + dynamic_arr
        assert c.sum() == 66  # 11 + 22 + 33

    assert foo()


def test_dynamic_plus_static():
    @zk_circuit
    def foo():
        dynamic_arr = np.promote_to_dynamic(np.asarray([10, 20, 30]))
        static_arr = np.asarray([1, 2, 3])
        c = dynamic_arr + static_arr
        assert c.sum() == 66

    assert foo()


# ── Chained binary ops ──────────────────────────────────────────────────

def test_chained_ops():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3]))
        b = np.promote_to_dynamic(np.asarray([4, 5, 6]))
        # (a + b) * a = [5, 14, 27] → sum = 46
        c = (a + b) * a
        assert c.sum() == 46

    assert foo()


# ── Broadcasting: 1D against 2D ─────────────────────────────────────────

def test_broadcast_1d_2d_add():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3]))
        b = np.promote_to_dynamic(np.asarray([[10, 20, 30], [40, 50, 60]]))
        c = a + b
        # [[11,22,33],[41,52,63]] → sum = 222
        assert c.sum() == 222

    assert foo()


# ── Comparison ops ───────────────────────────────────────────────────────

def test_comparison_gt():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 5, 3, 7]))
        b = np.promote_to_dynamic(np.asarray([2, 4, 3, 8]))
        mask = a > b  # [0, 1, 0, 0]
        assert mask.sum() == 1

    assert foo()


def test_comparison_eq():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3]))
        b = np.promote_to_dynamic(np.asarray([1, 5, 3]))
        mask = a == b  # [1, 0, 1]
        assert mask.sum() == 2

    assert foo()


# ── Unary ops ────────────────────────────────────────────────────────────

def test_unary_negation():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3]))
        b = -a
        assert b.sum() == -6

    assert foo()


def test_negate_then_add():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3]))
        b = np.promote_to_dynamic(np.asarray([10, 20, 30]))
        c = -a + b  # [-1+10, -2+20, -3+30] = [9, 18, 27]
        assert c.sum() == 54

    assert foo()


# ── np.add / np.subtract ────────────────────────────────────────────────

def test_np_add():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3]))
        b = np.promote_to_dynamic(np.asarray([10, 20, 30]))
        c = np.add(a, b)
        assert c.sum() == 66

    assert foo()


def test_np_subtract():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([10, 20, 30]))
        b = np.promote_to_dynamic(np.asarray([1, 2, 3]))
        c = np.subtract(a, b)
        assert c.sum() == 54

    assert foo()


def test_np_multiply():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([2, 3, 4]))
        b = np.promote_to_dynamic(np.asarray([5, 6, 7]))
        c = np.multiply(a, b)
        assert c.sum() == 56

    assert foo()


def test_np_negative():
    @zk_circuit
    def foo():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3]))
        b = np.negative(a)
        assert b.sum() == -6

    assert foo()
