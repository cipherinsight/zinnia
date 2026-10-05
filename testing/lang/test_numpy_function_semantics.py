"""NumPy-differential regression tests for operators that used to compute a
different function from NumPy (audit finding B5).

- ``/`` on integer operands was lowered to floor division; NumPy and Python
  ``/`` is true division and returns floats (cases 0452/0454).
- ``np.correlate(a, v, 'valid')`` with ``len(v) > len(a)`` swapped the
  operands without reversing the output as NumPy does (case 0422).
- ``np.unique`` returned the distinct values in first-occurrence order
  instead of ascending order (case 0298).

Every test compares the circuit's outputs with NumPy position by position.
"""
import numpy as np
import pytest

from testing.lang.mock_outputs import assert_matches_numpy, compile_kernel, numpy_kernel, run_mock


# ── true division ────────────────────────────────────────────────────────

@pytest.mark.parametrize("source, args", [
    ('def kernel(a: Integer[Array, "3"], b: Integer[Array, "3"]):\n    return a / b\n',
     (np.array([1, 2, 7]), np.array([10, 10, 2]))),
    ('def kernel(a: Integer[Array, "3"]):\n    return a / 2\n', (np.array([1, -3, 7]),)),
    ('def kernel(a: Integer[Array, "3"]):\n    return 7 / a\n', (np.array([1, 2, 4]),)),
    ('def kernel(a: Integer, b: Integer):\n    return a / b\n', (7, 2)),
    # Case 0452: row-normalise an integer matrix by its L1 norms.
    ('def kernel(X: Integer[Array, "2 3"]):\n'
     '    l1 = np.abs(X).sum(axis=1)\n'
     '    return X / l1.reshape(-1, 1)\n',
     (np.array([[1, -2, 3], [4, 5, -6]]),)),
])
def test_integer_true_division_matches_numpy(source, args):
    assert_matches_numpy(source, *args)


def test_integer_division_result_is_float():
    # `(a / b) * 10` stays exact only if the quotient is a float.
    assert_matches_numpy(
        'def kernel(a: Integer[Array, "2"]):\n    return (a / 4) * 10\n',
        np.array([1, 3]))


def test_floor_division_unchanged():
    assert_matches_numpy(
        'def kernel(a: Integer[Array, "3"]):\n    return a // 2\n', np.array([1, 4, 7]))


# ── np.correlate ─────────────────────────────────────────────────────────

CORRELATE_LONGER_V = (
    'def kernel(a: {t}[Array, "4"], b: {t}[Array, "3"]):\n'
    '    return np.correlate(a, np.hstack((b[1:], b)), mode="valid")\n')


@pytest.mark.parametrize("dtype, cast", [("Integer", int), ("Float", float)])
def test_correlate_with_longer_second_operand(dtype, cast):
    a = np.array([1, 2, 3, 4], dtype=cast)
    b = np.array([5, 6, 7], dtype=cast)
    outputs = assert_matches_numpy(CORRELATE_LONGER_V.format(t=dtype), a, b)
    assert outputs == [63, 59]  # NumPy order; the old lowering gave [59, 63]


def test_correlate_with_longer_first_operand():
    assert_matches_numpy(
        'def kernel(a: Integer[Array, "5"], b: Integer[Array, "2"]):\n'
        '    return np.correlate(a, b, mode="valid")\n',
        np.array([1, 2, 3, 4, 5]), np.array([5, -6]))


# ── np.unique ────────────────────────────────────────────────────────────

def _unique_outputs(values):
    program = compile_kernel('def kernel(a: Integer[Array, "5"]):\n    return np.unique(a)\n')
    satisfied, outputs = run_mock(program, np.array(values))
    assert satisfied
    # Layout: runtime length, runtime dim, then the 5-slot padded payload.
    length, dim, payload = outputs[0], outputs[1], outputs[2:]
    assert length == dim
    return payload[:length], payload[length:]


@pytest.mark.parametrize("values", [
    [3, 1, 2, 3, 2],
    [5, 5, 5, 5, 5],
    [-1, 4, -7, 0, 9],
    [2, -2, 2, -2, 0],
])
def test_unique_is_sorted(values):
    unique, padding = _unique_outputs(values)
    assert unique == list(np.unique(values))
    assert all(p == 0 for p in padding)


def test_unique_inverse_one_hot_matches_numpy():
    # Case 0298: one-hot rows built from `return_inverse` index the sorted
    # unique values, so the column order is NumPy's.
    source = (
        'def kernel(a: Float[Array, "5"]):\n'
        '    vals, idx = np.unique(a, return_inverse=True)\n'
        '    b = np.zeros((a.size, vals.size))\n'
        '    b[np.arange(a.size), idx] = 1\n'
        '    return b\n')
    a = np.array([3.0, 1.0, 2.0, 3.0, 2.0])
    satisfied, outputs = run_mock(compile_kernel(source), a)
    assert satisfied
    # Layout: runtime length (5*3), runtime column count (3), then the
    # 5x5 envelope row-major; the runtime 5x3 block sits in its top-left.
    assert outputs[:2] == [15, 3]
    envelope = np.array(outputs[2:]).reshape(5, 5)
    _, inverse = np.unique(a, return_inverse=True)
    expected = np.zeros((5, 3))
    expected[np.arange(5), inverse] = 1
    np.testing.assert_array_equal(envelope[:, :3], expected)


# ── constant folding agrees with the circuit ─────────────────────────────

@pytest.mark.parametrize("expr, args", [
    ("a % b", (-7, 3)),
    ("a % b", (7, 3)),
    ("a // b", (-7, 3)),
])
def test_folded_integer_remainder_matches_runtime(expr, args):
    # The builder folded `%` with Rust's truncating remainder (-7 % 3 = -1)
    # while every backend computes 2, as Python does.
    a, b = args
    folded = compile_kernel(f"def kernel():\n    a = {a}\n    b = {b}\n    return {expr}\n")
    runtime = compile_kernel(f"def kernel(a: Integer, b: Integer):\n    return {expr}\n")
    expected = eval(expr, {"a": a, "b": b})
    assert [e["Int"] for e in folded.get_elided_publics()] == [expected]
    assert run_mock(runtime, a, b) == (True, [expected])


def test_folded_float_remainder_has_divisor_sign():
    folded = compile_kernel("def kernel():\n    return 7.5 % -2.0\n")
    assert [e["Float"] for e in folded.get_elided_publics()] == [7.5 % -2.0]


# ── bitwise operators on integer arrays ──────────────────────────────────

@pytest.mark.parametrize("source, args", [
    # Cases 0425/0426: binary digits via `&` with powers of two. `&` on
    # integer arrays used to be the logical and.
    ('def kernel(a: Integer[Array, "4"]):\n'
     '    return ((a[:, None] & (1 << np.arange(5))[::-1]) > 0).astype(int)\n',
     (np.array([3, 22, -7, 9]),)),
    ('def kernel(a: Integer[Array, "4"], b: Integer[Array, "4"]):\n    return a | b\n',
     (np.array([1, 4, 6, 0]), np.array([2, 4, 3, 0]))),
    ('def kernel(a: Integer[Array, "4"]):\n    return a ^ 5\n', (np.array([1, 4, 6, 0]),)),
    # Booleans keep the logical meaning (and stay usable as a mask).
    ('def kernel(a: Integer[Array, "4"]):\n    return a[(a > 1) & (a < 6)]\n',
     (np.array([0, 2, 7, 5]),)),
])
def test_bitwise_operators_match_numpy(source, args):
    expected = np.asarray(numpy_kernel(source)(*[np.copy(a) for a in args])).ravel()
    satisfied, outputs = run_mock(compile_kernel(source), *args)
    assert satisfied
    if len(outputs) != expected.size:  # 1-D dynamic: length, dim, padded payload
        assert outputs[0] == expected.size
        outputs = outputs[2:2 + expected.size]
    assert outputs == list(expected)


# ── tuple indices address one axis per entry ─────────────────────────────

M23 = np.array([[1.0, 5.0, 2.0], [7.0, 3.0, 0.0]])


@pytest.mark.parametrize("source, extra", [
    ('def kernel(a: Float[Array, "2 3"]):\n    idx = (1, 0)\n    return a[idx]\n', ()),
    ('def kernel(a: Float[Array, "2 3"], i: Integer, j: Integer):\n    idx = (i, j)\n    return a[idx]\n',
     (1, 2)),
    # Case 0314: assignment through the tuple from np.unravel_index used to
    # be read as fancy rows and overwrote whole rows.
    ('def kernel(a: Float[Array, "2 3"]):\n'
     '    idx = np.unravel_index(a.argmax(), a.shape)\n'
     '    a[idx] = a.min()\n'
     '    return np.unravel_index(a.argmax(), a.shape)\n', ()),
    ('def kernel(a: Float[Array, "2 3"], i: Integer, j: Integer):\n'
     '    a[(i, j)] = -1.0\n    return a\n', (0, 1)),
])
def test_tuple_index_matches_numpy(source, extra):
    assert_matches_numpy(source, M23, *extra)


def test_list_index_is_still_fancy():
    assert_matches_numpy('def kernel(a: Float[Array, "2 3"]):\n    return a[[1, 0]]\n', M23)


# ── einsum relabelings are views ─────────────────────────────────────────

def test_einsum_diagonal_is_a_writeable_view():
    # Case 0478: the saved diagonal is written back through the einsum
    # result; with a copy every output was zero.
    assert_matches_numpy(
        'def kernel(a: Integer[Array, "3 3"]):\n'
        "    result = np.einsum('ii->i', a)\n"
        '    save = result.copy()\n'
        '    a[...] = 0\n'
        '    result[...] = save\n'
        '    return a\n',
        np.arange(9).reshape(3, 3))


@pytest.mark.parametrize("spec", ["ij->ji", "ii->i", "ij->ij", "ij->i"])
def test_einsum_single_operand_matches_numpy(spec):
    assert_matches_numpy(
        f"def kernel(a: Integer[Array, \"3 3\"]):\n    return np.einsum('{spec}', a)\n",
        np.arange(9).reshape(3, 3) - 4)
