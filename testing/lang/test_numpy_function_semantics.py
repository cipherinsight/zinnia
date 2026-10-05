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

from testing.lang.mock_outputs import assert_matches_numpy, compile_kernel, run_mock


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
