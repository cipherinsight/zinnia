"""NumPy-differential tests for operators with runtime-shaped results (audit M2).

Each case used to prove a result different from NumPy's on an honest input.
Outputs are compared position by position; a runtime-shaped (dynamic) result
is exposed as its runtime length and dims followed by the padded payload, and
only the runtime prefix is compared with NumPy.
"""
import numpy as np
import pytest

from testing.lang.mock_outputs import assert_matches_numpy, compile_kernel, numpy_kernel, run_mock

A23 = np.array([[1, 5, 0], [7, 3, 2]])


def dynamic_1d(source, *args):
    """(runtime length, runtime prefix of the payload) of a 1-D dynamic result."""
    satisfied, outputs = run_mock(compile_kernel(source), *args)
    assert satisfied
    length = outputs[0]
    return length, outputs[2:2 + length]


def test_one_argument_where_is_nonzero():
    # Used to return the condition itself, so `[0]` read its first element.
    source = 'def kernel(a: Integer[Array, "6"]):\n    return np.where(a > 2)[0]\n'
    a = np.array([1, 5, 0, 7, 3, 2])
    length, values = dynamic_1d(source, a)
    expected = np.where(a > 2)[0]
    assert length == len(expected) and values == list(expected)


@pytest.mark.parametrize("body", [
    "a[np.where(a > 2)] = 0",
    "a[np.where(a > 2)] = 0\n    a[np.where(a < 2)] = 30",
    "a[np.nonzero(a > 2)] = 9",
])
def test_assignment_through_nonzero_indices(body):
    assert_matches_numpy(f'def kernel(a: Integer[Array, "2 3"]):\n    {body}\n    return a\n', A23)
