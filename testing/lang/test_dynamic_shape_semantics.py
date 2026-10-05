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


def runtime_values(outputs, n):
    """The `n` result values: all outputs for a static result, the runtime
    prefix of the payload for a 1-D dynamic one (layout: runtime length,
    runtime dim, padded payload)."""
    if len(outputs) == n:
        return outputs
    assert outputs[0] == n, f"runtime length {outputs[0]} != {n}"
    return outputs[2:2 + n]


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


A42 = np.array([[1, 2], [3, 8], [-5, 5], [6, 6]])
SLICE_2D = 'def kernel(a: Integer[Array, "4 2"], r: Integer):\n    s = a[0:r, :]\n    return {}\n'


@pytest.mark.parametrize("expr", [
    "s.mean()", "s.min()", "s.max()", "s.argmax()", "s.prod()", "s.sum()",
    "len(s)", "s.shape[0]", "s.sum(axis=1)", "s.max(axis=0)",
])
def test_reduction_over_runtime_bounded_2d_slice(expr):
    # `a[0:r, :]` used to be the zero-padded 4x2 envelope; `.mean()` divided
    # by 8, `.min()` saw the padding zeros, `len` was 4.
    source = SLICE_2D.format(expr)
    expected = np.asarray(numpy_kernel(source)(A42.copy(), 2), dtype=float).ravel()
    satisfied, outputs = run_mock(compile_kernel(source), A42, 2)
    assert satisfied
    np.testing.assert_allclose(runtime_values(outputs, expected.size), expected)


def test_elementwise_over_runtime_bounded_2d_slice_keeps_runtime_shape():
    source = SLICE_2D.format("s + 1")
    satisfied, outputs = run_mock(compile_kernel(source), A42, 2)
    assert satisfied
    # Runtime length 4, runtime rows 2, then the padded 4x2 payload.
    assert outputs[:2] == [4, 2]
    assert outputs[2:6] == list((A42[:2] + 1).ravel())


@pytest.mark.parametrize("expr", ["np.cumsum(s)", "np.diff(y)", "np.flip(y)", "np.outer(y, y)"])
def test_static_shape_operation_on_runtime_shaped_array_is_rejected(expr):
    # These used to treat the runtime-shaped array as one scalar and return
    # it unchanged (or square it).
    source = ('def kernel(a: Integer[Array, "4 2"], r: Integer, x: Integer[Array, "6"]):\n'
              f'    s = a[0:r, :]\n    y = x[x > 0]\n    return {expr}\n')
    with pytest.raises(Exception, match="runtime shape"):
        compile_kernel(source)


@pytest.mark.parametrize("expr", [
    "y[y < 5]",                    # filter of a filter kept the padding
    "y.reshape(-1).mean()",        # runtime reshape claimed the envelope length
    "y[-1]",                       # negative index wrapped with the envelope size
    "np.argwhere(m).reshape(-1).max()",
])
def test_runtime_shaped_array_operations_match_numpy(expr):
    source = (f'def kernel(x: Integer[Array, "6"], m: Integer[Array, "2 3"]):\n'
              f'    y = x[x > 0]\n    return {expr}\n')
    x = np.array([1, -1, 2, 7, -3, 9])
    m = np.array([[0, 1, 0], [0, 0, 1]])
    expected = np.asarray(numpy_kernel(source)(x.copy(), m.copy()), dtype=float).ravel()
    satisfied, outputs = run_mock(compile_kernel(source), x, m)
    assert satisfied
    np.testing.assert_allclose(runtime_values(outputs, expected.size), expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("source, args", [
    ('def kernel(x: Integer[Array, "6"], i: Integer):\n    y = x[x > 0]\n    return y[i]\n',
     (np.array([1, -1, 2, 7, -3, 9]), 4)),
    ('def kernel(a: Integer[Array, "4 3"], r: Integer, i: Integer):\n    s = a[0:r, :]\n    return s[i]\n',
     (np.arange(12).reshape(4, 3), 2, 3)),  # row 3 of a 2-row slice: was unchecked
])
def test_out_of_range_index_into_runtime_shaped_array_is_rejected(source, args):
    satisfied, _ = run_mock(compile_kernel(source), *args)
    assert not satisfied
