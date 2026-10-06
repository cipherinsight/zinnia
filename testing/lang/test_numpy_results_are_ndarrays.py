"""NumPy calls return ndarrays, and complex arrays keep their complex parts.

Several `np.*` lowerings built their result as a Python list, so the result
picked up list semantics downstream: `2 * np.arctan2(a, b)` repeated the list
(pythran arc_distance proved 10 values for NumPy's 5), `np.outer(u, v) +
np.outer(w, z)` concatenated (npbench gemver was rejected), and
`np.vstack(([x.T], [y.T])).T` came out as a list of rows (DS-1000 448/449).
On a complex array `.real` returned the array itself, `.imag` zeros and
`.dtype` an integer dtype (DS-1000 413 proved an empty selection).

Every check compares the decoded public outputs, values and shapes, with
NumPy's; the outputs are public, so a proof of any other value is rejected.
"""
import numpy as np
import pytest

from testing.lang.mock_outputs import compile_kernel, numpy_outputs, run_mock_layout

X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
Y = X + 10.0

TWO_ARRAYS = [
    "2 * np.arctan2(x, y)",
    "2 * np.minimum(x, y)",
    "2 * np.maximum(x, 1.0)",
    "2 * np.fmax(x, y)",
    "2 * np.sqrt([x[0, 0], x[0, 1]])",
    "np.add([x[0, 0], x[0, 1]], 1.0) * 2",
    "2 * np.cumsum([x[0, 0], x[0, 1]])",
    "2 * np.diff([x[0, 0], x[0, 1], x[1, 1]])",
    "2 * np.clip([x[0, 0], x[0, 1]], 1.5, 3.0)",
    "2 * np.square([x[0, 0], x[0, 1]])",
    "2 * np.round([x[0, 0], x[0, 1]])",
    "np.outer(x[0], y[0]) + np.outer(y[1], x[1])",
    "np.where(x > 2, [1.0, 2.0, 3.0], 0.0) + np.where(x > 4, 1.0, 0.0)",
    "np.vstack(([x.T], [y.T])).T",
    "np.concatenate(([x], [y]))",
    "np.dstack((x, y))",
    "np.column_stack(([x[0, 0], x[0, 1]], [y[0, 0], y[0, 1]]))",
    "2 * np.transpose([[x[0, 0], x[0, 1]]])",
    "2 * np.reshape([x[0, 0], x[0, 1]], (2, 1))",
    "2 * np.tile([x[0, 0], x[0, 1]], 2)",
]


@pytest.mark.parametrize("expression", TWO_ARRAYS)
def test_numpy_call_results_are_ndarrays(expression):
    source = f"def kernel(x: NDArray[float, 2, 3], y: NDArray[float, 2, 3]):\n    return {expression}\n"
    satisfied, values, shapes = run_mock_layout(compile_kernel(source), X, Y)
    expected, expected_shapes = numpy_outputs(source, X, Y)
    assert satisfied
    assert shapes == expected_shapes
    np.testing.assert_allclose(values, expected, atol=1e-6)


Z = np.array([1 + 0j, 2 + 3j, -1 + 0j, 0.5 - 2j])

COMPLEX = [
    "return z.real",
    "return z.imag",
    "return z.conjugate()",
    "return z[z.imag != 0]",
    "return z.dtype == np.complex128",
    "out = np.zeros(z.shape, dtype=z.dtype)\n    out[0] = z[1]\n    out[1] = z[0]\n    return out",
]


@pytest.mark.parametrize("body", COMPLEX)
def test_complex_array_parts_and_dtype(body):
    source = f"def kernel(z: NDArray[complex, 4]):\n    {body}\n"
    satisfied, values, shapes = run_mock_layout(compile_kernel(source), Z)
    expected, expected_shapes = numpy_outputs(source, Z)
    assert satisfied
    assert shapes == expected_shapes
    np.testing.assert_allclose(values, expected, atol=1e-6)


@pytest.mark.parametrize("expression", ["np.array([[], [], []])", "np.zeros((3, 0))",
                                        "np.zeros((3, 0)).T + 1.0"])
def test_empty_arrays_keep_their_shape(expression):
    source = f"def kernel(x: NDArray[float, 2]):\n    return {expression}\n"
    satisfied, values, shapes = run_mock_layout(compile_kernel(source), np.array([1.0, 2.0]))
    expected, expected_shapes = numpy_outputs(source, np.array([1.0, 2.0]))
    assert satisfied and values == expected == []
    assert shapes == expected_shapes
