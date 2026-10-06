"""Index-array reads and writes, and np.round.

- A read with a constant index array (`v[[0, 2]]`, `M[[True, False, True]]`)
  was built as a nested Python list, so `2 * v[[0, 2]]` repeated it instead of
  doubling every element.
- A write through an index array with a runtime entry handed a 2-D value
  whole to every row (`M[[i, c]] = M[[c, i]]`, pythran lu, was rejected), and
  fed element `k` of a 1-D value with as many elements as indices to row `k`,
  where NumPy broadcasts the value to every row. The value's cells are read
  before any write, as NumPy evaluates the right-hand side first.
- `np.round(x, decimals)` ignored `decimals` (pythran goodexpomeasure) and
  rounded halves away from zero (`np.round(2.5)` proved 3, NumPy gives 2).
"""
import numpy as np
import pytest

from testing.lang.mock_outputs import compile_kernel, numpy_outputs, run_mock_layout

M = np.arange(6.0).reshape(3, 2)
V = np.array([1.0, 2.0, 3.0])

INDEX_ARRAYS = [
    "return M[[1, 2]] * 2",
    "return v[[0, 2]] * 2",
    "return M[np.array([1, 2])] * 2",
    "return v[[True, False, True]] + v[[0, 1]]",
    "M[[0, c]] = M[[c, 0]]\n    return M",
    "M[[1, 2]] = M[0:2]\n    return M",
    "M[[0, c]] = np.array([1.5, 2.5])\n    return M",
    "M[[c, 1]] = [[9.0, 8.0], [7.0, 6.0]]\n    return M",
    "M[[0, c], 1] = np.array([1.5, 2.5])\n    return M",
    "M[[0, 1]] = M[[1, 2]] * 2\n    return M",
]


@pytest.mark.parametrize("body", INDEX_ARRAYS)
def test_index_arrays_match_numpy(body):
    source = f"def kernel(M: NDArray[float, 3, 2], v: NDArray[float, 3], c: int):\n    {body}\n"
    for c in (1, 2):
        satisfied, values, shapes = run_mock_layout(compile_kernel(source), M, V, c)
        expected, expected_shapes = numpy_outputs(source, M, V, c)
        assert satisfied and shapes == expected_shapes
        np.testing.assert_allclose(values, expected, atol=1e-6)


X = np.array([1.23456, -2.71828, 2.5, -2.5, 0.5, -1.5, 3.4999, 1250.0])

ROUNDS = [
    "np.round(x)",
    "np.round(x, 2)",
    "np.round(x, decimals=3)",
    "np.around(x, 1)",
    "np.round(x, -2)",
    "np.round(x[0], 1) * 2",
    "np.round(i, 1)",
]


@pytest.mark.parametrize("expression", ROUNDS)
def test_round_matches_numpy(expression):
    source = f"def kernel(x: NDArray[float, 8], i: NDArray[int, 2]):\n    return {expression}\n"
    i = np.array([3, -7])
    satisfied, values, shapes = run_mock_layout(compile_kernel(source), X, i)
    expected, expected_shapes = numpy_outputs(source, X, i)
    assert satisfied and shapes == expected_shapes
    np.testing.assert_allclose(values, expected, atol=1e-6)


def test_a_half_rounds_to_even():
    source = "def kernel(x: NDArray[float, 6], claimed: NDArray[float, 6]):\n" \
             "    assert np.all(np.round(x) == claimed)\n"
    program = compile_kernel(source)
    x = np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5])
    assert run_mock_layout(program, x, np.round(x))[0]
    assert not run_mock_layout(program, x, np.array([1.0, 2.0, 3.0, -1.0, -2.0, -3.0]))[0]
