"""Slice and element assignment into arrays (`a[idx] = v`).

- NumPy broadcasts `v` to the region `a[idx]` selects, aligned at the
  trailing axes. A value whose element count divided the region's was written
  row by row instead: `z[:, :, 0] = x.reshape(-1, 1)` wrote only `z[i, 0, 0]`
  and `z[:, :, 1] = x` wrote `x[i]` where NumPy writes `x[j]` (pythran
  create_grid proved 480 of 768 values wrong). Shapes that do not broadcast
  are rejected with NumPy's message.
- A slice with a runtime bound into an array of more than 1024 cells was
  written with its bounds resolved to the whole axis: `a[i:i + 2] = 5.0`
  wrote every element. A slice of more than 1024 cells with constant bounds
  turned the array into a Python list (`2 * a` then repeated it).
- A constant index outside its axis was summed into the address unchecked:
  `a[0, 5] = v` on a 4x4 array wrote `a[1, 1]`.
- A complex array: a write with fewer indices than axes (`z[i] = row`)
  failed with an internal error, and a runtime index was not checked.
"""
import numpy as np
import pytest

from testing.lang.mock_outputs import compile_kernel, numpy_outputs, run_mock_layout
from zinnia.debug.exception.base import ZinniaException

CREATE_GRID = '''
def kernel(x: NDArray[float, 20]):
    N = x.shape[0]
    z = np.zeros((N, N, 3))
    z[:, :, 0] = x.reshape(-1, 1)
    z[:, :, 1] = x
    return z.reshape(N * N, 3)
'''

BROADCASTS = [
    (CREATE_GRID, (np.arange(20.0) / 3,)),
    ('''
def kernel(a: NDArray[float, 4, 3], b: NDArray[float, 3]):
    a[1:3] = b
    a[0] = 7.0
    a[3, :] = [1.0, 2.0, 3.0]
    return a
''', (np.zeros((4, 3)), np.arange(3.0))),
    ('''
def kernel(a: NDArray[float, 4, 3], b: NDArray[float, 4]):
    a[:, 1:2] = b.reshape(4, 1)
    a[:2, :] = np.ones((1, 3))
    return a
''', (np.zeros((4, 3)), np.arange(4.0))),
    ('''
def kernel(a: NDArray[float, 40, 40], b: NDArray[float, 40]):
    a[:, :] = b
    a[1:3] = b[:, None][1:3]
    return 2 * a, a + 1.0
''', (np.zeros((40, 40)), np.arange(40.0))),
]


@pytest.mark.parametrize("source, args", BROADCASTS)
def test_assigned_values_broadcast_like_numpy(source, args):
    satisfied, values, shapes = run_mock_layout(compile_kernel(source), *args)
    expected, expected_shapes = numpy_outputs(source, *args)
    assert satisfied and shapes == expected_shapes
    np.testing.assert_allclose(values, expected, atol=1e-6)


@pytest.mark.parametrize("body", [
    "a[:, 0] = np.ones(3)",
    "a[1:3] = np.ones((3, 3))",
    "a[:, :] = np.ones((2, 2))",
])
def test_shapes_that_do_not_broadcast_are_rejected(body):
    with pytest.raises(ZinniaException, match="could not broadcast input array from shape"):
        compile_kernel(f"def kernel(a: NDArray[float, 4, 3]):\n    {body}\n    return a\n")


RUNTIME_BOUND = '''
def kernel(a: NDArray[float, 1100], i: int, claimed: float):
    a[i:i + 2] = 5.0
    assert np.sum(a) == claimed
    return np.sum(a)
'''


def test_a_runtime_bounded_slice_of_a_large_array_writes_only_its_range():
    program = compile_kernel(RUNTIME_BOUND)
    a = np.zeros(1100)
    satisfied, values, _ = run_mock_layout(program, a, 3, 10.0)
    assert satisfied and values == [10.0]
    assert not run_mock_layout(program, a, 3, 5500.0)[0]


def test_a_constant_index_outside_its_axis_is_an_index_error():
    with pytest.raises(ZinniaException, match="index 5 is out of bounds for axis 1 with size 4"):
        compile_kernel("def kernel(a: NDArray[float, 4, 4]):\n    a[0, 5] = 1.0\n    return a\n")


def test_a_constant_index_outside_its_axis_under_a_runtime_condition_must_not_run():
    source = '''
def kernel(a: NDArray[float, 4], n: int):
    for i in range(4):
        if i < n:
            a[i + 1] = 2.0
    return a
'''
    program = compile_kernel(source)
    satisfied, values, _ = run_mock_layout(program, np.zeros(4), 3)
    assert satisfied and values == numpy_outputs(source, np.zeros(4), 3)[0]
    assert not run_mock_layout(program, np.zeros(4), 4)[0]  # NumPy: IndexError at a[4]


COMPLEX_WRITES = [
    "z[1] = w\n    return z",
    "z[0, 1] = w[0]\n    return z",
    "z[k] = w\n    return z",
    "z[k, -1] = w[1]\n    return z",
    "z[0, :] = w[0]\n    return z",
]


@pytest.mark.parametrize("body", COMPLEX_WRITES)
def test_complex_array_writes(body):
    source = f"def kernel(z: NDArray[complex, 2, 2], w: NDArray[complex, 2], k: int):\n    {body}\n"
    z = np.array([[1 + 1j, 2.0], [3j, -1 - 1j]])
    w = np.array([0.5 - 2j, 4 + 0j])
    for k in (0, 1, -1):
        satisfied, values, shapes = run_mock_layout(compile_kernel(source), z, w, k)
        expected, expected_shapes = numpy_outputs(source, z, w, k)
        assert satisfied and shapes == expected_shapes
        np.testing.assert_allclose(values, expected, atol=1e-6)


def proves(program, *args) -> bool:
    """Whether the mock proves the program on `args`. A witness that writes
    outside an array's memory stops the mock before its constraints are
    checked; that is a rejection too."""
    try:
        return run_mock_layout(program, *args)[0]
    except RuntimeError as error:
        if "Memory write out of bounds" in str(error):
            return False
        raise


def test_a_complex_write_at_a_runtime_index_outside_the_array_is_rejected():
    source = "def kernel(z: NDArray[complex, 2], k: int):\n    z[k] = 1j\n    return z\n"
    program = compile_kernel(source)
    z = np.array([1 + 1j, 2.0])
    assert proves(program, z, -2)
    assert not proves(program, z, 2)
    assert not proves(program, z, -3)
