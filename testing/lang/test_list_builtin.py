"""`list(x)` and `tuple(x)` follow Python: they collect what a `for` loop over
`x` visits (the rows of a 2-D array) and raise TypeError for a scalar.

Before, both returned `x` itself for anything but a list or tuple, so
`np.array(list(n), dtype=int)` on an integer `n` (DS-1000 case 0344, whose
reference raises in NumPy) compiled to the array `[n]`.
"""
import numpy as np
import pytest

from testing.lang.mock_outputs import compile_kernel, numpy_outputs, run_mock_layout
from zinnia.debug.exception.base import ZinniaException


@pytest.mark.parametrize("call", ["list(n)", "tuple(n)", "list(2.5)", "list(n > 0)"])
def test_list_of_a_scalar_raises_like_python(call):
    with pytest.raises(ZinniaException, match="TypeError: '(int|float|bool)' object is not iterable"):
        compile_kernel(f'''
def kernel(n: Integer):
    return np.array({call}, dtype=int)
''')


@pytest.mark.parametrize("body", [
    "return np.array(list(a))",
    "rows = list(a)\n    return rows[1] - rows[0]",
    "return np.array(tuple(a[0]))",
    "return np.array(list(a[:, 1]) + [7])",
    "return len(list(a))",
])
def test_list_of_an_array_matches_python(body):
    source = f'''
def kernel(a: NDArray[int, 2, 3]):
    {body}
'''
    a = np.array([[1, 2, 3], [4, 5, 9]])
    satisfied, values, shapes = run_mock_layout(compile_kernel(source), a)
    expected, expected_shapes = numpy_outputs(source, a)
    assert satisfied and values == expected and shapes == expected_shapes


def test_list_of_a_runtime_shaped_array_is_rejected():
    with pytest.raises(ZinniaException, match="runtime shape is not supported"):
        compile_kernel('''
def kernel(a: NDArray[int, 4]):
    return np.array(list(a[a > 0]))
''')
