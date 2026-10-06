"""`np.allclose` is `np.isclose(...).all()`, as in NumPy.

The old lowering compared compile-time constants in f64 but fell back to
exact equality for any private value, so `np.allclose(x, y)` proved `False`
for close-but-unequal private arrays (NumPy: `True`). It also ignored
`equal_nan` and positional tolerances beyond `rtol`/`atol`. Both functions
now share one lowering; list and tuple operands are converted like
`np.asarray` (they used to be compared as Python sequences by `np.isclose`).
"""
import numpy as np
import pytest

from testing.lang.mock_outputs import compile_kernel
from zinnia.debug.exception.base import ZinniaException


def accepts(program, *args) -> bool:
    return program.prove(*args, backend="mock").proof_bytes_hex == "mock_satisfied"


ALLCLOSE = '''
def kernel(x: NDArray[float, 3], y: NDArray[float, 3], expected: Boolean):
    assert np.allclose(x, y) == expected
'''


@pytest.mark.parametrize("y_offset", [
    np.array([1e-9, -2e-9, 0.0]),  # close but not equal: NumPy says True
    np.array([0.0, 0.0, 0.0]),     # equal
    np.array([0.0, 0.5, 0.0]),     # one element far
])
def test_private_arrays_follow_numpy(y_offset):
    x = np.array([1.0, 2.0, -3.0])
    y = x + y_offset
    truth = bool(np.allclose(x, y))
    program = compile_kernel(ALLCLOSE)
    assert accepts(program, x, y, truth)
    assert not accepts(program, x, y, not truth)


def test_tolerances_bind_like_numpy():
    program = compile_kernel('''
def kernel(x: NDArray[float, 2], y: NDArray[float, 2], expected: Boolean):
    assert np.allclose(x, y, 0.0, 0.25, False) == expected
''')
    x, y = np.array([1.0, 2.0]), np.array([1.2, 2.0])
    assert accepts(program, x, y, True) and not accepts(program, x, y, False)
    y = np.array([1.3, 2.0])
    assert accepts(program, x, y, False) and not accepts(program, x, y, True)


def test_scalar_broadcasts_against_an_array():
    program = compile_kernel('''
def kernel(x: NDArray[float, 3], c: Float, expected: Boolean):
    assert np.allclose(x, c) == expected
''')
    assert accepts(program, np.array([2.0, 2.0, 2.0]), 2.0, True)
    assert accepts(program, np.array([2.0, 2.1, 2.0]), 2.0, False)
    assert not accepts(program, np.array([2.0, 2.1, 2.0]), 2.0, True)


def test_isclose_on_lists_is_element_wise():
    program = compile_kernel('''
def kernel(expected: NDArray[int, 2]):
    assert np.all(np.isclose([1.0, 5.0], [1.0, 2.0]) == expected)
''')
    assert accepts(program, np.array([1, 0]))
    assert not accepts(program, np.array([1, 1]))


@pytest.mark.parametrize("flag", ["True", "False"])
def test_constant_equal_nan_is_accepted(flag):
    program = compile_kernel(f'''
def kernel(x: NDArray[float, 2], expected: Boolean):
    assert np.allclose(x, x, equal_nan={flag}) == expected
''')
    assert accepts(program, np.array([1.0, 2.0]), True)


def test_runtime_equal_nan_and_unknown_keywords_are_rejected():
    with pytest.raises(ZinniaException, match="np.allclose: equal_nan must be a compile-time"):
        compile_kernel('''
def kernel(x: NDArray[float, 2], flag: Boolean):
    return np.allclose(x, x, equal_nan=flag)
''')
    with pytest.raises(ZinniaException, match="np.allclose: keyword argument `tol` is not supported"):
        compile_kernel('''
def kernel(x: NDArray[float, 2]):
    return np.allclose(x, x, tol=0.1)
''')
    with pytest.raises(ZinniaException, match="np.allclose got multiple values for argument 'rtol'"):
        compile_kernel('''
def kernel(x: NDArray[float, 2]):
    return np.allclose(x, x, 0.1, rtol=0.1)
''')
