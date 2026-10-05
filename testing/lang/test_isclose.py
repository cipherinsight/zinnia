"""`np.isclose` follows NumPy's |a - b| <= atol + rtol * |b|.

Integer operands with `atol=0` were lowered to `a == b`, which differs from
NumPy once rtol * |b| >= 1: `isclose(100001, 100000, atol=0)` is True in
NumPy (audit X-2). Positional rtol/atol were ignored, and so was
`equal_nan` (now required to be a compile-time boolean; Zinnia's numbers
have no NaN, so both values give the same result).
"""
import numpy as np
import pytest

from testing.lang.mock_outputs import compile_kernel, numpy_outputs, run_mock_layout
from zinnia.debug.exception.base import ZinniaException


def check(source, *args):
    satisfied, values, shapes = run_mock_layout(compile_kernel(source), *args)
    expected, expected_shapes = numpy_outputs(source, *args)
    assert satisfied and values == expected and shapes == expected_shapes
    return values


INTEGERS = '''
def kernel(a: NDArray[int, 4], b: NDArray[int, 4]):
    return np.isclose(a, b, atol=0)
'''


def test_integers_with_zero_atol_use_the_relative_tolerance():
    a = np.array([100001, 100000, 7, 300003])
    b = np.array([100000, 100000, 8, 300000])
    assert check(INTEGERS, a, b) == [1, 1, 0, 1]
    checked = compile_kernel('''
def kernel(a: NDArray[int, 4], b: NDArray[int, 4], expected: NDArray[int, 4]):
    assert np.all(np.isclose(a, b, atol=0) == expected)
''')
    accepts = lambda e: checked.prove(a, b, np.array(e), backend="mock").proof_bytes_hex == "mock_satisfied"
    assert accepts([1, 1, 0, 1])
    assert not accepts([0, 1, 0, 1])  # what the `==` lowering claimed


@pytest.mark.parametrize("call", [
    "np.isclose(a, b, 0.5)",
    "np.isclose(a, b, 0.0, 1.0)",
    "np.isclose(a, b, rtol=0.0, atol=0.25)",
    "np.isclose(a, b, equal_nan=True)",
    "np.isclose(a, b, 1e-05, 1e-08, False)",
])
def test_positional_and_keyword_tolerances(call):
    source = f'''
def kernel(a: NDArray[float, 4], b: NDArray[float, 4]):
    return {call}
'''
    check(source, np.array([1.0, 2.0, 3.0, -4.0]), np.array([1.3, 2.9, 3.0, -3.2]))


def test_runtime_equal_nan_is_rejected():
    with pytest.raises(ZinniaException, match="equal_nan must be a compile-time"):
        compile_kernel('''
def kernel(a: NDArray[float, 2], b: NDArray[float, 2], flag: Boolean):
    return np.isclose(a, b, equal_nan=flag)
''')
