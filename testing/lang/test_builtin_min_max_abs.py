"""Python's `min`, `max` and `abs` on floats.

A float is a fixed-point cell (`v * 2^32`). The builtins `min(a, b, ...)`,
`max(a, b, ...)` and `abs(x)` took every operand with the integer compare and
select, so their result was an integer holding 2^32 times the value:
`max(1.0, x) + 1e-9` proved about `x * 2^32` (pythran lapl3d proved every
output wrong) and `x / abs(x)` proved 0 (pythran clip). `abs` of a float array
also took the integer absolute value of every cell.
"""
import numpy as np
import pytest

from testing.lang.mock_outputs import compile_kernel, numpy_outputs, run_mock_layout
from zinnia.debug.exception.base import ZinniaException

A = np.array([0.5, -2.0, 3.25, 0.0])
I = np.array([-3, 4, 0, 7])

EXPRESSIONS = [
    "max(1.0, a[0]) + 1e-9",
    "max(1.0, a[2]) + 1e-9",
    "min(a[1], 2, a[2])",
    "min(a[0], a[1], a[2], a[3])",
    "max(a[0], a[1], a[2], a[3])",
    "max(i[0], 3)",
    "min(True, i[1])",
    "max(i[0], a[0])",
    "abs(a[1])",
    "a[0] / abs(a[0])",
    "abs(a[1] - 1.0) / 2.0",
    "abs(a)",
    "abs(i)",
    "abs(i[0])",
    "abs(a[a > 0])",
]


@pytest.mark.parametrize("expression", EXPRESSIONS)
def test_builtins_match_python(expression):
    source = f"def kernel(a: NDArray[float, 4], i: NDArray[int, 4]):\n    return {expression}\n"
    satisfied, values, shapes = run_mock_layout(compile_kernel(source), A, I)
    expected, expected_shapes = numpy_outputs(source, A, I)
    assert satisfied
    assert shapes == expected_shapes
    np.testing.assert_allclose(values, expected, atol=1e-6)


def test_a_wrong_claim_about_max_is_rejected():
    source = '''
def kernel(a: NDArray[float, 4], claimed: float):
    assert max(1.0, a[2]) + 0.5 == claimed
'''
    program = compile_kernel(source)
    assert run_mock_layout(program, A, 3.75)[0]
    assert not run_mock_layout(program, A, 3.25 * 2**32 + 0.5)[0]
    assert not run_mock_layout(program, A, 1.5)[0]


def test_abs_of_complex_numbers_is_the_magnitude():
    source = '''
def kernel(z: NDArray[complex, 2]):
    return abs(z), abs(z[0])
'''
    z = np.array([3 + 4j, -1j])
    satisfied, values, shapes = run_mock_layout(compile_kernel(source), z)
    expected, expected_shapes = numpy_outputs(source, z)
    assert satisfied and shapes == expected_shapes
    np.testing.assert_allclose(values, expected, atol=1e-6)


@pytest.mark.parametrize("body, message", [
    ("return max(a, a)", r"max\(a, b, \.\.\.\) of arrays is not supported.*np\.maximum"),
    ("return min([a[0]], [a[1]])", r"min\(a, b, \.\.\.\) is supported for numbers only"),
    ("return abs([a[0], a[1]])", r"bad operand type for abs\(\)"),
])
def test_unsupported_operands_are_named(body, message):
    with pytest.raises(ZinniaException, match=message):
        compile_kernel(f"def kernel(a: NDArray[float, 4]):\n    {body}\n")
