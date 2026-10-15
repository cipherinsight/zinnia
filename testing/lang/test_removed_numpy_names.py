"""NumPy names that NumPy itself has removed ("Removed NumPy Names" in the
NumPy API documentation).

Zinnia supports NumPy 2.2 and later. The names NumPy 2.2 still provides and
later releases removed stay accepted with their former results: `np.in1d`
(removed in 2.4) is `np.isin(...).ravel()`, `np.trapz` (2.4) is
`np.trapezoid`, `np.row_stack` (2.5) is `np.vstack`. Each alias must compile
to the outputs of its current name, which NumPy computes as the reference.
The names NumPy 2.0 removed are rejected, and the error names the
replacement.
"""
import numpy as np
import pytest

from testing.lang.all_backends import assert_mock_matches_numpy
from testing.lang.mock_outputs import compile_kernel, run_mock
from zinnia.debug.exception.base import ZinniaException

A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([2, 5, 9, 5])
F = np.array([[0.5, -1.25, 3.0], [2.0, 0.0, 1.5]])
X = np.array([0.0, 0.5, 1.5])

KERNEL = 'def kernel(a: {a}, b: {b}):\n    return {body}\n'
INTS = ('Integer[Array, "2 3"]', 'Integer[Array, "4"]')
FLOATS = ('Float[Array, "2 3"]', 'Float[Array, "3"]')

# (alias, current spelling, argument types, arguments)
ALIASES = [
    ("np.in1d(a, b)", "np.isin(a, b).ravel()", INTS, (A, B)),
    ("np.in1d(a, b, invert=True)", "np.isin(a, b, invert=True).ravel()", INTS, (A, B)),
    ("np.trapz(a)", "np.trapezoid(a)", FLOATS, (F, X)),
    ("np.trapz(a, b, axis=1)", "np.trapezoid(a, b, axis=1)", FLOATS, (F, X)),
    ("np.trapz(a, dx=0.5, axis=0)", "np.trapezoid(a, dx=0.5, axis=0)", FLOATS, (F, X)),
    ("np.row_stack([a, a + 1])", "np.vstack([a, a + 1])", INTS, (A, B)),
    ("np.row_stack([b, b])", "np.vstack([b, b])", INTS, (A, B)),
]


@pytest.mark.parametrize("alias, current, types, args", ALIASES)
def test_alias_matches_its_current_name(alias, current, types, args):
    a_type, b_type = types
    source = KERNEL.format(a=a_type, b=b_type, body=current)
    _, expected = assert_mock_matches_numpy(source, *args, atol=1e-6)
    _, values = assert_mock_matches_numpy(KERNEL.format(a=a_type, b=b_type, body=alias), *args, atol=1e-6)
    assert values == expected


def test_alias_check_rejects_a_wrong_claim():
    source = ('def kernel(a: Integer[Array, "2 3"], expected: Integer[Array, "4 3"]):\n'
              '    assert np.all(np.row_stack([a, a + 1]) == expected)\n')
    program = compile_kernel(source)
    good = np.vstack([A, A + 1])
    assert run_mock(program, A, good)[0]
    wrong = good.copy()
    wrong[3, 2] += 1
    assert not run_mock(program, A, wrong)[0]


@pytest.mark.parametrize("body, name, replacement", [
    ("np.linalg.norm(f, np.Inf)", "Inf", "np.inf"),
    ("np.linalg.norm(f, -np.infty)", "infty", "np.inf"),
    ("f * np.PINF", "PINF", "np.inf"),
    ("np.product(f)", "product", "np.prod"),
    ("np.cumproduct(f)", "cumproduct", "np.cumprod"),
    ("np.round_(f)", "round_", "np.round"),
    ("np.sometrue(f > 0)", "sometrue", "np.any"),
    ("np.alltrue(f > 0)", "alltrue", "np.all"),
])
def test_names_removed_in_numpy_2_0_are_rejected_with_their_replacement(body, name, replacement):
    with pytest.raises(ZinniaException, match=rf"np\.{name} was removed in NumPy 2\.0; use {replacement}"):
        compile_kernel(f'def kernel(f: Float[Array, "3"]):\n    return {body}\n')
