"""`np.isin` and `np.in1d` follow NumPy (accept-gaps, 2026-10-06).

`np.isin` was not implemented and `np.in1d` took no keyword arguments. Both
now share one lowering: element-wise `x == t_1 or ... or x == t_k` over the
flattened second argument, which the optimizer's membership fusion turns
into a zero test of `(x - t_1)...(x - t_k)` (Horner's rule when many
subjects share constant `t_i`). The set may hold runtime values. `isin`
keeps the first argument's shape (a scalar gives a scalar), `in1d` flattens
it; `invert=True` negates; `assume_unique` and `kind` do not change the
result. Compared with NumPy on the mock, and halo2-base and Noir must prove
the mock's values.
"""
import numpy as np
import pytest

from testing.lang.all_backends import assert_all_backends_match_numpy, assert_mock_matches_numpy
from testing.lang.mock_outputs import compile_kernel, run_mock
from zinnia.debug.exception.base import ZinniaException

A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([2, 5, 9, 5])
F = np.array([0.5, -1.25, 3.0, 2.0])

CASES = [
    ('def kernel(a: Integer[Array, "2 3"], b: Integer[Array, "4"]):\n    return np.isin(a, b)\n', (A, B)),
    ('def kernel(a: Integer[Array, "2 3"], b: Integer[Array, "4"]):\n    return np.isin(a, b, invert=True)\n', (A, B)),
    ('def kernel(a: Integer[Array, "2 3"], b: Integer[Array, "4"]):\n    return np.in1d(a, b)\n', (A, B)),
    ('def kernel(a: Integer[Array, "2 3"], b: Integer[Array, "4"]):\n    return np.in1d(a, b, False, True)\n', (A, B)),
    ('def kernel(a: Integer[Array, "2 3"], b: Integer[Array, "4"]):\n'
     '    return np.isin(a, b, assume_unique=False, kind="sort")\n', (A, B)),
    # The set is a 2-D array, a list of runtime values, a scalar.
    ('def kernel(a: Integer[Array, "4"], b: Integer[Array, "2 3"]):\n    return np.isin(a, b)\n', (B, A)),
    ('def kernel(a: Integer[Array, "2 3"], x: Integer, y: Integer):\n    return np.isin(a, [x, y, 7])\n', (A, 3, 6)),
    ('def kernel(a: Integer[Array, "2 3"], x: Integer):\n    return np.isin(a, x)\n', (A, 4)),
    # A scalar element: a scalar result.
    ('def kernel(x: Integer, b: Integer[Array, "4"]):\n    return np.isin(x, b)\n', (5, B)),
    ('def kernel(x: Integer, b: Integer[Array, "4"]):\n    return np.in1d(x, b)\n', (7, B)),
    # Floats, and ints against floats.
    ('def kernel(a: Float[Array, "4"], b: Float[Array, "2"]):\n    return np.isin(a, b)\n', (F, np.array([3.0, -1.25]))),
    ('def kernel(a: Integer[Array, "2 3"], b: Float[Array, "4"]):\n    return np.isin(a, b)\n', (A, F)),
    # Used as a mask and in arithmetic (DS-1000 0442/0443 style).
    ('def kernel(a: Integer[Array, "2 3"], b: Integer[Array, "4"]):\n'
     '    m = np.isin(a, b)\n'
     '    return np.sum(m), np.where(~m, a, 0)\n', (A, B)),
    # Edge shapes: one element on each side.
    ('def kernel(a: Integer[Array, "1"], b: Integer[Array, "1"]):\n    return np.isin(a, b)\n',
     (np.array([3]), np.array([3]))),
]


@pytest.mark.parametrize("source, args", CASES)
def test_membership_matches_numpy(source, args):
    assert_mock_matches_numpy(source, *args, atol=0)


@pytest.mark.parametrize("source, args", [CASES[i] for i in (0, 3, 6, 8, 10, 12)])
def test_membership_on_every_backend(source, args):
    assert_all_backends_match_numpy(source, *args, atol=0)


def test_a_wrong_membership_claim_is_rejected():
    source = ('def kernel(a: Integer[Array, "2 3"], b: Integer[Array, "4"], expected: Integer[Array, "2 3"]):\n'
              '    assert np.all(np.isin(a, b) == expected)\n')
    program = compile_kernel(source)
    good = np.isin(A, B).astype(np.int64)
    assert run_mock(program, A, B, good)[0]
    for i in range(good.size):
        wrong = good.copy().ravel()
        wrong[i] = 1 - wrong[i]
        assert not run_mock(program, A, B, wrong.reshape(good.shape))[0]


@pytest.mark.parametrize("body, message", [
    ("np.isin(a, b, unknown=True)", "keyword argument `unknown` is not supported"),
    ("np.isin(a, b, kind='hash')", "kind must be None, 'sort' or 'table'"),
    ("np.isin(a)", "takes 2 to 4 positional arguments"),
])
def test_bad_arguments_are_rejected_by_name(body, message):
    with pytest.raises(ZinniaException, match=message):
        compile_kernel(f'def kernel(a: Integer[Array, "2 3"], b: Integer[Array, "4"]):\n    return {body}\n')
