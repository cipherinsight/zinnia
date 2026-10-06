"""`np.convolve` and `np.correlate` in every mode follow NumPy (accept-gaps).

`np.convolve` was not implemented and `np.correlate` took `mode='valid'`
only. Both are now direct sums with NumPy's index conventions (see
`np_correlate` in `numpy_funcs.rs`): `full`, `same` and `valid`, either
operand longer, integer operands summed exactly, float operands by one fused
product sum per output. Compared with NumPy on the mock over every mode and
a grid of operand lengths, and halo2-base and Noir must prove the mock's
values.
"""
import itertools

import numpy as np
import pytest

from testing.lang.all_backends import assert_all_backends_match_numpy, assert_mock_matches_numpy
from testing.lang.mock_outputs import compile_kernel
from zinnia.debug.exception.base import ZinniaException

RNG = np.random.default_rng(422)
MODES = ["full", "same", "valid"]
LENGTHS = [(1, 1), (1, 4), (4, 1), (2, 5), (5, 2), (3, 3), (4, 4), (6, 3), (3, 6), (7, 4)]


def source(function, n, m, mode, dtype="Float", mode_text=None):
    call = f"np.{function}(a, v)" if mode is None else f"np.{function}(a, v, mode={mode_text or repr(mode)})"
    return f'def kernel(a: {dtype}[Array, "{n}"], v: {dtype}[Array, "{m}"]):\n    return {call}\n'


def operands(n, m, dtype):
    if dtype == "Integer":
        return RNG.integers(-50, 50, n), RNG.integers(-50, 50, m)
    return RNG.normal(size=n), RNG.normal(size=m)


@pytest.mark.parametrize("function", ["convolve", "correlate"])
@pytest.mark.parametrize("mode", MODES + [None])
@pytest.mark.parametrize("n, m", LENGTHS)
def test_matches_numpy_on_float_operands(function, mode, n, m):
    a, v = operands(n, m, "Float")
    assert_mock_matches_numpy(source(function, n, m, mode), a, v, atol=1e-8)


@pytest.mark.parametrize("function", ["convolve", "correlate"])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("n, m", [(1, 3), (5, 2), (4, 4), (3, 6)])
def test_integer_operands_are_exact(function, mode, n, m):
    a, v = operands(n, m, "Integer")
    assert_mock_matches_numpy(source(function, n, m, mode, "Integer"), a, v, atol=0)


@pytest.mark.parametrize("function, mode, n, m, dtype", [
    ("convolve", "full", 5, 3, "Float"),
    ("convolve", "same", 2, 5, "Float"),
    ("correlate", "same", 6, 3, "Integer"),
    ("correlate", "full", 3, 6, "Float"),
    ("correlate", "valid", 7, 4, "Float"),
])
def test_every_backend_proves_the_mock_value(function, mode, n, m, dtype):
    a, v = operands(n, m, dtype)
    assert_all_backends_match_numpy(source(function, n, m, mode, dtype), a, v,
                                    atol=0 if dtype == "Integer" else 1e-8)


def test_ds1000_0422_circular_correlation():
    body = ('def kernel(a: Float[Array, "5"], b: Float[Array, "5"]):\n'
            "    return np.correlate(a, np.hstack((b[1:], b)), mode='valid')\n")
    assert_mock_matches_numpy(body, RNG.normal(size=5), RNG.normal(size=5), atol=1e-8)


@pytest.mark.parametrize("mode_text", ["'FULL'", "'s'", "'Valid'"])
def test_mode_names_follow_numpy(mode_text):
    a, v = operands(5, 3, "Float")
    assert_mock_matches_numpy(source("convolve", 5, 3, "x", mode_text=mode_text), a, v, atol=1e-8)


def test_lists_and_mixed_dtypes():
    body = ('def kernel(a: Integer[Array, "4"], x: Float):\n'
            "    return np.convolve(a, [x, 1.0, -x], 'same'), np.correlate([1, 2, 3], a, 'full')\n")
    assert_mock_matches_numpy(body, np.array([3, -1, 4, 2]), 0.75, atol=1e-8)


@pytest.mark.parametrize("body, message", [
    ("np.convolve(a, v, mode='circular')", "mode must be 'valid', 'same' or 'full'"),
    ("np.convolve(a, v, method='direct')", "keyword argument `method` is not supported"),
    ("np.correlate(np.ones((2, 2)), v)", "must be one-dimensional"),
])
def test_bad_arguments_are_rejected_by_name(body, message):
    with pytest.raises(ZinniaException, match=message):
        compile_kernel(f'def kernel(a: Float[Array, "4"], v: Float[Array, "2"]):\n    return {body}\n')
