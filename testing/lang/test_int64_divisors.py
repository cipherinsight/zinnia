"""An integer divisor must be an int64 (resid-noir).

The IR's integer `+ - *` are exact in the field, while NumPy's int64 arithmetic
wraps. A divisor that left the int64 window (a row sum of `|x|` reaching 2^63)
used to be divided by unwrapped on halo2-base and Noir, proving a value NumPy
does not give (0452's form: `0.5` where NumPy gives `-0.5`). Both backends now
check an integer divisor into `[-2^63, 2^63)`: such inputs have no proof, while
divisors just inside the window, and ordinary inputs, keep NumPy's result.
"""
import numpy as np
import pytest

from testing.lang.mock_outputs import compile_kernel
from testing.lang.test_dtype_conversions import halo2_accepts, noir_accepts

ROW_NORMALIZE = ('def kernel(X: Integer[Array, "2 4"], expected: Float[Array, "2 4"]):\n'
                 '    l1 = np.abs(X).sum(axis=1)\n'
                 '    assert np.all(X / l1.reshape(-1, 1) == expected)\n')

FLOOR_DIV = ('def kernel(x: Integer, y: Integer, z: Integer, expected: Integer):\n'
             '    assert z // (x + y) == expected\n')

MOD = ('def kernel(x: Integer, y: Integer, z: Integer, expected: Integer):\n'
       '    assert z % (x + y) == expected\n')

QUARTER = [0.25] * 4


def backends_accept(program, *args):
    return halo2_accepts(program, *args), noir_accepts(program, *args)


def test_row_sum_past_int64_has_no_proof():
    program = compile_kernel(ROW_NORMALIZE)
    X = np.array([[2**62, 2**62, 0, 0], [1, 1, 1, 1]])
    # The unwrapped quotient (what was proved) and NumPy's wrapped one.
    for row in ([0.5, 0.5, 0.0, 0.0], [-0.5, -0.5, 0.0, 0.0]):
        assert backends_accept(program, X, np.array([row, QUARTER])) == (False, False), row


def test_row_sum_inside_int64_keeps_numpy():
    program = compile_kernel(ROW_NORMALIZE)
    for X, row in [(np.array([[2**62 - 1, 2**62 - 1, 0, 0], [1, 1, 1, 1]]), [0.5, 0.5, 0.0, 0.0]),
                   (np.array([[3, -1, 0, 0], [1, 1, 1, 1]]), [0.75, -0.25, 0.0, 0.0])]:
        expected = np.array([row, QUARTER])
        assert np.array_equal(expected, X / np.abs(X).sum(axis=1).reshape(-1, 1))
        assert backends_accept(program, X, expected) == (True, True)
        wrong = expected.copy()
        wrong[0, 0] += 2.0**-32
        assert backends_accept(program, X, wrong) == (False, False)


@pytest.mark.parametrize("source", [FLOOR_DIV, MOD])
def test_floor_division_by_a_wrapped_sum_has_no_proof(source):
    program = compile_kernel(source)
    z = 2**62
    with np.errstate(over="ignore"):
        divisor = np.int64(2**62) + np.int64(2**62)  # wraps to -2^63
    numpy_result = int(np.int64(z) // divisor) if source is FLOOR_DIV else int(np.int64(z) % divisor)
    unwrapped = z // 2**63 if source is FLOOR_DIV else z % 2**63
    for claim in {numpy_result, unwrapped}:
        assert backends_accept(program, 2**62, 2**62, z, claim) == (False, False), claim


@pytest.mark.parametrize("source", [FLOOR_DIV, MOD])
def test_floor_division_inside_int64_keeps_numpy(source):
    program = compile_kernel(source)
    for x, y, z in [(2**62, 2**62 - 1, 2**62), (5, 2, -8), (-3, 1, 7)]:
        d = x + y
        good = z // d if source is FLOOR_DIV else z % d
        assert backends_accept(program, x, y, z, good) == (True, True)
        assert backends_accept(program, x, y, z, good + 1) == (False, False)
