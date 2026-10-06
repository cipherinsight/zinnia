"""`np.corrcoef` of integer arrays by exact integer sums (layer-arith).

For integer (or Boolean) inputs the statistics family lowers Pearson's `r`
as `A / sqrt(B * C)` with the exact integers `A = n*Sxy - Sx*Sy`,
`B = n*Sxx - Sx^2`, `C = n*Syy - Sy^2`: the root is `IR::SqrtProductIF`
(`floor(sqrt(B*C) * 2^32)`, admitted for `B < 2^64`, `B*C < 2^62`) and the quotient
`floor(A * 2^64 / root)`. The mock and every backend compute exactly this
value; it is within 2^-31 of NumPy's float64 result.

Each claim test asserts the result equal to a claimed Q32 value: the exact
formula's value is accepted on the mock, halo2-base and Noir, the next Q32
value is rejected on all three, and inputs outside the domain (a zero
variance, NumPy's nan; `B*C >= 2^62`) have no proof whatever the claim.
"""
import math

import numpy as np
import pytest

from testing.lang.mock_outputs import assert_matches_numpy, compile_kernel
from testing.lang.test_dtype_conversions import halo2_accepts, mock_accepts, noir_accepts

Q = 2**32

CLAIM = (
    'def kernel(x: Integer[Array, "4"], y: Integer[Array, "4"], expected: Float):\n'
    '    assert np.corrcoef(x, y)[0][1] == expected\n'
)

BRANCH = (
    'def kernel(x: Integer[Array, "4"], y: Integer[Array, "4"], flag: Integer, expected: Float):\n'
    '    r = 0.5\n'
    '    if flag > 0:\n'
    '        r = np.corrcoef(x, y)[0][1]\n'
    '    assert r == expected\n'
)


def exact_r(x, y):
    """The raw Q32 `r` of the exact-sums construction, or None (no proof)."""
    n = len(x)
    sx, sy = sum(x), sum(y)
    a = n * sum(p * q for p, q in zip(x, y)) - sx * sy
    b = n * sum(p * p for p in x) - sx * sx
    c = n * sum(q * q for q in y) - sy * sy
    if not 0 <= b < 2**64 or not 1 <= b * c < 2**62:
        return None
    root = math.isqrt(b * c * 2**64)
    return (a * 2**64) // root


def accepted_everywhere(program, *args) -> bool:
    return mock_accepts(program, *args) and halo2_accepts(program, *args) and noir_accepts(program, *args)


def rejected_everywhere(program, *args) -> bool:
    return not (mock_accepts(program, *args) or halo2_accepts(program, *args) or noir_accepts(program, *args))


@pytest.fixture(scope="module")
def claim_program():
    return compile_kernel(CLAIM)


INPUTS = [
    ([14, 2, 17, 10], [153, 486, 588, 232]),
    ([10, 11, 14, 12], [687, 336, 768, 775]),
    ([1, 2, 3, 4], [2, 4, 6, 8]),          # r = 1 exactly
    ([1, 2, 3, 4], [8, 6, 4, 2]),          # r = -1 exactly
    ([-5, 0, 7, -3], [100, -100, 3, 0]),
    ([11585, -11585, 11585, -11585], [11585, 11585, -11585, -11585]),  # B*C near 2^62
]


@pytest.mark.parametrize("x, y", INPUTS)
def test_the_exact_formula_is_proved_and_its_neighbour_rejected(claim_program, x, y):
    raw = exact_r(x, y)
    assert raw is not None
    assert abs(raw / Q - np.corrcoef(x, y)[0][1]) < 2**-30
    assert accepted_everywhere(claim_program, np.array(x), np.array(y), raw / Q)
    assert rejected_everywhere(claim_program, np.array(x), np.array(y), (raw + 1) / Q)


@pytest.mark.parametrize("x, y", [
    ([3, 3, 3, 3], [1, 2, 3, 4]),            # var(x) = 0: NumPy gives nan
    ([1, 2, 3, 4], [-7, -7, -7, -7]),        # var(y) = 0
    ([0, 0, 0, 2**20], [0, 0, 0, 2**13]),    # B*C = 3^2 * 2^66 >= 2^62
    ([2**31, -2**31, 2**31, -2**31], [1, 2, 3, 4]),  # B = 2^66
])
def test_outside_the_domain_there_is_no_proof(claim_program, x, y):
    assert exact_r(x, y) is None
    for claim in (0.0, 1.0, -1.0, 0.5):
        assert rejected_everywhere(claim_program, np.array(x), np.array(y), claim)


def test_checks_hold_only_where_the_call_runs():
    """A zero variance in a branch that does not run is no error (NumPy never
    evaluates it); in a branch that runs it has no proof."""
    program = compile_kernel(BRANCH)
    x, y = np.array([3, 3, 3, 3]), np.array([1, 2, 3, 4])
    assert accepted_everywhere(program, x, y, 0, 0.5)
    assert rejected_everywhere(program, x, y, 1, 0.5)
    good_x = np.array([1, 5, 2, 8])
    raw = exact_r([1, 5, 2, 8], [1, 2, 3, 4])
    assert accepted_everywhere(program, good_x, y, 1, raw / Q)
    assert rejected_everywhere(program, good_x, y, 1, (raw - 1) / Q)


@pytest.mark.parametrize("n", [2, 3, 7])
def test_integer_corrcoef_matches_numpy_on_other_lengths(n):
    rng = np.random.default_rng(n)
    x = rng.integers(-1000, 1000, n)
    y = rng.integers(-1000, 1000, n)
    source = (f'def kernel(x: Integer[Array, "{n}"], y: Integer[Array, "{n}"]):\n'
              '    return np.corrcoef(x, y)\n')
    assert_matches_numpy(source, x, y)


def test_float_corrcoef_keeps_the_fixed_point_construction():
    source = ('def kernel(x: Float[Array, "5"], y: Float[Array, "5"]):\n'
              '    return np.corrcoef(x, y)[0][1]\n')
    assert_matches_numpy(source, np.array([0.5, -1.25, 2.0, 3.5, 0.0]), np.array([1.0, 0.25, -2.0, 4.0, 1.5]))


def test_constant_inputs_fold_to_the_same_value():
    source = ('def kernel(z: Integer):\n'
              '    x = np.array([14, 2, 17, 10]) + z\n'
              '    return np.corrcoef(x, np.array([153, 486, 588, 232]))[0][1]\n')
    folded = ('def kernel(z: Integer):\n'
              '    return np.corrcoef(np.array([14, 2, 17, 10]), np.array([153, 486, 588, 232]))[0][1] + z\n')
    raw = exact_r([14, 2, 17, 10], [153, 486, 588, 232])
    for src in (source, folded):
        program = compile_kernel(src)
        proof = program.prove(0, backend="mock")
        assert proof.proof_bytes_hex == "mock_satisfied"
        value = int(proof.public_values[0].partition("=")[2])
        assert value == raw, src


# ── divisor guards: `d != 0` before `x / d` is implied by the division ──

@pytest.mark.parametrize("op", ["x / d", "x // d", "x % d"])
def test_a_zero_divisor_still_has_no_proof(op):
    """The asserted precondition `d != 0` emits no constraint of its own on
    halo2-base and Noir (the division's remainder bound `|r| <= |d| - 1`
    implies it); a zero divisor must still have no proof anywhere."""
    for dtype, x in (("Float", 7.5), ("Integer", 7)):
        if dtype == "Integer" and op == "x / d":
            continue
        program = compile_kernel(f"def kernel(x: {dtype}, d: {dtype}, expected: Float):\n"
                                 f"    assert {op} == expected\n")
        d = 2.0 if dtype == "Float" else 2
        good = float(eval(op, {"x": x, "d": d}))
        assert accepted_everywhere(program, x, d, good), (dtype, op)
        assert rejected_everywhere(program, x, d, good + 1.0), (dtype, op)
        zero = 0.0 if dtype == "Float" else 0
        for claim in (0.0, good):
            assert rejected_everywhere(program, x, zero, claim), (dtype, op)
