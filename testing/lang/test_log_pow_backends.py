"""`np.log` and `**` with a float exponent prove the same value on the mock,
halo2-base and Noir (accept-gaps, 2026-10-07).

Before, halo2-base had neither (`log_f` unsupported; `**` only with a
constant integer-valued exponent, through a different construction than the
mock's), and the mock evaluated `log` in f64 while Noir's `qlog` was fixed
point. Now `kernel::fp_log` (mantissa/exponent split, `2 atanh` series) and
`kernel::fp_pow` (`exp(b * log |a|)` with NumPy's rules for negative and zero
bases) are the mock's, halo2-base's and Noir's value, bit for bit. Where NumPy
gives `nan` or `inf` (log of `x <= 0`, a negative base with a fractional
exponent, zero to a negative power) no backend has a proof.
"""
import numpy as np
import pytest

from testing.lang.all_backends import (
    BN254_P,
    assert_all_backends_match_numpy,
    halo2_field_publics,
    mock_field_publics,
    nargo_available,
    noir_field_publics,
)
from testing.lang.mock_outputs import compile_kernel

Q = 2**32


def test_log_matches_numpy_on_every_backend():
    # Inputs on the Q32 grid, so NumPy sees the values the circuit reads.
    x = np.round(np.array([1.0, 0.5, 3.0, 1e-6, 2.0**-31, 12345.678, 1.5e9, 2.0, 0.999999]) * Q) / Q
    values = assert_all_backends_match_numpy('def kernel(x: Float[Array, "9"]):\n    return np.log(x)\n',
                                             x, atol=2e-9)
    assert np.max(np.abs(np.asarray(values) - np.log(x))) < 1.5e-9


POW_CASES = [
    # Runtime base and exponent.
    ('def kernel(a: Float[Array, "6"], b: Float[Array, "6"]):\n    return a ** b\n',
     (np.array([2.0, 0.3, 5.0, -2.0, -0.5, 0.0]), np.array([0.5, 7.5, -1.25, 3.0, -2.0, 2.5]))),
    # npbench deriche: a constant base, a runtime exponent.
    ('def kernel(alpha: Float):\n    return 2.0 ** (-alpha)\n', (0.25,)),
    # np.power, a runtime scalar exponent broadcast over an array.
    ('def kernel(a: Float[Array, "3"], b: Float):\n    return np.power(a, b)\n', (np.array([1.5, 4.0, 0.25]), 1.5)),
    # A constant fractional exponent and a constant negative one.
    ('def kernel(a: Float[Array, "3"]):\n    return a ** 0.5 + a ** -2.0\n', (np.array([1.5, 4.0, 0.25]),)),
    # Zero base: 0 ** 0 == 1, 0 ** b == 0 for b > 0.
    ('def kernel(a: Float, b: Float, c: Float):\n    return a ** b, a ** c\n', (0.0, 0.0, 3.5)),
]


@pytest.mark.parametrize("source, args", POW_CASES)
def test_pow_matches_numpy_on_every_backend(source, args):
    assert_all_backends_match_numpy(source, *args, atol=1e-7)


def accepts(program, backend, *args) -> bool:
    try:
        if backend == "mock":
            return program.prove(*args, backend="mock").proof_bytes_hex == "mock_satisfied"
        if backend == "halo2-base":
            return program.verify(program.prove(*args, backend="halo2-base"))
        return noir_field_publics(program, *args) is not None
    except Exception:  # noqa: BLE001 - an unsatisfiable witness raises at proving
        return False


BACKENDS = ["mock", "halo2-base"] + (["noir"] if nargo_available() else [])


@pytest.mark.parametrize("function, args", [
    ("np.log(a)", (0.75,)),
    ("np.log(a)", (37.5,)),
    ("a ** b", (2.5, -1.75)),
    ("a ** b", (-1.5, 3.0)),
])
def test_only_the_kernel_value_is_accepted(function, args):
    params = ", ".join(f"{n}: Float" for n in "ab"[: len(args)])
    value_program = compile_kernel(f"def kernel({params}):\n    return {function}\n")
    raw = mock_field_publics(value_program, *args)[0]
    raw = raw - BN254_P if raw > BN254_P // 2 else raw  # signed reading
    claim = compile_kernel(f"def kernel({params}, expected: Float):\n    assert {function} == expected\n")
    for backend in BACKENDS:
        assert accepts(claim, backend, *args, raw / Q), backend
        assert not accepts(claim, backend, *args, (raw + 1) / Q), backend
        assert not accepts(claim, backend, *args, (raw - 1) / Q), backend


@pytest.mark.parametrize("source, args", [
    ('def kernel(a: Float):\n    return np.log(a)\n', (0.0,)),
    ('def kernel(a: Float):\n    return np.log(a)\n', (-2.0,)),
    ('def kernel(a: Float, b: Float):\n    return a ** b\n', (-2.0, 0.5)),   # NumPy: nan
    ('def kernel(a: Float, b: Float):\n    return a ** b\n', (0.0, -1.0)),   # NumPy: inf
    ('def kernel(a: Float, b: Float):\n    return a ** b\n', (10.0, 10.0)),  # beyond the window
])
def test_nan_inf_and_overflow_have_no_proof(source, args):
    program = compile_kernel(source)
    assert program.prove(*args, backend="mock").proof_bytes_hex != "mock_satisfied"
    with pytest.raises(Exception):
        halo2_field_publics(program, *args)
    if nargo_available():
        with pytest.raises(Exception):
            noir_field_publics(program, *args)
