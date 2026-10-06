"""Powers with a compile-time integer exponent are exact multiplication chains.

`x ** e` (and `np.power(x, e)`) with an integer-valued constant `e`,
`|e| <= 64`, written as an int (`x ** -2`) or a float (`x ** -2.0`), lowers
in the IR to square-and-multiply `MulF`s, and for `e < 0` to the
fixed-point reciprocal `1.0 / x ** |e|`. The mock and every circuit run the
same IR, so they agree bit for bit; no `exp`/`log` is involved. Non-negative
exponents were already chains; negative ones went through `PowF`
(`exp(b * log a)`, approximate and costlier) or `PowFI` (backend-specific
chains that rounded differently on Noir). `0 ** e` for `e < 0` (NumPy's
`inf`) has no proof. Fractional and runtime exponents keep `PowF`.
"""
import numpy as np
import pytest

from testing.lang.all_backends import assert_all_backends_match_numpy, halo2_field_publics, nargo_available, noir_field_publics
from testing.lang.mock_outputs import compile_kernel, run_mock

X = np.array([1.5, -0.75, 3.0, -2.25])


def ir_classes(source):
    return [s["ir_instance"]["__class__"] for s in compile_kernel(source).get_ir_stmts()]


def kernel(body):
    return f'def kernel(x: Float[Array, "4"]):\n    return {body}\n'


@pytest.mark.parametrize("body", [
    "x ** 2.0", "x ** 3", "x ** -2.0", "x ** -2", "x ** -1.0", "np.power(x, -3)", "np.power(x, -3.0)",
    "x ** 0.0", "x ** -5", "x ** 64.0",
])
def test_constant_integer_powers_are_chains(body):
    classes = ir_classes(kernel(body))
    assert "PowFIR" not in classes and "PowFIIR" not in classes
    assert "ExpFIR" not in classes and "LogFIR" not in classes


@pytest.mark.parametrize("body", ["x ** 0.5", "x ** 65.0", "x ** -1.5"])
def test_other_constant_exponents_keep_exp_log(body):
    assert "PowFIR" in ir_classes(kernel(body))


@pytest.mark.parametrize("body", ["x ** -2.0", "x ** -2", "np.power(x, -1)", "x ** -3.0 + x ** 2.0", "x ** -7"])
def test_negative_constant_powers_match_numpy_on_every_backend(body):
    assert_all_backends_match_numpy(kernel(body), X, atol=1e-8)


def test_the_chain_is_the_product_of_fixed_point_multiplications():
    # x ** -2.0 is 1 / (x * x), each step rounded as MulF / DivF: the same
    # values as writing the operations out.
    assert run_mock(compile_kernel(kernel("x ** -2.0")), X) == run_mock(compile_kernel(kernel("1.0 / (x * x)")), X)
    assert run_mock(compile_kernel(kernel("x ** 3.0")), X) == run_mock(compile_kernel(kernel("x * x * x")), X)


@pytest.mark.parametrize("body", ["x ** -2.0", "x ** -1"])
def test_zero_to_a_negative_power_has_no_proof(body):
    program = compile_kernel(f"def kernel(x: Float):\n    return {body}\n")
    assert program.prove(0.0, backend="mock").proof_bytes_hex != "mock_satisfied"
    with pytest.raises(Exception):
        halo2_field_publics(program, 0.0)
    if nargo_available():
        with pytest.raises(Exception):
            noir_field_publics(program, 0.0)
