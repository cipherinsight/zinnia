"""Powers with a compile-time integer exponent are exact multiplication chains.

`x ** e` (and `np.power(x, e)`) with an integer-valued constant `e`,
`|e| <= 64`, written as an int (`x ** -2`) or a float (`x ** -2.0`), lowers
in the IR to square-and-multiply `MulF`s, for `e < 0` on the fixed-point
reciprocal: `(1.0 / x) ** |e|`. The mock and every circuit run the
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


@pytest.mark.parametrize("body", ["x ** 0.25", "x ** 65.0", "x ** -1.5"])
def test_other_constant_exponents_keep_exp_log(body):
    assert "PowFIR" in ir_classes(kernel(body))


@pytest.mark.parametrize("body", ["x ** -2.0", "x ** -2", "np.power(x, -1)", "x ** -3.0 + x ** 2.0", "x ** -7"])
def test_negative_constant_powers_match_numpy_on_every_backend(body):
    assert_all_backends_match_numpy(kernel(body), X, atol=1e-8)


def test_the_chain_is_the_product_of_fixed_point_multiplications():
    # x ** -2.0 is (1 / x) * (1 / x), each step rounded as DivF / MulF: the
    # same values as writing the operations out.
    assert run_mock(compile_kernel(kernel("x ** -2.0")), X) == run_mock(compile_kernel(kernel("(1.0 / x) * (1.0 / x)")), X)
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


# ── runtime integer exponents (PowFI): one definition on every backend ───

def runtime_kernel():
    return 'def kernel(x: Float[Array, "4"], n: Integer):\n    return x ** n\n', None


@pytest.mark.parametrize("n", [0, 1, 5, -3, 13])
def test_runtime_integer_powers_match_numpy_on_every_backend(n):
    source, _ = runtime_kernel()
    assert_all_backends_match_numpy(source, X, n, atol=1e-6 * max(1.0, float(np.max(np.abs(X) ** abs(n)))))


@pytest.mark.parametrize("n", [3, -2, 7])
def test_a_runtime_exponent_gives_the_constant_chains_value(n):
    runtime = run_mock(compile_kernel(runtime_kernel()[0]), X, n)
    constant = run_mock(compile_kernel(kernel(f"x ** {n}")), X)
    assert runtime == constant


def test_runtime_zero_to_a_negative_power_has_no_proof():
    program = compile_kernel('def kernel(x: Float, n: Integer):\n    return x ** n\n')
    assert program.prove(0.0, -2, backend="mock").proof_bytes_hex != "mock_satisfied"
    with pytest.raises(Exception):
        halo2_field_publics(program, 0.0, -2)
    if nargo_available():
        with pytest.raises(Exception):
            noir_field_publics(program, 0.0, -2)
    assert program.prove(0.0, 0, backend="mock").proof_bytes_hex == "mock_satisfied"  # 0 ** 0 == 1


def test_the_exponent_width_comes_from_its_proven_range():
    from zinnia import ZKCircuit
    narrow = ZKCircuit.from_source(
        "kernel", "@requires(lambda n: 0 <= n <= 6)\ndef kernel(x: Float, n: Integer):\n    return x ** n\n").compile()
    wide = compile_kernel('def kernel(x: Float, n: Integer):\n    return x ** n\n')
    bits = lambda p: [s["ir_instance"]["ir_data"]["exponent_bits"] for s in p.get_ir_stmts()
                      if s["ir_instance"]["__class__"] == "PowFIIR"]
    assert bits(narrow) == [3]
    assert bits(wide) == [64]


# ── integer ** negative integer: NumPy raises ────────────────────────────

def test_integer_to_a_negative_constant_power_is_rejected():
    from zinnia.debug.exception.base import ZinniaException
    with pytest.raises(ZinniaException, match="negative integer powers are not allowed"):
        compile_kernel('def kernel(k: Integer):\n    return k ** -2\n')


def test_integer_to_a_negative_runtime_power_has_no_proof():
    program = compile_kernel('def kernel(k: Integer, n: Integer):\n    return k ** n\n')
    assert program.prove(3, 2, backend="mock").proof_bytes_hex == "mock_satisfied"
    assert program.prove(3, -1, backend="mock").proof_bytes_hex != "mock_satisfied"
    with pytest.raises(Exception):
        halo2_field_publics(program, 3, -1)


# ── a constant ** 0.5 is the square root ─────────────────────────────────

def test_half_power_is_the_exact_square_root():
    classes = ir_classes(kernel("np.abs(x) ** 0.5"))
    assert "SqrtFIR" in classes and "PowFIR" not in classes
    y = np.abs(X)
    assert_all_backends_match_numpy('def kernel(x: Float[Array, "4"]):\n    return x ** 0.5\n', y, atol=1e-9)
    assert run_mock(compile_kernel(kernel("x ** 0.5")), y) == run_mock(compile_kernel(kernel("np.sqrt(x)")), y)


def test_half_power_of_a_negative_number_has_no_proof():
    program = compile_kernel('def kernel(x: Float):\n    return x ** 0.5\n')
    assert program.prove(-2.0, backend="mock").proof_bytes_hex != "mock_satisfied"
    with pytest.raises(Exception):
        halo2_field_publics(program, -2.0)
    assert program.prove(0.0, backend="mock").proof_bytes_hex == "mock_satisfied"
