"""`np.exp` proves the same value on the mock, halo2-base and Noir (accept-gaps).

Before, halo2-base had no `exp` ("does not yet support exp_f") and the mock
evaluated `exp` in f64, while Noir's `qexp` ran a fixed-point algorithm: the
backends agreed only within a tolerance. Now one algorithm,
`kernel::fp_exp` (`y = x / ln 2` split into `floor(y) + f`, `2^f` by a
degree-12 polynomial, scaled by `2^floor(y)`), is the mock's value, the
halo2-base gadget's and Noir's `qexp`, bit for bit. Every prover hint of the
gadget is pinned (Rust: `exp_is_pinned_and_equals_the_kernel`). Its error
against libm is below 1e-9 relative for results of at least 1 and 1e-9
absolute below 1. The domain: the result must fit the signed 64-bit Q32
window (`exp(x) < 2^31`, `x <= 21.48`); beyond it no backend has a proof.
"""
import numpy as np
import pytest

from testing.lang.all_backends import (
    assert_all_backends_match_numpy,
    halo2_field_publics,
    mock_field_publics,
    nargo_available,
    noir_field_publics,
)
from testing.lang.mock_outputs import compile_kernel

Q = 2**32

X = np.array([0.0, 1.0, -1.0, 0.5, -0.3, 2.75, -7.25, 10.0, -20.0, -100.0, 21.0, 1e-9])


def test_exp_matches_numpy_and_every_backend_proves_the_mock_value():
    source = 'def kernel(x: Float[Array, "12"]):\n    return np.exp(x)\n'
    values = assert_all_backends_match_numpy(source, X, atol=1e-9 * 2**31)
    exact = np.exp(X)
    big = exact >= 1
    assert np.max(np.abs(np.asarray(values)[big] - exact[big]) / exact[big]) < 2e-9
    assert np.max(np.abs(np.asarray(values)[~big] - exact[~big])) < 2e-9


@pytest.mark.parametrize("source, args", [
    # npbench softmax: exp of x - max(x).
    ('def kernel(x: Float[Array, "2 5"]):\n'
     '    m = np.max(x, axis=-1, keepdims=True)\n'
     '    e = np.exp(x - m)\n'
     '    return e / np.sum(e, axis=-1, keepdims=True)\n',
     (np.array([[0.1, -2.0, 3.5, 0.0, 1.25], [-50.0, 0.3, 0.2, -0.1, 4.0]]),)),
    # pythran vibr_energy / log_likelihood shapes: exp of a negated product.
    ('def kernel(x: Float[Array, "4"], s: Float):\n    return np.exp(-x * x / s)\n',
     (np.array([0.0, 0.5, -2.0, 3.0]), 1.5)),
    # A scalar.
    ('def kernel(x: Float):\n    return np.exp(x) - 1.0\n', (0.125,)),
    # sinh, cosh, tanh are built from exp on every backend (npbench go_fast: tanh).
    ('def kernel(x: Float[Array, "5"]):\n    return np.sinh(x), np.cosh(x), np.tanh(x)\n',
     (np.array([0.0, 0.75, -3.5, 4.0, -0.001]),)),
])
def test_exp_programs_on_every_backend(source, args):
    assert_all_backends_match_numpy(source, *args, atol=1e-7)


CLAIM = 'def kernel(x: Float, expected: Float):\n    assert np.exp(x) == expected\n'


def accepts(program, backend, *args) -> bool:
    try:
        if backend == "mock":
            return program.prove(*args, backend="mock").proof_bytes_hex == "mock_satisfied"
        if backend == "halo2-base":
            proof = program.prove(*args, backend="halo2-base")
            return program.verify(proof)
        return noir_field_publics(program, *args) is not None
    except Exception:  # noqa: BLE001 - an unsatisfiable witness raises at proving
        return False


BACKENDS = ["mock", "halo2-base"] + (["noir"] if nargo_available() else [])


# Claims are passed as floats: results below 2^21 keep every Q32 unit exact.
@pytest.mark.parametrize("x", [1.0, -3.3, 12.5, -40.0])
def test_only_the_kernel_value_is_accepted(x):
    program = compile_kernel(CLAIM)
    value_program = compile_kernel('def kernel(x: Float):\n    return np.exp(x)\n')
    raw = mock_field_publics(value_program, x)[0]
    for backend in BACKENDS:
        assert accepts(program, backend, x, raw / Q), backend
        assert not accepts(program, backend, x, (raw + 1) / Q), backend
        assert not accepts(program, backend, x, (raw - 1) / Q), backend


@pytest.mark.parametrize("x", [21.5, 30.0, 1e6])
def test_results_beyond_the_window_have_no_proof(x):
    program = compile_kernel('def kernel(x: Float):\n    return np.exp(x)\n')
    assert program.prove(x, backend="mock").proof_bytes_hex != "mock_satisfied"
    with pytest.raises(Exception):
        halo2_field_publics(program, x)
    if nargo_available():
        with pytest.raises(Exception):
            noir_field_publics(program, x)
