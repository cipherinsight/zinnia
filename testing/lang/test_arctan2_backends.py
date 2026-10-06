"""`np.arctan2` proves the same value on the mock, halo2-base and Noir (npb-gaps).

Before, halo2-base had no `arctan2` ("does not yet support arctan2_f") and the
mock evaluated it in f64, while Noir's `qarctan2` ran a fixed-point algorithm:
the backends agreed only within a tolerance. Now one algorithm,
`kernel::fp_atan2` (the ratio of the smaller to the larger magnitude, the odd
atan polynomial, then the octant, half-plane and sign adjustments), is the
mock's value, the halo2-base gadget's and Noir's `qarctan2`, bit for bit. Every
prover hint of the gadget is pinned (Rust: `arctan2_is_pinned_and_equals_the_kernel`).
Its error against libm is below 3.5e-9. An operand outside the int64 window
has no proof.
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

Y = np.array([1.0, 2.0, -1.0, -2.0, 1.0, 0.0, 3.5, 0.0, -0.25, 0.0, 1e-9, -7.0])
X = np.array([2.0, 1.0, 2.0, -1.0, -2.0, 1.0, 0.0, -1.0, 0.0, 0.0, 5.0, -7.0])


def test_arctan2_matches_numpy_and_every_backend_proves_the_mock_value():
    source = 'def kernel(y: Float[Array, "12"], x: Float[Array, "12"]):\n    return np.arctan2(y, x)\n'
    values = assert_all_backends_match_numpy(source, Y, X, atol=3.5e-9)
    assert np.max(np.abs(np.asarray(values) - np.arctan2(Y, X))) < 3.5e-9


def test_arc_distance_shape():
    """The pythran/npbench arc_distance kernel."""
    source = (
        'def kernel(theta_1: Float[Array, "4"], phi_1: Float[Array, "4"], theta_2: Float[Array, "4"], phi_2: Float[Array, "4"]):\n'
        '    temp = np.sin((theta_2 - theta_1) / 2) ** 2 + np.cos(theta_1) * np.cos(theta_2) * np.sin((phi_2 - phi_1) / 2) ** 2\n'
        '    return 2 * np.arctan2(np.sqrt(temp), np.sqrt(1 - temp))\n'
    )
    rng = np.random.default_rng(7)
    args = [rng.uniform(-3, 3, 4) for _ in range(4)]
    assert_all_backends_match_numpy(source, *args, atol=1e-7)


CLAIM = 'def kernel(y: Float, x: Float, expected: Float):\n    assert np.arctan2(y, x) == expected\n'


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


@pytest.mark.parametrize("y, x", [(1.0, 2.0), (-2.0, -1.0), (0.0, -1.0), (0.5, 0.0)])
def test_only_the_kernel_value_is_accepted(y, x):
    program = compile_kernel(CLAIM)
    raw = mock_field_publics(compile_kernel('def kernel(y: Float, x: Float):\n    return np.arctan2(y, x)\n'), y, x)[0]
    if raw > BN254_P // 2:  # a negative angle
        raw -= BN254_P
    for backend in BACKENDS:
        assert accepts(program, backend, y, x, raw / Q), backend
        assert not accepts(program, backend, y, x, (raw + 1) / Q), backend
        assert not accepts(program, backend, y, x, (raw - 1) / Q), backend


def test_operands_outside_the_window_have_no_proof():
    # 2^31 + 2^31 in Q32 is 2^64: outside the signed 64-bit window.
    program = compile_kernel('def kernel(y: Float, x: Float):\n    return np.arctan2(y + y, x)\n')
    args = (2.0**31 - 1, 1.0)
    assert program.prove(*args, backend="mock").proof_bytes_hex != "mock_satisfied"
    with pytest.raises(Exception):
        halo2_field_publics(program, *args)
    if nargo_available():
        with pytest.raises(Exception):
            noir_field_publics(program, *args)
