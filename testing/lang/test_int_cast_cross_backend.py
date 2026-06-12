"""Cross-backend value + soundness regression for the float->int cast.

C02: halo2-base `int_cast` used to compute the quotient host-side from the
witness and load it as a *free* witness cell with no gate tying it to the
input — a malicious prover could substitute any cast result. It also used
Euclidean (flooring) division, disagreeing with every other backend (which
truncate toward zero, matching Python `int()`) on negative inputs.

This module pins both facets:
  * `int(x)` agrees across mock / halo2-base / halo2 (legacy) for negative x
    (the value criterion — int(-2.5) == -2, NOT floor == -3), and
  * the constrained lowering keeps proving — legacy halo2 proves AND
    verifies the cast; halo2-base produces the correct constrained public
    output.

Notes on backend reachability from the Python API:
  * halo2-base `verify()` is pre-existing-broken through this entry point
    (it fails with ConstraintSystemFailure even for a trivial `v + 1`
    circuit that never touches int_cast — same keygen-without-witness shape
    family as the T01 in-crate test breakage). We therefore assert on the
    constrained public *value* halo2-base emits rather than on verify().
  * The malicious-witness tamper regression lives as a Rust test in
    `src/prove/halo2_base/synthesizer.rs` behind the T01 cfg gate; see the
    C02 report. From Python the strongest reachable statements are the
    cross-backend value agreement here plus legacy-halo2 prove+verify.
"""
import pytest

from zinnia import *

# Tiny circuit -> small k keeps prove fast; k must clear the 32-bit range
# lookup (lookup_bits=8 -> the int_cast |r| range check), so k=8 is too small.
HALO2_PARAMS = {"k": 12, "precision_bits": 32, "lookup_bits": 8}

# bn256 scalar field modulus (halo2 IPA curve used by the kernel).
_BN256_FR = 0x30644E72E131A029B85045B68181585D2833E84879B9709143E1F593F0000001


def _decode_field_int(hexstr):
    """Decode a little-endian field-repr hex string to a signed integer."""
    val = int.from_bytes(bytes.fromhex(hexstr), "little")
    if val > _BN256_FR // 2:
        val -= _BN256_FR
    return val


def _mock_value(public_values):
    """Mock backend tags public values as `<label>=<num>`."""
    assert public_values, "circuit exposed no public values"
    tail = public_values[-1]
    assert "=" in tail, f"unexpected mock public value: {tail!r}"
    return int(tail.rsplit("=", 1)[1])


# Truncation toward zero (Python int()), NOT floor.
CASES = [
    (-2.5, -2),
    (-0.5, 0),
    (-3.9, -3),
    (2.5, 2),
    (7.0, 7),
]


def _cast_circuit():
    @zk_circuit
    def cast(v: Public[Float]) -> Integer:
        return int(v)

    return ZKCircuit.from_method(cast)


@pytest.mark.parametrize("x, expected", CASES)
def test_int_cast_truncates_toward_zero_mock(x, expected):
    proof = _cast_circuit().prove(x, backend="mock")
    assert proof.proof_bytes_hex == "mock_satisfied"
    assert _mock_value(proof.public_values) == expected


@pytest.mark.parametrize("x, expected", CASES)
def test_int_cast_value_halo2_base(x, expected):
    """halo2-base emits the truncated value through its constrained lowering."""
    proof = _cast_circuit().prove(x, backend="halo2-base", params=HALO2_PARAMS)
    got = _decode_field_int(proof.public_values[-1])
    assert got == expected, f"halo2-base int({x}) = {got}, expected {expected}"


@pytest.mark.parametrize("x, expected", CASES)
def test_int_cast_proves_and_verifies_legacy_halo2(x, expected):
    """Legacy halo2 proves, verifies, and agrees on the truncated value."""
    circuit = _cast_circuit()
    proof = circuit.prove(x, backend="halo2", params=HALO2_PARAMS)
    assert circuit.verify(proof), f"legacy halo2 proof did not verify for int({x})"
    assert _decode_field_int(proof.public_values[-1]) == expected


def test_int_cast_negative_agrees_across_backends():
    """Canonical case int(-2.5) == -2 agrees mock / halo2-base / legacy halo2."""
    circuit = _cast_circuit()
    mock = circuit.prove(-2.5, backend="mock")
    base = circuit.prove(-2.5, backend="halo2-base", params=HALO2_PARAMS)
    legacy = circuit.prove(-2.5, backend="halo2", params=HALO2_PARAMS)
    assert circuit.verify(legacy)
    assert _mock_value(mock.public_values) == -2
    assert _decode_field_int(base.public_values[-1]) == -2
    assert _decode_field_int(legacy.public_values[-1]) == -2
