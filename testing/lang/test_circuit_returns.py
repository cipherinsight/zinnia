import pytest

from zinnia import *


def test_circuit_annotated_integer_return_is_public_mock_value():
    @zk_circuit
    def add_one(x: Public[Integer]) -> Integer:
        return x + 1

    proof = ZKCircuit.from_method(add_one).prove(6, backend="mock")
    assert proof.proof_bytes_hex == "mock_satisfied"
    assert any(v.endswith("=7") for v in proof.public_values)


def test_circuit_annotated_float_return_is_public_mock_value():
    @zk_circuit
    def half_plus_one(x: Public[Float]) -> Float:
        return x * 0.5 + 1.0

    proof = ZKCircuit.from_method(half_plus_one).prove(4.0, backend="mock")
    assert proof.proof_bytes_hex == "mock_satisfied"
    assert proof.public_values


def test_circuit_annotated_return_supports_early_return():
    @zk_circuit
    def clamp_positive(x: Public[Integer]) -> Integer:
        if x > 0:
            return x
        return 0

    positive = ZKCircuit.from_method(clamp_positive).prove(5, backend="mock")
    zero = ZKCircuit.from_method(clamp_positive).prove(-3, backend="mock")
    assert positive.proof_bytes_hex == "mock_satisfied"
    assert zero.proof_bytes_hex == "mock_satisfied"
    assert any(v.endswith("=5") for v in positive.public_values)
    assert any(v.endswith("=0") for v in zero.public_values)


def test_circuit_annotated_integer_return_halo2_base_proves_and_verifies():
    @zk_circuit
    def add_one(x: Public[Integer]) -> Integer:
        return x + 1

    circuit = ZKCircuit.from_method(add_one)
    proof = circuit.prove(6, backend="halo2-base")
    assert proof.public_values
    assert circuit.verify(proof)


def test_circuit_bare_scalar_return_is_inferred_and_public():
    @zk_circuit
    def add_one(x: Public[Integer]):
        return x + 1

    proof = ZKCircuit.from_method(add_one).prove(6, backend="mock")
    assert proof.proof_bytes_hex == "mock_satisfied"
    assert any(v.endswith("=7") for v in proof.public_values)


def test_circuit_bare_tuple_return_is_flattened_to_public_outputs():
    @zk_circuit
    def pair(x: Public[Integer]):
        return (x, x + 1)

    proof = ZKCircuit.from_method(pair).prove(6, backend="mock")
    assert proof.proof_bytes_hex == "mock_satisfied"
    assert any(v.endswith("=6") for v in proof.public_values)
    assert any(v.endswith("=7") for v in proof.public_values)


def test_circuit_without_return_outputs_the_arguments_it_writes():
    # A function that returns None outputs the array arguments it writes
    # (see test_in_place_outputs.py); one that writes nothing outputs nothing.
    @zk_circuit
    def mutate_only(x: Integer[Array, "3"]):
        x[0] = x[1] + 1

    proof = ZKCircuit.from_method(mutate_only).prove([1, 2, 3], backend="mock")
    assert proof.proof_bytes_hex == "mock_satisfied"
    from testing.lang.mock_outputs import mock_publics
    assert mock_publics(proof) == [3, 2, 3]

    @zk_circuit
    def read_only(x: Integer[Array, "3"]):
        assert x[0] < 100

    proof = ZKCircuit.from_method(read_only).prove([1, 2, 3], backend="mock")
    assert proof.proof_bytes_hex == "mock_satisfied"
    assert proof.public_values == []


def test_circuit_bare_complex_return_exposes_real_and_imag_parts():
    @zk_circuit
    def complex_pair(x: Public[Float]):
        return x + 2j

    proof = ZKCircuit.from_method(complex_pair).prove(3.0, backend="mock")
    assert proof.proof_bytes_hex == "mock_satisfied"
    assert len(proof.public_values) == 2
    assert proof.public_values


def test_circuit_annotated_partial_return_rejected():
    @zk_circuit
    def partial(x: Public[Integer]) -> Integer:
        if x > 0:
            return x

    with pytest.raises(Exception, match="requires every path to return"):
        ZKCircuit.from_method(partial).compile()


def test_circuit_unannotated_partial_return_rejected():
    @zk_circuit
    def partial(x: Public[Integer]):
        if x > 0:
            return x

    with pytest.raises(Exception, match="partial returns without an explicit return annotation"):
        ZKCircuit.from_method(partial).compile()


def test_circuit_mixed_none_and_value_returns_rejected():
    @zk_circuit
    def mixed(x: Public[Integer]):
        if x > 0:
            return x
        return

    with pytest.raises(Exception, match="mixing bare returns with value returns"):
        ZKCircuit.from_method(mixed).compile()


def test_circuit_mixed_explicit_none_and_value_returns_rejected():
    @zk_circuit
    def mixed(x: Public[Integer]):
        if x > 0:
            return x
        return None

    with pytest.raises(Exception, match="mixing bare returns with value returns"):
        ZKCircuit.from_method(mixed).compile()


def test_circuit_annotated_return_type_mismatch_rejected():
    @zk_circuit
    def bad(x: Public[Integer]) -> Integer:
        return (x, x + 1)

    with pytest.raises(Exception, match="Circuit return type mismatch"):
        ZKCircuit.from_method(bad).compile()


def test_circuit_annotated_array_return_type_mismatch_rejected():
    @zk_circuit
    def bad(x: Public[Integer]) -> Float[Array, "N<=8"]:
        return x + 1.0

    with pytest.raises(Exception, match="Circuit return type mismatch"):
        ZKCircuit.from_method(bad).compile()


def test_circuit_dynamic_ndarray_return_exposes_runtime_metadata_and_bounded_payload():
    @zk_circuit
    def dyn_return(x: Public[Float[Array, "N<=4"]]):
        return x

    proof = ZKCircuit.from_method(dyn_return).prove(
        np.asarray([2.0, 3.0]), backend="mock"
    )
    assert proof.proof_bytes_hex == "mock_satisfied"
    # The runtime length (also the only dim) and the 4-slot payload.
    assert len(proof.public_values) == 5
    assert any(v.endswith("=2") for v in proof.public_values)


def test_boolean_returned_as_integer_is_zero_or_one():
    """`bool` is an `int` subtype: a chip or circuit annotated `-> Integer`
    may return a comparison, and the result is the integer 0 or 1 (it used
    to fail with "return type mismatch: expected Integer, got Boolean")."""

    @zk_chip
    def is_thirteen(x) -> Integer:
        return x == 13

    @zk_circuit
    def count_thirteens(x: Integer, y: Integer, expected: Integer):
        assert is_thirteen(x + y) + is_thirteen(x) == expected

    for x, y in [(1, 12), (13, 0), (13, 5), (2, 2)]:
        truth = int(x + y == 13) + int(x == 13)
        circuit = ZKCircuit.from_method(count_thirteens, chips=[is_thirteen])
        assert circuit.prove(x, y, truth, backend="mock").proof_bytes_hex == "mock_satisfied"
        assert circuit.prove(x, y, truth + 1, backend="mock").proof_bytes_hex != "mock_satisfied"

    @zk_circuit
    def is_three(x: Integer) -> Integer:
        return x == 3

    for x in (3, 4):
        proof = ZKCircuit.from_method(is_three).prove(x, backend="mock")
        assert proof.proof_bytes_hex == "mock_satisfied"
        assert proof.public_values[-1].endswith(f"={int(x == 3)}")
