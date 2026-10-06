"""The legacy halo2 (IPA) backend is retired.

Its comparisons are limited to operands within 2^56 of each other, so it
cannot satisfy the 64-bit input-domain checks every circuit with an integer
or float input carries. Selecting it, for proving or for verifying an old
artifact, is an error that names the replacement.
"""
import pytest

from zinnia import Integer, zk_circuit
from zinnia.api.zk_circuit import ZKCircuit
from zinnia.exec.proof_result import ZKProofResult


@zk_circuit
def add_one(x: Integer) -> Integer:
    return x + 1


@pytest.mark.parametrize("backend", ["halo2", "halo2-ipa"])
def test_proving_with_the_legacy_backend_names_halo2_base(backend):
    with pytest.raises(RuntimeError, match="legacy halo2 backend .* use backend='halo2-base'"):
        ZKCircuit.from_method(add_one).prove(6, backend=backend)


def test_verifying_a_legacy_artifact_names_halo2_base():
    circuit = ZKCircuit.from_method(add_one)
    legacy = ZKProofResult.from_json(circuit.prove(6, backend="halo2-base").to_json())
    legacy.backend = "halo2-ipa"
    with pytest.raises(RuntimeError, match="use backend='halo2-base'"):
        circuit.verify(legacy)


def test_halo2_base_still_proves_the_same_circuit():
    circuit = ZKCircuit.from_method(add_one)
    assert circuit.verify(circuit.prove(6, backend="halo2-base"))
