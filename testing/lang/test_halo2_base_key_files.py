"""halo2-base keys written to a directory and read back in another role.

The benchmark harness runs setup, every prove and every verify in its own
process, as the handwritten halo2 baseline does; the keys travel through a key
directory (`keys.save`, `ZKCompiledProgram.load_keys`). These tests check that
keys read back prove and verify like the generated ones, that a verifier's
keys reject a changed public value and cannot prove, and that each read and
each proving phase is timed.
"""
import pytest

from zinnia import *

PARAMS = {"k": 10, "lookup_bits": 8, "precision_bits": 32}
FR = 0x30644E72E131A029B85045B68181585D2833E84879B9709143E1F593F0000001
SCALE = 2**32


def _decode(hexstr):
    value = int.from_bytes(bytes.fromhex(hexstr), "little")
    return value - FR if value > FR // 2 else value


def _bump_first_public(proof):
    first = (int.from_bytes(bytes.fromhex(proof["public_values"][0]), "little") + 1) % FR
    return dict(proof, public_values=[first.to_bytes(32, "little").hex(),
                                      *proof["public_values"][1:]])


@zk_circuit
def scaled_sum(xs: NDArray[Float, 4], c: Float) -> Float:
    total = 0.0
    for x in xs:
        total += x * c
    return total


@pytest.fixture(scope="module")
def compiled():
    return ZKCircuit.from_method(scaled_sum).compile()


@pytest.fixture(scope="module")
def key_dir(compiled, tmp_path_factory):
    directory = tmp_path_factory.mktemp("keys")
    compiled.setup(params=PARAMS).save(directory)
    return directory


def test_save_writes_params_both_keys_and_pinning(key_dir):
    assert sorted(p.name for p in key_dir.iterdir()) == ["keys.json", "params.bin", "pk.bin",
                                                          "vk.bin"]


def test_loaded_keys_prove_and_verify_across_roles(compiled, key_dir):
    prover = ZKCompiledProgram.load_keys(key_dir, role="prover")
    verifier = ZKCompiledProgram.load_keys(key_dir, role="verifier")
    assert (prover.k, prover.lookup_bits, prover.precision_bits) == (10, 8, 32)
    assert prover.has_proving_key and not verifier.has_proving_key
    assert prover.keygen_seconds is None
    assert set(prover.load_seconds) == {"params", "proving_key"}
    assert set(verifier.load_seconds) == {"params", "verifying_key"}
    assert prover.vk_bytes() == verifier.vk_bytes()

    proof = compiled.prove_with_keys(prover, [1.0, 2.0, -0.5, 0.25], 2.0)
    assert _decode(proof["public_values"][-1]) == round(5.5 * SCALE)
    for phase in ("parse_inputs_seconds", "preprocess_seconds", "witness_seconds",
                  "thread_pool_seconds", "prove_seconds"):
        assert proof[phase] >= 0, phase
    report = ZKCompiledProgram.verify_with_keys_report(verifier, proof)
    assert report["valid"] and report["error"] is None and report["verify_seconds"] > 0


def test_verifier_keys_reject_a_changed_public_value(compiled, key_dir):
    prover = ZKCompiledProgram.load_keys(key_dir, role="prover")
    verifier = ZKCompiledProgram.load_keys(key_dir, role="verifier")
    proof = compiled.prove_with_keys(prover, [0.5, 0.5, 0.5, 0.5], 3.0)
    assert ZKCompiledProgram.verify_with_keys(verifier, proof)
    report = ZKCompiledProgram.verify_with_keys_report(verifier, _bump_first_public(proof))
    assert not report["valid"] and "verify_proof failed" in report["error"]


def test_loaded_keys_agree_with_generated_keys(compiled, key_dir):
    """A proof from read-back keys verifies against freshly generated ones and back."""
    generated = compiled.setup(params=PARAMS)
    prover = ZKCompiledProgram.load_keys(key_dir, role="prover")
    verifier = ZKCompiledProgram.load_keys(key_dir, role="verifier")
    assert generated.vk_bytes() == verifier.vk_bytes()
    args = ([3.0, -1.0, 0.0, 2.0], -1.5)
    assert ZKCompiledProgram.verify_with_keys(generated, compiled.prove_with_keys(prover, *args))
    assert ZKCompiledProgram.verify_with_keys(verifier, compiled.prove_with_keys(generated, *args))


def test_verifier_keys_cannot_prove_or_be_saved(compiled, key_dir, tmp_path):
    verifier = ZKCompiledProgram.load_keys(key_dir, role="verifier")
    with pytest.raises(RuntimeError, match="only a verifying key"):
        compiled.prove_with_keys(verifier, [1.0, 1.0, 1.0, 1.0], 1.0)
    with pytest.raises(RuntimeError, match="only a verifying key"):
        verifier.save(tmp_path / "copy")


def test_keys_of_another_circuit_are_refused(key_dir):
    @zk_circuit
    def other(xs: NDArray[Float, 4], c: Float) -> Float:
        return xs[0] - c

    prover = ZKCompiledProgram.load_keys(key_dir, role="prover")
    with pytest.raises(ValueError, match="different circuit"):
        ZKCircuit.from_method(other).compile().prove_with_keys(prover, [1.0, 1.0, 1.0, 1.0], 1.0)


def test_bad_role_and_missing_directory_are_errors(tmp_path):
    with pytest.raises(ValueError, match="role"):
        ZKCompiledProgram.load_keys(tmp_path, role="auditor")
    with pytest.raises(RuntimeError, match="keys.json"):
        ZKCompiledProgram.load_keys(tmp_path / "missing", role="prover")
