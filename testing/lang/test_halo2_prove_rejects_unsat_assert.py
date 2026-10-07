"""C18: the halo2-base prove path must reject an unsatisfiable witness.

halo2's `create_proof` does NOT check that the witness satisfies the gate
polynomials — left unguarded it emits a valid-looking artifact for an
`assert(False)` circuit that only `verify` would later reject. The backend
runs an always-on, O(asserts) witness-satisfaction check during synthesis
(assert/bit gate cells vs. the concrete witness) and raises at prove time
instead. (The legacy halo2-IPA backend, which had the same check, is
retired.)

A kernel asserting a predicate the witness violates (`v == 3`, proved with
`v = 7`) compiles fine — the violation is only visible once a concrete witness
flows through — so the failure must surface at prove(), not compile(). (An
always-false predicate such as `v == v + 1` is rejected earlier, at compile
time: `AlwaysSatisfiedElimination` proves it false from its definitions.)
"""
import pytest

from zinnia import zk_circuit, Integer
from zinnia.api.zk_circuit import ZKCircuit

HALO2_PARAMS = {"k": 12, "precision_bits": 32, "lookup_bits": 8}


@zk_circuit
def asserts_false(v: Integer) -> Integer:
    # False for the witness below: pins an `assert` gate cell to 0, which the
    # gate requires to be 1.
    assert v == 3
    return v


@zk_circuit
def asserts_always_false(v: Integer) -> Integer:
    assert v == v + 1
    return v


def test_always_false_assert_is_rejected_at_compile_time():
    with pytest.raises(Exception, match="provably unsatisfiable"):
        ZKCircuit.from_method(asserts_always_false).compile()


def test_halo2_prove_rejects_unsat_assert():
    backend = "halo2-base"
    circuit = ZKCircuit.from_method(asserts_false)
    # Compilation succeeds — the assertion is witness-dependent.
    circuit.compile()
    with pytest.raises(RuntimeError) as excinfo:
        circuit.prove(7, backend=backend, params=HALO2_PARAMS)
    msg = str(excinfo.value)
    assert "Witness does not satisfy" in msg, (
        f"expected a witness-satisfaction error from {backend}, got: {msg}"
    )


def test_halo2_prove_accepts_satisfiable_assert():
    """No-regression guard: a satisfiable assert still proves and verifies."""
    backend = "halo2-base"

    @zk_circuit
    def asserts_true(v: Integer) -> Integer:
        assert v == v
        return v

    circuit = ZKCircuit.from_method(asserts_true)
    proof = circuit.prove(7, backend=backend, params=HALO2_PARAMS)
    assert circuit.verify(proof), f"{backend}: satisfiable circuit must verify"
