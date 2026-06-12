"""C18: the halo2 prove path must reject an unsatisfiable witness.

halo2's `create_proof` does NOT check that the witness satisfies the gate
polynomials — left unguarded it emits a valid-looking artifact for an
`assert(False)` circuit that only `verify` would later reject. The legacy
halo2-IPA backend used to run MockProver for this check; it was dropped for
perf (commit bb67c32) and never replaced, so prove() silently accepted
unsat witnesses. Both halo2 backends now run an always-on, O(asserts)
witness-satisfaction check during synthesis (assert/bit gate cells vs. the
concrete witness) and raise at prove time instead.

A kernel asserting an always-false predicate (`v == v + 1`) compiles fine —
the contradiction is only visible once a concrete witness flows through — so
the failure must surface at prove(), not compile().
"""
import pytest

from zinnia import zk_circuit, Integer
from zinnia.api.zk_circuit import ZKCircuit

HALO2_PARAMS = {"k": 12, "precision_bits": 32, "lookup_bits": 8}


@zk_circuit
def asserts_false(v: Integer) -> Integer:
    # Always false for any v: pins an `assert` gate cell to 0, which the
    # gate requires to be 1.
    assert v == v + 1
    return v


@pytest.mark.parametrize("backend", ["halo2", "halo2-base"])
def test_halo2_prove_rejects_unsat_assert(backend):
    circuit = ZKCircuit.from_method(asserts_false)
    # Compilation succeeds — the assertion is witness-dependent.
    circuit.compile()
    with pytest.raises(RuntimeError) as excinfo:
        circuit.prove(7, backend=backend, params=HALO2_PARAMS)
    msg = str(excinfo.value)
    assert "Witness does not satisfy" in msg, (
        f"expected a witness-satisfaction error from {backend}, got: {msg}"
    )


@pytest.mark.parametrize("backend", ["halo2", "halo2-base"])
def test_halo2_prove_accepts_satisfiable_assert(backend):
    """No-regression guard: a satisfiable assert still proves and verifies."""

    @zk_circuit
    def asserts_true(v: Integer) -> Integer:
        assert v == v
        return v

    circuit = ZKCircuit.from_method(asserts_true)
    proof = circuit.prove(7, backend=backend, params=HALO2_PARAMS)
    assert circuit.verify(proof), f"{backend}: satisfiable circuit must verify"
