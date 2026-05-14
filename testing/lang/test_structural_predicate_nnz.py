"""Tests for the `nnz` predicate end-to-end (Python surface side).

The Rust side carries the deeper assertions (Z3 actually derives `k <= len(x)`
from the structural-predicate facts — see
`src/optim/predicates/tests/discharge_tests.rs::full_resolver_chain_proves_k_bound_via_smt`).

This file covers the Python surface:

- `@requires(lambda x, k: nnz(x) == k)` is recognised and emits the right
  IR atom.
- Walkthrough 1 (`nonzero_indices`) rejects with the expected
  shape-axis diagnostic under default `smt=off`. The `smt=on` flip is
  blocked by the ShapeAxis chokepoint extension (the resolver chain
  produces the bound — verified Rust-side — but `np_fill` does not yet
  fall back to dyn-ndarray construction). See the nnz card status.
"""

import pytest
import numpy as np

from zinnia import zk_circuit, requires, NDArray, Float, Integer, ZinniaException
from zinnia.spec.predicates import nnz
from zinnia.api.zk_circuit import ZKCircuit


def _dump_ir(decorated):
    return ZKCircuit.from_method(decorated).compile().get_ir_stmts()


def _structural_predicates(stmts):
    return [
        s["ir_instance"]["ir_data"]
        for s in stmts
        if s.get("ir_instance", {}).get("__class__") == "StructuralPredicateIR"
    ]


# ── Surface emission of the nnz atom ─────────────────────────────────────


@zk_circuit
@requires(lambda x, k: nnz(x) == k)
def _nnz_eq_k_tiny(x: NDArray[Float, 8], k: int):
    pass


def test_nnz_atom_emitted_with_correct_payload():
    sps = _structural_predicates(_dump_ir(_nnz_eq_k_tiny))
    assert len(sps) == 1
    assert sps[0] == {"kind": "nnz", "args": ["x"], "op": "==", "bound": "k"}


# ── Walkthrough 1 — kill-criterion reject half ──────────────────────────


@zk_circuit
@requires(lambda x, k: nnz(x) == k)
def _walkthrough_1_nonzero_indices(x: NDArray[Float, 1024], k: int):
    out = np.zeros(k, dtype=Integer)
    j = 0
    for i in range(1024):
        if x[i] != 0.0:
            out[j] = i
            j += 1
    _zinnia_result = out


def test_walkthrough_1_rejects_under_smt_off():
    """Kill-criterion reject-half: with `ZINNIA_SMT_ENABLE=0`, the
    ShapeAxis chokepoint sees `k` as a non-constant runtime int (no
    bound derivable without SMT) and panics with the cited diagnostic.
    """
    import os
    prev = os.environ.get("ZINNIA_SMT_ENABLE")
    os.environ["ZINNIA_SMT_ENABLE"] = "0"
    try:
        with pytest.raises(ZinniaException, match=r"(?i)shape|static|constant"):
            ZKCircuit.from_method(_walkthrough_1_nonzero_indices).compile()
    finally:
        if prev is None:
            os.environ.pop("ZINNIA_SMT_ENABLE", None)
        else:
            os.environ["ZINNIA_SMT_ENABLE"] = prev


@zk_circuit
@requires(lambda x, k: nnz(x) == k)
def _nnz_witness_demo(x: NDArray[Integer, 4], k: int):
    pass


def test_nnz_witness_emits_indicator_chain_and_assert():
    """A circuit with `@requires(lambda x, k: nnz(x) == k)` must, beyond
    the compile-time IR atom, emit a witness-time circuit that the
    prover satisfies. For x of length 4: four indicator chains
    (`NeI` + `IntCast`), four `AddI`s accumulating the count, a final
    `EqI(count, k)`, and an `IR::Assert`. Without this chain a malicious
    prover supplying mismatched (x, k) would not be caught.
    """
    stmts = _dump_ir(_nnz_witness_demo)
    classes = [s.get("ir_instance", {}).get("__class__") for s in stmts]
    # The StructuralPredicate atom is the compile-time fact.
    assert "StructuralPredicateIR" in classes
    # The witness chain produces:
    #   - one `NotEqualIIR` per element (4)
    #   - one `IntCastIR` per element (4)
    #   - one `AddIIR` per element (4)
    #   - one `EqualIIR` for the final comparison
    #   - one `AssertIR` (the enforcement)
    assert classes.count("NotEqualIIR") >= 4
    assert classes.count("IntCastIR") >= 4
    # 4 elements produce 4 AddI calls, but constant-folding collapses
    # `AddI(0, indicator)` into the indicator itself, leaving 3 AddI in
    # the dump. Accept ≥ 3 to remain robust to the folding pass.
    assert classes.count("AddIIR") >= 3
    assert classes.count("EqualIIR") >= 1
    assert classes.count("AssertIR") >= 1
    # The Assert must come after the EqI in the IR order.
    assert_idx = next(i for i, c in enumerate(classes) if c == "AssertIR")
    eq_idx = next(i for i, c in enumerate(classes) if c == "EqualIIR")
    assert eq_idx < assert_idx, "EqI must precede Assert in the witness chain"


def test_walkthrough_1_compiles_under_smt_on():
    """Kill-criterion compile-half: under `ZINNIA_SMT_ENABLE=1`, Walkthrough 1
    no longer rejects at the ShapeAxis chokepoint. The bounded-aware
    `require_static_or_bounded_int` returns `Bounded { min: 0, max: 16 }`
    (from `nnz(x) == k` plus `len(x) == 16`), and `np_fill` routes through
    `dyn_fill_with_active` to build a dyn-ndarray with active size = k.
    """
    import os
    prev = os.environ.get("ZINNIA_SMT_ENABLE")
    os.environ["ZINNIA_SMT_ENABLE"] = "1"
    try:
        program = ZKCircuit.from_method(_walkthrough_1_nonzero_indices).compile()
        stmts = program.get_ir_stmts()
        classes = [s.get("ir_instance", {}).get("__class__") for s in stmts]
        # The compile succeeded — that's the kill criterion. Sanity-check
        # the IR contains both the structural-predicate atom and the
        # witness-time assert.
        assert "StructuralPredicateIR" in classes
        assert "AssertIR" in classes
        # No `DynamicNDArrayMetaIR` because `dyn_fill_with_active` uses
        # low-level `AllocateMemory` + `WriteMemory` to back the dyn
        # array. Just make sure the compile didn't fall back to the
        # rejection path.
    finally:
        if prev is None:
            os.environ.pop("ZINNIA_SMT_ENABLE", None)
        else:
            os.environ["ZINNIA_SMT_ENABLE"] = prev


def test_walkthrough_1_emits_structural_predicate_atom():
    """Even if Walkthrough 1 cannot compile yet, the surface must still
    emit the precondition's IR atom before the chokepoint trips.
    Verified by parsing the AST directly (avoids the chokepoint panic).
    """
    import inspect
    from zinnia.compile.zinnia_compiler import ZinniaCompiler

    src = inspect.getsource(_walkthrough_1_nonzero_indices)
    # Pull the inner function source out of the closure.
    for cell in _walkthrough_1_nonzero_indices.__closure__ or ():
        c = cell.cell_contents
        if isinstance(c, str) and "def _walkthrough_1_nonzero_indices" in c:
            src = c
            break

    ast_dict = ZinniaCompiler.circuit_ast_parse(src, "_walkthrough_1_nonzero_indices")
    requires = ast_dict.get("requires", [])
    assert len(requires) == 1
    assert requires[0]["kind"] == "nnz"
    assert requires[0]["args"] == ["x"]
    assert requires[0]["op"] == "=="
    assert requires[0]["bound"] == "k"
