"""Tests for the ordering-family predicates (is_sorted,
is_monotone_nondecreasing, max_run) — Python surface side.

The Rust side carries the deeper assertions (registry contains the
predicates, witness emitters add the right IR ops). This file verifies
the end-to-end Python → IR pipeline for circuits decorated with these
preconditions.
"""

import pytest
import numpy as np

from zinnia import zk_circuit, requires, NDArray, Integer, ZinniaException
from zinnia.spec.predicates import is_sorted, is_monotone_nondecreasing, max_run
from zinnia.api.zk_circuit import ZKCircuit


def _dump_ir(decorated):
    return ZKCircuit.from_method(decorated).compile().get_ir_stmts()


def _structural_predicates(stmts):
    return [
        s["ir_instance"]["ir_data"]
        for s in stmts
        if s.get("ir_instance", {}).get("__class__") == "StructuralPredicateIR"
    ]


# ── is_sorted: atom emission and witness chain ─────────────────────────


@zk_circuit
@requires(lambda x: is_sorted(x))
def _sorted_input(x: NDArray[Integer, 8]):
    pass


def test_is_sorted_atom_emitted_with_correct_payload():
    sps = _structural_predicates(_dump_ir(_sorted_input))
    assert len(sps) == 1
    assert sps[0] == {"kind": "is_sorted", "args": ["x"], "op": None, "bound": None}


def test_is_sorted_witness_emits_adjacent_pair_asserts():
    """For an 8-element sorted-input precondition, the witness emitter
    should produce 7 adjacent-pair comparisons + 7 asserts.
    """
    stmts = _dump_ir(_sorted_input)
    classes = [s.get("ir_instance", {}).get("__class__") for s in stmts]
    assert classes.count("LessThanOrEqualIIR") >= 7, \
        f"expected ≥7 LteI ops; got {classes.count('LessThanOrEqualIIR')}"
    assert classes.count("AssertIR") >= 7, \
        f"expected ≥7 Assert ops; got {classes.count('AssertIR')}"


# ── is_monotone_nondecreasing: alias of is_sorted ───────────────────────


@zk_circuit
@requires(lambda row_ptr: is_monotone_nondecreasing(row_ptr))
def _monotone_input(row_ptr: NDArray[Integer, 5]):
    pass


def test_is_monotone_emits_same_witness_as_is_sorted():
    stmts = _dump_ir(_monotone_input)
    sps = _structural_predicates(stmts)
    assert len(sps) == 1
    assert sps[0]["kind"] == "is_monotone_nondecreasing"
    # 5 elements → 4 adjacent pairs → 4 LteI + 4 Assert.
    classes = [s.get("ir_instance", {}).get("__class__") for s in stmts]
    assert classes.count("LessThanOrEqualIIR") >= 4
    assert classes.count("AssertIR") >= 4


# ── max_run with `<=` op ─────────────────────────────────────────────────


@zk_circuit
@requires(lambda row_ptr, K: max_run(row_ptr) <= K)
def _max_run_input(row_ptr: NDArray[Integer, 5], K: int):
    pass


def test_max_run_atom_emitted_with_correct_payload():
    sps = _structural_predicates(_dump_ir(_max_run_input))
    assert len(sps) == 1
    assert sps[0] == {"kind": "max_run", "args": ["row_ptr"], "op": "<=", "bound": "K"}


def test_max_run_witness_emits_per_pair_difference_and_assert():
    """For a 5-element monotone array with `max_run <= K`, the witness
    emitter should produce 4 (subtraction + LteI + Assert) chains.
    """
    stmts = _dump_ir(_max_run_input)
    classes = [s.get("ir_instance", {}).get("__class__") for s in stmts]
    assert classes.count("SubIIR") >= 4
    assert classes.count("LessThanOrEqualIIR") >= 4
    assert classes.count("AssertIR") >= 4


# ── max_run with unsupported op falls back gracefully ────────────────────


@zk_circuit
@requires(lambda row_ptr, K: max_run(row_ptr) == K)
def _max_run_eq_unsupported(row_ptr: NDArray[Integer, 4], K: int):
    pass


def test_max_run_with_eq_does_not_emit_witness_chain():
    """`max_run(arr) == K` is not yet supported by the witness emitter
    (computing the actual max gap is harder than the `<=` per-pair check).
    The compile succeeds — the IR atom is still emitted — but no
    witness-time enforcement is produced. Documented as a soundness
    gap in the card status.
    """
    stmts = _dump_ir(_max_run_eq_unsupported)
    sps = _structural_predicates(stmts)
    assert len(sps) == 1
    assert sps[0]["op"] == "=="
    # No SubI/LteI emitted because the emitter short-circuits.
    classes = [s.get("ir_instance", {}).get("__class__") for s in stmts]
    assert classes.count("SubIIR") == 0
    assert classes.count("AssertIR") == 0


# ── Compatibility: no preconditions ─────────────────────────────────────


@zk_circuit
def _plain_ordering(x: NDArray[Integer, 5]):
    pass


def test_plain_circuit_has_no_ordering_atoms():
    sps = _structural_predicates(_dump_ir(_plain_ordering))
    assert sps == []
