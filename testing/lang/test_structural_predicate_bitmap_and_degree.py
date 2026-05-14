"""Tests for the bitmap/degree predicates (popcount, forall_*) — Python
surface side.

Verifies:
- `popcount(b)` registered as an alias of `nnz` for boolean arrays.
- `forall(arr, P)` extracted to a synthesized `forall_<inner>` IR kind
  with the inner element-predicate's args inlined.
- Witness emitters produce the per-element check chains.
"""

import pytest

from zinnia import zk_circuit, requires, NDArray, Integer, Boolean
from zinnia.spec.predicates import (
    popcount, forall, is_nonneg, lt_bound, eq_const, in_range,
)
from zinnia.api.zk_circuit import ZKCircuit


def _dump_ir(decorated):
    return ZKCircuit.from_method(decorated).compile().get_ir_stmts()


def _structural_predicates(stmts):
    return [
        s["ir_instance"]["ir_data"]
        for s in stmts
        if s.get("ir_instance", {}).get("__class__") == "StructuralPredicateIR"
    ]


# ── popcount ─────────────────────────────────────────────────────────────


@zk_circuit
@requires(lambda b, k: popcount(b) == k)
def _popcount_input(b: NDArray[Integer, 4], k: int):
    pass


def test_popcount_atom_emitted_with_correct_payload():
    sps = _structural_predicates(_dump_ir(_popcount_input))
    assert len(sps) == 1
    assert sps[0] == {"kind": "popcount", "args": ["b"], "op": "==", "bound": "k"}


def test_popcount_witness_reuses_nnz_indicator_chain():
    """popcount uses emit_nnz directly: 4 NeI per element + 4 IntCast +
    ≥3 AddI + 1 EqI + 1 Assert. Same shape as the nnz card's witness
    chain.
    """
    stmts = _dump_ir(_popcount_input)
    classes = [s.get("ir_instance", {}).get("__class__") for s in stmts]
    assert classes.count("NotEqualIIR") >= 4
    assert classes.count("IntCastIR") >= 4
    assert classes.count("AddIIR") >= 3
    assert classes.count("AssertIR") >= 1


# ── forall(arr, is_nonneg) — bare-name form ──────────────────────────────


@zk_circuit
@requires(lambda x: forall(x, is_nonneg))
def _forall_is_nonneg_input(x: NDArray[Integer, 4]):
    pass


def test_forall_is_nonneg_flattens_to_synthesized_kind():
    sps = _structural_predicates(_dump_ir(_forall_is_nonneg_input))
    assert len(sps) == 1
    assert sps[0] == {
        "kind": "forall_is_nonneg", "args": ["x"], "op": None, "bound": None,
    }


def test_forall_is_nonneg_witness_emits_per_element_ge_assert():
    stmts = _dump_ir(_forall_is_nonneg_input)
    classes = [s.get("ir_instance", {}).get("__class__") for s in stmts]
    assert classes.count("GreaterThanOrEqualIIR") >= 4
    assert classes.count("AssertIR") >= 4


# ── forall(arr, lt_bound(B)) ─────────────────────────────────────────────


@zk_circuit
@requires(lambda col_idx, N: forall(col_idx, lt_bound(N)))
def _forall_lt_bound_input(col_idx: NDArray[Integer, 3], N: int):
    pass


def test_forall_lt_bound_flattens_with_inner_arg():
    sps = _structural_predicates(_dump_ir(_forall_lt_bound_input))
    assert len(sps) == 1
    assert sps[0] == {
        "kind": "forall_lt_bound", "args": ["col_idx", "N"], "op": None, "bound": None,
    }


def test_forall_lt_bound_witness_emits_per_element_lt_assert():
    stmts = _dump_ir(_forall_lt_bound_input)
    classes = [s.get("ir_instance", {}).get("__class__") for s in stmts]
    assert classes.count("LessThanIIR") >= 3
    assert classes.count("AssertIR") >= 3


# ── forall(arr, eq_const(C)) ─────────────────────────────────────────────


@zk_circuit
@requires(lambda arr, C: forall(arr, eq_const(C)))
def _forall_eq_const_input(arr: NDArray[Integer, 3], C: int):
    pass


def test_forall_eq_const_flattens_with_inner_arg():
    sps = _structural_predicates(_dump_ir(_forall_eq_const_input))
    assert len(sps) == 1
    assert sps[0]["kind"] == "forall_eq_const"
    assert sps[0]["args"] == ["arr", "C"]


# ── forall(arr, in_range(lo, hi)) — 3 args total ─────────────────────────


@zk_circuit
@requires(lambda arr, lo, hi: forall(arr, in_range(lo, hi)))
def _forall_in_range_input(arr: NDArray[Integer, 3], lo: int, hi: int):
    pass


def test_forall_in_range_flattens_with_three_args():
    sps = _structural_predicates(_dump_ir(_forall_in_range_input))
    assert len(sps) == 1
    assert sps[0]["kind"] == "forall_in_range"
    assert sps[0]["args"] == ["arr", "lo", "hi"]


def test_forall_in_range_witness_emits_two_asserts_per_element():
    """N=3 → 3 GteI + 3 LteI + 6 Asserts."""
    stmts = _dump_ir(_forall_in_range_input)
    classes = [s.get("ir_instance", {}).get("__class__") for s in stmts]
    assert classes.count("GreaterThanOrEqualIIR") >= 3
    assert classes.count("LessThanOrEqualIIR") >= 3
    assert classes.count("AssertIR") >= 6


# ── Error: unknown inner predicate ───────────────────────────────────────


def test_forall_with_unknown_inner_predicate_errors_at_compile():
    from zinnia.debug.exception import ZinniaException

    @zk_circuit
    @requires(lambda arr: forall(arr, mystery))  # noqa: F821 — intentional
    def bad(arr: NDArray[Integer, 3]):
        pass

    with pytest.raises(ZinniaException, match=r"forall.*inner predicate.*mystery"):
        ZKCircuit.from_method(bad).compile()


# ── Compatibility ────────────────────────────────────────────────────────


@zk_circuit
def _plain_bitmap(arr: NDArray[Integer, 3]):
    pass


def test_plain_circuit_has_no_bitmap_atoms():
    sps = _structural_predicates(_dump_ir(_plain_bitmap))
    assert sps == []
