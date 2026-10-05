"""Tests for the permutation-family predicates (is_permutation,
cycle_count, fixed_point_count) — Python surface side.
"""

import pytest

from zinnia import zk_circuit, requires, NDArray, Integer
from zinnia.spec.predicates import is_permutation, cycle_count, fixed_point_count
from zinnia.api.zk_circuit import ZKCircuit
from zinnia.debug.exception import ZinniaException


def _dump_ir(decorated):
    return ZKCircuit.from_method(decorated).compile().get_ir_stmts()


def _structural_predicates(stmts):
    return [
        s["ir_instance"]["ir_data"]
        for s in stmts
        if s.get("ir_instance", {}).get("__class__") == "StructuralPredicateIR"
    ]


# ── is_permutation: atom emission and quadratic witness chain ────────────


@zk_circuit
@requires(lambda p: is_permutation(p))
def _perm_input(p: NDArray[Integer, 4]):
    pass


def test_is_permutation_atom_emitted_with_correct_payload():
    sps = _structural_predicates(_dump_ir(_perm_input))
    assert len(sps) == 1
    assert sps[0] == {
        "kind": "is_permutation", "args": ["p"], "op": None, "bound": None,
    }


def test_is_permutation_witness_emits_quadratic_chain():
    """For N=4, the witness emitter emits:
    - 4 GteI (range lower) + 4 LtI (range upper) + 4*(4-1)/2 = 6 NeI
      (injectivity pairs)
    - 4 + 4 + 6 = 14 IR::Assert
    """
    stmts = _dump_ir(_perm_input)
    classes = [s.get("ir_instance", {}).get("__class__") for s in stmts]
    assert classes.count("LessThanIIR") >= 8  # PatternMatchOptim: >=, <=, > are a (negated) <
    assert classes.count("NotEqualIIR") >= 6
    assert classes.count("AssertIR") >= 14


# ── fixed_point_count: atom + indicator sum chain ────────────────────────


@zk_circuit
@requires(lambda p, k: fixed_point_count(p) == k)
def _fpc_input(p: NDArray[Integer, 4], k: int):
    pass


def test_fixed_point_count_atom_emitted_with_correct_payload():
    sps = _structural_predicates(_dump_ir(_fpc_input))
    assert len(sps) == 1
    assert sps[0] == {
        "kind": "fixed_point_count", "args": ["p"], "op": "==", "bound": "k",
    }


def test_fixed_point_count_witness_emits_indicator_chain_and_assert():
    """For N=4 with `fixed_point_count(p) == k`, witness emits:
    - 4 EqI for per-index `p[i] == i` (booleans reinterpreted as 0/1
      integers; no `IntCast`, the float->int conversion that maps 1 to 0)
    - ≥3 AddI for the running sum (4 calls, but post-fold AddI(0, ind_0) = ind_0)
    - 1 EqI for the final `count == k`
    - 1 IR::Assert
    """
    stmts = _dump_ir(_fpc_input)
    classes = [s.get("ir_instance", {}).get("__class__") for s in stmts]
    # 4 per-index + 1 final = 5 EqI total in absence of folding.
    assert classes.count("EqualIIR") >= 4
    assert classes.count("IntCastIR") == 0
    assert classes.count("AddIIR") >= 3
    assert classes.count("AssertIR") >= 1


# ── cycle_count: no in-circuit encoding, rejected ────────────────────────


@zk_circuit
@requires(lambda p, c: cycle_count(p) == c)
def _cycle_count_input(p: NDArray[Integer, 4], c: int):
    pass


def test_cycle_count_is_rejected():
    """`cycle_count` has no in-circuit encoding (graph traversal). It used
    to compile as an unenforced compile-time fact; it is now rejected."""
    with pytest.raises(ZinniaException, match="cannot be enforced"):
        _dump_ir(_cycle_count_input)


# ── multiple permutation preconditions stack ─────────────────────────────


@zk_circuit
@requires(lambda p: is_permutation(p))
@requires(lambda p, f: fixed_point_count(p) == f)
def _multi_perm_input(p: NDArray[Integer, 3], f: int):
    pass


def test_multiple_permutation_preconditions_compose():
    sps = _structural_predicates(_dump_ir(_multi_perm_input))
    kinds = [sp["kind"] for sp in sps]
    assert "is_permutation" in kinds
    assert "fixed_point_count" in kinds


# ── Compatibility: no preconditions ─────────────────────────────────────


@zk_circuit
def _plain_perm(p: NDArray[Integer, 3]):
    pass


def test_plain_circuit_has_no_permutation_atoms():
    sps = _structural_predicates(_dump_ir(_plain_perm))
    assert sps == []
