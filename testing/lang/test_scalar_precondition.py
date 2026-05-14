"""Tests for the scalar / arithmetic / logical precondition surface.

Each test verifies that:
1. The Python AST extractor builds an `ASTScalarRequires` spec with the
   correct `ContractTerm`-shaped JSON.
2. The Rust ir-gen emits the corresponding `IR::ScalarPrecondition` atom
   plus the witness-time enforcement IR chain ending in `IR::Assert`.
"""

import pytest

from zinnia import zk_circuit, requires, NDArray, Integer, ZinniaException
from zinnia.api.zk_circuit import ZKCircuit


def _dump_ir(decorated):
    return ZKCircuit.from_method(decorated).compile().get_ir_stmts()


def _scalar_atoms(stmts):
    return [
        s["ir_instance"]["ir_data"]
        for s in stmts
        if s.get("ir_instance", {}).get("__class__") == "ScalarPreconditionIR"
    ]


def _assert_count(stmts):
    return sum(
        1 for s in stmts
        if s.get("ir_instance", {}).get("__class__") == "AssertIR"
    )


# ── 1. Simple comparison: k >= 0 ────────────────────────────────────────


@zk_circuit
@requires(lambda k: k >= 0)
def _simple_cmp(k: int):
    pass


def test_simple_comparison_emits_scalar_atom():
    atoms = _scalar_atoms(_dump_ir(_simple_cmp))
    assert len(atoms) == 1
    import json
    term = json.loads(atoms[0]["term_json"])
    assert term["Cmp"]["op"] == "Ge"
    assert term["Cmp"]["lhs"] == {"Var": {"Input": "k"}}
    assert term["Cmp"]["rhs"] == {"LitInt": 0}


def test_simple_comparison_witness_chain():
    stmts = _dump_ir(_simple_cmp)
    classes = [s.get("ir_instance", {}).get("__class__") for s in stmts]
    assert "GreaterThanOrEqualIIR" in classes
    assert _assert_count(stmts) >= 1


# ── 2. Chained comparison: 0 <= k <= 1024 ───────────────────────────────


@zk_circuit
@requires(lambda k: 0 <= k <= 1024)
def _chained_cmp(k: int):
    pass


def test_chained_comparison_lowers_to_and():
    atoms = _scalar_atoms(_dump_ir(_chained_cmp))
    assert len(atoms) == 1
    import json
    term = json.loads(atoms[0]["term_json"])
    assert "BoolComb" in term
    assert term["BoolComb"]["op"] == "And"
    assert len(term["BoolComb"]["operands"]) == 2


def test_chained_comparison_witness_emits_two_comparisons_and_logical_and():
    stmts = _dump_ir(_chained_cmp)
    classes = [s.get("ir_instance", {}).get("__class__") for s in stmts]
    assert classes.count("LessThanOrEqualIIR") >= 2
    assert "LogicalAndIR" in classes
    assert _assert_count(stmts) >= 1


# ── 3. Arithmetic: a + b == c ───────────────────────────────────────────


@zk_circuit
@requires(lambda a, b, c: a + b == c)
def _arith(a: int, b: int, c: int):
    pass


def test_arithmetic_inside_comparison():
    atoms = _scalar_atoms(_dump_ir(_arith))
    assert len(atoms) == 1
    import json
    term = json.loads(atoms[0]["term_json"])
    assert "Cmp" in term
    assert term["Cmp"]["op"] == "Eq"
    assert "Arith" in term["Cmp"]["lhs"]
    assert term["Cmp"]["lhs"]["Arith"]["op"] == "Add"


def test_arithmetic_witness_emits_addi_and_eqi():
    stmts = _dump_ir(_arith)
    classes = [s.get("ir_instance", {}).get("__class__") for s in stmts]
    assert "AddIIR" in classes
    assert "EqualIIR" in classes


# ── 4. Modular: k % 2 == 0 (parity) ─────────────────────────────────────


@zk_circuit
@requires(lambda k: k % 2 == 0)
def _modular(k: int):
    pass


def test_modular_arithmetic_lowering():
    atoms = _scalar_atoms(_dump_ir(_modular))
    assert len(atoms) == 1


def test_modular_witness_emits_modi_and_eqi():
    stmts = _dump_ir(_modular)
    classes = [s.get("ir_instance", {}).get("__class__") for s in stmts]
    assert "ModIIR" in classes
    assert "EqualIIR" in classes


# ── 5. Logical AND: a > 0 and b > 0 ─────────────────────────────────────


@zk_circuit
@requires(lambda a, b: a > 0 and b > 0)
def _logical_and(a: int, b: int):
    pass


def test_logical_and_lowering():
    atoms = _scalar_atoms(_dump_ir(_logical_and))
    assert len(atoms) == 1
    import json
    term = json.loads(atoms[0]["term_json"])
    assert term["BoolComb"]["op"] == "And"


def test_logical_and_witness():
    stmts = _dump_ir(_logical_and)
    classes = [s.get("ir_instance", {}).get("__class__") for s in stmts]
    assert classes.count("GreaterThanIIR") >= 2
    assert "LogicalAndIR" in classes


# ── 6. Logical OR ───────────────────────────────────────────────────────


@zk_circuit
@requires(lambda a, b: a > 0 or b > 0)
def _logical_or(a: int, b: int):
    pass


def test_logical_or_lowering():
    atoms = _scalar_atoms(_dump_ir(_logical_or))
    import json
    term = json.loads(atoms[0]["term_json"])
    assert term["BoolComb"]["op"] == "Or"


def test_logical_or_witness():
    stmts = _dump_ir(_logical_or)
    classes = [s.get("ir_instance", {}).get("__class__") for s in stmts]
    assert "LogicalOrIR" in classes


# ── 7. Negation: not (k == 0) ───────────────────────────────────────────


@zk_circuit
@requires(lambda k: not (k == 0))
def _negation(k: int):
    pass


def test_negation_lowering():
    atoms = _scalar_atoms(_dump_ir(_negation))
    import json
    term = json.loads(atoms[0]["term_json"])
    assert "Not" in term


def test_negation_witness():
    stmts = _dump_ir(_negation)
    classes = [s.get("ir_instance", {}).get("__class__") for s in stmts]
    assert "LogicalNotIR" in classes
    assert "EqualIIR" in classes


# ── 8. Range membership: k in range(10) ─────────────────────────────────


@zk_circuit
@requires(lambda k: k in range(10))
def _range_membership(k: int):
    pass


def test_range_membership_desugars_to_and_of_bounds():
    atoms = _scalar_atoms(_dump_ir(_range_membership))
    assert len(atoms) == 1
    import json
    term = json.loads(atoms[0]["term_json"])
    assert term["BoolComb"]["op"] == "And"
    # Should have at least two comparisons: lo <= k and k < hi.
    assert len(term["BoolComb"]["operands"]) >= 2


def test_range_membership_witness():
    stmts = _dump_ir(_range_membership)
    classes = [s.get("ir_instance", {}).get("__class__") for s in stmts]
    # `0 <= k` → LteI; `k < 10` → LtI.
    assert "LessThanOrEqualIIR" in classes
    assert "LessThanIIR" in classes


# ── 9. Set membership: k in {1, 3, 5} ───────────────────────────────────


@zk_circuit
@requires(lambda k: k in {1, 3, 5})
def _set_membership(k: int):
    pass


def test_set_membership_desugars_to_or_of_equalities():
    atoms = _scalar_atoms(_dump_ir(_set_membership))
    assert len(atoms) == 1
    import json
    term = json.loads(atoms[0]["term_json"])
    assert term["BoolComb"]["op"] == "Or"
    assert len(term["BoolComb"]["operands"]) == 3


def test_set_membership_witness():
    stmts = _dump_ir(_set_membership)
    classes = [s.get("ir_instance", {}).get("__class__") for s in stmts]
    assert classes.count("EqualIIR") >= 3
    assert "LogicalOrIR" in classes


# ── 10. not in membership ───────────────────────────────────────────────


@zk_circuit
@requires(lambda k: k not in {0})
def _not_in_membership(k: int):
    pass


def test_not_in_membership_wraps_in_not():
    atoms = _scalar_atoms(_dump_ir(_not_in_membership))
    import json
    term = json.loads(atoms[0]["term_json"])
    assert "Not" in term


# ── 11. Boolean literal True (trivially satisfied) ──────────────────────


@zk_circuit
@requires(lambda: True)
def _bool_lit_true():
    pass


def test_bool_literal_true_compiles():
    # Just compiling without error is the success condition. The
    # witness chain reduces to `assert True` which the optimizer
    # eliminates as always-satisfied.
    _ = _dump_ir(_bool_lit_true)


# ── 12. Mixing structural + scalar via separate @requires clauses ───────


@zk_circuit
@requires(lambda x, k: __import__('zinnia').spec.predicates.nnz(x) == k)
def _placeholder(x: NDArray[Integer, 4], k: int):
    """Just a sanity check that the existing structural-predicate path
    is unchanged by the scalar-precondition extension. This test is a
    smoke test for the layered behaviour: structural goes to
    `requires`, scalar goes to `scalar_requires`."""
    pass


# (No assert — we just need this to compile without error.)


# ── Error: unsupported operator ─────────────────────────────────────────


def test_bitwise_operator_in_precondition_rejected():
    @zk_circuit
    @requires(lambda a, b: (a & b) == 0)
    def bad(a: int, b: int):
        pass

    with pytest.raises(ZinniaException, match=r"(?i)bitwise|bv|unsupported"):
        ZKCircuit.from_method(bad).compile()


def test_set_membership_variable_member_rejected():
    @zk_circuit
    @requires(lambda k, x: k in {x, 1})
    def bad(k: int, x: int):
        pass

    with pytest.raises(ZinniaException, match=r"(?i)literal|variable"):
        ZKCircuit.from_method(bad).compile()


# ── Load-bearing: pure scalar bound unlocks np.zeros(k) via SMT ─────────


import numpy as np


@zk_circuit
@requires(lambda k: 0 <= k <= 16)
def _scalar_bound_unlocks_zeros(k: int):
    out = np.zeros(k, dtype=Integer)
    _zinnia_result = out


def test_pure_scalar_bound_unlocks_dyn_ndarray_construction():
    """The bigger pay-off of scalar preconditions: a circuit with NO
    structural inputs but a pure scalar bound on `k` (`0 <= k <= 16`)
    derives the same bound as a `nnz`-driven precondition. The
    ShapeAxis chokepoint admits `np.zeros(k, ...)` as a dyn-ndarray.

    With SMT enabled (default), this should compile end-to-end. With
    SMT disabled (ZINNIA_SMT_ENABLE=0), the chokepoint rejects.
    """
    import os
    prev = os.environ.get("ZINNIA_SMT_ENABLE")
    os.environ["ZINNIA_SMT_ENABLE"] = "1"
    try:
        # Just compiling without raising is the success condition.
        program = ZKCircuit.from_method(_scalar_bound_unlocks_zeros).compile()
        stmts = program.get_ir_stmts()
        # IR must contain at least the scalar-precondition atom + assert.
        classes = [s.get("ir_instance", {}).get("__class__") for s in stmts]
        assert "ScalarPreconditionIR" in classes
        assert "AssertIR" in classes
    finally:
        if prev is None:
            os.environ.pop("ZINNIA_SMT_ENABLE", None)
        else:
            os.environ["ZINNIA_SMT_ENABLE"] = prev


def test_pure_scalar_bound_rejects_under_smt_off():
    import os
    prev = os.environ.get("ZINNIA_SMT_ENABLE")
    os.environ["ZINNIA_SMT_ENABLE"] = "0"
    try:
        with pytest.raises(ZinniaException, match=r"(?i)shape|static|constant"):
            ZKCircuit.from_method(_scalar_bound_unlocks_zeros).compile()
    finally:
        if prev is None:
            os.environ.pop("ZINNIA_SMT_ENABLE", None)
        else:
            os.environ["ZINNIA_SMT_ENABLE"] = prev
