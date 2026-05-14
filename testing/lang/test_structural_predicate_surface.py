"""Surface tests for ``@requires`` / ``@zk_circuit(requires=[...])``.

Covers the exit criteria of `compiler.structural-predicate-surface`:

  - Both decorator forms emit `IR::StructuralPredicate` atoms with the
    correct payload (kind / args / op / bound).
  - Three predicate families (nnz, is_sorted, is_permutation) round-trip
    from source through to dumped IR.
  - Predicate-marker symbols raise at runtime if called by accident.
  - Decoration-time errors fire for misspelled predicates and lambda
    parameters that aren't function parameters.
  - Compatibility: a circuit with no preconditions emits no
    `StructuralPredicateIR` atoms in its dump.
"""

import pytest

from zinnia import zk_circuit, requires, NDArray, Float, Integer
from zinnia.spec.predicates import (
    nnz, is_sorted, is_permutation, in_range,
    _PredicateMarker,
)
from zinnia.api.zk_circuit import ZKCircuit
from zinnia.debug.exception import ZinniaException


def _dump_ir(decorated):
    circuit = ZKCircuit.from_method(decorated)
    program = circuit.compile()
    return program.get_ir_stmts()


def _structural_predicates(stmts):
    return [
        s["ir_instance"]["ir_data"]
        for s in stmts
        if s.get("ir_instance", {}).get("__class__") == "StructuralPredicateIR"
    ]


# ── Stacked-form, nnz family ────────────────────────────────────────────────

@zk_circuit
@requires(lambda x, k: nnz(x) == k)
def _nnz_eq_k(x: NDArray[Float, 4], k: int):
    pass


def test_stacked_requires_nnz_eq():
    sps = _structural_predicates(_dump_ir(_nnz_eq_k))
    assert len(sps) == 1
    assert sps[0] == {"kind": "nnz", "args": ["x"], "op": "==", "bound": "k"}


# ── Block-form, is_sorted family ────────────────────────────────────────────

@zk_circuit(requires=[lambda x: is_sorted(x)])
def _block_is_sorted(x: NDArray[Integer, 8]):
    pass


def test_block_requires_is_sorted():
    sps = _structural_predicates(_dump_ir(_block_is_sorted))
    assert len(sps) == 1
    assert sps[0] == {"kind": "is_sorted", "args": ["x"], "op": None, "bound": None}


# ── Permutation family ──────────────────────────────────────────────────────

@zk_circuit
@requires(lambda p: is_permutation(p))
def _perm(p: NDArray[Integer, 8]):
    pass


def test_stacked_requires_is_permutation():
    sps = _structural_predicates(_dump_ir(_perm))
    assert len(sps) == 1
    assert sps[0] == {"kind": "is_permutation", "args": ["p"], "op": None, "bound": None}


# ── Multiple preconditions, mixed forms ─────────────────────────────────────

@zk_circuit(requires=[lambda x, k: nnz(x) == k, lambda k: in_range(k, 0, 1024)])
def _multi(x: NDArray[Float, 1024], k: int):
    pass


def test_block_multi_preconditions():
    sps = _structural_predicates(_dump_ir(_multi))
    assert len(sps) == 2
    assert sps[0] == {"kind": "nnz", "args": ["x"], "op": "==", "bound": "k"}
    assert sps[1] == {"kind": "in_range", "args": ["k", "0", "1024"],
                      "op": None, "bound": None}


@zk_circuit
@requires(lambda x: is_sorted(x))
@requires(lambda x: is_permutation(x))
def _stacked_two(x: NDArray[Integer, 10]):
    pass


def test_stacked_multi_preconditions_preserve_order():
    sps = _structural_predicates(_dump_ir(_stacked_two))
    # Python applies decorators bottom-up, so the inner decorator
    # (`is_permutation`) is closer to the function; the AST `decorator_list`
    # walks top-to-bottom, so the surface order in source is preserved.
    assert [sp["kind"] for sp in sps] == ["is_sorted", "is_permutation"]


# ── Compatibility: no preconditions ─────────────────────────────────────────

@zk_circuit
def _plain(x: NDArray[Integer, 5]):
    pass


def test_plain_circuit_emits_no_structural_predicates():
    sps = _structural_predicates(_dump_ir(_plain))
    assert sps == []


# ── Predicate markers raise on direct call ──────────────────────────────────

def test_predicate_markers_raise_on_call():
    with pytest.raises(RuntimeError, match="only valid inside"):
        nnz([1, 2, 3])

    with pytest.raises(RuntimeError, match="only valid inside"):
        is_sorted([1, 2, 3])


def test_all_predicate_names_registered_as_markers():
    """Every user-surface predicate name has a callable marker symbol.

    Synthesized IR-level kinds (e.g., `forall_lt_bound`) are excluded
    — they appear in IR dumps after AST-extractor flattening but are
    not user-facing.
    """
    from zinnia.spec.predicates import _USER_PREDICATE_NAMES
    import zinnia.spec.predicates as pred_module
    for name in _USER_PREDICATE_NAMES:
        assert isinstance(getattr(pred_module, name), _PredicateMarker)


# ── Decoration-time / compile-time error paths ──────────────────────────────

def test_misspelled_predicate_raises_at_compile_time():
    """A misspelled predicate (`nzz` instead of `nnz`) is rejected at
    compile time. The exact diagnostic depends on whether the lambda
    body looks predicate-shaped (single Compare with a Call left)
    versus general-shaped (composite scalar expression). Either way
    the error names the offending identifier.
    """
    @zk_circuit
    @requires(lambda x: nzz(x) == 5)   # noqa: F821 — intentional typo
    def bad(x: NDArray[Integer, 4]):
        pass

    with pytest.raises(ZinniaException, match=r"(?i)nzz"):
        ZKCircuit.from_method(bad).compile()


def test_lambda_param_not_in_signature_raises():
    @zk_circuit
    @requires(lambda y: nnz(y) == 5)  # y is not a parameter of the function
    def bad(x: NDArray[Integer, 4]):
        pass

    with pytest.raises(ZinniaException, match="not a parameter of"):
        ZKCircuit.from_method(bad).compile()


def test_requires_rejects_non_callable_at_decoration():
    with pytest.raises(TypeError, match="expects a single lambda"):
        @zk_circuit
        @requires("not a lambda")
        def f(x: NDArray[Integer, 4]):
            pass
