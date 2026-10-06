"""Executable thesis-validation rows for preconditioned tensor programs."""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest

from zinnia import *
from zinnia.api.zk_circuit import ZKCircuit
from zinnia.debug.exception.base import ZinniaException
from zinnia.spec.predicates import is_sorted


@contextmanager
def _scoped_env(**updates: str | None):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _compile(source: str, name: str):
    return ZKCircuit.from_source(name, source).compile()


def _prove(source: str, name: str, *args):
    return ZKCircuit.from_source(name, source).prove(*args, backend="mock")


def test_row1_bounded_runtime_allocation_positive():
    source = """
@requires(lambda n: 0 <= n <= 6)
def row1_runtime_allocation(n: int):
    arr = np.zeros((2, n), dtype=int)
    assert arr.sum() == 0
"""

    proof = _prove(source, "row1_runtime_allocation", 4)
    assert proof.proof_bytes_hex == "mock_satisfied"


def test_row1_bounded_runtime_allocation_no_bound_rejects():
    source = """
def row1_runtime_allocation_unbounded(n: int):
    arr = np.zeros((2, n), dtype=int)
    _zinnia_result = arr.sum()
"""

    with pytest.raises(Exception, match="shape element at axis 1"):
        _compile(source, "row1_runtime_allocation_unbounded")


def test_row3_domain_safe_numeric_transform_positive():
    source = """
@requires(lambda x: x >= 0)
def row3_domain_safe_numeric_transform(x: int):
    y = np.sqrt(x)
    assert y * y == x
"""

    proof = _prove(source, "row3_domain_safe_numeric_transform", 9)
    assert proof.proof_bytes_hex == "mock_satisfied"


def test_row3_domain_safe_numeric_transform_missing_precondition_rejects_in_strict_mode():
    source = """
def row3_domain_safe_numeric_transform_unbounded(x: int):
    y = np.sqrt(x)
    _zinnia_result = y
"""

    with _scoped_env(ZINNIA_OP_REQUIRES_STRICT="1"):
        with pytest.raises(Exception, match="sqrt|requires|domain|non-negative|>= 0"):
            _compile(source, "row3_domain_safe_numeric_transform_unbounded")


def test_row4_sorted_aggregate_strategy_positive():
    @zk_circuit
    @requires(lambda x: is_sorted(x))
    def row4_sorted_max(x: NDArray[Integer, 4]):
        dyn = np.promote_to_dynamic(x)
        assert dyn.max() == 9

    proof = ZKCircuit.from_method(row4_sorted_max).prove([1, 4, 7, 9], backend="mock")
    assert proof.proof_bytes_hex == "mock_satisfied"


def test_row4_sorted_aggregate_strategy_without_sorted_fact_still_compiles():
    @zk_circuit
    def row4_sorted_max_generic(x: NDArray[Integer, 4]):
        dyn = np.promote_to_dynamic(x)
        assert dyn.max() == 9

    proof = ZKCircuit.from_method(row4_sorted_max_generic).prove([1, 4, 7, 9], backend="mock")
    assert proof.proof_bytes_hex == "mock_satisfied"


def test_row5_bounded_dynamic_index_lookup_positive():
    source = """
@requires(lambda i: 0 <= i < 4)
def row5_bounded_dynamic_index_lookup(i: int):
    arr = np.asarray([10, 20, 30, 40])
    assert arr[i] >= 10
"""

    proof = _prove(source, "row5_bounded_dynamic_index_lookup", 2)
    assert proof.proof_bytes_hex == "mock_satisfied"


def test_row5_bounded_dynamic_index_lookup_missing_bound_rejects_in_strict_mode():
    source = """
def row5_bounded_dynamic_index_lookup_unbounded(i: int):
    arr = np.asarray([10, 20, 30, 40])
    _zinnia_result = arr[i]
    """

    with _scoped_env(ZINNIA_OP_REQUIRES_STRICT="1"):
        with pytest.raises(Exception, match="requires precondition not provable|static_array_read|strict mode"):
            _compile(source, "row5_bounded_dynamic_index_lookup_unbounded")


def test_row6_shape_preserving_pipeline_operator_support_enough():
    @zk_circuit
    def row6_shape_pipeline():
        a = np.promote_to_dynamic(np.asarray([1, 2, 3]))
        b = np.promote_to_dynamic(np.asarray([4, 5, 6]))
        joined = np.concatenate([a, b])
        parts = np.split(joined, 2)
        stacked = np.stack(parts)
        assert stacked.sum() == 21
        assert stacked[1, 2] == 6

    assert row6_shape_pipeline()


def test_row7_static_common_operator_row():
    @zk_circuit
    def row7_static_common_operator_row():
        x = np.asarray([[3, 1, 2], [9, 7, 7]])
        sorted_x = np.sort(x, axis=-1)
        order = np.argsort(x, axis=1)
        taken = np.take(x, [2, 0], axis=1)
        filled = np.full((2, 2), 4, dtype=int)
        filled_like = np.full_like(filled, 3)
        coords = np.argwhere(np.asarray([[0, 2, 3], [0, 0, 4]]))
        rows, cols = np.nonzero(np.asarray([[0, 2, 3], [0, 0, 4]]))
        assert sorted_x[0, 0] == 1
        assert order[0, 0] == 1
        assert taken[0, 0] == 2
        assert filled.sum() == 16
        assert filled_like.sum() == 12
        assert coords.sum() == 6
        assert rows.sum() == 1
        assert cols.sum() == 5

    assert row7_static_common_operator_row()


def test_row7_dynamic_argwhere_boundary_rejects():
    source = """
def row7_dynamic_argwhere_boundary():
    dyn = np.promote_to_dynamic(np.asarray([0, 5, 0, 7]))
    coords = np.argwhere(dyn)
    _zinnia_result = coords.sum()
"""

    with pytest.raises(Exception, match="DynamicNDArray input is deferred"):
        _compile(source, "row7_dynamic_argwhere_boundary")


def test_row8_explicit_linalg_witness_helper_row():
    @zk_circuit
    def row8_explicit_linalg_witness_helper_row():
        a = np.asarray([[1, 0], [0, 1]])
        inv = np.asarray([[1, 0], [0, 1]])
        solved = np.asarray([3, 4])
        b = np.asarray([3, 4])
        out_inv = np.linalg.verify_inverse(a, inv)
        out_solve = np.linalg.verify_solve(a, solved, b)
        assert out_inv[1, 1] == 1
        assert out_solve[0] == 3

    assert row8_explicit_linalg_witness_helper_row()


def test_row8_standard_computed_linalg_rejects_with_helper_guidance():
    source = """
def row8_standard_computed_linalg():
    a = np.asarray([[1, 0], [0, 1]])
    y = np.linalg.inv(a)
    _zinnia_result = y[0, 0]
"""

    with pytest.raises(Exception, match="verify_inverse/verify_solve/verify_cholesky"):
        _compile(source, "row8_standard_computed_linalg")


def test_row9_dynamic_family_boundary_sort_rejects():
    source = """
def row9_dynamic_family_boundary_sort():
    dyn = np.promote_to_dynamic(np.asarray([3, 1, 2]))
    out = np.sort(dyn)
    _zinnia_result = out[0]
"""

    with pytest.raises(Exception, match="dynamic sort/argsort is not yet implemented"):
        _compile(source, "row9_dynamic_family_boundary_sort")


def test_row9_dynamic_family_boundary_complex_stats_rejects():
    source = """
def row9_dynamic_family_boundary_complex_stats():
    a = np.asarray([1 + 0j, 2 + 0j])
    out = np.var(a)
    _zinnia_result = out.real
"""

    with pytest.raises(Exception, match="Complex StaticArray input is deferred"):
        _compile(source, "row9_dynamic_family_boundary_complex_stats")


def test_row10_dynamic_boolean_mask_indexing_matches_numpy():
    """Row 10 used to be a rejection boundary: boolean-mask indexing of a
    bounded-dynamic array panicked with an index out of bounds, and the row
    asserted that panic. The compiler now supports it, so the row checks
    that NumPy's result is accepted and a wrong one rejected, for a constant
    mask (the original program) and for a private one."""
    import numpy as np

    constant_mask = """
def row10_constant_mask(expected: int):
    a = np.promote_to_dynamic(np.asarray([10, 20, 30, 40, 50]))
    mask = np.asarray([True, False, True, False, True])
    out = a[mask]
    assert out.sum() == expected
"""
    values = np.array([10, 20, 30, 40, 50])
    truth = int(values[np.array([True, False, True, False, True])].sum())
    assert _prove(constant_mask, "row10_constant_mask", truth).proof_bytes_hex == "mock_satisfied"
    assert _prove(constant_mask, "row10_constant_mask", truth + 1).proof_bytes_hex != "mock_satisfied"

    private_mask = """
def row10_private_mask(x: Integer[Array, "5"], m: Integer[Array, "5"], expected: int):
    a = np.promote_to_dynamic(x)
    out = a[m > 0]
    assert out.sum() == expected
"""
    mask = np.array([1, 0, 1, 1, 0])
    truth = int(values[mask > 0].sum())
    circuit = ZKCircuit.from_source("row10_private_mask", private_mask)
    assert circuit.prove(values, mask, truth, backend="mock").proof_bytes_hex == "mock_satisfied"
    assert circuit.prove(values, mask, truth + 1, backend="mock").proof_bytes_hex != "mock_satisfied"
