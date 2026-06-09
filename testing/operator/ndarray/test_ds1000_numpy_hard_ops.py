import pytest

from zinnia import *
from zinnia.debug.exception import ZinniaException


def test_percentile_correlate_corrcoef_static_subsets_compile():
    @zk_circuit
    @requires(lambda p: 0 <= p <= 100)
    def foo(a: Float[Array, "20"], p: Integer, x: Float[Array, "5"], y: Float[Array, "5"]):
        pct = np.percentile(a, p)
        corr = np.correlate(x, np.hstack((y[1:], y)), mode="valid")
        coef = np.corrcoef(x, y)[0][1]
        return pct, corr, coef

    compiled = ZKCircuit.from_method(foo).compile()
    assert compiled is not None


def test_einsum_diagonal_and_tensordot_static_subsets_compile():
    @zk_circuit
    def foo(a: Integer[Array, "4 4"], A: Float[Array, "5 6 3"], B: Float[Array, "3 3"]):
        diag = np.einsum("ii->i", a)
        contracted = np.tensordot(A, B, axes=((2), (0)))
        return diag, contracted

    compiled = ZKCircuit.from_method(foo).compile()
    assert compiled is not None


def test_unique_returns_bounded_dynamic_with_inverse():
    """np.unique compaction lands in a DynamicNDArray with total_bound ==
    input.size; return_inverse produces a same-shape StaticArray of
    positions in the unique output (first-occurrence order)."""
    source = """
def foo(a: Float[Array, "5"]):
    vals, idx = np.unique(a, return_inverse=True)
    return vals
"""
    assert ZKCircuit.from_source("foo", source).compile() is not None


def test_unique_without_return_inverse_compiles():
    source = """
def foo(a: Integer[Array, "8"]):
    return np.unique(a)
"""
    assert ZKCircuit.from_source("foo", source).compile() is not None


def test_unique_axis_kwarg_rejected():
    source = """
def foo(a: Integer[Array, "2 3"]):
    return np.unique(a, axis=0)
"""
    with pytest.raises(ZinniaException, match="np.unique: kwarg `axis`"):
        ZKCircuit.from_source("foo", source).compile()


def test_unique_return_index_kwarg_rejected():
    source = """
def foo(a: Integer[Array, "4"]):
    return np.unique(a, return_index=True)
"""
    with pytest.raises(ZinniaException, match="np.unique: kwarg `return_index`"):
        ZKCircuit.from_source("foo", source).compile()

