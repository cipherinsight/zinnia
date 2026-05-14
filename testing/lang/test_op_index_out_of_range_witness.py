"""Group 5a lang smoke tests for `discharge_index_in_range`.

The `good` function pairs a satisfying `@requires(0 <= i < 10)` with a
`arr[i]` read; the precondition discharges Proved and compilation
proceeds with no witness check. The `bad` function omits the annotation;
under the lenient default the discharge is Unknown and a witness
assertion is emitted, so compile succeeds and the prover would refuse to
forge a witness for an out-of-range input.
"""
import os
import pytest

from zinnia import zk_circuit, requires, NDArray, Integer
from zinnia.api.zk_circuit import ZKCircuit


def test_index_in_range_good_compiles_with_satisfying_requires():
    """`@requires(0 <= i < 10)` discharges the index bound Proved on the
    dyn-ndarray subscript chokepoint.
    """
    @zk_circuit
    @requires(lambda arr, i: i >= 0)
    @requires(lambda arr, i: i < 10)
    def good(arr: NDArray[Integer, 10], i: int):
        y = arr[i]
        _zinnia_result = y

    _ = ZKCircuit.from_method(good).compile()


def test_index_in_range_bad_compiles_in_lenient_default():
    """Without an annotation, lenient mode emits a witness assertion and
    compilation still succeeds.
    """
    @zk_circuit
    def bad(arr: NDArray[Integer, 10], i: int):
        y = arr[i]
        _zinnia_result = y

    prev = os.environ.pop("ZINNIA_OP_REQUIRES_STRICT", None)
    try:
        _ = ZKCircuit.from_method(bad).compile()
    finally:
        if prev is not None:
            os.environ["ZINNIA_OP_REQUIRES_STRICT"] = prev
