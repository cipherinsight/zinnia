"""Lang smoke test for `compiler.op-fact-group-3b-reductions-relay-interval`.

Confirms `np.sum`, `np.max`, `np.min` compile cleanly on a static array
whose elements have a `forall(arr, in_range(lo, hi))` precondition. The
relay helper inspects per-element fact-derived bounds and deposits the
output interval on the reduction's value bucket.

Rust unit tests (`relay_sum_static_array_yields_n_times_input_interval`
et al.) cover the fact-deposition mechanics; this lang test is the
functional end-to-end check that the ops compile and consume their
result without regression.
"""
import numpy as np

from zinnia import zk_circuit, requires, NDArray, Integer
from zinnia.spec.predicates import forall, in_range
from zinnia.api.zk_circuit import ZKCircuit


def test_np_sum_on_bounded_static_array_compiles():
    """Per-element bound `[0, 5]` over a 4-element array. `np.sum`'s
    relay deposits Output ∈ [0, 20] on the reduction's value bucket.
    """
    @zk_circuit
    @requires(lambda arr: forall(arr, in_range(0, 5)))
    def foo(arr: NDArray[Integer, 4]):
        _zinnia_result = np.sum(arr)

    _ = ZKCircuit.from_method(foo).compile()


def test_np_max_on_bounded_static_array_compiles():
    """Per-element bound `[-3, 7]` over a 4-element array. `np.max`'s
    relay deposits Output ∈ [-3, 7] (multiplier 1).
    """
    @zk_circuit
    @requires(lambda arr: forall(arr, in_range(-3, 7)))
    def foo(arr: NDArray[Integer, 4]):
        _zinnia_result = np.max(arr)

    _ = ZKCircuit.from_method(foo).compile()


def test_np_min_on_bounded_static_array_compiles():
    """Per-element bound `[2, 9]` over a 3-element array. `np.min`'s
    relay deposits Output ∈ [2, 9] (multiplier 1).
    """
    @zk_circuit
    @requires(lambda arr: forall(arr, in_range(2, 9)))
    def foo(arr: NDArray[Integer, 3]):
        _zinnia_result = np.min(arr)

    _ = ZKCircuit.from_method(foo).compile()
