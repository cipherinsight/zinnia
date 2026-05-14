"""Lang smoke test for `compiler.op-fact-group-5c-gather-content-relay`.

Fancy indexing (`arr[[i, j, k]]`) and `np.take(arr, [i, j, k])` should
propagate `forall_eq_const(in, k)` from a uniform-constant source through
the gather op to the output. Group 3d's `sum` strategy then specializes
the surrounding `np.sum` to the closed form.

Mirrors `test_op_strategy_sum_on_slice_of_zeros.py`: compile-and-no-throw
is the exit criterion.
"""
import numpy as np

from zinnia import zk_circuit
from zinnia.api.zk_circuit import ZKCircuit


def test_sum_on_take_of_zeros_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.sum(np.take(np.zeros(10), [2, 5, 8]))

    _ = ZKCircuit.from_method(foo).compile()


def test_sum_on_take_of_ones_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.sum(np.take(np.ones(10), [2, 5, 8]))

    _ = ZKCircuit.from_method(foo).compile()


def test_sum_on_fancy_index_of_zeros_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.sum(np.zeros(10)[[2, 5, 8]])

    _ = ZKCircuit.from_method(foo).compile()


def test_sum_on_fancy_index_of_ones_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.sum(np.ones(10)[[2, 5, 8]])

    _ = ZKCircuit.from_method(foo).compile()
