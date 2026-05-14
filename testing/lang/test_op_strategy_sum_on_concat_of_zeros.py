"""Lang smoke test for `compiler.op-fact-group-7-concat-stack-relay`.

`np.sum(np.concatenate([...]))` and the stack family (`vstack`, `hstack`,
`dstack`, `column_stack`, `stack`) should compile end-to-end when every
input is provably `forall_eq_const(k)`. The multi-input relay forwards
the content fact onto the merged output, and Group 3d's `sum` strategy
specializes to a compile-time constant.

Mirrors `test_op_strategy_sum_on_tile_of_zeros.py`: compile-and-no-throw
is the exit criterion. Static-array constructor paths without `value_id`
won't relay, but the dyn / composite paths exercise the wired relay.
"""
import numpy as np

from zinnia import zk_circuit
from zinnia.api.zk_circuit import ZKCircuit


def test_sum_on_concatenate_of_zeros_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.sum(np.concatenate([np.zeros(3), np.zeros(4)]))

    _ = ZKCircuit.from_method(foo).compile()


def test_sum_on_concatenate_of_ones_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.sum(np.concatenate([np.ones(3), np.ones(4)]))

    _ = ZKCircuit.from_method(foo).compile()


def test_sum_on_stack_of_zeros_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.sum(np.stack([np.zeros(3), np.zeros(3)]))

    _ = ZKCircuit.from_method(foo).compile()


def test_sum_on_vstack_of_ones_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.sum(np.vstack([np.ones(3), np.ones(3)]))

    _ = ZKCircuit.from_method(foo).compile()


def test_sum_on_hstack_of_zeros_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.sum(np.hstack([np.zeros(3), np.zeros(4)]))

    _ = ZKCircuit.from_method(foo).compile()


def test_sum_on_dstack_of_ones_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.sum(np.dstack([np.ones(3), np.ones(3)]))

    _ = ZKCircuit.from_method(foo).compile()


def test_sum_on_column_stack_of_zeros_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.sum(np.column_stack([np.zeros(3), np.zeros(3)]))

    _ = ZKCircuit.from_method(foo).compile()
