"""Lang smoke test for `compiler.op-fact-group-6-shape-preserving-relay`.

Shape-preserving ops (transpose, reshape, squeeze, expand_dims,
broadcast_to, flatten, swapaxes, moveaxis) relay `forall_eq_const` from
their input to their output. Composed with the `sum` strategy from
Group 3d, programs like `np.sum(np.transpose(np.zeros((3, 4))))` compile
to a constant.

Compile-and-no-throw is the exit criterion — mirrors
`test_op_strategy_sum_on_tile_of_zeros.py`.
"""
import numpy as np

from zinnia import zk_circuit
from zinnia.api.zk_circuit import ZKCircuit


def test_sum_on_transpose_of_zeros_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.sum(np.transpose(np.zeros((3, 4))))

    _ = ZKCircuit.from_method(foo).compile()


def test_sum_on_reshape_of_ones_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.sum(np.reshape(np.ones(12), (3, 4)))

    _ = ZKCircuit.from_method(foo).compile()


def test_sum_on_squeeze_of_zeros_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.sum(np.squeeze(np.zeros((1, 3, 1))))

    _ = ZKCircuit.from_method(foo).compile()


def test_sum_on_expand_dims_of_ones_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.sum(np.expand_dims(np.ones(4), 0))

    _ = ZKCircuit.from_method(foo).compile()


def test_sum_on_broadcast_to_of_ones_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.sum(np.broadcast_to(np.ones(3), (2, 3)))

    _ = ZKCircuit.from_method(foo).compile()


def test_sum_on_swapaxes_of_zeros_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.sum(np.swapaxes(np.zeros((2, 3)), 0, 1))

    _ = ZKCircuit.from_method(foo).compile()
