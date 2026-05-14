"""Lang smoke test for `compiler.op-fact-group-4c-tile-repeat-content-relay`.

`np.sum(np.tile(np.zeros(3), 4))` and `np.sum(np.repeat(np.zeros(3), 4))`
should compile end-to-end: the content fact `forall_eq_const(in, 0)` is
relayed from the inner constructor through tile/repeat to their outputs,
and Group 3d's `sum` strategy specializes to constant 0.

Mirrors `test_op_strategy_sum_on_zeros.py`: compile-and-no-throw is the
exit criterion. Static-array constructor paths without `value_id` won't
relay, but the dyn / composite paths exercise the wired relay.
"""
import numpy as np

from zinnia import zk_circuit
from zinnia.api.zk_circuit import ZKCircuit


def test_sum_on_tile_of_zeros_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.sum(np.tile(np.zeros(3), 4))

    _ = ZKCircuit.from_method(foo).compile()


def test_sum_on_tile_of_ones_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.sum(np.tile(np.ones(3), 4))

    _ = ZKCircuit.from_method(foo).compile()


def test_sum_on_repeat_of_zeros_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.sum(np.repeat(np.zeros(3), 4))

    _ = ZKCircuit.from_method(foo).compile()


def test_sum_on_repeat_of_ones_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.sum(np.repeat(np.ones(3), 4))

    _ = ZKCircuit.from_method(foo).compile()
