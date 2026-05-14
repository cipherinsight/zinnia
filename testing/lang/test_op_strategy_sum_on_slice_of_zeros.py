"""Lang smoke test for `compiler.op-fact-group-5b-slice-content-relay`.

`np.sum(np.zeros(N)[a:b])` and similar slice variants should compile
end-to-end: the content fact `forall_eq_const(in, 0)` is relayed from the
inner constructor through slice helpers to the sliced output, and Group
3d's `sum` strategy specializes to constant 0.

Mirrors `test_op_strategy_sum_on_tile_of_zeros.py`: compile-and-no-throw
is the exit criterion. Slice with step is deferred (Tier-2).
"""
import numpy as np

from zinnia import zk_circuit
from zinnia.api.zk_circuit import ZKCircuit


def test_sum_on_slice_of_zeros_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.sum(np.zeros(10)[2:5])

    _ = ZKCircuit.from_method(foo).compile()


def test_sum_on_slice_of_ones_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.sum(np.ones(10)[2:5])

    _ = ZKCircuit.from_method(foo).compile()


def test_sum_on_open_start_slice_of_zeros_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.sum(np.zeros(10)[:5])

    _ = ZKCircuit.from_method(foo).compile()


def test_sum_on_open_stop_slice_of_zeros_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.sum(np.zeros(10)[3:])

    _ = ZKCircuit.from_method(foo).compile()


def test_sum_on_full_slice_of_zeros_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.sum(np.zeros(10)[:])

    _ = ZKCircuit.from_method(foo).compile()
