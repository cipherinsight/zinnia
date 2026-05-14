"""Lang smoke test for `compiler.op-fact-group-4a-fill-constructors-forall-eq-const`.

Confirms `np.zeros` / `np.ones` / `np.zeros_like` / `np.ones_like` compile
cleanly with the new `forall_eq_const(out, k)` content fact wired into
their constructors. The Rust unit tests cover the fact-deposition
mechanics; this lang test is the end-to-end check that the ops compile
without regression.
"""
import numpy as np

from zinnia import zk_circuit
from zinnia.api.zk_circuit import ZKCircuit


def test_np_zeros_static_compiles():
    @zk_circuit
    def foo():
        arr = np.zeros(8)
        _zinnia_result = arr[0]

    _ = ZKCircuit.from_method(foo).compile()


def test_np_ones_static_compiles():
    @zk_circuit
    def foo():
        arr = np.ones(8)
        _zinnia_result = arr[0]

    _ = ZKCircuit.from_method(foo).compile()


def test_np_zeros_like_compiles():
    @zk_circuit
    def foo():
        src = np.asarray([1, 2, 3, 4])
        arr = np.zeros_like(src)
        _zinnia_result = arr[0]

    _ = ZKCircuit.from_method(foo).compile()


def test_np_ones_like_compiles():
    @zk_circuit
    def foo():
        src = np.asarray([1, 2, 3, 4])
        arr = np.ones_like(src)
        _zinnia_result = arr[0]

    _ = ZKCircuit.from_method(foo).compile()
