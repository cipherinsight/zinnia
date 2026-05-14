"""Lang smoke test for `compiler.op-fact-group-8a-matmul-zero-short-circuit`.

Confirms `np.matmul(np.zeros((m, k)), arr)` / `arr @ np.zeros((k, n))` /
the 1D@1D variant all compile cleanly with the new Phase F strategy-
selection wiring on `matmul`. The Rust unit tests in
`optim/tests/strategy_tests.rs` cover the dispatch mechanics (precondition
matching, MulI/AddI statement counts); this lang test is the end-to-end
check that programs using matmul over a provably-all-zeros operand compile
without regression.

The Group 4a `zeros_content` ensure plants the `forall_eq_const(_, 0)` fact
that the matmul strategy queries.
"""
import numpy as np

from zinnia import zk_circuit
from zinnia.api.zk_circuit import ZKCircuit


def test_matmul_lhs_zeros_2d_2d_compiles():
    @zk_circuit
    def foo():
        b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        _zinnia_result = np.matmul(np.zeros((3, 4), dtype=int), b)

    _ = ZKCircuit.from_method(foo).compile()


def test_matmul_rhs_zeros_2d_2d_compiles():
    @zk_circuit
    def foo():
        a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        _zinnia_result = a @ np.zeros((4, 2), dtype=int)

    _ = ZKCircuit.from_method(foo).compile()


def test_matmul_zeros_1d_1d_compiles():
    @zk_circuit
    def foo():
        a = np.array([1, 2, 3, 4])
        _zinnia_result = np.matmul(np.zeros(4, dtype=int), a)

    _ = ZKCircuit.from_method(foo).compile()
