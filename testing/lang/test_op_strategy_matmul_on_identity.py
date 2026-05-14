"""Lang smoke test for `compiler.op-fact-group-8b-matmul-identity-short-circuit`.

Confirms `np.matmul(np.identity(N), arr)` / `arr @ np.identity(N)` /
the 2D@1D variant all compile cleanly with the new Phase F strategy-
selection wiring on `matmul`. The Rust unit tests in
`optim/tests/strategy_tests.rs` cover the dispatch mechanics (precondition
matching, MulI/AddI statement counts); this lang test is the end-to-end
check that programs using matmul over a `np.identity(N)` operand compile
without regression.

The Group 8b `identity_content` ensure plants the `is_identity(_)` fact
that the matmul strategy queries.
"""
import numpy as np

from zinnia import zk_circuit
from zinnia.api.zk_circuit import ZKCircuit


def test_matmul_lhs_identity_2d_2d_compiles():
    @zk_circuit
    def foo():
        b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        _zinnia_result = np.matmul(np.identity(4), b)

    _ = ZKCircuit.from_method(foo).compile()


def test_matmul_rhs_identity_2d_2d_compiles():
    @zk_circuit
    def foo():
        a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        _zinnia_result = a @ np.identity(4)

    _ = ZKCircuit.from_method(foo).compile()


def test_matmul_lhs_identity_2d_1d_compiles():
    @zk_circuit
    def foo():
        v = np.array([1, 2, 3, 4])
        _zinnia_result = np.matmul(np.identity(4), v)

    _ = ZKCircuit.from_method(foo).compile()
