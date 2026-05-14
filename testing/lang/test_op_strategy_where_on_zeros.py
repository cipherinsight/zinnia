"""Lang smoke test for `compiler.op-fact-group-5e-where-arm-elision`.

Confirms `np.where(np.ones(N, bool), a, b)` / `np.where(np.zeros(N, bool), a, b)`
compile cleanly with the new Phase F strategy-selection wiring on np_where.
The Rust unit tests in `optim/tests/strategy_tests.rs` cover the dispatch
mechanics (precondition matching, SelectI statement counts); this lang test
is the end-to-end check that programs using `np.where` over a provably-
all-true or all-false cond compile without regression.

The Group 4a `zeros_content` / `ones_content` ensures plant the
`forall_eq_const` facts that the where strategy queries.
"""
import numpy as np

from zinnia import zk_circuit
from zinnia.api.zk_circuit import ZKCircuit


def test_where_on_ones_compiles():
    @zk_circuit
    def foo():
        a = np.array([10, 20, 30, 40])
        b = np.array([1, 2, 3, 4])
        _zinnia_result = np.where(np.ones(4, bool), a, b)

    _ = ZKCircuit.from_method(foo).compile()


def test_where_on_zeros_compiles():
    @zk_circuit
    def foo():
        a = np.array([10, 20, 30, 40])
        b = np.array([1, 2, 3, 4])
        _zinnia_result = np.where(np.zeros(4, bool), a, b)

    _ = ZKCircuit.from_method(foo).compile()
