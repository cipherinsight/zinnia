"""Lang smoke test for `compiler.op-fact-group-3d-reductions-forall-eq-const-strategy`.

Confirms `np.sum(np.zeros(N))` / `np.prod(np.ones(N))` / `np.mean(np.zeros(N))`
compile cleanly with the new strategy-selection wiring. The Rust unit
tests in `optim/tests/strategy_tests.rs` cover the dispatch mechanics
(precondition matching, IR statement counts); this lang test is the
end-to-end check that programs using these reductions compile without
regression. The static-array constructor path (`np.zeros(8)` →
`Value::StaticArray`) doesn't currently carry a `value_id`, so the
strategy fires on the dyn-ndarray / composite paths only — but the
program must still compile end-to-end on the static path.
"""
import numpy as np

from zinnia import zk_circuit
from zinnia.api.zk_circuit import ZKCircuit


def test_sum_on_zeros_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.sum(np.zeros(8))

    _ = ZKCircuit.from_method(foo).compile()


def test_sum_on_ones_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.sum(np.ones(8))

    _ = ZKCircuit.from_method(foo).compile()


def test_prod_on_zeros_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.prod(np.zeros(8))

    _ = ZKCircuit.from_method(foo).compile()


def test_prod_on_ones_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.prod(np.ones(8))

    _ = ZKCircuit.from_method(foo).compile()


def test_mean_on_zeros_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.mean(np.zeros(8))

    _ = ZKCircuit.from_method(foo).compile()


def test_mean_on_ones_compiles():
    @zk_circuit
    def foo():
        _zinnia_result = np.mean(np.ones(8))

    _ = ZKCircuit.from_method(foo).compile()


def test_sum_on_zeros_static_path_compiles():
    """Parallel to `test_sum_on_zeros_compiles`, but exercises the
    multi-dim static-only path (`np.zeros((3, 4))` lands as
    `Value::StaticArray` rather than promoting to dyn). With
    `compiler.value-static-array-value-id`, the Group 3d sum-on-zero
    strategy now fires on this path too — fact-anchoring was previously
    a no-op because `Value::StaticArray` carried no `value_id`."""
    @zk_circuit
    def foo():
        _zinnia_result = np.sum(np.zeros((3, 4)))

    _ = ZKCircuit.from_method(foo).compile()
