"""Lang regression test for the `np.mean(np.cos(...))` dispatch fix.

Phase 3a differential fuzzer (run 2026-05-15) surfaced programs where
`np.mean(np.cos(x))` panicked at compile time with "static inference:
assertion ... is provably unsatisfiable" because the named-attr handler
at `src/ir_gen/named_attr.rs:394` dispatched `mean` through the same
catch-all arm as `sum`/`prod`/`min`/`max`/`any`/`all`, routing it into
`helpers::ndarray::builtin_reduce`, which has no `"mean"` arm and
returned `Value::None`. The `Value::None` was then coerced to constant
`0` by `IRBuilder::ensure_ptr`, so a downstream `out - C > -0.001`
constant-folded to `false` and tripped `AlwaysSatisfiedElimination`.

The fix de-shadows `mean` from that catch-all so `np.mean(...)` reaches
the dedicated `np_mean` handler (which has correct float-mean lowering
plus strategy-set wiring).

This test compiles a circuit that exercises the exact failing surface
(`out = np.mean(np.cos(x))` on a `Float` NDArray, then an inequality
assert) and confirms the program compiles without panicking. The Rust
unit `np_mean_on_list_of_floats_returns_real_value` in
`strategy_tests.rs` complements this by locking the `Value::None` exit
at the dispatch surface.
"""
import numpy as np

from zinnia import zk_circuit, NDArray, Float
from zinnia.api.zk_circuit import ZKCircuit


def test_mean_cos_chain_compiles():
    """`np.mean(np.cos(x))` over a 4-element Float array: must compile
    cleanly after the named-attr dispatch fix. Before the fix this
    panicked at static inference time with "provably unsatisfiable"
    because `Value::None → 0` constant-folded the assert to `false`.
    """
    @zk_circuit
    def mean_cos_simple(x: NDArray[Float, 4]):
        y = np.cos(x)
        out = np.mean(y)
        # `np.mean(cos(x))` is bounded in [-1, 1]; -2 is a generous slack
        # that the resolver doesn't need to prove — we only need the
        # compile to succeed (i.e., not panic in AlwaysSatisfiedElimination).
        assert out > -2.0

    _ = ZKCircuit.from_method(mean_cos_simple).compile()


def test_mean_sin_chain_compiles():
    """Sister case: `np.mean(np.sin(x))`. The original fuzz repro at
    `tools/fuzz_reports/20260515-002952/0029-compile_failure-B.json`
    used `sin` not `cos`; same root cause, same fix exercises it.
    """
    @zk_circuit
    def mean_sin_simple(x: NDArray[Float, 4]):
        y = np.sin(x)
        out = np.mean(y)
        assert out > -2.0

    _ = ZKCircuit.from_method(mean_sin_simple).compile()


def test_mean_cos_with_inner_poly_compiles():
    """Original fuzz-found shape: `np.mean(np.cos(x*x + 1.0))`. The
    polynomial inner is incidental to the bug but kept here as a more
    faithful regression check against the exact failing program.
    """
    @zk_circuit
    def mean_cos_poly(x: NDArray[Float, 2]):
        y = np.cos(x * x + 1.0)
        out = np.mean(y)
        diff = out - 0.1750237645509226
        assert diff > -2.0

    _ = ZKCircuit.from_method(mean_cos_poly).compile()
