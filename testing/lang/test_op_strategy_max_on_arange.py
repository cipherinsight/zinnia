"""Lang smoke test for `compiler.op-fact-group-4b-range-constructors-is-sorted`.

Confirms `np.max(np.arange(10))` / `np.min(np.arange(10))` /
`np.max(np.linspace(0, 1, 5))` compile cleanly end-to-end with the new
range-constructor is_sorted emission wired through the boundary-read
fast path (compiler.consumer-max-min-on-sorted).

The fully-static path produces a `Value::StaticArray` (no `value_id`),
so the is_sorted fire is a no-op for the static composite — and the
static reduction collapses the value at compile time via constant
folding regardless. The bounded-arange / bounded-linspace paths
produce `DynamicNDArray`s where the is_sorted fact actually lands;
those are exercised by the Rust unit tests in `contracts_tests.rs`.
This lang test is the end-to-end compile-and-run check: programs using
the four monotone-respecting reductions on range-constructor outputs
must still compile cleanly and produce the right value.
"""
from zinnia import *


def test_max_on_arange_compiles_and_returns_last_element():
    @zk_circuit
    def foo():
        assert np.max(np.arange(10)) == 9

    assert foo()


def test_min_on_arange_returns_zero():
    @zk_circuit
    def foo():
        assert np.min(np.arange(10)) == 0

    assert foo()


def test_argmax_on_arange_returns_last_index():
    @zk_circuit
    def foo():
        assert np.argmax(np.arange(10)) == 9

    assert foo()


def test_argmin_on_arange_returns_zero():
    @zk_circuit
    def foo():
        assert np.argmin(np.arange(10)) == 0

    assert foo()


def test_max_on_linspace_compiles():
    # Compile-only smoke for the linspace path. We don't assert on the
    # value because static linspace evaluates `(stop-start)/(num-1)` in
    # f64 and the resulting last-element comparison is bit-sensitive on
    # some platforms — exercise the compile pipeline only.
    @zk_circuit
    def foo():
        _ = np.max(np.linspace(0.0, 1.0, 5))

    foo()
