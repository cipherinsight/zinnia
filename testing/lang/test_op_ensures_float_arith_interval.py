"""Lang smoke test for `compiler.op-fact-group-9-float-arith-interval`.

Confirms a function whose float array elements carry forall-derived
bounds compiles cleanly when its body composes `+`, `-`, and `*` over
those elements. The interval-E helper at `ir_add_f / sub_f / mul_f`
deposits the output bound onto the result value, letting downstream
consumers see a finite range through the arith chain.

Rust unit tests (`interval_float_add_yields_sum_of_bounds` et al.)
cover the fact-deposition mechanics; this lang test is the functional
end-to-end check that the wired call-sites compile without regression.
"""
from zinnia import zk_circuit, requires, NDArray, Float
from zinnia.spec.predicates import forall, in_range
from zinnia.api.zk_circuit import ZKCircuit


def test_float_add_on_bounded_inputs_compiles():
    """Per-element `[0, 5]` over a float array; `arr[0] + arr[1]` lets the
    interval-E helper deposit Output ∈ [0, 10] on the sum's value bucket.
    """
    @zk_circuit
    @requires(lambda arr: forall(arr, in_range(0, 5)))
    def foo(arr: NDArray[Float, 4]):
        _zinnia_result = arr[0] + arr[1]

    _ = ZKCircuit.from_method(foo).compile()


def test_float_sub_on_bounded_inputs_compiles():
    """Per-element `[0, 5]` over a float array; `arr[0] - arr[1]` deposits
    Output ∈ [-5, 5]."""
    @zk_circuit
    @requires(lambda arr: forall(arr, in_range(0, 5)))
    def foo(arr: NDArray[Float, 4]):
        _zinnia_result = arr[0] - arr[1]

    _ = ZKCircuit.from_method(foo).compile()


def test_float_mul_on_bounded_inputs_compiles():
    """Per-element `[-2, 3]` over a float array; `arr[0] * arr[1]`
    deposits Output ∈ [-6, 9] via the corner-min-max rule."""
    @zk_circuit
    @requires(lambda arr: forall(arr, in_range(-2, 3)))
    def foo(arr: NDArray[Float, 4]):
        _zinnia_result = arr[0] * arr[1]

    _ = ZKCircuit.from_method(foo).compile()


def test_float_arith_composition_compiles():
    """Compose `(arr[0] + arr[1]) * (arr[2] - arr[3])`: each intermediate
    deposits its own bound, and the final mul composes through them."""
    @zk_circuit
    @requires(lambda arr: forall(arr, in_range(0, 5)))
    def foo(arr: NDArray[Float, 4]):
        _zinnia_result = (arr[0] + arr[1]) * (arr[2] - arr[3])

    _ = ZKCircuit.from_method(foo).compile()
