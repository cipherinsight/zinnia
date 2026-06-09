"""Regression tests for the @zk_chip helper requirement.

A `@zk_circuit` body that calls a plain `def` helper fails to compile because
`@zk_circuit` only captures the source of the decorated function — the helper
is invisible to the AST → IR transformer. Decorating the helper with
`@zk_chip` makes it auto-discoverable from caller locals
(`zk_circuit.py:182-191`) and the circuit compiles successfully.
"""
from zinnia import *
import pytest


def test_circuit_calls_zk_chip_scalar_helper():
    """A scalar-returning helper, decorated with @zk_chip, is reachable from
    the circuit and the program produces the correct result."""

    @zk_chip
    def double(x) -> Integer:
        return x * 2

    @zk_circuit
    def add_doubled(a: Public[Integer], b: Private[Integer]):
        assert double(a) + double(b) == 30

    assert add_doubled(5, 10)


def test_zk_chip_omitted_param_and_return_types_infer_from_call_site():
    """A chip without param or return annotations uses call-site values and
    lets the return type be inferred from the returned expression."""

    @zk_chip
    def increment(x):
        return x + 1

    @zk_circuit
    def check(a: Public[Integer]):
        y = increment(a)
        assert y == 8

    assert check(7)


def test_nested_helper_without_annotations_is_lifted_and_inferred():
    @zk_circuit
    def check(a: Public[Integer]):
        def increment(x):
            return x + 1

        assert increment(a) == 8

    assert check(7)


def test_zk_chip_annotated_partial_return_rejected():
    @zk_chip
    def partial(x) -> Integer:
        if x > 0:
            return x

    @zk_circuit
    def check(a: Public[Integer]):
        assert partial(a) == 8

    with pytest.raises(Exception, match="Chip return annotation requires every path to return"):
        ZKCircuit.from_method(check, chips=[partial]).compile()


def test_zk_chip_unannotated_partial_return_rejected():
    @zk_chip
    def partial(x):
        if x > 0:
            return x

    @zk_circuit
    def check(a: Public[Integer]):
        assert partial(a) == 8

    with pytest.raises(Exception, match="Chip return inference does not support partial returns"):
        ZKCircuit.from_method(check, chips=[partial]).compile()


def test_zk_chip_annotated_return_type_mismatch_rejected():
    @zk_chip
    def bad(x) -> Integer:
        return (x, x + 1)

    @zk_circuit
    def check(a: Public[Integer]):
        assert bad(a) == 8

    with pytest.raises(Exception, match="Chip return type mismatch"):
        ZKCircuit.from_method(check, chips=[bad]).compile()


def test_zk_chip_mixed_none_and_value_returns_rejected():
    @zk_chip
    def mixed(x):
        if x > 0:
            return x
        return

    @zk_circuit
    def check(a: Public[Integer]):
        assert mixed(a) == 8

    with pytest.raises(Exception, match="mixing bare returns with value returns"):
        ZKCircuit.from_method(check, chips=[mixed]).compile()


def test_zk_chip_mixed_explicit_none_and_value_returns_rejected():
    @zk_chip
    def mixed(x):
        if x > 0:
            return x
        return None

    @zk_circuit
    def check(a: Public[Integer]):
        assert mixed(a) == 8

    with pytest.raises(Exception, match="mixing bare returns with value returns"):
        ZKCircuit.from_method(check, chips=[mixed]).compile()


def test_circuit_calls_zk_chip_ndarray_helper():
    """An NDArray-returning helper (relu-like) decorated with @zk_chip
    composes cleanly inside a circuit body."""

    @zk_chip
    def relu(x) -> NDArray[Float, 4]:
        return np.maximum(x, 0)

    @zk_circuit
    def sum_relu(x: NDArray[Float, 4]):
        y = relu(x)
        assert np.sum(y) >= 0

    assert sum_relu([-1.0, 2.0, -3.0, 4.0])
