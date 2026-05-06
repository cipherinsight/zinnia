"""Regression tests for the @zk_chip helper requirement.

A `@zk_circuit` body that calls a plain `def` helper fails to compile because
`@zk_circuit` only captures the source of the decorated function — the helper
is invisible to the AST → IR transformer. Decorating the helper with
`@zk_chip` makes it auto-discoverable from caller locals
(`zk_circuit.py:182-191`) and the circuit compiles successfully.
"""
from zinnia import *


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
