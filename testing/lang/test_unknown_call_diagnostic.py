"""Diagnostic for unknown function call inside @zk_circuit.

Helpers should be decorated with @zk_chip; the error message should say so.
"""
import pytest

from zinnia import *


def test_undecorated_helper_call_suggests_zk_chip():
    def my_helper(x):
        return x * 2

    @zk_circuit
    def foo(x: NDArray[Integer, 4]):
        y = my_helper(x[0])
        assert y == 0

    with pytest.raises(Exception) as exc_info:
        ZKCircuit.from_method(foo).compile()

    msg = str(exc_info.value)
    assert "my_helper" in msg
    assert "@zk_chip" in msg
