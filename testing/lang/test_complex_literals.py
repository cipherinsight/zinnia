"""Pure complex constant literals — arithmetic (including unary `-`) is part
of the compiler.complex-arithmetic follow-up card.
"""
import pytest

from zinnia import zk_circuit, ZKCircuit, Complex
from zinnia.debug.exception import ZinniaException


def test_pure_imaginary_literal():
    @zk_circuit
    def foo():
        c = 1j
        _ = c

    ZKCircuit.from_method(foo).compile()


def test_imaginary_literal_in_assignment():
    @zk_circuit
    def foo():
        c = 2j
        _ = c

    ZKCircuit.from_method(foo).compile()


def test_complex_param_passthrough_compiles():
    @zk_circuit
    def foo(c: complex):
        _ = c

    ZKCircuit.from_method(foo).compile()


def test_arithmetic_on_complex_literal_still_fails_until_next_card():
    """Arithmetic involving a complex literal is the next card's territory."""
    @zk_circuit
    def foo():
        c = 1j + 1   # arithmetic on complex — not yet supported
        _ = c

    with pytest.raises(Exception):
        ZKCircuit.from_method(foo).compile()
