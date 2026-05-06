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


def test_literal_followed_by_simple_arithmetic_now_works():
    """The deferred arithmetic case from this card's original scope was lifted
    by compiler.complex-arithmetic. Keep as a smoke regression."""
    @zk_circuit
    def foo():
        c = 1j + 1
        _ = c

    ZKCircuit.from_method(foo).compile()
