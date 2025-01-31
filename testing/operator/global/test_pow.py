import pytest

from zinnia import *


def test_simple_pow():
    @zk_circuit
    def foo(x: int, exponent: int, result: int):
        assert x ** exponent == result

    for x in range(2, 5):
        for exp in range(1, 5):
            assert foo(x, exp, x ** exp)


def test_pow_method():
    @zk_circuit
    def foo(x: int, exponent: int, result: int):
        assert pow(x, exponent) == result

    for x in range(2, 5):
        for exp in range(1, 5):
            assert foo(x, exp, x ** exp)


def test_pow_method_with_mod():
    @zk_circuit
    def foo(x: int, exponent: int, md: int, result: int):
        assert pow(x, exponent, md) == result

    for x in range(2, 5):
        for exp in range(1, 5):
            for mod in range(2, 5):
                assert foo(x, exp, mod, x ** exp % mod)


def test_simple_pow_float():
    @zk_circuit
    def foo(x: float, exponent: float, result: float):
        assert pow(x, exponent) == result

    for x in range(1, 10):
        for exp in range(1, 10):
            _x = 0.5 + x * 0.3
            _exp = 0.5 + exp * 0.3
            assert foo(_x, _exp, _x ** _exp)


def test_pow_negative_base():
    @zk_circuit
    def foo(x: int, exponent: int, result: int):
        assert pow(x, exponent) == result

    assert foo(-2, 3, -8)


@pytest.mark.skip("A known bug. It will produce a complex number here")
def test_pow_negative_base_with_float_exp():
    @zk_circuit
    def foo(x: int, exponent: float, result: int):
        assert pow(x, exponent) == result

    assert foo(-2, 2.3, -8)


