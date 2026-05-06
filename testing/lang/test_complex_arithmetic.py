"""Component-wise arithmetic on Complex operands.

Per compiler.complex-arithmetic. Each operator is dispatched at the top of
`apply_binary_op` and lowered through ir_*_f gates on the underlying
2-tuple representation.
"""
import pytest

from zinnia import zk_circuit, ZKCircuit


def test_complex_add_two_complex_values():
    @zk_circuit
    def foo():
        c = (1 + 2j) + (3 + 4j)  # 4 + 6j
        d = c + 0j
        _ = d

    ZKCircuit.from_method(foo).compile()


def test_complex_sub():
    @zk_circuit
    def foo():
        c = (5 + 3j) - (2 + 1j)  # 3 + 2j
        _ = c

    ZKCircuit.from_method(foo).compile()


def test_complex_mul():
    @zk_circuit
    def foo():
        # (1 + 2j)(3 + 4j) = (3 - 8) + (4 + 6)j = -5 + 10j
        c = (1 + 2j) * (3 + 4j)
        _ = c

    assert foo()


def test_complex_div_real():
    @zk_circuit
    def foo():
        c = (4 + 0j) / (2 + 0j)  # 2 + 0j
        _ = c

    assert foo()


def test_complex_eq_true():
    @zk_circuit
    def foo():
        a = 1 + 2j
        b = 1 + 2j
        assert a == b

    assert foo()


def test_complex_eq_false_returns_false():
    @zk_circuit
    def foo():
        a = 1 + 2j
        b = 1 + 3j
        assert (a == b) == False

    assert foo()


def test_complex_ne():
    @zk_circuit
    def foo():
        a = 1 + 2j
        b = 1 + 3j
        assert a != b

    assert foo()


def test_complex_with_real_int():
    @zk_circuit
    def foo():
        c = (3 + 4j) + 5  # 8 + 4j
        _ = c

    assert foo()


def test_complex_with_real_float():
    @zk_circuit
    def foo():
        c = (3 + 4j) * 2.0  # 6 + 8j
        _ = c

    assert foo()


def test_complex_unary_neg():
    @zk_circuit
    def foo():
        c = -(2 + 3j)  # -2 - 3j
        _ = c

    assert foo()


def test_complex_pow_static_int():
    @zk_circuit
    def foo():
        c = (1 + 0j) ** 3  # 1
        _ = c

    assert foo()


def test_complex_lt_rejected():
    @zk_circuit
    def foo():
        c = (1 + 0j) < (2 + 0j)
        _ = c

    with pytest.raises(Exception):
        ZKCircuit.from_method(foo).compile()
