import pytest

from zenopy import zk_circuit, ZKCircuit, zk_chip, Integer


def test_while():
    @zk_circuit
    def foo():
        the_sum = 0
        i = 0
        while i < 10:
            the_sum += i
            i += 1
        assert the_sum == 45

    ZKCircuit.from_method(foo).compile()


def test_while_break():
    @zk_circuit
    def foo():
        the_sum = 0
        i = 0
        while i < 10:
            the_sum += i
            i += 1
            if i == 5:
                break
        assert the_sum == 10

    ZKCircuit.from_method(foo).compile()


def test_while_continue():
    @zk_circuit
    def foo():
        the_sum = 0
        i = 0
        while i < 10:
            i += 1
            if i % 2 == 0:
                continue
            the_sum += i
        assert the_sum == 25

    ZKCircuit.from_method(foo).compile()


def test_infinite_while_with_break():
    @zk_circuit
    def foo():
        the_sum = 0
        i = 0
        while True:
            the_sum += i
            i += 1
            if i == 5:
                break
        assert the_sum == 10

    ZKCircuit.from_method(foo).compile()


def test_infinite_while_with_return():
    @zk_chip
    def while_chip() -> Integer:
        the_sum = 0
        i = 0
        while True:
            the_sum += i
            i += 1
            if i == 5:
                return the_sum

    @zk_circuit
    def foo():
        the_sum = while_chip()
        assert the_sum == 10

    ZKCircuit.from_method(foo, chips=[while_chip]).compile()
