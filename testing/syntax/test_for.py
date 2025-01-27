import pytest

from zinnia import zk_circuit, ZKCircuit
from zinnia.debug.exception import ZenoPyException


def test_for_in_range_1():
    """This test is to check the basic for loop with range"""
    @zk_circuit
    def foo():
        the_sum = 0
        for i in range(0, 10):
            the_sum += i
        assert the_sum == 45

    ZKCircuit.from_method(foo).compile()


def test_for_in_range_2():
    """This test is to check the basic for loop with range"""
    @zk_circuit
    def foo():
        the_sum = 0
        for i in range(0, 10, 2):
            the_sum += i
        assert the_sum == 20

    ZKCircuit.from_method(foo).compile()


def test_for_in_list():
    """This test is to check if the for loop with list is executed correctly"""
    @zk_circuit
    def foo():
        the_sum = 0
        for i in [1, 2, 3, 4]:
            the_sum += i
        assert the_sum == 10

    ZKCircuit.from_method(foo).compile()


def test_for_in_tuple():
    """This test is to check if the for loop with tuple is executed correctly"""
    @zk_circuit
    def foo():
        the_sum = 0
        for i in (1, 2, 3, 4):
            the_sum += i
        assert the_sum == 10

    ZKCircuit.from_method(foo).compile()


def test_for_in_unpacking():
    """This test is to check if the for loop with unpacking is executed correctly"""
    @zk_circuit
    def foo():
        the_sum_a = 0
        the_sum_b = 0
        for a, b in [(1, 2), (3, 4), (5, 6), (7, 8)]:
            the_sum_a += a
            the_sum_b += b
        assert the_sum_a == 16
        assert the_sum_b == 20

    ZKCircuit.from_method(foo).compile()


def test_for_with_continue():
    """This test is to check if the for loop with continue is executed correctly"""
    @zk_circuit
    def foo():
        the_sum = 0
        for i in range(0, 10):
            if i % 2 == 0:
                continue
            the_sum += i
        assert the_sum == 25

    ZKCircuit.from_method(foo).compile()


def test_for_with_break():
    """This test is to check if the for loop with break is executed correctly"""
    @zk_circuit
    def foo():
        the_sum = 0
        for i in range(0, 10):
            if i == 5:
                break
            the_sum += i
        assert the_sum == 10

    ZKCircuit.from_method(foo).compile()


def test_for_with_else():
    """This test is to check if the for loop with else is executed correctly"""
    @zk_circuit
    def foo():
        the_sum = 0
        for i in range(0, 10):
            the_sum += i
        else:
            the_sum += 10
        assert the_sum == 55

    ZKCircuit.from_method(foo).compile()


def test_for_with_else_and_break():
    """This test is to check if the for loop with else is executed correctly"""
    @zk_circuit
    def foo():
        the_sum = 0
        for i in range(0, 10):
            the_sum += i
            if i == 5:
                break
        else:
            the_sum += 10
        assert the_sum == 15

    ZKCircuit.from_method(foo).compile()


def test_for_with_nested():
    """This test is to check if the nested for loop is executed correctly"""
    @zk_circuit
    def foo():
        the_sum = 0
        for i in range(0, 10):
            for j in range(0, 10):
                the_sum += i + j
        assert the_sum == 900

    ZKCircuit.from_method(foo).compile()


def test_for_with_nested_break():
    """This test is to check if the nested for loop with break is executed correctly"""
    @zk_circuit
    def foo():
        the_sum = 0
        for i in range(0, 10):
            for j in range(0, 10):
                if j == 5:
                    break
                the_sum += i + j
        assert the_sum == 325

    ZKCircuit.from_method(foo).compile()


def test_for_with_nested_continue():
    """This test is to check if the nested for loop with continue is executed correctly"""
    @zk_circuit
    def foo():
        the_sum = 0
        for i in range(0, 10):
            for j in range(0, 10):
                if j % 2 == 0:
                    continue
                the_sum += i + j
        assert the_sum == 475

    ZKCircuit.from_method(foo).compile()


def test_for_error_not_iterable():
    """This test is to check if the error is raised when the for loop is not iterable"""
    @zk_circuit
    def foo():
        the_sum = 0
        for i in 10:
            the_sum += i
        assert the_sum == 10

    with pytest.raises(ZenoPyException) as e:
        ZKCircuit.from_method(foo).compile()
    assert "is not iterable" in str(e.value)


def test_for_type_mismatch():
    """This test is to check when the for loop contains objects that are not the same type"""
    @zk_circuit
    def foo():
        the_sum = 0
        for i in [(1, 2, 3), (1, 2)]:
            the_sum += i[0]
        assert the_sum == 2

    ZKCircuit.from_method(foo).compile()
