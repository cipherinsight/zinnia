import pytest

from zinnia import zk_circuit, ZKCircuit, NDArray
from zinnia.debug.exception import ZenoPyException


def test_if():
    """This test is to check if the if statement is correctly parsed."""
    @zk_circuit
    def foo():
        condition_1, condition_2 = 1, 0
        the_sum = 0
        if condition_1:
            the_sum += 1
        if condition_2:
            the_sum += 1
        assert the_sum == 1

    ZKCircuit.from_method(foo).compile()


def test_if_else():
    """This test is to check if the if-else statement is correctly parsed."""
    @zk_circuit
    def foo():
        condition_1, condition_2 = 1, 0
        the_sum = 0
        if condition_1:
            the_sum += 1
        else:
            the_sum += 2
        if condition_2:
            the_sum += 1
        else:
            the_sum += 2
        assert the_sum == 3

    ZKCircuit.from_method(foo).compile()


def test_elif():
    """This test is to check if the if-elif-else statement is correctly parsed."""
    @zk_circuit
    def foo():
        condition_1, condition_2 = 1, 0
        the_sum = 0
        if condition_1:
            the_sum += 1
        elif condition_2:
            the_sum += 2
        else:
            the_sum += 3
        assert the_sum == 1
        if condition_2:
            the_sum += 1
        elif condition_1:
            the_sum += 2
        else:
            the_sum += 3
        assert the_sum == 3
        if not condition_2:
            the_sum += 1
        elif condition_1:
            the_sum += 2
        else:
            the_sum += 3
        assert the_sum == 4

    ZKCircuit.from_method(foo).compile()


def test_nested_if():
    """This test is to check if the nested if statement is correctly parsed."""
    @zk_circuit
    def foo():
        condition_1, condition_2 = 1, 0
        the_sum = 0
        if condition_1:
            the_sum += 1
            if condition_2:
                the_sum += 1
        assert the_sum == 1
        if condition_2:
            the_sum += 1
            if condition_1:
                the_sum += 1
        assert the_sum == 1
        if not condition_2:
            the_sum += 1
            if condition_1:
                the_sum += 1
        assert the_sum == 3

    ZKCircuit.from_method(foo).compile()


def test_condition_with_list_and_tuple():
    """This test is to check if the tuple/list condition is correctly parsed."""
    @zk_circuit
    def foo():
        tuple_condition = (1, 2)
        if tuple_condition:
            assert True
        else:
            assert False
        list_condition = [1, 2]
        if list_condition:
            assert True
        else:
            assert False

    ZKCircuit.from_method(foo).compile()


def test_invalid_condition_with_ndarray():
    """This test is to check if the invalid condition is correctly parsed."""
    @zk_circuit
    def foo():
        ndarray_condition = NDArray.ones((2, 2))
        if ndarray_condition:
            assert False

    with pytest.raises(ZenoPyException) as e:
        ZKCircuit.from_method(foo).compile()
    assert "The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()" in str(e.value)


def test_valid_condition_with_ndarray():
    """This test is to check if the invalid condition is correctly parsed."""
    @zk_circuit
    def foo():
        ndarray_condition = NDArray.ones((1, 1))
        if ndarray_condition:
            assert True
        else:
            assert False

    ZKCircuit.from_method(foo).compile()
