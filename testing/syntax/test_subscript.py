import pytest

from zenopy import zk_circuit, ZKCircuit, Integer, Public
from zenopy.debug.exception import ZenoPyException


def test_subscript_in_tuple():
    """This test is to check if the subscript in a tuple is working."""
    @zk_circuit
    def foo():
        the_tuple = (1, 2, 3)
        assert the_tuple[0] == 1
        assert the_tuple[1] == 2
        assert the_tuple[2] == 3
        assert the_tuple[-1] == 3
        assert the_tuple[-2] == 2
        assert the_tuple[-3] == 1

    ZKCircuit.from_method(foo, {}).compile()


def test_subscript_in_tuple_range():
    """This test is to check if the subscript in a tuple is working."""
    @zk_circuit
    def foo():
        the_tuple = (1, 2, 3, 4, 5, 6)
        assert the_tuple[0:3] == (1, 2, 3)
        assert the_tuple[1:4] == (2, 3, 4)
        assert the_tuple[2:] == (3, 4, 5, 6)
        assert the_tuple[:4] == (1, 2, 3, 4)
        assert the_tuple[1:4:2] == (2, 4)
        assert the_tuple[1:-2:2] == (2, 4)

    ZKCircuit.from_method(foo, {}).compile()


def test_subscript_in_list():
    """This test is to check if the subscript in a list is working."""
    @zk_circuit
    def foo():
        the_list = [1, 2, 3]
        assert the_list[0] == 1
        assert the_list[1] == 2
        assert the_list[2] == 3
        assert the_list[-1] == 3
        assert the_list[-2] == 2
        assert the_list[-3] == 1

    ZKCircuit.from_method(foo, {}).compile()


def test_subscript_in_list_range():
    """This test is to check if the subscript in a list is working."""
    @zk_circuit
    def foo():
        the_list = [1, 2, 3, 4, 5, 6]
        assert the_list[0:3] == [1, 2, 3]
        assert the_list[1:4] == [2, 3, 4]
        assert the_list[2:] == [3, 4, 5, 6]
        assert the_list[:4] == [1, 2, 3, 4]
        assert the_list[1:4:2] == [2, 4]
        assert the_list[1:-2:2] == [2, 4]

    ZKCircuit.from_method(foo, {}).compile()


def test_subscript_assign_list():
    """This test is to check if the subscript assignment in a list is working."""
    @zk_circuit
    def foo():
        the_list = [1, 2, 3, 4, 5, 6]
        the_list[0] = 10
        assert the_list[0] == 10
        the_list[1] = 20
        assert the_list[1] == 20
        the_list[2] = 30
        assert the_list[2] == 30
        the_list[-1] = 60
        assert the_list[-1] == 60
        the_list[-2] = 50
        assert the_list[-2] == 50
        the_list[-3] = 40
        assert the_list[-3] == 40
        assert the_list == [10, 20, 30, 40, 50, 60]

    ZKCircuit.from_method(foo, {}).compile()


def test_subscript_error_assign_tuple():
    """This test is to check if the subscript assignment in a tuple should not work."""
    @zk_circuit
    def foo():
        the_tuple = (1, 2, 3, 4, 5, 6)
        the_tuple[0] = 10

    with pytest.raises(ZenoPyException) as e:
        ZKCircuit.from_method(foo, {}).compile()
    assert "does not support item assignment" in str(e.value)


def test_subscript_error_assign_tuple_range():
    """This test is to check if the subscript assignment in a tuple should not work."""
    @zk_circuit
    def foo():
        the_tuple = (1, 2, 3, 4, 5, 6)
        the_tuple[0:3] = (10, 20, 30)

    with pytest.raises(ZenoPyException) as e:
        ZKCircuit.from_method(foo, {}).compile()
    assert "does not support item assignment" in str(e.value)


def test_subscript_list_with_variable_index():
    """This test is to check if the subscript in a list is working with variable index."""
    @zk_circuit
    def foo(idx: Public[Integer]):
        the_list = [1, 2, 3]
        assert the_list[idx] == 2

    ZKCircuit.from_method(foo, {}).compile()
    # TODO: mock execution this circuit

