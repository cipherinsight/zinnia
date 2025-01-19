import pytest

from zenopy import zk_circuit, ZKCircuit


def test_basic_generator_tuple():
    """This test is to check if the generator in a tuple is working."""
    @zk_circuit
    def foo():
        the_tuple = (i for i in range(5))
        assert the_tuple == (0, 1, 2, 3, 4)

    ZKCircuit.from_method(foo, {}).compile()


def test_basic_generator_list():
    """This test is to check if the generator in a list is working."""
    @zk_circuit
    def foo():
        the_list = [i for i in range(5)]
        assert the_list == [0, 1, 2, 3, 4]

    ZKCircuit.from_method(foo, {}).compile()


def test_generator_with_if():
    """This test is to check if the generator with if is working."""
    @zk_circuit
    def foo():
        the_list = [i for i in range(10) if i % 2 == 0]
        assert the_list == [0, 2, 4, 6, 8]

    ZKCircuit.from_method(foo, {}).compile()


def test_generator_with_many_if():
    """This test is to check if the generator with many ifs is working."""
    @zk_circuit
    def foo():
        the_list = [i for i in range(10) if i % 2 == 0 if i % 3 == 0]
        assert the_list == [0, 6]

    ZKCircuit.from_method(foo, {}).compile()


def test_multiple_generators_1():
    """This test is to check if the multiple generators are working."""
    @zk_circuit
    def foo():
        the_list = [i for i in range(10) if i % 2 == 0 for j in range(2)]
        assert the_list == [0, 0, 2, 2, 4, 4, 6, 6, 8, 8]

    ZKCircuit.from_method(foo, {}).compile()


def test_multiple_generators_2():
    """This test is to check if the multiple generators are working."""
    @zk_circuit
    def foo():
        the_list = [i for i in range(10) if i % 2 == 0 for j in range(2) if j % 2 == 0]
        assert the_list == [0, 2, 4, 6, 8]

    ZKCircuit.from_method(foo, {}).compile()


def test_multiple_generators_3():
    """This test is to check if the multiple generators are working."""
    @zk_circuit
    def foo():
        the_list = [i * j for i in range(10) if i % 2 == 0 for j in range(2) if j % 2 == 0]
        assert the_list == [0, 0, 0, 0, 0]

    ZKCircuit.from_method(foo, {}).compile()


def test_multiple_generators_with_same_target():
    """This test is to check if the multiple generators are working."""
    @zk_circuit
    def foo():
        the_list = [i for i in range(10) for i in range(2)]
        assert the_list == [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    ZKCircuit.from_method(foo, {}).compile()


def test_generator_with_different_datatype():
    """This test is to check if the generator with different datatype is working."""
    @zk_circuit
    def foo():
        the_list = [elem * 2 for elem in [(1, ), (1, 2), (1, 2, 3)]]
        assert the_list == [(1, 1), (1, 2, 1, 2), (1, 2, 3, 1, 2, 3)]

    ZKCircuit.from_method(foo, {}).compile()


def test_generator_with_different_datatype_and_if():
    """This test is to check if the generator with different datatype is working."""
    @zk_circuit
    def foo():
        the_list = [elem * 2 for elem in [(1, ), (1, 2), (1, 2, 3)] if len(elem) == 2]
        assert the_list == [(1, 2, 1, 2)]

    ZKCircuit.from_method(foo, {}).compile()
