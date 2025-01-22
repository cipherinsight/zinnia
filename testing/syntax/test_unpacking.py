import pytest

from zenopy import zk_circuit, ZKCircuit
from zenopy.debug.exception import ZenoPyException


def test_tuple_unpacking():
    """This test is to check if the tuple unpacking is done correctly"""
    @zk_circuit
    def foo():
        tup = (1, 2, 3)
        a, b, c = tup
        assert a == 1
        assert b == 2
        assert c == 3

    ZKCircuit.from_method(foo).compile()


def test_tuple_unpacking_with_star_1():
    """This test is to check if the tuple unpacking with star is done correctly"""
    @zk_circuit
    def foo():
        tup = (1, 2, 3, 4)
        a, *b, c = tup
        assert a == 1
        assert b == [2, 3]
        assert c == 4

    ZKCircuit.from_method(foo).compile()


def test_tuple_unpacking_with_star_2():
    """This test is to check if the tuple unpacking with star only is done correctly"""
    @zk_circuit
    def foo():
        tup = (1, 2, 3, 4)
        *a, = tup
        assert a == [1, 2, 3, 4]

    ZKCircuit.from_method(foo).compile()


def test_tuple_unpacking_with_star_3():
    """This test is to check if the tuple unpacking with star is done correctly"""
    @zk_circuit
    def foo():
        tup = (1, (2, 3), (4, ))
        a, *b, c = tup
        assert a == 1
        assert b == (2, 3)
        assert c == (4, )

    ZKCircuit.from_method(foo).compile()


def test_tuple_unpacking_with_error_1():
    """This test is to check if the tuple unpacking with star is done correctly"""
    @zk_circuit
    def foo():
        tup = (1, 2, 3, 4)
        a, b, c = tup

    with pytest.raises(ZenoPyException) as e:
        ZKCircuit.from_method(foo).compile()
    assert "TupleUnpackingError" in str(e.value)


def test_tuple_unpacking_with_error_2():
    """This test is to check if the tuple unpacking with star is done correctly"""
    @zk_circuit
    def foo():
        tup = (1, 2, 3, 4)
        a, b, c, d, e = tup

    with pytest.raises(ZenoPyException) as e:
        ZKCircuit.from_method(foo).compile()
    assert "TupleUnpackingError" in str(e.value)


def test_tuple_unpacking_with_error_3():
    """This test is to check if the tuple unpacking with star is done correctly"""
    @zk_circuit
    def foo():
        tup = (1, (2, 3), 4, 5)
        a, b, c, (d, e) = tup
        assert a == 1
        assert b == 2
        assert c == 3
        assert d == 4
        assert e == 5

    with pytest.raises(ZenoPyException) as e:
        ZKCircuit.from_method(foo).compile()
    assert "TypeInferenceError: Integer is not iterable" in str(e.value)


def test_tuple_unpacking_with_inner():
    """This test is to check if the tuple unpacking with star is done correctly"""
    @zk_circuit
    def foo():
        tup = (1, (2, 3), 4, 5)
        a, (b, c), d, e = tup
        assert a == 1
        assert b == 2
        assert c == 3
        assert d == 4
        assert e == 5

    ZKCircuit.from_method(foo).compile()
