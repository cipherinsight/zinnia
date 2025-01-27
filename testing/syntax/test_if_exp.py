import pytest

from zinnia import ZKCircuit, zk_circuit
from zinnia.debug.exception import ZinniaException


def test_basic_if_exp_1():
    @zk_circuit
    def foo():
        cond = 1
        the_value = 20 if cond else 10
        assert the_value == 20

    ZKCircuit.from_method(foo).compile()


def test_basic_if_exp_2():
    @zk_circuit
    def foo():
        cond = 0
        the_value = 20 if cond else 10
        assert the_value == 10

    ZKCircuit.from_method(foo).compile()


def test_if_exp_with_different_types():
    @zk_circuit
    def foo():
        cond = 1
        the_value = 20 if cond else 10.0
        assert the_value == 20

    with pytest.raises(ZinniaException) as e:
        ZKCircuit.from_method(foo).compile()
    assert "must have the same type" in str(e.value)


def test_nested_if_exp():
    @zk_circuit
    def foo():
        cond_1, cond_2 = 1, 0
        the_value = 20 if cond_1 else 10 if cond_2 else 5
        assert the_value == 20
        the_value = 20 if cond_2 else 10 if cond_1 else 5
        assert the_value == 10
        the_value = 20 if cond_2 else 10 if cond_2 else 5
        assert the_value == 5

    ZKCircuit.from_method(foo).compile()
