import pytest

from zinnia import *


def test_list_remove():
    @zk_circuit
    def foo():
        lst = [1, 2, 2, 3]
        lst.remove(2)
        assert lst == [1, 2, 3]

    assert foo()


def test_list_remove_dynamic_value():
    @zk_circuit
    def foo(value: int, result: List[int, int, int]):
        lst = [1, 2, 2, 3]
        lst.remove(value)
        assert lst == result

    assert foo(2, [1, 2, 3])
    assert foo(1, [2, 2, 3])
    assert foo(3, [1, 2, 2])


def test_list_remove_not_exists():
    @zk_circuit
    def foo():
        lst = [1, 2, 2, 3]
        lst.remove(4)
        assert lst == [1, 2, 2, 3]

    with pytest.raises(ZinniaException) as e:
        assert foo()
    assert "Value not found in list" in str(e)


def test_list_remove_not_exists_dynamic():
    @zk_circuit
    def foo(value: int):
        lst = [1, 2, 2, 3]
        lst.remove(value)

    assert not foo(4)
