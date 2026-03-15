import pytest

from zinnia import *


def test_list_pop():
    @zk_circuit
    def foo():
        lst = [1, 2, 3]
        lst.pop()
        assert lst == [1, 2]

    assert foo()


def test_list_pop_different_indices():
    @zk_circuit
    def foo():
        lst = [1, 2, 3]
        lst.pop(0)
        assert lst == [2, 3]
        lst = [1, 2, 3]
        lst.pop(-3)
        assert lst == [2, 3]
        lst = [1, 2, 3]
        lst.pop(1)
        assert lst == [1, 3]
        lst = [1, 2, 3]
        lst.pop(-2)
        assert lst == [1, 3]
        lst = [1, 2, 3]
        lst.pop(2)
        assert lst == [1, 2]
        lst = [1, 2, 3]
        lst.pop(-1)
        assert lst == [1, 2]

    assert foo()


def test_list_pop_invalid_indices():
    @zk_circuit
    def foo():
        lst = [1, 2, 3]
        lst.pop(4)

    with pytest.raises(ZinniaException) as e:
        assert foo()
    assert "pop index out of range" in str(e)


def test_list_pop_invalid_indices_negative():
    @zk_circuit
    def foo():
        lst = [1, 2, 3]
        lst.pop(-4)

    with pytest.raises(ZinniaException) as e:
        assert foo()
    assert "pop index out of range" in str(e)


def test_list_pop_dynamic_indices():
    @zk_circuit
    def foo(index: int, result: Public[List[int, int]]):
        lst = [1, 2, 3]
        lst.pop(index)
        assert lst == result

    assert foo(0, [2, 3])
    assert foo(1, [1, 3])
    assert foo(2, [1, 2])
    assert foo(-3, [2, 3])
    assert foo(-2, [1, 3])
    assert foo(-1, [1, 2])


def test_list_pop_dynamic_indices_invalid():
    @zk_circuit
    def foo(index: int):
        lst = [1, 2, 3]
        lst.pop(index)

    assert not foo(4)
    assert not foo(-4)
