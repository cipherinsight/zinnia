from zinnia import *


def test_list_index():
    @zk_circuit
    def foo(value: int, result: int):
        lst = [1, 2, 2, 3]
        assert lst.index(value) == result

    assert foo(1, 0)
    assert foo(2, 1)
    assert foo(3, 3)
    assert not foo(4, 0)


def test_list_index_with_start():
    @zk_circuit
    def foo(value: int, start: int, result: int):
        lst = [1, 2, 2, 3]
        assert lst.index(value, start) == result

    assert foo(2, 1, 1)
    assert foo(3, 1, 3)
