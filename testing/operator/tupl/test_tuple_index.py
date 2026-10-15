from zinnia import *


def test_tuple_index():
    @zk_circuit
    def foo(value: int, result: int):
        tpl = (1, 2, 2, 3)
        assert tpl.index(value) == result

    assert foo(1, 0)
    assert foo(2, 1)
    assert foo(3, 3)
    assert not foo(4, 0)


def test_tuple_index_with_start():
    @zk_circuit
    def foo(value: int, start: int, result: int):
        tpl = (1, 2, 2, 3)
        assert tpl.index(value, start) == result

    assert foo(2, 1, 1)
    assert foo(3, 1, 3)
    assert foo(2, 2, 2)
    assert foo(2, -2, 2)
    assert foo(1, -9, 0)
    assert not foo(2, 2, 1)
    assert not foo(1, 1, 0)
    assert not foo(2, 9, 0)


def test_tuple_index_with_start_and_stop():
    @zk_circuit
    def foo(value: int, start: int, stop: int, result: int):
        tpl = (1, 2, 2, 3)
        assert tpl.index(value, start, stop) == result

    assert foo(2, 0, 2, 1)
    assert foo(3, -1, 9, 3)
    assert foo(2, -3, -1, 1)
    assert not foo(3, 0, 3, 3)
    assert not foo(3, 0, -1, 3)
    assert not foo(2, 2, 2, 2)
