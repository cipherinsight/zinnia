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
