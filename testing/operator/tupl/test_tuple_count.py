from zinnia import *


def test_tuple_count():
    @zk_circuit
    def foo(value: int, result: int):
        tpl = (1, 2, 2, 3)
        assert tpl.count(value) == result

    assert foo(1, 1)
    assert foo(2, 2)
    assert foo(3, 1)
    assert foo(4, 0)
