from zinnia import *


def test_list_count():
    @zk_circuit
    def foo(value: int, result: int):
        lst = [1, 2, 2, 3]
        assert lst.count(value) == result

    assert foo(1, 1)
    assert foo(2, 2)
    assert foo(3, 1)
    assert foo(4, 0)
