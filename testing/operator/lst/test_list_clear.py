from zinnia import *


def test_list_clear():
    @zk_circuit
    def foo():
        lst = [1, 2, 3]
        lst.clear()
        assert lst == []

    assert foo()
