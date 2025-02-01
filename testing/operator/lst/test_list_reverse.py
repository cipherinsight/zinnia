from zinnia import *


def test_list_reverse():
    @zk_circuit
    def foo():
        lst = [1, 2, 3]
        lst.reverse()
        assert lst == [3, 2, 1]

    assert foo()
