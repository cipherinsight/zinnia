from zinnia import *


def test_list_extend():
    @zk_circuit
    def foo():
        lst = [1, 2, 3, 4]
        list_2 = lst.copy()
        list_2[1] = 9
        assert lst == [1, 2, 3, 4]
        assert list_2 == [1, 9, 3, 4]

    assert foo()
