from zinnia import *


def test_list_extend():
    @zk_circuit
    def foo():
        lst = [1, 2]
        lst.extend([3, 4, 5])
        assert lst == [1, 2, 3, 4, 5]
        lst.extend([6])
        assert lst == [1, 2, 3, 4, 5, 6]

    assert foo()


def test_list_extend_with_condition():
    @zk_circuit
    def foo():
        lst = [1, 2]
        if True:
            lst.extend([3, 4, 5])
        else:
            lst.extend([6, 7, 8])
        assert lst == [1, 2, 3, 4, 5]

    assert foo()
