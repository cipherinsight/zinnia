from zinnia import *


def test_list_append():
    @zk_circuit
    def foo():
        lst = []
        lst.append(1)
        lst.append(2)
        assert lst == [1, 2]

    assert foo()


def test_list_append_with_different_dtype():
    @zk_circuit
    def foo():
        lst = [1, 2, 3]
        lst.append([4])
        assert lst == [1, 2, 3, [4]]

    assert foo()
