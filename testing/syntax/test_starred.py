from zinnia import *


def test_starred_in_list():
    @zk_circuit
    def foo():
        a = [1, 2, 3, 4, 5, 6]
        b = [*a, 7, 8, 9]
        assert b == [1, 2, 3, 4, 5, 6, 7, 8, 9]

    assert foo()


def test_starred_in_tuple():
    @zk_circuit
    def foo():
        a = [7, 8, 9]
        b = (1, 2, 3, 4, 5, 6, *a)
        assert b == (1, 2, 3, 4, 5, 6, 7, 8, 9)

    assert foo()


def test_starred_in_args():
    @zk_chip
    def my_add(x, y, z, u) -> Integer:
        return x + y + z + u

    @zk_circuit
    def foo():
        args = [2, 3]
        assert my_add(1, *args, 4) == 10

    assert foo()
