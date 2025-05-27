import pytest

from zinnia import *


def test_set_single_item_by_constants_1():
    @zk_circuit
    def foo():
        array = np.zeros((3, 3), dtype=int)
        array[0, 0] = 1
        array[0, 1] = 2
        array[0, 2] = 3
        array[1, 0] = 4
        array[1, 1] = 5
        array[1, 2] = 6
        array[2, 0] = 7
        array[2, 1] = 8
        array[2, 2] = 9
        assert array[0][0] == 1
        assert array[0][1] == 2
        assert array[0][2] == 3
        assert array[1][0] == 4
        assert array[1][1] == 5
        assert array[1][2] == 6
        assert array[2][0] == 7
        assert array[2][1] == 8
        assert array[2][2] == 9

    assert foo()


@pytest.mark.skip("A known bug. We should implement NDArrayView to fix this.")
def test_set_single_item_by_constants_2():
    @zk_circuit
    def foo():
        array = np.zeros((3, 3), dtype=int)
        array[0][0] = 1
        array[0][1] = 2
        array[0][2] = 3
        array[1][0] = 4
        array[1][1] = 5
        array[1][2] = 6
        array[2][0] = 7
        array[2][1] = 8
        array[2][2] = 9
        assert array[0, 0] == 1
        assert array[0, 1] == 2
        assert array[0, 2] == 3
        assert array[1, 0] == 4
        assert array[1, 1] == 5
        assert array[1, 2] == 6
        assert array[2, 0] == 7
        assert array[2, 1] == 8
        assert array[2, 2] == 9

    assert foo()


def test_set_single_item_by_variable():
    @zk_circuit
    def foo(x: int, y: int):
        array = np.zeros((4, 4), dtype=int)
        array[x, y] = 1
        assert array[x, y] == 1
        for i in range(4):
            for j in range(4):
                if i != x or j != y:
                    assert array[i, j] == 0

    for i in range(4):
        for j in range(4):
            assert foo(i, j)


def test_set_item_by_slice():
    @zk_circuit
    def foo():
        array = np.zeros((3, 3), dtype=int)
        array[0, :] = [1, 2, 3]
        array[1, :] = [4, 5, 6]
        array[2, :] = [7, 8, 9]
        assert (array[0, :] == [1, 2, 3]).all()
        assert (array[1, :] == [4, 5, 6]).all()
        assert (array[2, :] == [7, 8, 9]).all()
        array[:, 0] = [1, 4, 7]
        array[:, 1] = [2, 5, 8]
        array[:, 2] = [3, 6, 9]
        assert (array[:, 0] == [1, 4, 7]).all()
        assert (array[:, 1] == [2, 5, 8]).all()
        assert (array[:, 2] == [3, 6, 9]).all()

    assert foo()


def test_set_item_by_slice_with_variable():
    @zk_circuit
    def foo(x: int):
        array = np.zeros((4, 4), dtype=int)
        array[x, :] = [1, 2, 3, 4]
        assert (array[x, :] == [1, 2, 3, 4]).all()
        for i in range(4):
            if i != x:
                assert (array[i, :] == [0, 0, 0, 0]).all()

    for i in range(4):
        assert foo(i)


def test_set_item_by_different_dtype_1():
    @zk_circuit
    def foo():
        array = np.zeros((3, 3), dtype=int)
        array[0, :] = [1.2, 2.1, 3.0]
        array[1, :] = [4.2, 5.1, 6.0]
        array[2, :] = [7.2, 8.1, 9.0]
        assert (array[0, :] == [1, 2, 3]).all()
        assert (array[1, :] == [4, 5, 6]).all()
        assert (array[2, :] == [7, 8, 9]).all()
        assert array.dtype == int

    assert foo()


def test_set_item_by_different_dtype_2():
    @zk_circuit
    def foo():
        array = np.zeros((3, 3), dtype=float)
        array[0, :] = [1, 2, 3]
        array[1, :] = [4, 5, 6]
        array[2, :] = [7, 8, 9]
        assert (array[0, :] == [1, 2, 3]).all()
        assert (array[1, :] == [4, 5, 6]).all()
        assert (array[2, :] == [7, 8, 9]).all()
        assert array.dtype == float

    assert foo()


def test_set_single_item_by_different_dtype_1():
    @zk_circuit
    def foo():
        array = np.zeros((3, 3), dtype=int)
        array[1, 1] = 3.3
        assert array[1, 1] == 3
        assert array.dtype == int

    assert foo()


def test_set_single_item_by_different_dtype_2():
    @zk_circuit
    def foo():
        array = np.zeros((3, 3), dtype=float)
        array[1, 1] = 3
        assert array[1, 1] == 3.0
        assert array.dtype == float

    assert foo()


def test_set_item_by_broadcasting():
    @zk_circuit
    def foo():
        array = np.zeros((3, 3), dtype=int)
        array[0, :] = 1
        array[1, :] = 2
        array[2, :] = 3
        assert (array[0, :] == [1, 1, 1]).all()
        assert (array[1, :] == [2, 2, 2]).all()
        assert (array[2, :] == [3, 3, 3]).all()

    assert foo()
