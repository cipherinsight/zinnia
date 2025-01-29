from zinnia import *


def test_get_single_item_by_constants():
    @zk_circuit
    def foo():
        array = NDArray.asarray([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        assert array[0, 0] == array[0][0] == 1
        assert array[0, 1] == array[0][1] == 2
        assert array[0, 2] == array[0][2] == 3
        assert array[1, 0] == array[1][0] == 4
        assert array[1, 1] == array[1][1] == 5
        assert array[1, 2] == array[1][2] == 6
        assert array[2, 0] == array[2][0] == 7
        assert array[2, 1] == array[2][1] == 8
        assert array[2, 2] == array[2][2] == 9

    assert foo()


def test_get_single_item_by_non_constants():
    @zk_circuit
    def foo():
        array = NDArray.identity(10)
        for i in range(10):
            for j in range(10):
                assert array[i, j] == array[i][j] == (1 if i == j else 0)

    assert foo()


def test_get_single_item_by_variables():
    @zk_circuit
    def foo(x: int, y: int):
        array = NDArray.identity(4)
        assert array[x, y] == array[x][y] == (1 if x == y else 0)

    for i in range(4):
        for j in range(4):
            assert foo(i, j)


def test_get_item_by_slice():
    @zk_circuit
    def foo():
        array = NDArray.asarray([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        assert (array[0, :] == array[0][:]).all()
        assert (array[0, :] == [1, 2, 3]).all()
        assert (array[1, :] == array[1][:]).all()
        assert (array[1, :] == [4, 5, 6]).all()
        assert (array[2, :] == array[2][:]).all()
        assert (array[2, :] == [7, 8, 9]).all()
        assert (array[:, 0] == [1, 4, 7]).all()
        assert (array[:, 1] == [2, 5, 8]).all()
        assert (array[:, 2] == [3, 6, 9]).all()

    assert foo()