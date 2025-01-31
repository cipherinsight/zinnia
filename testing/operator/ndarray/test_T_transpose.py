import pytest

from zinnia import *


def test_T_1_dim():
    @zk_circuit
    def foo():
        a = NDArray.asarray([1, 2, 3, 4, 5, 6])
        assert (a.T == NDArray.asarray([1, 2, 3, 4, 5, 6])).all()

    assert foo()


def test_T_2_dim():
    @zk_circuit
    def foo():
        a = NDArray.asarray([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        assert (a.T == NDArray.asarray([
            [1, 4, 7],
            [2, 5, 8],
            [3, 6, 9]
        ])).all()

    assert foo()


def test_T_3_dim():
    @zk_circuit
    def foo():
        a = NDArray.asarray([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]],
        ])
        assert (a.T == NDArray.asarray([
            [[1, 5, 9], [3, 7, 11]],
            [[2, 6, 10], [4, 8, 12]]
        ])).all()

    assert foo()


def test_transpose_1_dim():
    @zk_circuit
    def foo():
        a = NDArray.asarray([1, 2, 3, 4, 5, 6])
        assert (a.transpose() == NDArray.asarray([1, 2, 3, 4, 5, 6])).all()

    assert foo()


def test_transpose_2_dim():
    @zk_circuit
    def foo():
        a = NDArray.asarray([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        assert (a.transpose() == NDArray.asarray([
            [1, 4, 7],
            [2, 5, 8],
            [3, 6, 9]
        ])).all()

    assert foo()


def test_transpose_3_dim():
    @zk_circuit
    def foo():
        a = NDArray.asarray([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ])
        assert (a.transpose() == NDArray.asarray([
            [[1, 5], [3, 7]],
            [[2, 6], [4, 8]]
        ])).all()

    assert foo()


def test_transpose_3_dim_axes_specified():
    import numpy as np

    @zk_circuit
    def foo_1(input_array: NDArray[Float, 2, 2, 2], result: Public[NDArray[Float, 2, 2, 2]]):
        assert (input_array.transpose(axes=(0, 1, 2)) == result).all()

    @zk_circuit
    def foo_2(input_array: NDArray[Float, 2, 2, 2], result: Public[NDArray[Float, 2, 2, 2]]):
        assert (input_array.transpose(axes=(2, 1, 0)) == result).all()

    @zk_circuit
    def foo_3(input_array: NDArray[Float, 2, 2, 2], result: Public[NDArray[Float, 2, 2, 2]]):
        assert (input_array.transpose(axes=(1, 0, 2)) == result).all()

    @zk_circuit
    def foo_4(input_array: NDArray[Float, 2, 2, 2], result: Public[NDArray[Float, 2, 2, 2]]):
        assert (input_array.transpose(axes=(1, 2, 0)) == result).all()

    @zk_circuit
    def foo_5(input_array: NDArray[Float, 2, 2, 2], result: Public[NDArray[Float, 2, 2, 2]]):
        assert (input_array.transpose(axes=(0, 2, 1)) == result).all()

    @zk_circuit
    def foo_6(input_array: NDArray[Float, 2, 2, 2], result: Public[NDArray[Float, 2, 2, 2]]):
        assert (input_array.transpose(axes=(2, 0, 1)) == result).all()

    input_array = [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ]

    assert foo_1(input_array, np.transpose(input_array, (0, 1, 2)))
    assert foo_2(input_array, np.transpose(input_array, (2, 1, 0)))
    assert foo_3(input_array, np.transpose(input_array, (1, 0, 2)))
    assert foo_4(input_array, np.transpose(input_array, (1, 2, 0)))
    assert foo_5(input_array, np.transpose(input_array, (0, 2, 1)))
    assert foo_6(input_array, np.transpose(input_array, (2, 0, 1)))


def test_transpose_3_dim_axes_specified_error():
    @zk_circuit
    def foo_1(input_array: NDArray[Float, 2, 2, 2]):
        input_array.transpose(axes=(2, 2, 2))

    @zk_circuit
    def foo_2(input_array: NDArray[Float, 2, 2, 2]):
        input_array.transpose(axes=(0, 1, 2, 3))

    @zk_circuit
    def foo_3(input_array: NDArray[Float, 2, 2, 2]):
        input_array.transpose(axes=(0, 1))

    @zk_circuit
    def foo_4(input_array: NDArray[Float, 2, 2, 2]):
        input_array.transpose(axes=(5, 6, 7))

    @zk_circuit
    def foo_5(input_array: NDArray[Float, 2, 2, 2]):
        input_array.transpose(axes=(-3, -2, -4))

    input_array = [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ]

    with pytest.raises(ZinniaException) as e:
        assert foo_1(input_array)
    assert "should be a permutation of 0 to 2" in str(e)

    with pytest.raises(ZinniaException) as e:
        foo_2(input_array)
    assert "Length of `axes` should be equal to the number of dimensions of the array" in str(e)

    with pytest.raises(ZinniaException) as e:
        foo_3(input_array)
    assert "Length of `axes` should be equal to the number of dimensions of the array" in str(e)

    with pytest.raises(ZinniaException) as e:
        foo_4(input_array)
    assert "Invalid axis value" in str(e)

    with pytest.raises(ZinniaException) as e:
        foo_5(input_array)
    assert "Invalid axis value" in str(e)


def test_transpose_negative_axes():
    import numpy as np

    @zk_circuit
    def foo_1(input_array: NDArray[Float, 2, 2, 2], result: Public[NDArray[Float, 2, 2, 2]]):
        assert (input_array.transpose(axes=(-3, -2, -1)) == result).all()

    @zk_circuit
    def foo_2(input_array: NDArray[Float, 2, 2, 2], result: Public[NDArray[Float, 2, 2, 2]]):
        assert (input_array.transpose(axes=(-1, -2, -3)) == result).all()

    @zk_circuit
    def foo_3(input_array: NDArray[Float, 2, 2, 2], result: Public[NDArray[Float, 2, 2, 2]]):
        assert (input_array.transpose(axes=(-2, -3, -1)) == result).all()

    @zk_circuit
    def foo_4(input_array: NDArray[Float, 2, 2, 2], result: Public[NDArray[Float, 2, 2, 2]]):
        assert (input_array.transpose(axes=(-2, -1, -3)) == result).all()

    @zk_circuit
    def foo_5(input_array: NDArray[Float, 2, 2, 2], result: Public[NDArray[Float, 2, 2, 2]]):
        assert (input_array.transpose(axes=(-3, -1, -2)) == result).all()

    @zk_circuit
    def foo_6(input_array: NDArray[Float, 2, 2, 2], result: Public[NDArray[Float, 2, 2, 2]]):
        assert (input_array.transpose(axes=(-1, -3, -2)) == result).all()

    input_array = [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ]

    assert foo_1(input_array, np.transpose(input_array, (-3, -2, -1)))
    assert foo_2(input_array, np.transpose(input_array, (-1, -2, -3)))
    assert foo_3(input_array, np.transpose(input_array, (-2, -3, -1)))
    assert foo_4(input_array, np.transpose(input_array, (-2, -1, -3)))
    assert foo_5(input_array, np.transpose(input_array, (-3, -1, -2)))
    assert foo_6(input_array, np.transpose(input_array, (-1, -3, -2)))
