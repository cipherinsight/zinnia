import pytest

from zinnia import *


def test_stack():
    @zk_circuit
    def foo():
        array1 = NDArray.asarray([1, 2, 3])
        array2 = NDArray.asarray([4, 5, 6])
        assert stack([array1, array2]).tolist() == [[1, 2, 3], [4, 5, 6]]
        assert stack([array1, array2], axis=1).tolist() == [[1, 4], [2, 5], [3, 6]]

    assert foo()


def test_stack_different_type():
    @zk_circuit
    def foo():
        array1 = NDArray.asarray([1, 2, 3])
        array2 = NDArray.asarray([4.5, 5.5, 6.5])
        assert stack([array1, array2]).tolist() == [[1.0, 2.0, 3.0], [4.5, 5.5, 6.5]]
        assert stack([array1, array2], axis=1).tolist() == [[1.0, 4.5], [2.0, 5.5], [3.0, 6.5]]

    assert foo()


def test_stack_axis_out_of_bound():
    @zk_circuit
    def foo_1():
        array1 = NDArray.asarray([1, 2, 3])
        array2 = NDArray.asarray([4, 5, 6])
        stack([array1, array2], axis=2)

    @zk_circuit
    def foo_2():
        array1 = NDArray.asarray([1, 2, 3])
        array2 = NDArray.asarray([4, 5, 6])
        stack([array1, array2], axis=-1)

    with pytest.raises(ZinniaException) as e:
        assert foo_1()
    assert "is out of bounds for array" in str(e)
    with pytest.raises(ZinniaException) as e:
        assert foo_2()
    assert "is out of bounds for array" in str(e)
