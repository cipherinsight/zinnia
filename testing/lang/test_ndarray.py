from zinnia import *


def test_create_ndarray_instance():
    array = NDArray.asarray([[1, 2], [3, 4]])
    assert array.shape == (2, 2)


def test_use_ndarray_as_input():
    @zk_circuit
    def foo(x: NDArray[Integer, 2, 2]):
        assert x.sum() == 10

    array = NDArray.asarray([[1, 2], [3, 4]])
    assert foo(array)


def test_ndarray_subscript_1():
    array = NDArray.ones((10, 10))
    array[:, 0] = 2
    for i in range(10):
        assert array[i, 0] == 2
    for i in range(10):
        for j in range(10):
            if j != 0:
                assert array[i, j] == 1
