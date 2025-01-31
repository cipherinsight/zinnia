from zinnia import *


def test_argmax():
    @zk_circuit
    def foo():
        array = NDArray.asarray([1, 2, 3, 4, 5])
        assert array.argmax() == 4

    assert foo()


def test_argmin():
    @zk_circuit
    def foo():
        array = NDArray.asarray([1, 2, 3, 4, 5])
        assert array.argmin() == 0

    assert foo()


def test_argmax_with_axis():
    @zk_circuit
    def foo():
        array = NDArray.asarray([[1, 2, 3], [4, 5, 6]])
        assert array.argmax(axis=0).tolist() == [1, 1, 1]
        assert array.argmax(axis=1).tolist() == [2, 2]

    assert foo()


def test_argmin_with_axis():
    @zk_circuit
    def foo():
        array = NDArray.asarray([[1, 2, 3], [4, 5, 6]])
        assert array.argmin(axis=0).tolist() == [0, 0, 0]
        assert array.argmin(axis=1).tolist() == [0, 0]

    assert foo()


def test_argmax_with_multidim_array():
    @zk_circuit
    def foo():
        array = NDArray.asarray([[1, 2, 3], [4, 5, 6]])
        assert array.argmax() == 5
        assert array.argmax(axis=-1) == 5

    assert foo()


def test_argmin_with_multidim_array():
    @zk_circuit
    def foo():
        array = NDArray.asarray([[1, 2, 3], [4, 5, 6]])
        assert array.argmin() == 0
        assert array.argmin(axis=-1) == 0

    assert foo()
