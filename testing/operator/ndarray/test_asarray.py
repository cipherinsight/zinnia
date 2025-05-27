import pytest

from zinnia import zk_circuit, NDArray, ZinniaException


def test_asarray_1():
    @zk_circuit
    def foo(x: int, y: int):
        array = np.asarray([x, y])
        assert array.shape == (2,)

    assert foo(1, 2)


def test_asarray_2():
    @zk_circuit
    def foo(x: int, y: int):
        array = np.asarray([[[x, y], [x, y]], [[x, y], [x, y]], [[x, y], [x, y]]])
        assert array.shape == (3, 2, 2)

    assert foo(1, 2)


def test_asarray_3():
    @zk_circuit
    def foo(x: int, y: int):
        array = np.asarray((([x, y], [x, y]), [[x, y], [x, y]], ((x, y), [x, y])))
        assert array.shape == (3, 2, 2)

    assert foo(1, 2)


def test_asarray_dtype_inference_1():
    @zk_circuit
    def foo(x: int, y: int):
        array = np.asarray([x, y])
        assert array.shape == (2,)
        assert array.dtype == int
        assert array.dtype != float

    assert foo(1, 2)


def test_asarray_dtype_inference_2():
    @zk_circuit
    def foo(x: int, y: float):
        array = np.asarray([x, y])
        assert array.shape == (2,)
        assert array.dtype == float
        assert array.dtype != int

    assert foo(1, 2.3)


def test_asarray_from_ndarray():
    @zk_circuit
    def foo(x: int, y: float):
        array = np.asarray([x, y])
        array = np.asarray(array)
        assert array.shape == (2,)

    assert foo(1, 2.3)


def test_asarray_bad_array():
    @zk_circuit
    def foo(x: int, y: float):
        array = np.asarray([x, y, [x, y]])
        assert array.shape == (3,)
        assert array.dtype == int

    with pytest.raises(ZinniaException) as e:
        assert foo(1, 2.3)
    assert "To convert to NDArray, all sub-lists should be of the same shape" in str(e)


def test_asarray_with_dtype():
    @zk_circuit
    def foo():
        array = np.asarray([1.5, 2.5], dtype=int)
        assert array.dtype == int
        assert array[0] == 1
        assert array[1] == 2

    assert foo()
