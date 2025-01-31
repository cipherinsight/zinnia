from zinnia import *


def test_ndarray_compare_different_dtype_1():
    @zk_circuit
    def foo():
        array1 = np.asarray([1, 2, 3])
        array2 = np.asarray([1.5, 2.5, 3.5])
        assert (array1 < array2).all()

    foo()


def test_ndarray_compare_different_dtype_2():
    @zk_circuit
    def foo():
        array1 = np.asarray([1, 2, 3])
        array2 = np.asarray([1.5, 2.5, 3.5])
        assert (array1 != array2).all()

    foo()


def test_ndarray_compare_different_dtype_3():
    @zk_circuit
    def foo():
        array1 = np.asarray([1, 2, 3])
        array2 = [1.5, 2.5, 3.5]
        assert (array1 != array2).all()

    foo()
