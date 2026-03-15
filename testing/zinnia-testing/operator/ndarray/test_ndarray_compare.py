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


def test_ndarray_chained_compare_compiles_and_executes():
    @zk_circuit
    def foo():
        array = np.asarray([8000, 9200, 9800, 6100])
        lower = 9000
        upper = 10000
        flags = lower <= array < upper
        assert flags.tolist() == [0, 1, 1, 0]

    foo()
