from zinnia import *


def test_max():
    @zk_circuit
    def foo():
        array = np.asarray([1, 2, 3, 4, 5])
        assert array.max() == 5

    assert foo()


def test_min():
    @zk_circuit
    def foo():
        array = np.asarray([1, 2, 3, 4, 5])
        assert array.min() == 1

    assert foo()


def test_max_with_axis():
    @zk_circuit
    def foo():
        array = np.asarray([[1, 2, 3], [4, 5, 6]])
        assert array.max(axis=0).tolist() == [4, 5, 6]
        assert array.max(axis=1).tolist() == [3, 6]

    assert foo()


def test_min_with_axis():
    @zk_circuit
    def foo():
        array = np.asarray([[1, 2, 3], [4, 5, 6]])
        assert array.min(axis=0).tolist() == [1, 2, 3]
        assert array.min(axis=1).tolist() == [1, 4]

    assert foo()


def test_max_with_multidim_array():
    @zk_circuit
    def foo():
        array = np.asarray([[1, 2, 3], [4, 5, 6]])
        assert array.max() == 6
        assert array.max(axis=None) == 6

    assert foo()


def test_min_with_multidim_array():
    @zk_circuit
    def foo():
        array = np.asarray([[1, 2, 3], [4, 5, 6]])
        assert array.min() == 1
        assert array.min(axis=None) == 1

    assert foo()
