from zinnia import *


def test_sum():
    @zk_circuit
    def foo():
        array = np.asarray([1, 2, 3, 4, 5])
        assert array.sum() == 15

    assert foo()


def test_sum_over_axis():
    @zk_circuit
    def foo():
        array = np.asarray([[1, 2, 3], [4, 5, 6]])
        assert array.sum(axis=0).tolist() == [5, 7, 9]
        assert array.sum(axis=1).tolist() == [6, 15]

    assert foo()


def test_sum_over_axis_default():
    @zk_circuit
    def foo():
        array = np.asarray([[1, 2, 3], [4, 5, 6]])
        assert array.sum() == 21
        assert array.sum() == 21

    assert foo()
