import pytest

from zinnia import *


def test_concat():
    @zk_circuit
    def foo():
        array1 = np.asarray([[1, 2], [3, 4]])
        array2 = np.asarray([[5, 6], [7, 8]])
        assert np.concatenate([array1, array2]).tolist() == [[1, 2], [3, 4], [5, 6], [7, 8]]
        assert np.concatenate([array1, array2], axis=1).tolist() == [[1, 2, 5, 6], [3, 4, 7, 8]]

    assert foo()


def test_concat_different_type():
    @zk_circuit
    def foo():
        array1 = np.asarray([[1, 2], [3, 4]])
        array2 = np.asarray([[5.5, 6.5], [7.5, 8.5]])
        assert np.concatenate((array1, array2)).tolist() == [[1.0, 2.0], [3.0, 4.0], [5.5, 6.5], [7.5, 8.5]]

    assert foo()


def test_concat_invalid_axis():
    @zk_circuit
    def foo_1():
        array1 = np.asarray([[1, 2], [3, 4]])
        array2 = np.asarray([[5, 6], [7, 8]])
        np.concatenate([array1, array2], axis=-1)

    @zk_circuit
    def foo_2():
        array1 = np.asarray([[1, 2], [3, 4]])
        array2 = np.asarray([[5, 6], [7, 8]])
        np.concatenate([array1, array2], axis=2)

    with pytest.raises(ZinniaException) as e:
        assert foo_1()
    assert "out of bounds for array with" in str(e)
    with pytest.raises(ZinniaException) as e:
        assert foo_2()
    assert "out of bounds for array with" in str(e)
