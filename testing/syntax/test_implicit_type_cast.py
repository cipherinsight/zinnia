from zinnia import *


def test_compare_ndarray_1():
    @zk_circuit
    def foo():
        ary1 = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
        assert ary1 == [1, 2, 3, 4, 5]

    assert foo()


def test_compare_ndarray_2():
    @zk_circuit
    def foo():
        ary1 = np.asarray([1, 2, 3, 4, 5], dtype=int)
        assert ary1 == [1.0, 2.0, 3.0, 4.0, 5.0]

    assert foo()


def test_compare_ndarray_subscript_1():
    @zk_circuit
    def foo():
        ary1 = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
        assert ary1[:2] == [1, 2]

    assert foo()


def test_compare_ndarray_subscript_2():
    @zk_circuit
    def foo():
        ary1 = np.asarray([1, 2, 3, 4, 5], dtype=int)
        assert ary1[:2] == [1.0, 2.0]

    assert foo()


def test_compare_ndarray_multidim_1():
    @zk_circuit
    def foo():
        ary1 = np.asarray([[1, 2], [3, 4]], dtype=int)
        assert ary1 == [(1, 2), [3.0, 4.0]]

    assert foo()


def test_compare_ndarray_multidim_2():
    @zk_circuit
    def foo():
        ary1 = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
        assert ary1 == [(1, 2), [3.0, 4.0]]

    assert foo()


def test_ndarray_list_add_1():
    @zk_circuit
    def foo():
        ary1 = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
        assert ary1 + [[-1, -2], [-3, -4]] == 0


def test_ndarray_list_add_2():
    @zk_circuit
    def foo():
        ary1 = np.asarray([[1, 2], [3, 4]], dtype=int)
        assert ary1 + [[-1.0, -2.0], [-3.0, -4.0]] == 0
