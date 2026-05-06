from zinnia import *


def test_np_square_int():
    @zk_circuit
    def foo(x: NDArray[Integer, 4]):
        s = np.square(x)
        assert s[0] == 1
        assert s[3] == 16

    assert foo(np.asarray([1, 2, 3, 4]))


def test_np_square_float():
    @zk_circuit
    def foo(x: NDArray[Float, 3]):
        s = np.square(x)
        assert s[1] == 4.0

    assert foo(np.asarray([1.0, 2.0, 3.0]))


def test_np_diff_1d():
    @zk_circuit
    def foo(x: NDArray[Integer, 5]):
        d = np.diff(x)
        assert d[0] == 1
        assert d[3] == 1

    assert foo(np.asarray([1, 2, 3, 4, 5]))


def test_np_diff_2d_along_last_axis():
    @zk_circuit
    def foo(x: NDArray[Integer, 2, 4]):
        d = np.diff(x)
        # diff along last axis: d[0] = [1,1,1], d[1] = [1,1,1]
        assert d[0, 0] == 1
        assert d[1, 2] == 1

    assert foo(np.asarray([[1, 2, 3, 4], [10, 11, 12, 13]]))


def test_np_diff_n_2():
    @zk_circuit
    def foo(x: NDArray[Integer, 5]):
        d = np.diff(x, 2)
        # second-order diff of [1, 4, 9, 16, 25] (squares) yields [2, 2, 2]
        assert d[0] == 2
        assert d[2] == 2

    assert foo(np.asarray([1, 4, 9, 16, 25]))


def test_np_outer_int():
    @zk_circuit
    def foo(a: NDArray[Integer, 3], b: NDArray[Integer, 4]):
        o = np.outer(a, b)
        assert o[0, 0] == 10
        assert o[2, 3] == 120

    assert foo(np.asarray([1, 2, 3]), np.asarray([10, 20, 30, 40]))


def test_np_outer_float():
    @zk_circuit
    def foo(a: NDArray[Float, 2], b: NDArray[Float, 2]):
        o = np.outer(a, b)
        assert o[0, 0] == 1.0
        assert o[1, 1] == 6.0  # 2.0 * 3.0

    assert foo(np.asarray([1.0, 2.0]), np.asarray([1.0, 3.0]))
