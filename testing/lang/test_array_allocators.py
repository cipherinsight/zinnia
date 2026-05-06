from zinnia import *


def test_np_empty_int():
    @zk_circuit
    def foo():
        a = np.empty((4, 4), dtype=int)
        a[0, 0] = 7
        assert a[0, 0] == 7

    assert foo()


def test_np_empty_float():
    @zk_circuit
    def foo():
        a = np.empty((3, 3), dtype=float)
        a[1, 1] = 2.5
        assert a[1, 1] == 2.5

    assert foo()


def test_np_zeros_like_float():
    @zk_circuit
    def foo(x: NDArray[Float, 4, 4]):
        z = np.zeros_like(x)
        assert z[0, 0] == 0.0
        assert z[3, 3] == 0.0

    array = np.asarray([[1.0] * 4 for _ in range(4)])
    assert foo(array)


def test_np_ones_like_int():
    @zk_circuit
    def foo(x: NDArray[Integer, 5]):
        o = np.ones_like(x)
        assert o[0] == 1
        assert o[4] == 1

    assert foo(np.asarray([0, 0, 0, 0, 0]))


def test_np_empty_like_inherits_dtype():
    @zk_circuit
    def foo(x: NDArray[Float, 3]):
        e = np.empty_like(x)
        e[0] = 3.5
        assert e[0] == 3.5

    assert foo(np.asarray([0.0, 0.0, 0.0]))
