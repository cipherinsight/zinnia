from zinnia import *


def test_np_pi():
    @zk_circuit
    def foo():
        x = np.pi
        assert x > 3.14
        assert x < 3.142

    assert foo()


def test_np_e():
    @zk_circuit
    def foo():
        x = np.e
        assert x > 2.71

    assert foo()


def test_dtype_float64_kwarg():
    @zk_circuit
    def foo():
        a = np.zeros((3, 3), dtype=np.float64)
        assert a[0, 0] == 0.0

    assert foo()


def test_dtype_uint32_kwarg():
    @zk_circuit
    def foo():
        a = np.zeros((4,), dtype=np.uint32)
        assert a[0] == 0

    assert foo()


def test_dtype_bool_underscore():
    @zk_circuit
    def foo():
        a = np.zeros((2,), dtype=np.bool_)
        assert a[0] == 0

    assert foo()


def test_np_ndarray_alias_of_empty():
    @zk_circuit
    def foo():
        a = np.ndarray((2, 2), dtype=np.float32)
        a[0, 0] = 1.5
        assert a[0, 0] == 1.5

    assert foo()


def test_np_array_alias_of_asarray():
    @zk_circuit
    def foo():
        a = np.array([[1, 2], [3, 4]])
        assert a[1, 1] == 4

    assert foo()
