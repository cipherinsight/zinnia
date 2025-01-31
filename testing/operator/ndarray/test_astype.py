from zinnia import zk_circuit, NDArray


def test_astype_1():
    @zk_circuit
    def foo(x: int, y: int):
        array = np.asarray([x, y])
        assert array.dtype == int
        array = array.astype(float)
        assert array.dtype == float

    assert foo(1, 2)


def test_astype_2():
    @zk_circuit
    def foo(x: float, y: float):
        array = np.asarray([x, y])
        assert array.dtype == float
        array = array.astype(float)
        assert array.dtype == float
        assert (array == [x, y]).all()

    assert foo(1.5, 2.5)


def test_astype_3():
    @zk_circuit
    def foo(x: float, y: float):
        array = np.asarray([x, y])
        assert array.dtype == float
        array = array.astype(int)
        assert array.dtype == int
        assert all(array == [int(1.5), int(2.5)])

    assert foo(1.5, 2.5)
