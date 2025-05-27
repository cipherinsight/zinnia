from zinnia import zk_circuit, NDArray


def test_sum():
    @zk_circuit
    def foo():
        array = np.asarray([1, 2, 3, 4, 5])
        assert sum(array) == 15

    assert foo()


def test_sum_with_start():
    @zk_circuit
    def foo():
        array = np.asarray([1, 2, 3, 4, 5])
        assert sum(array, 10) == 25

    assert foo()


def test_sum_over_ndarray():
    @zk_circuit
    def foo():
        array = np.asarray([[1, 2], [3, 4]])
        assert sum(array, 10).tolist() == [14, 16]

    assert foo()
