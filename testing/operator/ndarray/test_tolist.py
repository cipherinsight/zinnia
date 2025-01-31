from zinnia import zk_circuit, NDArray


def test_to_list_1():
    @zk_circuit
    def foo():
        array = np.zeros((2, 2), int)
        assert array.tolist() == [[0, 0], [0, 0]]

    assert foo()


def test_to_list_2():
    @zk_circuit
    def foo():
        array = np.identity(3, int)
        assert array.tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    assert foo()
