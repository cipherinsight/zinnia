from zinnia import *


def test_ndarray_moveaxis():
    @zk_circuit
    def foo():
        a = np.asarray([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ])
        expected = np.asarray([
            [[1, 5], [2, 6]],
            [[3, 7], [4, 8]],
        ])
        assert (a.moveaxis(0, 2) == expected).all()  # type: ignore[attr-defined]

    assert foo()


def test_np_moveaxis():
    @zk_circuit
    def foo():
        a = np.asarray([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ])
        expected = np.asarray([
            [[1, 2], [5, 6]],
            [[3, 4], [7, 8]],
        ])
        assert (np.moveaxis(a, 0, 1) == expected).all()

    assert foo()


def test_np_transpose():
    @zk_circuit
    def foo():
        a = np.asarray([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ])
        expected = np.asarray([
            [[1, 3], [5, 7]],
            [[2, 4], [6, 8]],
        ])
        assert (np.transpose(a, axes=(2, 0, 1)) == expected).all()

    assert foo()


def test_axis_wise_prod_reduction():
    @zk_circuit
    def foo():
        a = np.asarray([
            [1, 2, 3],
            [4, 5, 6],
        ])
        assert a.prod(axis=0).tolist() == [4, 10, 18]
        assert a.prod(axis=1).tolist() == [6, 120]

    assert foo()
