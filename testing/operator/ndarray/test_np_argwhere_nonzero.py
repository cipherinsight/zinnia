"""Static-array coverage for np.argwhere / np.nonzero dynamic outputs."""

from zinnia import *


def test_np_argwhere_static_1d_returns_dynamic_coordinate_cells():
    @zk_circuit
    def foo():
        coords = np.argwhere(np.asarray([0, 5, 0, 7]))
        assert coords.sum() == 4  # coordinates [1, 3]

    assert foo()


def test_np_argwhere_static_2d_returns_dynamic_coordinate_cells():
    @zk_circuit
    def foo():
        coords = np.argwhere(np.asarray([[0, 2, 3], [0, 0, 4]]))
        assert coords.sum() == 6  # rows [[0, 1], [0, 2], [1, 2]]

    assert foo()


def test_np_nonzero_static_2d_returns_per_axis_dynamic_arrays():
    @zk_circuit
    def foo():
        rows, cols = np.nonzero(np.asarray([[0, 2, 3], [0, 0, 4]]))
        assert rows.sum() == 1  # [0, 0, 1]
        assert cols.sum() == 5  # [1, 2, 2]

    assert foo()
