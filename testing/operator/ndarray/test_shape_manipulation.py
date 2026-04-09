"""
Static-NDArray shape-manipulation coverage:

  - generalized stack/concatenate (axis > 1, negative axes)
  - swapaxes / moveaxis / transpose
  - flip / flipud / fliplr / rot90
  - squeeze / expand_dims / broadcast_to / atleast_1d/2d/3d
  - tile / repeat / reshape
  - vstack / hstack / dstack / column_stack / row_stack

These pin down behaviour for both newly-added ops and ops that previously
existed but had no test coverage (moveaxis, reshape, repeat, axis>1 stack).
"""

from zinnia import *


# ───────────────────────────────────────────────────────────────────────
# stack / concatenate — extended axes
# ───────────────────────────────────────────────────────────────────────

def test_concatenate_axis_minus_one():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2], [3, 4]])
        b = np.asarray([[5, 6], [7, 8]])
        out = np.concatenate([a, b], axis=-1)
        assert (out == np.asarray([[1, 2, 5, 6], [3, 4, 7, 8]])).all()

    assert foo()


def test_concatenate_axis_2_three_d():
    @zk_circuit
    def foo():
        a = np.asarray([[[1, 2], [3, 4]]])      # (1, 2, 2)
        b = np.asarray([[[5, 6], [7, 8]]])      # (1, 2, 2)
        out = np.concatenate([a, b], axis=2)     # (1, 2, 4)
        assert (out == np.asarray([[[1, 2, 5, 6], [3, 4, 7, 8]]])).all()

    assert foo()


def test_stack_axis_2_three_d():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2], [3, 4]])
        b = np.asarray([[5, 6], [7, 8]])
        out = np.stack([a, b], axis=2)           # (2, 2, 2)
        assert (out == np.asarray([
            [[1, 5], [2, 6]],
            [[3, 7], [4, 8]],
        ])).all()

    assert foo()


def test_stack_negative_axis():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        b = np.asarray([4, 5, 6])
        out = np.stack([a, b], axis=-1)
        assert (out == np.asarray([[1, 4], [2, 5], [3, 6]])).all()

    assert foo()


# ───────────────────────────────────────────────────────────────────────
# Permutation: swapaxes / moveaxis / transpose
# ───────────────────────────────────────────────────────────────────────

def test_swapaxes_2d():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        out = np.swapaxes(a, 0, 1)
        assert (out == np.asarray([[1, 4], [2, 5], [3, 6]])).all()

    assert foo()


def test_swapaxes_method_3d():
    @zk_circuit
    def foo():
        a = np.asarray([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ])  # (2, 2, 2)
        out = a.swapaxes(0, 2)
        assert (out == np.asarray([
            [[1, 5], [3, 7]],
            [[2, 6], [4, 8]],
        ])).all()

    assert foo()


def test_moveaxis_basic():
    @zk_circuit
    def foo():
        a = np.asarray([[[1, 2, 3], [4, 5, 6]]])  # (1, 2, 3)
        out = a.moveaxis(0, 2)                     # (2, 3, 1)
        assert (out == np.asarray([[[1], [2], [3]], [[4], [5], [6]]])).all()

    assert foo()


def test_transpose_function_form():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        out = np.transpose(a)
        assert (out == np.asarray([[1, 4], [2, 5], [3, 6]])).all()

    assert foo()


# ───────────────────────────────────────────────────────────────────────
# flip / flipud / fliplr / rot90
# ───────────────────────────────────────────────────────────────────────

def test_flip_default_all_axes():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        out = np.flip(a)
        assert (out == np.asarray([[6, 5, 4], [3, 2, 1]])).all()

    assert foo()


def test_flip_axis_0():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        out = np.flip(a, axis=0)
        assert (out == np.asarray([[4, 5, 6], [1, 2, 3]])).all()

    assert foo()


def test_flip_axis_1():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        out = np.flip(a, axis=1)
        assert (out == np.asarray([[3, 2, 1], [6, 5, 4]])).all()

    assert foo()


def test_flipud_fliplr():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2], [3, 4]])
        assert (np.flipud(a) == np.asarray([[3, 4], [1, 2]])).all()
        assert (np.fliplr(a) == np.asarray([[2, 1], [4, 3]])).all()

    assert foo()


def test_rot90_once():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2], [3, 4]])
        # 90° CCW: [[1,2],[3,4]] -> [[2,4],[1,3]]
        out = np.rot90(a)
        assert (out == np.asarray([[2, 4], [1, 3]])).all()

    assert foo()


def test_rot90_four_times_is_identity():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        out = np.rot90(a, k=4)
        assert (out == a).all()

    assert foo()


# ───────────────────────────────────────────────────────────────────────
# squeeze / expand_dims / broadcast_to / atleast_Nd
# ───────────────────────────────────────────────────────────────────────

def test_squeeze_all():
    @zk_circuit
    def foo():
        a = np.asarray([[[1, 2, 3]]])     # (1, 1, 3)
        out = np.squeeze(a)
        assert (out == np.asarray([1, 2, 3])).all()

    assert foo()


def test_squeeze_specific_axis():
    @zk_circuit
    def foo():
        a = np.asarray([[[1], [2], [3]]])   # (1, 3, 1)
        out = np.squeeze(a, axis=0)          # (3, 1)
        assert (out == np.asarray([[1], [2], [3]])).all()

    assert foo()


def test_expand_dims_front_and_back():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        front = np.expand_dims(a, 0)
        back = np.expand_dims(a, 1)
        assert (front == np.asarray([[1, 2, 3]])).all()
        assert (back == np.asarray([[1], [2], [3]])).all()

    assert foo()


def test_broadcast_to_basic():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        out = np.broadcast_to(a, (4, 3))
        assert (out == np.asarray([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])).all()

    assert foo()


def test_atleast_2d_promotes_1d():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        out = np.atleast_2d(a)
        assert (out == np.asarray([[1, 2, 3]])).all()

    assert foo()


def test_atleast_3d_promotes_1d():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        out = np.atleast_3d(a)
        assert (out == np.asarray([[[1, 2, 3]]])).all()

    assert foo()


# ───────────────────────────────────────────────────────────────────────
# tile / repeat / reshape
# ───────────────────────────────────────────────────────────────────────

def test_tile_int_reps_1d():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        out = np.tile(a, 3)
        assert (out == np.asarray([1, 2, 3, 1, 2, 3, 1, 2, 3])).all()

    assert foo()


def test_tile_tuple_reps_2d():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2], [3, 4]])
        out = np.tile(a, (2, 3))
        expected = np.asarray([
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
        ])
        assert (out == expected).all()

    assert foo()


def test_tile_promotes_lower_rank():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        out = np.tile(a, (2, 1))
        assert (out == np.asarray([[1, 2, 3], [1, 2, 3]])).all()

    assert foo()


def test_repeat_no_axis_flattens():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2], [3, 4]])
        out = a.repeat(2)
        assert (out == np.asarray([1, 1, 2, 2, 3, 3, 4, 4])).all()

    assert foo()


def test_repeat_axis_0():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2], [3, 4]])
        out = a.repeat(2, axis=0)
        assert (out == np.asarray([[1, 2], [1, 2], [3, 4], [3, 4]])).all()

    assert foo()


def test_reshape_with_minus_one():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        out = a.reshape(3, -1)
        assert (out == np.asarray([[1, 2], [3, 4], [5, 6]])).all()

    assert foo()


# ───────────────────────────────────────────────────────────────────────
# vstack / hstack / dstack / column_stack / row_stack
# ───────────────────────────────────────────────────────────────────────

def test_vstack_1d_inputs():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        b = np.asarray([4, 5, 6])
        out = np.vstack([a, b])
        assert (out == np.asarray([[1, 2, 3], [4, 5, 6]])).all()

    assert foo()


def test_vstack_2d_inputs():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2], [3, 4]])
        b = np.asarray([[5, 6], [7, 8]])
        out = np.vstack([a, b])
        assert (out == np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]])).all()

    assert foo()


def test_hstack_1d_inputs():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        b = np.asarray([4, 5, 6])
        out = np.hstack([a, b])
        assert (out == np.asarray([1, 2, 3, 4, 5, 6])).all()

    assert foo()


def test_hstack_2d_inputs():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2], [3, 4]])
        b = np.asarray([[5, 6], [7, 8]])
        out = np.hstack([a, b])
        assert (out == np.asarray([[1, 2, 5, 6], [3, 4, 7, 8]])).all()

    assert foo()


def test_column_stack_1d_inputs():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        b = np.asarray([4, 5, 6])
        out = np.column_stack([a, b])
        assert (out == np.asarray([[1, 4], [2, 5], [3, 6]])).all()

    assert foo()


def test_row_stack_alias():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        b = np.asarray([4, 5, 6])
        out = np.row_stack([a, b])
        assert (out == np.asarray([[1, 2, 3], [4, 5, 6]])).all()

    assert foo()


def test_dstack_1d_inputs():
    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        b = np.asarray([4, 5, 6])
        out = np.dstack([a, b])
        # (1, 3, 2)
        assert (out == np.asarray([[[1, 4], [2, 5], [3, 6]]])).all()

    assert foo()


def test_dstack_2d_inputs():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2], [3, 4]])
        b = np.asarray([[5, 6], [7, 8]])
        out = np.dstack([a, b])
        # (2, 2, 2)
        assert (out == np.asarray([
            [[1, 5], [2, 6]],
            [[3, 7], [4, 8]],
        ])).all()

    assert foo()
