from zinnia import *


def test_static_shape_still_works_for_zeros_and_reshape():
    @zk_circuit
    def foo():
        arr = np.zeros((2, 3), int)
        reshaped = arr.reshape((3, 2))
        assert reshaped.shape == (3, 2)

    assert foo()


def test_dynamic_shape_zeros_auto_promotes_with_smt_max_bound():
    @zk_circuit
    def foo(flag: Integer):
        rows = 1 if flag > 0 else 2
        cols = 5 if flag > 0 else 4
        arr = np.zeros((rows, cols), int)
        # Dynamic shape gets promoted to bounded flat storage.
        assert arr.shape == (8,)

    assert foo(1)
    assert foo(0)


def test_dynamic_shape_ones_auto_promotes_with_runtime_len_gating():
    @zk_circuit
    def foo(flag: Integer):
        rows = 1 if flag > 0 else 2
        cols = 5 if flag > 0 else 4
        arr = np.ones((rows, cols), int)
        assert arr.shape == (8,)
        # ones() initializes the full bounded buffer.
        assert arr.sum() == 8

    assert foo(1)
    assert foo(0)


def test_dynamic_shape_eye_auto_promotes_with_smt_max_bound():
    @zk_circuit
    def foo(flag: Integer):
        rows = 1 if flag > 0 else 2
        cols = 5 if flag > 0 else 4
        arr = np.eye(rows, cols, int)
        assert arr.shape == (8,)
        assert arr.sum() == (1 if flag > 0 else 2)

    assert foo(1)
    assert foo(0)


def test_dynamic_shape_identity_auto_promotes_with_smt_max_bound():
    @zk_circuit
    def foo(flag: Integer):
        n = 2 if flag > 0 else 3
        arr = np.identity(n, int)
        assert arr.shape == (9,)
        assert arr.sum() == n

    assert foo(1)
    assert foo(0)


def test_static_ndarray_dynamic_axis_sum_offloads_to_dynamic_ndarray():
    @zk_circuit
    def foo(flag: Integer):
        arr = np.asarray([
            [1, 2, 4],
            [3, 5, 7],
        ])
        axis = 0 if flag > 0 else 1
        reduced = arr.sum(axis=axis)
        assert reduced[0] == (4 if flag > 0 else 7)
        assert reduced[1] == (7 if flag > 0 else 15)
        assert reduced[2] == (11 if flag > 0 else 0)

    assert foo(1)
    assert foo(0)


def test_static_ndarray_dynamic_axis_prod_offloads_to_dynamic_ndarray():
    @zk_circuit
    def foo(flag: Integer):
        arr = np.asarray([
            [1, 2, 4],
            [3, 5, 7],
        ])
        axis = 0 if flag > 0 else 1
        reduced = arr.prod(axis=axis)
        assert reduced[0] == (3 if flag > 0 else 8)
        assert reduced[1] == (10 if flag > 0 else 105)
        assert reduced[2] == (28 if flag > 0 else 0)

    assert foo(1)
    assert foo(0)


def test_static_ndarray_dynamic_axis_max_min_offloads_to_dynamic_ndarray():
    @zk_circuit
    def foo(flag: Integer):
        arr = np.asarray([
            [1, 2, 4],
            [3, 5, 7],
        ])
        axis = 0 if flag > 0 else 1

        reduced_max = arr.max(axis=axis)
        assert reduced_max[0] == (3 if flag > 0 else 4)
        assert reduced_max[1] == (5 if flag > 0 else 7)
        assert reduced_max[2] == (7 if flag > 0 else 0)

        reduced_min = arr.min(axis=axis)
        assert reduced_min[0] == (1 if flag > 0 else 1)
        assert reduced_min[1] == (2 if flag > 0 else 3)
        assert reduced_min[2] == (4 if flag > 0 else 0)

    assert foo(1)
    assert foo(0)


def test_static_ndarray_dynamic_axis_argmax_argmin_offloads_to_dynamic_ndarray():
    @zk_circuit
    def foo(flag: Integer):
        arr = np.asarray([
            [1, 2, 4],
            [3, 5, 7],
        ])
        axis = 0 if flag > 0 else 1

        reduced_argmax = arr.argmax(axis=axis)
        assert reduced_argmax[0] == (1 if flag > 0 else 2)
        assert reduced_argmax[1] == (1 if flag > 0 else 2)
        assert reduced_argmax[2] == (1 if flag > 0 else 0)

        reduced_argmin = arr.argmin(axis=axis)
        assert reduced_argmin[0] == 0
        assert reduced_argmin[1] == 0
        assert reduced_argmin[2] == 0

    assert foo(1)
    assert foo(0)


def test_static_ndarray_dynamic_axis_all_any_offloads_to_dynamic_ndarray():
    @zk_circuit
    def foo(flag: Integer):
        arr = np.asarray([
            [1, 0, 4],
            [3, 5, 0],
        ])
        axis = 0 if flag > 0 else 1

        reduced_all = arr.all(axis=axis)
        assert reduced_all[0] == (1 if flag > 0 else 0)
        assert reduced_all[1] == 0
        assert reduced_all[2] == (0 if flag > 0 else 0)

        reduced_any = arr.any(axis=axis)
        assert reduced_any[0] == 1
        assert reduced_any[1] == 1
        assert reduced_any[2] == (1 if flag > 0 else 0)

    assert foo(1)
    assert foo(0)


def test_static_ndarray_dynamic_axes_transpose_offloads_to_dynamic_ndarray():
    @zk_circuit
    def foo(flag: Integer):
        arr = np.asarray([
            [1, 2, 3],
            [4, 5, 6],
        ])
        a0 = 0 if flag > 0 else 1
        a1 = 1 if flag > 0 else 0
        t = np.transpose(arr, axes=(a0, a1))
        assert t[0] == 1
        assert t[1] == (2 if flag > 0 else 4)
        assert t[2] == (3 if flag > 0 else 2)
        assert t[3] == (4 if flag > 0 else 5)

    assert foo(1)
    assert foo(0)


def test_static_ndarray_dynamic_axis_moveaxis_offloads_to_dynamic_ndarray():
    @zk_circuit
    def foo(flag: Integer):
        arr = np.asarray([
            [1, 2, 3],
            [4, 5, 6],
        ])
        src = 0 if flag > 0 else 1
        dst = 1 if flag > 0 else 0
        moved = np.moveaxis(arr, src, dst)
        assert moved[0] == 1
        assert moved[1] == (4 if flag > 0 else 2)
        assert moved[2] == (2 if flag > 0 else 3)
        assert moved[3] == (5 if flag > 0 else 4)

    assert foo(1)
    assert foo(0)


def test_static_ndarray_dynamic_axes_transpose_method_offloads_to_dynamic_ndarray():
    @zk_circuit
    def foo(flag: Integer):
        arr = np.asarray([
            [1, 2, 3],
            [4, 5, 6],
        ])
        a0 = 0 if flag > 0 else 1
        a1 = 1 if flag > 0 else 0
        t = arr.transpose((a0, a1))  # type: ignore[attr-defined]
        assert t[0] == 1
        assert t[1] == (2 if flag > 0 else 4)
        assert t[2] == (3 if flag > 0 else 2)
        assert t[3] == (4 if flag > 0 else 5)

    assert foo(1)
    assert foo(0)


def test_static_ndarray_dynamic_axis_moveaxis_method_offloads_to_dynamic_ndarray():
    @zk_circuit
    def foo(flag: Integer):
        arr = np.asarray([
            [1, 2, 3],
            [4, 5, 6],
        ])
        src = 0 if flag > 0 else 1
        dst = 1 if flag > 0 else 0
        moved = arr.moveaxis(src, dst)  # type: ignore[attr-defined]
        assert moved[0] == 1
        assert moved[1] == (4 if flag > 0 else 2)
        assert moved[2] == (2 if flag > 0 else 3)
        assert moved[3] == (5 if flag > 0 else 4)

    assert foo(1)
    assert foo(0)


def test_dynamic_shape_transpose_moveaxis_preserve_sum_runtime_dynamic_axis():
    @zk_circuit
    def foo(shape_flag: Integer, axis_flag: Integer):
        rows = 1 if shape_flag > 0 else 2
        cols = 4 if shape_flag > 0 else 3
        arr = np.eye(rows, cols, int)

        a0 = 1 if axis_flag > 0 else 0
        a1 = 0 if axis_flag > 0 else 1
        t = np.transpose(arr, axes=(a0, a1))

        src = 0 if axis_flag > 0 else 1
        dst = 1 if axis_flag > 0 else 0
        moved = np.moveaxis(arr, src, dst)

        assert t.sum() == arr.sum()
        assert moved.sum() == arr.sum()

    assert foo(1, 1)
    assert foo(1, 0)
    assert foo(0, 1)
    assert foo(0, 0)


def test_dynamic_array_helpers_metadata_and_conversion():
    @zk_circuit
    def foo(flag: Integer):
        rows = 1 if flag > 0 else 2
        cols = 4 if flag > 0 else 3
        arr = np.eye(rows, cols, int)

        assert arr.ndim == 1
        assert arr.shape == (6,)
        assert arr.size == 6
        assert arr.dtype == int

        arr_t = arr.T
        assert arr_t.shape == arr.shape
        assert arr_t.sum() == arr.sum()

        as_float = arr.astype(float)
        assert as_float.dtype == float
        assert as_float.sum() == arr.sum()

        flat_view = arr.flat
        assert flat_view.shape == (6,)
        assert flat_view.sum() == arr.sum()

        flat_list = arr.flatten()
        assert len(flat_list) == 6

        listed = arr.tolist()
        assert len(listed) == 6

    assert foo(1)
    assert foo(0)


def test_dynamic_shape_concatenate_offloads_to_dynamic_ndarray():
    @zk_circuit
    def foo(flag: Integer):
        rows = 1 if flag > 0 else 2
        cols = 4 if flag > 0 else 3
        arr1 = np.eye(rows, cols, int)
        arr2 = np.ones((rows, cols), int)

        joined = np.concatenate([arr1, arr2], axis=0)
        assert joined.shape == (12,)
        assert joined.sum() == arr1.sum() + arr2.sum()

        joined_alias = np.concat([arr1, arr2], axis=0)
        assert joined_alias.shape == (12,)
        assert joined_alias.sum() == joined.sum()

    assert foo(1)
    assert foo(0)


def test_dynamic_shape_stack_offloads_to_dynamic_ndarray():
    @zk_circuit
    def foo(flag: Integer):
        rows = 1 if flag > 0 else 2
        cols = 4 if flag > 0 else 3
        arr1 = np.eye(rows, cols, int)
        arr2 = np.ones((rows, cols), int)

        stacked0 = np.stack([arr1, arr2], axis=0)
        assert stacked0.shape == (12,)
        assert stacked0.sum() == arr1.sum() + arr2.sum()

        stacked1 = np.stack([arr1, arr2], axis=1)
        assert stacked1.shape == (12,)
        assert stacked1.sum() == arr1.sum() + arr2.sum()
        # axis controls source interleaving pattern in bounded flat storage
        assert stacked0[1] == 0
        assert stacked1[1] == 1

    assert foo(1)
    assert foo(0)


def test_dynamic_shape_concatenate_runtime_dynamic_axis():
    @zk_circuit
    def foo(shape_flag: Integer, axis_flag: Integer):
        rows = 1 if shape_flag > 0 else 2
        cols = 4 if shape_flag > 0 else 3
        arr1 = np.eye(rows, cols, int)
        arr2 = np.ones((rows, cols), int)
        axis = 0 if axis_flag > 0 else -1

        joined = np.concatenate([arr1, arr2], axis=axis)
        assert joined.sum() == arr1.sum() + arr2.sum()

    assert foo(1, 1)
    assert foo(1, 0)
    assert foo(0, 1)
    assert foo(0, 0)


def test_dynamic_shape_stack_runtime_dynamic_axis():
    @zk_circuit
    def foo(shape_flag: Integer, axis_flag: Integer):
        rows = 1 if shape_flag > 0 else 2
        cols = 4 if shape_flag > 0 else 3
        arr1 = np.eye(rows, cols, int)
        arr2 = np.ones((rows, cols), int)
        axis = 0 if axis_flag > 0 else -1

        stacked = np.stack([arr1, arr2], axis=axis)
        assert stacked.sum() == arr1.sum() + arr2.sum()

    assert foo(1, 1)
    assert foo(1, 0)
    assert foo(0, 1)
    assert foo(0, 0)


def test_dynamic_shape_stack_broadcast_compatible_inputs():
    @zk_circuit
    def foo(flag: Integer):
        rows = 1 if flag > 0 else 2
        cols = 4 if flag > 0 else 3
        arr_dyn = np.eye(rows, cols, int)
        scalar_channel = np.asarray([9])

        stacked = np.stack([arr_dyn, scalar_channel], axis=0)
        assert stacked.shape == (12,)
        assert stacked.sum() == arr_dyn.sum() + arr_dyn.size * 9

    assert foo(1)
    assert foo(0)


def test_static_array_boolean_mask_filtering():
    @zk_circuit
    def foo():
        arr = np.asarray([1, -2, 3, 0])
        filtered = arr[arr > 0]
        assert filtered.shape == (4,)
        assert filtered[0] == 1
        assert filtered[1] == 3
        assert filtered.sum() == 4

    assert foo()


def test_dynamic_array_boolean_mask_filtering():
    @zk_circuit
    def foo(flag: Integer):
        rows = 1 if flag > 0 else 2
        cols = 4 if flag > 0 else 3
        arr = np.eye(rows, cols, int)
        filtered = arr[arr > 0]
        assert filtered.shape == (6,)
        assert filtered[0] == 1
        assert filtered[1] == (0 if flag > 0 else 1)
        assert filtered.sum() == arr.sum()

    assert foo(1)
    assert foo(0)


def test_dynamic_mask_filtering_respects_runtime_length_not_bounded_payload():
    @zk_circuit
    def foo(flag: Integer):
        rows = 1 if flag > 0 else 2
        cols = 6 if flag > 0 else 3
        dyn = np.eye(rows, cols, int)

        # Mixed static+dynamic boolean composition can produce a dynamic mask
        # whose bounded payload exceeds the filtered vector length.
        stat = np.asarray([1, 2, 3, 4, 5, 6])
        dyn_mask = np.logical_or((dyn + np.asarray([1])) > np.asarray([2]), stat > np.asarray([4]))

        picked = stat[dyn_mask]
        assert picked.shape == (6,)
        assert picked.sum() > 0
        assert picked.sum() <= stat.sum()

    assert foo(1)
    assert foo(0)


def test_static_array_dynamic_index_uses_dynamic_offload():
    @zk_circuit
    def foo(flag: Integer):
        arr = np.asarray([10, 20, 30, 40])
        idx = 1 if flag > 0 else 3
        assert arr[idx] == (20 if flag > 0 else 40)

    assert foo(1)
    assert foo(0)


def test_static_array_dynamic_slice_offloads_and_tracks_runtime_length():
    @zk_circuit
    def foo(flag: Integer):
        arr = np.asarray([10, 20, 30, 40])
        start = 1 if flag > 0 else 2
        sliced = arr[start:4:1]
        assert sliced[0] == (20 if flag > 0 else 30)
        assert sliced.sum() == (90 if flag > 0 else 70)

    assert foo(1)
    assert foo(0)


def test_dynamic_array_dynamic_index_uses_zkram_read_path():
    @zk_circuit
    def foo(flag: Integer):
        rows = 1 if flag > 0 else 2
        cols = 4 if flag > 0 else 3
        arr = np.eye(rows, cols, int)
        idx = 0 if flag > 0 else 4
        assert arr[idx] == 1

    assert foo(1)
    assert foo(0)


def test_dynamic_broadcast_scalar_arithmetic_and_compare():
    @zk_circuit
    def foo(flag: Integer):
        rows = 1 if flag > 0 else 2
        cols = 4 if flag > 0 else 3
        arr = np.eye(rows, cols, int)

        add_res = arr + 5
        mul_res = arr * 3
        cmp_res = arr > 0

        assert add_res.sum() == arr.sum() + arr.size * 5
        assert mul_res.sum() == arr.sum() * 3
        assert cmp_res.sum() == arr.sum()

    assert foo(1)
    assert foo(0)


def test_dynamic_broadcast_len1_dynamic_rhs():
    @zk_circuit
    def foo(flag: Integer):
        rows = 1 if flag > 0 else 2
        cols = 4 if flag > 0 else 3
        arr = np.eye(rows, cols, int)

        # Build a runtime-length dynamic vector with length=1.
        base = np.asarray([7, 9])
        start = 0 if flag > 0 else 0
        rhs = base[start:1:1]

        out = arr + rhs
        assert rhs.size == 1
        assert out.sum() == arr.sum() + arr.size * 7

    assert foo(1)
    assert foo(0)


def test_dynamic_logical_broadcasting():
    @zk_circuit
    def foo(flag: Integer):
        rows = 1 if flag > 0 else 2
        cols = 4 if flag > 0 else 3
        arr = np.eye(rows, cols, int)
        mask = arr > 0
        out = np.logical_and(mask, 1)
        assert out.sum() == mask.sum()

    assert foo(1)
    assert foo(0)
