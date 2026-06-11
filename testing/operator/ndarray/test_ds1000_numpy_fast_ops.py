import pytest

from zinnia import *
from zinnia.debug.exception import ZinniaException


def test_deg2rad_and_degrees_scalar():
    @zk_circuit
    def foo(x: Integer) -> Float:
        return np.degrees(np.deg2rad(x))

    proof = ZKCircuit.from_method(foo).prove(30, backend="mock")
    assert proof.proof_bytes_hex == "mock_satisfied"
    assert proof.public_values


def test_ravel_swapaxes_imag_diag_indices_static_arrays():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 2, 3], [4, 5, 6]])
        flat = a.ravel()
        assert flat.shape == (6,)
        assert flat[4] == 5

        sw = a.reshape(1, 2, 3).swapaxes(0, 2)
        assert sw.shape == (3, 2, 1)
        assert sw[2, 1, 0] == 6

        imag = a.imag
        assert imag.shape == (2, 3)
        assert imag.sum() == 0.0

        d = np.diag(a)
        assert d.shape == (2,)
        assert d[0] == 1
        assert d[1] == 5

        grid = np.indices((2, 3))
        assert grid.shape == (2, 2, 3)
        assert grid[0, 1, 2] == 1
        assert grid[1, 1, 2] == 2

    assert foo()


def test_isclose_and_array_equal_static_arrays():
    @zk_circuit
    def foo():
        a = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        b = np.asarray([[1.0, 2.001], [3.0, 4.0]])
        close = np.isclose(a, b, atol=0.01)
        assert close.all()
        assert np.array_equal(a, a)
        assert not np.array_equal(a, b)

    assert foo()


def test_isclose_equal_nan_and_arcsin_compile_under_current_numeric_model():
    source_equal_nan = """
def foo(x: Float[Array, "2"]):
    y = np.isclose(x, x, equal_nan=True)
    assert y.all()
"""
    assert ZKCircuit.from_source("foo", source_equal_nan).compile() is not None

    source_arcsin = """
def foo(x: Float):
    y = np.arcsin(x)
    return y
"""
    assert ZKCircuit.from_source("foo", source_arcsin).compile() is not None


def test_unravel_index_delete_and_nonzero_medium_static_ops():
    @zk_circuit
    def foo(x: Float[Array, "2 3"]):
        idx_c = np.unravel_index(x.argmax(), x.shape)
        idx_f = np.unravel_index(x.argmax(), x.shape, order="F")
        trimmed = np.delete(x, 1, axis=1)
        many = np.delete(np.asarray([[1, 2, 3], [4, 5, 6]]), np.array([0, 2]), axis=1)
        return idx_c, idx_f, trimmed, many

    compiled = ZKCircuit.from_method(foo).compile()
    assert compiled is not None


def test_nonzero_method_medium_dynamic_return_compiles():
    source = """
def foo(y: Integer[Array, "5"]):
    nz = (y == 3).nonzero()[0]
    return nz
"""
    compiled = ZKCircuit.from_source("foo", source).compile()
    assert compiled is not None


def test_take_along_axis_static_runtime_indices():
    @zk_circuit
    @requires(lambda b: 0 <= b <= 1)
    def foo(a: Float[Array, "2 2 2"], b: Integer[Array, "2 2"]):
        result = np.take_along_axis(a, b[..., np.newaxis], axis=-1)[..., 0]
        return result

    compiled = ZKCircuit.from_method(foo).compile()
    assert compiled is not None


def test_insert_pad_ravel_multi_index_and_in1d_static_ops():
    @zk_circuit
    @requires(lambda pos: 0 <= pos <= 4)
    def foo(a: Float[Array, "4"], pos: Integer):
        inserted = np.insert(a, pos, 9)
        padded = np.pad(np.asarray([[1, 2], [3, 4]]), ((1, 0), (0, 1)), "constant")
        idx = np.ravel_multi_index(np.array([1, 2]), (3, 4), order="C")
        mask = np.in1d(np.asarray([1, 2, 3, 4]), np.asarray([2, 4]))
        return inserted, padded, idx, mask

    compiled = ZKCircuit.from_method(foo).compile()
    assert compiled is not None


def test_pad_ravel_multi_index_and_bincount_boundaries():
    source_pad = """
@requires(lambda n: 0 <= n <= 8)
def foo(a: Float[Array, "4"], n: Integer):
    return np.pad(a, (0, n))
"""
    assert ZKCircuit.from_source("foo", source_pad).compile() is not None

    # An UNBOUNDED runtime pad width is rejected: the old behavior
    # silently capped it at 128 with a proof-time witness assert, which
    # narrowed the contract without telling the user. The compiler now
    # demands a @requires bound (soundness over convenience).
    source_pad_unbounded = """
def foo(a: Float[Array, "4"], n: Integer):
    return np.pad(a, (0, n))
"""
    with pytest.raises(ZinniaException, match="not statically bounded"):
        ZKCircuit.from_source("foo", source_pad_unbounded).compile()

    # Dynamic dims (rank statically known, values runtime) is supported:
    # strides are emitted as runtime IR products and per-axis index bounds
    # are discharged against the dim wires.
    source_ravel_multi_index_dynamic_dims = """
def foo(i: Integer[Array, "2"], dims: Integer[Array, "2"]):
    return np.ravel_multi_index(i, dims)
"""
    assert (
        ZKCircuit.from_source("foo", source_ravel_multi_index_dynamic_dims).compile()
        is not None
    )

    # Vectorized 2-D multi_index of shape (rank, N) → 1-D output of shape (N,).
    source_ravel_multi_index_vectorized = """
def foo(idx: Integer[Array, "3 4"]):
    return np.ravel_multi_index(idx, (5, 6, 7))
"""
    assert (
        ZKCircuit.from_source("foo", source_ravel_multi_index_vectorized).compile()
        is not None
    )

    # Unbounded bincount inputs are rejected (the old behavior silently
    # capped the output at 256 with a proof-time witness assert).
    source_bincount = """
def foo(x: Integer[Array, "4"]):
    return np.bincount(x)
"""
    with pytest.raises(ZinniaException, match="not statically bounded"):
        ZKCircuit.from_source("foo", source_bincount).compile()

    source_bincount_weighted = """
def foo(x: Integer[Array, "4"], w: Integer[Array, "4"]):
    return np.bincount(x, weights=w)
"""
    with pytest.raises(ZinniaException, match="not statically bounded"):
        ZKCircuit.from_source("foo", source_bincount_weighted).compile()


def test_ds1000_runtime_pad_width_bounded_dynamic_subset():
    source_right_pad = """
@requires(lambda length: 5 <= length <= 16)
def foo(A: Float[Array, "5"], length: Integer):
    return np.pad(A, (0, length - A.shape[0]), "constant")
"""
    assert ZKCircuit.from_source("foo", source_right_pad).compile() is not None

    # Unbounded target shapes are rejected — see the pad-width policy above.
    source_2d_target_shape = """
def foo(a: Float[Array, "42 10"], shape: Integer[Array, "2"], element: Integer):
    return np.pad(
        a,
        ((0, shape[0] - a.shape[0]), (0, shape[1] - a.shape[1])),
        "constant",
        constant_values=element,
    )
"""
    with pytest.raises(ZinniaException, match="not statically bounded"):
        ZKCircuit.from_source("foo", source_2d_target_shape).compile()

    source_centered = """
def foo(a: Float[Array, "42 10"], shape: Integer[Array, "2"]):
    y_, x_ = shape
    y, x = a.shape
    y_pad = y_ - y
    x_pad = x_ - x
    return np.pad(
        a,
        ((y_pad // 2, y_pad // 2 + y_pad % 2), (x_pad // 2, x_pad // 2 + x_pad % 2)),
        mode="constant",
    )
"""
    with pytest.raises(ZinniaException, match="not statically bounded"):
        ZKCircuit.from_source("foo", source_centered).compile()


def test_flatnonzero_reaches_dynamic_return_boundary():
    source = """
def foo(x: Integer[Array, "4"]):
    return np.flatnonzero(x)
"""
    compiled = ZKCircuit.from_source("foo", source).compile()
    assert compiled is not None


def test_ds1000_indexing_rows_do_not_panic_in_optimization_rebuilds():
    sources = [
        """
def foo(a: Float[Array, "8 6"], permutation: Integer[Array, "6"]):
    c = np.empty_like(permutation)
    c[permutation] = np.arange(len(permutation))
    a = a[:, c]
    return a
""",
        """
def foo(a: Float[Array, "8 6 8"], permutation: Integer[Array, "8"]):
    c = np.empty_like(permutation)
    c[permutation] = np.arange(len(permutation))
    result = a[c, :, :]
    return result
""",
        """
def foo(a: Integer[Array, "4 4"], zero_rows: Integer[Array, "2"], zero_cols: Integer[Array, "2"]):
    a[zero_rows, :] = 0
    a[:, zero_cols] = 0
    return a
""",
    ]
    for source in sources:
        compiled = ZKCircuit.from_source("foo", source).compile()
        assert compiled is not None


def test_ds1000_keepdims_integer_truthy_reduction_broadcast():
    @zk_circuit
    def foo(a: Float[Array, "10 5"]):
        max_mask = a.max(axis=1, keepdims=1) == a
        min_mask = a.min(axis=1, keepdims=1) == a
        return max_mask, min_mask

    compiled = ZKCircuit.from_method(foo).compile()
    assert compiled is not None


def test_ds1000_np_multiply_reduce_ufunc_method_lowers_to_prod():
    @zk_circuit
    @requires(lambda row: 1 <= row <= 8)
    @requires(lambda divide_number: divide_number != 0)
    def foo(a: Float[Array, "8 6"], row: Integer, divide_number: Integer):
        a[row - 1, :] /= divide_number
        return np.multiply.reduce(a[row - 1, :])

    compiled = ZKCircuit.from_method(foo).compile()
    assert compiled is not None


def test_ds1000_static_array_runtime_boolean_mask_promotes_to_dynamic_filter():
    @zk_circuit
    def mask_expr(A: Integer[Array, "20"], B: Integer[Array, "3"]):
        return A[np.logical_and(A > B[0], A < B[1]) | np.logical_and(A > B[1], A < B[2])]

    @zk_circuit
    def mask_assignment(a: Integer[Array, "20"]):
        selection = np.ones(len(a), dtype=bool)
        selection[1:] = a[1:] != a[:-1]
        selection &= a != 0
        return a[selection]

    @zk_circuit
    def mask_assignment_2d(a: Integer[Array, "20 1"]):
        selection = np.ones((len(a), 1), dtype=bool)
        selection[1:] = a[1:] != a[:-1]
        selection &= a != 0
        return a[selection].reshape(-1, 1)

    assert ZKCircuit.from_method(mask_expr).compile() is not None
    assert ZKCircuit.from_method(mask_assignment).compile() is not None
    assert ZKCircuit.from_method(mask_assignment_2d).compile() is not None
