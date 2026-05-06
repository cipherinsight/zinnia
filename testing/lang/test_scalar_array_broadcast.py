from zinnia import *


def test_int_times_2d_array():
    @zk_circuit
    def foo(a: NDArray[Integer, 3, 4]):
        b = 9 * a
        assert b[0, 0] == 9
        assert b[2, 3] == 9 * 12

    assert foo(np.asarray([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]))


def test_int_times_2d_array_rhs():
    @zk_circuit
    def foo(a: NDArray[Integer, 2, 3]):
        b = a * 5
        assert b[0, 0] == 5
        assert b[1, 2] == 30

    assert foo(np.asarray([[1, 2, 3], [4, 5, 6]]))


def test_float_times_2d_array():
    @zk_circuit
    def foo(a: NDArray[Float, 2, 2]):
        b = 4.0 * a
        assert b[0, 0] == 4.0
        assert b[1, 1] == 16.0

    assert foo(np.asarray([[1.0, 2.0], [3.0, 4.0]]))


def test_int_times_1d_array_still_works():
    # Sanity: 1-D scalar × array (the lapl3d / specialconvolve fix path)
    @zk_circuit
    def foo(a: NDArray[Integer, 4]):
        b = 3 * a
        assert b[0] == 3
        assert b[3] == 12

    assert foo(np.asarray([1, 2, 3, 4]))


def test_specialconvolve_pattern():
    # Direct repro of the specialconvolve shape that previously broke:
    # `9 * a[1:-1, 1:-1]` should give a same-shape slice, not a tiled one.
    @zk_circuit
    def foo(a: NDArray[Integer, 4, 4]):
        sliced = a[1:-1, 1:-1]
        scaled = 9 * sliced
        assert scaled[0, 0] == 9 * a[1, 1]
        assert scaled[1, 1] == 9 * a[2, 2]

    assert foo(np.asarray([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]))


def test_runtime_shape_array_creation_diagnostic():
    # np.zeros((n, m)) with runtime n must produce a clear, sourced diagnostic
    # (was: silent 0 then "chunk size must be non-zero").
    import pytest
    @zk_circuit
    def foo(d: int):
        x = np.ones((4, d))
        return x[0, 0]

    with pytest.raises(Exception) as excinfo:
        foo(3)
    msg = str(excinfo.value)
    assert "compile-time constant" in msg, f"Expected actionable diagnostic, got: {msg}"


