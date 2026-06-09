"""Common array operator coverage package A."""

import pytest

from zinnia import *
from zinnia.api.zk_circuit import ZKCircuit


def test_np_sort_1d_values_and_argsort_stable_ties():
    @zk_circuit
    def foo():
        x = np.asarray([3, 1, 2, 1])
        s = np.sort(x)
        p = np.argsort(x)
        assert s[0] == 1
        assert s[1] == 1
        assert s[2] == 2
        assert s[3] == 3
        assert p[0] == 1
        assert p[1] == 3
        assert p[2] == 2
        assert p[3] == 0

    assert foo()


def test_np_sort_and_argsort_2d_last_axis_are_row_local():
    @zk_circuit
    def foo():
        x = np.asarray([[3, 1, 2], [9, 7, 7]])
        s = np.sort(x, axis=-1)
        p = np.argsort(x, axis=1)
        assert s[0, 0] == 1
        assert s[0, 1] == 2
        assert s[0, 2] == 3
        assert s[1, 0] == 7
        assert s[1, 1] == 7
        assert s[1, 2] == 9
        assert p[0, 0] == 1
        assert p[0, 1] == 2
        assert p[0, 2] == 0
        assert p[1, 0] == 1
        assert p[1, 1] == 2
        assert p[1, 2] == 0

    assert foo()


def test_np_sort_rejects_deferred_axis_none_flatten_for_2d():
    source = """
def sort_axis_none():
    x = np.asarray([[3, 1], [2, 0]])
    y = np.sort(x, axis=None)
    _zinnia_result = y[0]
"""

    with pytest.raises(Exception, match="axis=None flattening is not yet implemented"):
        ZKCircuit.from_source("sort_axis_none", source).compile()


def test_np_take_static_2d_axis_and_negative_axis():
    @zk_circuit
    def foo():
        x = np.asarray([[10, 20, 30], [40, 50, 60]])
        cols = np.take(x, [2, 0], axis=1)
        rows = np.take(x, [1, 0], axis=-2)
        assert cols[0, 0] == 30
        assert cols[0, 1] == 10
        assert cols[1, 0] == 60
        assert cols[1, 1] == 40
        assert rows[0, 0] == 40
        assert rows[1, 0] == 10

    assert foo()


def test_np_take_axis_out_of_range_is_explicit():
    source = """
def take_bad_axis():
    x = np.asarray([[1, 2], [3, 4]])
    y = np.take(x, [0], axis=2)
    _zinnia_result = y.sum()
"""

    with pytest.raises(Exception, match="axis 2 is out of bounds"):
        ZKCircuit.from_source("take_bad_axis", source).compile()


def test_np_full_and_full_like_static_scalar_fill():
    @zk_circuit
    def foo():
        a = np.full((2, 3), 7, dtype=int)
        b = np.full_like(a, 4)
        c = np.full((2,), 1.5, dtype=float)
        assert a.sum() == 42
        assert b.sum() == 24
        assert c[0] == 1.5

    assert foo()


def test_np_full_bounded_dynamic_runtime_fill_compiles_without_static_content_fact():
    source = """
@requires(lambda n: 0 <= n <= 5)
def full_runtime_fill(n: int, k: int):
    arr = np.full((2, n), k, dtype=int)
    _zinnia_result = arr.sum()
"""

    program = ZKCircuit.from_source("full_runtime_fill", source).compile()
    assert len(program.get_ir_stmts()) > 0


def test_np_full_rejects_explicit_complex_dtype():
    source = """
def full_complex_dtype():
    arr = np.full((2,), 1, dtype=complex)
    _zinnia_result = arr[0]
"""

    with pytest.raises(Exception, match="Complex fill is not yet implemented"):
        ZKCircuit.from_source("full_complex_dtype", source).compile()


def test_np_full_like_rejects_explicit_complex_dtype():
    source = """
def full_like_complex_dtype():
    base = np.asarray([1, 2])
    arr = np.full_like(base, 1, dtype=complex)
    _zinnia_result = arr[0]
"""

    with pytest.raises(Exception, match="Complex fill is not yet implemented"):
        ZKCircuit.from_source("full_like_complex_dtype", source).compile()
