"""Writes through chained subscripts and through views (audit finding X4).

`y[k][j] = v` with a private `k` used to write into a temporary copy, so the
circuit accepted the unmodified array. `row = y[k]` with a private `k` is a
copy in Zinnia but a view in NumPy, so a write through either one would make
the other disagree with NumPy; reading the stale one is now rejected.
"""
import numpy as np
import pytest

from zinnia import ZKCircuit
from zinnia.debug.exception import ZinniaException
from studies.user_study.validate import checked_source

X = np.array([[1, 2, 3], [4, 5, 6]])


def compile_checked(source, shape):
    return ZKCircuit.from_source("kernel", checked_source(source, shape)).compile()


def accepts(program, *args):
    return program.prove(*args, backend="mock").proof_bytes_hex == "mock_satisfied"


def test_chained_write_with_private_first_index():
    program = compile_checked('''def kernel(x: Integer[Array, "2 3"]):
    y = np.zeros((2, 3), dtype=int)
    k = x[0, 0] % 2
    y[k][1] = 7
    return y
''', (2, 3))
    assert accepts(program, X, np.array([[0, 0, 0], [0, 7, 0]]))
    assert not accepts(program, X, np.zeros((2, 3), dtype=int))
    assert not accepts(program, X, np.array([[0, 7, 0], [0, 0, 0]]))
    x = X.copy()
    x[0, 0] = 2
    assert accepts(program, x, np.array([[0, 7, 0], [0, 0, 0]]))


def test_chained_write_with_private_inner_and_outer_index():
    program = compile_checked('''def kernel(x: Integer[Array, "2 3"]):
    y = x.copy()
    y[x[0, 0] % 2][x[1, 0] % 3] = 9
    return y
''', (2, 3))
    expected = X.copy()
    expected[1, 1] = 9
    assert accepts(program, X, expected)
    assert not accepts(program, X, X)


def test_chained_write_through_constant_views():
    program = compile_checked('''def kernel(x: Integer[Array, "2 3"]):
    y = x.copy()
    y[1][2] = 9
    y[0:2][0][1] = 8
    return y
''', (2, 3))
    assert accepts(program, X, np.array([[1, 8, 3], [4, 5, 9]]))
    assert not accepts(program, X, X)


def test_chained_write_into_a_copy_is_rejected():
    with pytest.raises(ZinniaException, match="chained subscript assignment"):
        ZKCircuit.from_source("kernel", '''def kernel(x: Integer[Array, "4"]):
    y = x.copy()
    y[::-1][0] = 9
    return y
''').compile()


def test_write_through_private_row_then_read_base_is_rejected():
    with pytest.raises(ZinniaException, match="Reading `y` is not supported"):
        ZKCircuit.from_source("kernel", '''def kernel(x: Integer[Array, "2 3"]):
    y = np.zeros((2, 3), dtype=int)
    k = x[0, 0] % 2
    row = y[k]
    row[1] = 7
    return y
''').compile()


def test_write_to_base_then_read_private_row_is_rejected():
    with pytest.raises(ZinniaException, match="Reading `row` is not supported"):
        ZKCircuit.from_source("kernel", '''def kernel(x: Integer[Array, "2 3"]):
    y = x.copy()
    k = x[0, 0] % 2
    row = y[k]
    y[k, 0] = 100
    return row
''').compile()


def test_private_row_copy_without_aliasing_writes_is_fine():
    program = compile_checked('''def kernel(x: Integer[Array, "2 3"]):
    y = np.zeros((2, 3), dtype=int)
    k = x[0, 0] % 2
    row = y[k]
    row[1] = 7
    return row
''', (3,))
    assert accepts(program, X, np.array([0, 7, 0]))
    assert not accepts(program, X, np.array([0, 0, 0]))
