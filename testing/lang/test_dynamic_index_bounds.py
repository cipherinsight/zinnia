"""Runtime indices must be bounds-checked on every axis (audit findings M1, M2).

Before the fix a dynamic read of a static N-D array checked only the flat
address, so ``a[0, 5]`` on a 4x4 array read ``a[1, 1]``; a dynamic row read
``a[i]`` checked nothing; a runtime index into a Python list returned the
last element when out of range, and a runtime list assignment out of range
was ignored. NumPy and Python raise ``IndexError`` in all these cases, so the
circuit must reject such a witness (the mock prover reports the failed
assertion; halo2-base refuses to prove).

In-range reads (including negative indices) must still match NumPy.
"""
import numpy as np
import pytest

from testing.lang.mock_outputs import assert_matches_numpy, compile_kernel, run_mock

M = np.arange(16).reshape(4, 4)

ELEMENT = 'def kernel(a: Integer[Array, "4 4"], i: Integer, j: Integer):\n    return a[i, j]\n'
ROW = 'def kernel(a: Integer[Array, "4 4"], i: Integer):\n    return a[i]\n'
ROW_VIEW = 'def kernel(a: Integer[Array, "4 4"], i: Integer):\n    return a[1][i]\n'
COLUMN_VIEW = 'def kernel(a: Integer[Array, "4 4"], i: Integer):\n    return a[:, 1][i]\n'
ROW_SLICE = 'def kernel(a: Integer[Array, "4 4"], i: Integer):\n    return a[i, 1:3]\n'
FANCY = 'def kernel(a: Integer[Array, "4 4"], i: Integer):\n    return a[[0, i], 2]\n'
LIST = 'def kernel(i: Integer):\n    xs = [0, 2, 4]\n    return xs[i]\n'
FLOAT_LIST = 'def kernel(i: Integer):\n    xs = [0.5, 1.5, 2.5]\n    return xs[i]\n'
NESTED_LIST = 'def kernel(i: Integer):\n    xs = [[1, 2], [3, 4], [5, 6]]\n    return xs[i][1:]\n'
LIST_STORE = ('def kernel(i: Integer):\n    xs = [0, 2, 4]\n    xs[i] = 9\n'
              '    return xs[0] + xs[1] + xs[2]\n')


@pytest.mark.parametrize("source, args", [
    (ELEMENT, (M, 1, 2)),
    (ELEMENT, (M, -1, 2)),
    (ELEMENT, (M, 3, -4)),
    (ROW, (M, 2)),
    (ROW, (M, -1)),
    (ROW_VIEW, (M, -4)),
    (COLUMN_VIEW, (M, 2)),
    (COLUMN_VIEW, (M, -1)),
    (ROW_SLICE, (M, -2)),
    (FANCY, (M, -1)),
    (LIST, (-2,)),
    (FLOAT_LIST, (1,)),
    (FLOAT_LIST, (-3,)),
    (NESTED_LIST, (-1,)),
    (LIST_STORE, (-1,)),
])
def test_in_range_index_matches_numpy(source, args):
    assert_matches_numpy(source, *args)


@pytest.mark.parametrize("source, args", [
    (ELEMENT, (M, 0, 5)),    # flat address 5 is inside the array
    (ELEMENT, (M, 1, -5)),
    (ELEMENT, (M, 4, 0)),
    (ELEMENT, (M, -5, 0)),
    (ROW, (M, 4)),
    (ROW, (M, -5)),
    (ROW_VIEW, (M, -5)),     # would read row 0 through the view's offset
    (COLUMN_VIEW, (M, 4)),
    (ROW_SLICE, (M, 7)),
    (FANCY, (M, 4)),
    (LIST, (3,)),
    (LIST, (-4,)),
    (FLOAT_LIST, (7,)),
    (NESTED_LIST, (3,)),
    (LIST_STORE, (3,)),
])
def test_out_of_range_index_is_rejected(source, args):
    with pytest.raises(IndexError):
        from testing.lang.mock_outputs import numpy_kernel
        numpy_kernel(source)(*args)
    satisfied, _ = run_mock(compile_kernel(source), *args)
    assert not satisfied, "an out-of-range index must not prove"


def test_halo2_base_refuses_out_of_range_element_read():
    program = compile_kernel(ELEMENT)
    params = {"k": 12, "precision_bits": 32, "lookup_bits": 8}
    program.prove(M, 1, 2, backend="halo2-base", params=params)  # honest: proves
    with pytest.raises(RuntimeError, match="Witness does not satisfy"):
        program.prove(M, 0, 5, backend="halo2-base", params=params)


@pytest.mark.parametrize("body", ["return a[0, 5]", "return a[-5]", "return a[1][4]"])
def test_out_of_range_literal_index_is_a_compile_error(body):
    with pytest.raises(Exception, match="out of bounds"):
        compile_kernel(f'def kernel(a: Integer[Array, "4 4"]):\n    {body}\n')
