"""Arrays as SSA versions (`docs/array-ssa.md`): writes, reads after writes and
views equal NumPy on the mock, and halo2-base and Noir prove the mock's values.

Covers the wrong outputs the array-ssa inventory found (B1-B9 in the design
document), each of which used to verify:

- B1: an in-place kernel writing at a runtime index exposed no output;
- B2, B3: a masked write through a transposed or column view wrote the wrong
  cells (the view's strides were ignored);
- B4: a row read at a runtime index of a transposed view read a row of the base;
- B5: `flatten` returned a view (NumPy copies);
- B6: `reshape` of a contiguous view at an offset copied (NumPy returns a view);
- B7, B8: a non-integral `np.full` array read at a runtime index read a
  truncated fill value;
- B9: a complex `out=` through a shared view failed to compile.

and the write families under a runtime execution predicate (the branch not
taken changes nothing), in loops, with negative and out-of-range indices.
"""
import numpy as np
import pytest

from testing.lang.all_backends import assert_all_backends_match_numpy
from testing.lang.mock_outputs import compile_kernel, numpy_in_place_outputs, run_mock_layout

M34 = np.arange(12, dtype=np.int64).reshape(3, 4)
F33 = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0]])
X6 = np.array([10, 20, 30, 40, 50, 60], dtype=np.int64)


# ── Views (B2-B6) ────────────────────────────────────────────────────────

VIEWS = {
    "mask write through a transposed view (B2)":
        ('def kernel(m: Float[Array, "3 3"]):\n    t = m.T\n    t[t < 0] = 0.0\n    return m\n', (F33,)),
    "mask write through a column view (B3)":
        ('def kernel(m: Float[Array, "3 3"]):\n    c = m[:, 1]\n    c[c < 0] = 0.0\n    return m\n', (F33,)),
    "runtime row of a transposed view (B4)":
        ('def kernel(m: Integer[Array, "3 4"], i: Integer):\n    t = m.T\n    return t[i]\n', (M34, 2)),
    "flatten copies (B5)":
        ('def kernel(x: Integer[Array, "3 4"]):\n    f = x.flatten()\n    f[0] = 99\n    return x, f\n', (M34,)),
    "ravel of a contiguous array is a view":
        ('def kernel(x: Integer[Array, "3 4"]):\n    f = x.ravel()\n    f[5] = 99\n    return x\n', (M34,)),
    "reshape of an offset view writes through (B6)":
        ('def kernel(x: Integer[Array, "6"]):\n    y = x[2:6].reshape(2, 2)\n    y[0, 1] = 99\n    return x\n', (X6,)),
    "reshape of an offset view sees writes to the base (B6)":
        ('def kernel(x: Integer[Array, "6"]):\n    y = x[2:6].reshape(2, 2)\n    x[3] = 99\n    return y\n', (X6,)),
    "runtime write through a slice view":
        ('def kernel(x: Integer[Array, "6"], i: Integer):\n    y = x[2:5]\n    y[i] = 77\n    return x[i + 2] + x[3], x\n', (X6, 1)),
    "runtime write through a transposed view":
        ('def kernel(m: Integer[Array, "3 4"], i: Integer, j: Integer):\n    t = m.T\n    t[i, j] = -1\n    return m\n', (M34, 3, 1)),
}


@pytest.mark.parametrize("name", list(VIEWS))
def test_views_match_numpy_on_every_backend(name):
    source, args = VIEWS[name]
    assert_all_backends_match_numpy(source, *args)


# ── Fill values (B7, B8) and complex out= (B9) ──────────────────────────

def test_a_non_integral_fill_is_read_exactly_at_a_runtime_index():
    source = 'def kernel(i: Integer):\n    a = np.full(5, 2.5)\n    a[1] = 0.25\n    return a[i]\n'
    for i in (0, 1, 4):
        assert_all_backends_match_numpy(source, i)


def test_a_complex_out_through_a_view_writes_both_parts():
    source = ('def kernel(a: Float[Array, "4"]):\n'
              '    z = np.zeros(4, dtype=complex)\n'
              '    w = z[1:3]\n'
              '    np.multiply(a[:2], 1j, out=w)\n'
              '    return z.real, z.imag\n')
    assert_all_backends_match_numpy(source, np.array([1.0, -2.0, 3.0, 4.0]))


# ── In-place kernels (B1) ────────────────────────────────────────────────

def test_an_in_place_write_at_a_runtime_index_is_an_output():
    source = 'def kernel(A: Integer[Array, "6"], k: Integer):\n    A[k] = 3\n'
    program = compile_kernel(source)
    assert [part["output"]["argument"] for part in program.get_output_layout()] == ["A"]
    for k in (0, 4, -1):
        satisfied, values, _ = run_mock_layout(program, X6, k)
        expected, _ = numpy_in_place_outputs(source, ["A"], X6, k)
        assert satisfied and values == expected


# ── Writes under a runtime predicate, in loops, at any index ─────────────

WRITES = {
    "element write in a branch":
        ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean):\n'
         '    if c:\n        x[i] = 99\n    return x\n'),
    "row write in a branch":
        ('def kernel(m: Integer[Array, "3 4"], i: Integer, c: Boolean):\n'
         '    if c:\n        m[i, :] = 7\n    return m\n'),
    "column update in a branch":
        ('def kernel(m: Integer[Array, "3 4"], i: Integer, c: Boolean):\n'
         '    if c:\n        m[:, i] *= 3\n    return m\n'),
    "mask write in a branch":
        ('def kernel(m: Integer[Array, "3 4"], i: Integer, c: Boolean):\n'
         '    if c:\n        m[m > i] = 0\n    return m\n'),
    "writes in a loop with a break":
        ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean):\n'
         '    for k in range(6):\n'
         '        if x[k] > 10 * i:\n            break\n'
         '        x[(i + k) % 6] = x[k] + 1\n'
         '    return x\n'),
    "reads after writes at runtime indices":
        ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean):\n'
         '    s = 0\n'
         '    for k in range(3):\n'
         '        x[(i + k) % 6] = x[(i + k + 1) % 6] * 2\n'
         '        s = s + x[i] + x[k]\n'
         '    return s, x\n'),
    "two runtime axes":
        ('def kernel(m: Integer[Array, "3 4"], i: Integer, c: Boolean):\n'
         '    m[i % 3, (i + 1) % 4] = 50\n'
         '    if c:\n        m[(i + 1) % 3, i % 4] += m[i % 3, (i + 1) % 4]\n'
         '    return m\n'),
    "runtime-bounded slice write":
        ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean):\n'
         '    if c:\n        x[i:i + 2] = 5\n    return x\n'),
}


@pytest.mark.parametrize("name", list(WRITES))
@pytest.mark.parametrize("i, c", [(1, False), (1, True), (2, True)])
def test_writes_match_numpy_on_every_backend(name, i, c):
    source = WRITES[name]
    arg = M34 if "m: Integer" in source else X6
    assert_all_backends_match_numpy(source, arg, i, c)


def test_negative_runtime_indices_count_from_the_end():
    source = 'def kernel(x: Integer[Array, "6"], i: Integer):\n    x[i] = 99\n    return x, x[i] + x[-1]\n'
    for i in (-1, -6, 2):
        assert_all_backends_match_numpy(source, X6, i)


def test_an_out_of_range_runtime_write_has_no_witness():
    # NumPy raises IndexError; the circuit has no witness. Under a branch not
    # taken the write does not run, and any index is accepted.
    program = compile_kernel('def kernel(x: Integer[Array, "6"], i: Integer):\n    x[i] = 99\n    return x\n')
    for i in (6, -7, 100):
        assert not run_mock_layout(program, X6, i)[0]
    program = compile_kernel('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean):\n'
                             '    if c:\n        x[i] = 99\n    return x\n')
    assert run_mock_layout(program, X6, 100, False)[0]
    assert not run_mock_layout(program, X6, 100, True)[0]
