"""Regression tests for the soundness review of array-ssa
(`eval-redo/layer/array-ssa-review/REVIEW.md`): every reproducer, on the mock,
halo2-base and Noir.

- F1: a write under a runtime condition to an array that was rebound under the
  same condition ran where the condition is false (the merged array holds the
  old cells there). With `c = False` the program must return the old array.
- F4a: functions NumPy answers with a view of their argument. A real view where
  Zinnia can give one (`np.swapaxes`, `np.asarray`, `np.atleast_1d/2d`); where
  it copies (`np.flip`, `np.fliplr`, `np.flipud`, `np.rot90`), a write through
  either array is a compile error, never a result that misses the write.
- F4b: runtime-shaped arrays: `.copy()` and `np.array(y)` copy; `.T` of a
  runtime-strided array is rejected when written through either name.
- F5: the NumPy reference keeps scalar arguments scalars, and a program that
  used to hang the compiler compiles.

F2 (the RAM trace's per-memory bound) is a MockProver test in Rust:
`prove::halo2_base::synthesizer::tests::ram_trace_rejects_a_position_outside_its_own_memory`.
"""
import time

import numpy as np
import pytest

from testing.lang.all_backends import assert_all_backends_match_numpy
from testing.lang.mock_outputs import compile_kernel, numpy_outputs
from zinnia.debug.exception.base import ZinniaException

X6 = np.array([10, 20, 30, 40, 50, 60], dtype=np.int64)
M34 = np.arange(12, dtype=np.int64).reshape(3, 4)
F3 = np.array([0.5, 1.5, 2.5])
F6 = np.array([1.5, -2.0, 3.25, -4.0, 5.0, 0.5])
M34F = np.array([[1.0, -2.0, 3.0, 0.5], [-4.0, 5.0, -6.0, 2.0], [7.0, -8.0, 9.0, -1.5]])

# ── F1: writes after a rebinding under the same condition ────────────────

REBIND = 'def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean):\n    t = x[:3] * 1\n    if c:\n        t = x[3:] * 1\n'
F1 = {
    "minimal (constant index, fresh arrays)":
        ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean):\n    t = np.zeros(3, dtype=int)\n'
         '    if c:\n        t = np.ones(3, dtype=int)\n        t[1] = 7\n    return t\n', X6),
    "constant index": (REBIND + '        t[1] = 7\n    return t\n', X6),
    "runtime index": (REBIND + '        t[i] = 7\n    return t\n', X6),
    "augmented, constant index": (REBIND + '        t[1] += 7\n    return t\n', X6),
    "augmented, runtime index": (REBIND + '        t[i] += 7\n    return t\n', X6),
    "slice": (REBIND + '        t[1:] = 7\n    return t\n', X6),
    "fancy": (REBIND + '        t[np.array([0, 2])] = 7\n    return t\n', X6),
    "whole array": (REBIND + '        t += 7\n    return t\n', X6),
    "mask": (REBIND + '        t[t > 45] = 7\n    return t\n', X6),
    "out=":
        ('def kernel(x: Float[Array, "3"], i: Integer, c: Boolean):\n    t = x * 1.0\n'
         '    if c:\n        t = x + 1.0\n        np.add(t, 1.0, out=t)\n    return t\n', F3),
    "float, runtime index":
        ('def kernel(x: Float[Array, "3"], i: Integer, c: Boolean):\n    t = x * 1.0\n'
         '    if c:\n        t = x + 1.0\n        t[i] = 0.25\n    return t\n', F3),
    "else branch":
        ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean):\n    t = x[:3] * 1\n'
         '    if c:\n        t[0] = 1\n    else:\n        t = x[3:] * 1\n        t[2] = 9\n    return t\n', X6),
    "nested if":
        ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean):\n    t = x[:3] * 1\n'
         '    if c:\n        if x[0] > 5:\n            t = x[3:] * 1\n            t[i] = 9\n    return t\n', X6),
    "2-D row":
        ('def kernel(m: Integer[Array, "3 4"], i: Integer, c: Boolean):\n    t = m * 1\n'
         '    if c:\n        t = m + 1\n        t[i, :] = 0\n    return t\n', M34),
    "2-D column":
        ('def kernel(m: Integer[Array, "3 4"], i: Integer, c: Boolean):\n    t = m * 1\n'
         '    if c:\n        t = m + 1\n        t[:, i] = -5\n    return t\n', M34),
    "runtime-shaped":
        ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean):\n    y = x[x > 0]\n'
         '    if c:\n        y = x[x > 30]\n        y[0] = 100\n    return y\n', X6),
    "runtime-shaped, runtime index":
        ('def kernel(x: Integer[Array, "6"], i: Integer, c: Boolean):\n    y = x[x > 0]\n'
         '    if c:\n        y = x[x > 30]\n        y[i] = 100\n    return y\n', X6),
}


@pytest.mark.parametrize("name", list(F1))
@pytest.mark.parametrize("c", [False, True])
def test_a_write_after_a_conditional_rebinding_keeps_its_condition(name, c):
    source, x = F1[name]
    assert_all_backends_match_numpy(source, x, 1, c)


LOOPS = {
    "break, runtime index":
        'def kernel(x: Integer[Array, "6"], i: Integer):\n    t = np.zeros(6, dtype=int)\n    for k in range(4):\n'
        '        if x[k] > 25:\n            break\n        t = np.zeros(6, dtype=int)\n        t[(i + k) % 6] = k + 1\n    return t\n',
    "break, constant index":
        'def kernel(x: Integer[Array, "6"], i: Integer):\n    t = np.zeros(3, dtype=int)\n    for k in range(4):\n'
        '        if x[k] > 25:\n            break\n        t = np.zeros(3, dtype=int)\n        t[1] = k + 1\n    return t\n',
    "break, copy":
        'def kernel(x: Integer[Array, "6"], i: Integer):\n    t = x.copy()\n    for k in range(4):\n'
        '        if x[k] > 25:\n            break\n        t = x.copy()\n        t[0] = 100 + k\n    return t\n',
    "break, augmented rebinding":
        'def kernel(x: Integer[Array, "6"], i: Integer):\n    t = np.zeros(3, dtype=int)\n    for k in range(4):\n'
        '        if x[k] > 25:\n            break\n        t = t + 1\n        t[0] = 50 + k\n    return t\n',
    "continue":
        'def kernel(x: Integer[Array, "6"], i: Integer):\n    t = np.zeros(3, dtype=int)\n    for k in range(4):\n'
        '        if x[k] > 25:\n            continue\n        t = np.zeros(3, dtype=int)\n        t[2] = k + 1\n    return t\n',
    "conditional rebinding":
        'def kernel(x: Integer[Array, "6"], i: Integer):\n    t = np.zeros(3, dtype=int)\n    for k in range(4):\n'
        '        if x[k] < 25:\n            t = np.zeros(3, dtype=int)\n            t[1] = k + 1\n    return t\n',
    "return in loop":
        'def kernel(x: Integer[Array, "6"], i: Integer):\n    t = np.zeros(3, dtype=int)\n    for k in range(4):\n'
        '        if x[k] > 25:\n            return t\n        t = np.zeros(3, dtype=int)\n        t[2] = k + 1\n    return t\n',
}


@pytest.mark.parametrize("name", list(LOOPS))
@pytest.mark.parametrize("i", [1, 4])
def test_a_write_after_a_rebinding_in_a_loop_keeps_its_condition(name, i):
    assert_all_backends_match_numpy(LOOPS[name], X6, i)


def test_a_rebinding_of_a_large_array_keeps_its_condition():
    # 1,100 cells: the write is a RAM access (halo2-base trace, Noir array).
    source = ('def kernel(x: Integer[Array, "1100"], i: Integer, c: Boolean):\n    t = x * 1\n'
              '    if c:\n        t = x + 1\n        t[i] = -7\n    return t[0], t[5], t[i]\n')
    for c in (False, True):
        assert_all_backends_match_numpy(source, np.arange(1100, dtype=np.int64), 5, c)


# ── F4a: functions NumPy answers with a view ─────────────────────────────

VIEWS = {
    "np.swapaxes, write through it": 'def kernel(m: Integer[Array, "3 4"]):\n    t = np.swapaxes(m, 0, 1)\n    t[2, 1] = 99\n    return m\n',
    "np.swapaxes, sees the base": 'def kernel(m: Integer[Array, "3 4"]):\n    t = np.swapaxes(m, 0, 1)\n    m[1, 2] = 99\n    return t\n',
    "np.asarray, write through it": 'def kernel(x: Integer[Array, "6"]):\n    t = np.asarray(x)\n    t[2] = 99\n    return x\n',
    "np.asarray, sees the base": 'def kernel(x: Integer[Array, "6"]):\n    t = np.asarray(x)\n    x[0] = 99\n    return t\n',
    "np.array copies": 'def kernel(x: Integer[Array, "6"]):\n    t = np.array(x)\n    t[2] = 99\n    return x, t\n',
    "np.atleast_1d": 'def kernel(x: Integer[Array, "6"]):\n    t = np.atleast_1d(x)\n    t[2] = 99\n    return x\n',
    "np.atleast_2d": 'def kernel(x: Integer[Array, "6"]):\n    t = np.atleast_2d(x)\n    t[0, 2] = 99\n    return x, t\n',
}


@pytest.mark.parametrize("name", list(VIEWS))
def test_functions_numpy_answers_with_a_view_share_the_array(name):
    arg = M34 if "m:" in VIEWS[name] else X6
    assert_all_backends_match_numpy(VIEWS[name], arg)


COPIED_VIEWS = {
    "np.flip, write through it": 'def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    t[2] = 99\n    return x\n',
    "np.flip, write to the base": 'def kernel(x: Integer[Array, "6"]):\n    t = np.flip(x)\n    x[0] = 99\n    return t\n',
    "np.fliplr": 'def kernel(m: Integer[Array, "3 4"]):\n    t = np.fliplr(m)\n    t[1, 2] = 99\n    return m\n',
    "np.flipud": 'def kernel(m: Integer[Array, "3 4"]):\n    t = np.flipud(m)\n    t[1, 2] = 99\n    return m\n',
    "np.rot90": 'def kernel(m: Integer[Array, "3 4"]):\n    t = np.rot90(m)\n    t[2, 1] = 99\n    return m\n',
    "runtime-shaped .T, write to the base":
        'def kernel(m: Float[Array, "3 4"]):\n    y = m[m[:, 0] > 0.0]\n    w = y.T\n    y[1, 2] = 0.0\n    return w\n',
    "runtime-shaped .T, write through it":
        'def kernel(m: Float[Array, "3 4"]):\n    y = m[m[:, 0] > 0.0]\n    w = y.T\n    w[2, 1] = 0.0\n    return y\n',
}


@pytest.mark.parametrize("name", list(COPIED_VIEWS))
def test_a_write_through_a_copied_view_is_rejected(name):
    with pytest.raises(ZinniaException, match="linked by NumPy view semantics"):
        compile_kernel(COPIED_VIEWS[name])


@pytest.mark.parametrize("name", ["np.flip", "np.fliplr", "np.rot90"])
def test_a_copied_view_nobody_writes_is_fine(name):
    source = {"np.flip": 'def kernel(m: Integer[Array, "3 4"]):\n    return np.flip(m) + m\n',
              "np.fliplr": 'def kernel(m: Integer[Array, "3 4"]):\n    return np.fliplr(m) + m\n',
              "np.rot90": 'def kernel(m: Integer[Array, "3 4"]):\n    return np.rot90(m).sum() + m\n'}[name]
    assert_all_backends_match_numpy(source, M34)


# ── F4b: copies of runtime-shaped arrays ─────────────────────────────────

DYN_COPIES = {
    ".copy(), write to the copy": 'def kernel(x: Float[Array, "6"]):\n    y = x[x > 1.0]\n    w = y.copy()\n    w[1] = 0.0\n    return y\n',
    ".copy(), runtime index": 'def kernel(x: Float[Array, "6"], i: Integer):\n    y = x[x > 1.0]\n    w = y.copy()\n    w[i] = 0.0\n    return y\n',
    ".copy(), write to the original": 'def kernel(x: Float[Array, "6"]):\n    y = x[x > 1.0]\n    w = y.copy()\n    y[1] = 0.0\n    return w\n',
    "np.array": 'def kernel(x: Float[Array, "6"]):\n    y = x[x > 1.0]\n    w = np.array(y)\n    w[1] = 0.0\n    return y\n',
    ".copy() then slice write": 'def kernel(x: Float[Array, "6"]):\n    y = x[x > 1.0]\n    w = y.copy()\n    w[:2] = 0.0\n    return y, w\n',
}


@pytest.mark.parametrize("name", list(DYN_COPIES))
def test_a_copy_of_a_runtime_shaped_array_is_a_new_array(name):
    source = DYN_COPIES[name]
    args = (F6, 1) if "i: Integer" in source else (F6,)
    assert_all_backends_match_numpy(source, *args)


def test_a_copy_of_a_runtime_shaped_2d_array_is_a_new_array():
    source = 'def kernel(m: Float[Array, "3 4"]):\n    y = m[m[:, 0] > 0.0]\n    w = y.copy()\n    w[1, 1] = 0.0\n    return y\n'
    assert_all_backends_match_numpy(source, M34F)


# ── F5 ───────────────────────────────────────────────────────────────────

def test_the_numpy_reference_keeps_scalar_arguments_scalars():
    # `m[i][j] = 0` writes through the row view `m[i]` in NumPy; with `i` a
    # 0-d array (what `np.copy` makes of a scalar) it would write a copy.
    source = 'def kernel(m: Integer[Array, "3 4"], i: Integer, j: Integer):\n    m[i][j] = 0\n    return m\n'
    values, _ = numpy_outputs(source, M34, 1, 2)
    assert values[6] == 0
    assert_all_backends_match_numpy(source, M34, 1, 2)


def test_a_program_that_hung_the_compiler_compiles():
    # Fuzz program #1534 (`fuzz_views_12`): `SelectMerge` computed an
    # interval's width in i64, which overflowed and made it enumerate 2^64
    # points.
    source = '''def kernel(a: Integer[Array, "4 5"], b: Integer[Array, "6"], i: Integer, j: Integer, c: Boolean):
    s = 0
    for k in range(2):
        a[(j % 4 - 4), 0] = a[(i % 4 - 4), (i + 2) % 5] + 2
    b[-2] = a[(i + 1) % 4, -1] + -2
    s = s + a[(j % 4 - 4), (j % 5 - 5)] + 0
    if c:
        a[(i + 1) % 4, (j + 2) % 5] = b[(j + 3) % 6] + 1
    b[1:] = b[:-1] + 1
    b[b > 3] = 6
    if c:
        a[(i + 1) % 4, :] = -7
    return s, a, b
'''
    started = time.time()
    compile_kernel(source)
    assert time.time() - started < 60
    assert_all_backends_match_numpy(source, np.arange(20, dtype=np.int64).reshape(4, 5), X6, 3, 7, True)
