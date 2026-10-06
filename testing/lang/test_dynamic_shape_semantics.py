"""NumPy-differential tests for operators with runtime-shaped results (audit M2).

Each case used to prove a result different from NumPy's on an honest input.
Outputs are compared position by position; a runtime-shaped (dynamic) result
is exposed as its runtime length and dims followed by the padded payload, and
only the runtime prefix is compared with NumPy.
"""
import numpy as np
import pytest

from testing.lang.mock_outputs import assert_matches_numpy, compile_kernel, numpy_kernel, run_mock

A23 = np.array([[1, 5, 0], [7, 3, 2]])


def dynamic_1d(source, *args):
    """(runtime length, runtime prefix of the payload) of a 1-D dynamic result."""
    satisfied, outputs = run_mock(compile_kernel(source), *args)
    assert satisfied
    length = outputs[0]
    return length, outputs[2:2 + length]


def runtime_values(outputs, n):
    """The `n` result values: all outputs for a static result, the runtime
    prefix of the payload for a 1-D dynamic one (layout: runtime length,
    runtime dim, padded payload)."""
    if len(outputs) == n:
        return outputs
    assert outputs[0] == n, f"runtime length {outputs[0]} != {n}"
    return outputs[2:2 + n]


def test_one_argument_where_is_nonzero():
    # Used to return the condition itself, so `[0]` read its first element.
    source = 'def kernel(a: Integer[Array, "6"]):\n    return np.where(a > 2)[0]\n'
    a = np.array([1, 5, 0, 7, 3, 2])
    length, values = dynamic_1d(source, a)
    expected = np.where(a > 2)[0]
    assert length == len(expected) and values == list(expected)


@pytest.mark.parametrize("body", [
    "a[np.where(a > 2)] = 0",
    "a[np.where(a > 2)] = 0\n    a[np.where(a < 2)] = 30",
    "a[np.nonzero(a > 2)] = 9",
])
def test_assignment_through_nonzero_indices(body):
    assert_matches_numpy(f'def kernel(a: Integer[Array, "2 3"]):\n    {body}\n    return a\n', A23)


A42 = np.array([[1, 2], [3, 8], [-5, 5], [6, 6]])
SLICE_2D = 'def kernel(a: Integer[Array, "4 2"], r: Integer):\n    s = a[0:r, :]\n    return {}\n'


@pytest.mark.parametrize("expr", [
    "s.mean()", "s.min()", "s.max()", "s.argmax()", "s.prod()", "s.sum()",
    "len(s)", "s.shape[0]", "s.sum(axis=1)", "s.max(axis=0)",
])
def test_reduction_over_runtime_bounded_2d_slice(expr):
    # `a[0:r, :]` used to be the zero-padded 4x2 envelope; `.mean()` divided
    # by 8, `.min()` saw the padding zeros, `len` was 4.
    source = SLICE_2D.format(expr)
    expected = np.asarray(numpy_kernel(source)(A42.copy(), 2), dtype=float).ravel()
    satisfied, outputs = run_mock(compile_kernel(source), A42, 2)
    assert satisfied
    np.testing.assert_allclose(runtime_values(outputs, expected.size), expected)


def test_elementwise_over_runtime_bounded_2d_slice_keeps_runtime_shape():
    source = SLICE_2D.format("s + 1")
    satisfied, outputs = run_mock(compile_kernel(source), A42, 2)
    assert satisfied
    # Runtime length 4, runtime rows 2, then the padded 4x2 payload.
    assert outputs[:2] == [4, 2]
    assert outputs[2:6] == list((A42[:2] + 1).ravel())


@pytest.mark.parametrize("expr", ["np.cumsum(s)", "np.diff(y)", "np.flip(y)", "np.outer(y, y)"])
def test_static_shape_operation_on_runtime_shaped_array_is_rejected(expr):
    # These used to treat the runtime-shaped array as one scalar and return
    # it unchanged (or square it).
    source = ('def kernel(a: Integer[Array, "4 2"], r: Integer, x: Integer[Array, "6"]):\n'
              f'    s = a[0:r, :]\n    y = x[x > 0]\n    return {expr}\n')
    with pytest.raises(Exception, match="runtime shape"):
        compile_kernel(source)


@pytest.mark.parametrize("expr", [
    "y[y < 5]",                    # filter of a filter kept the padding
    "y.reshape(-1).mean()",        # runtime reshape claimed the envelope length
    "y[-1]",                       # negative index wrapped with the envelope size
    "np.argwhere(m).reshape(-1).max()",
])
def test_runtime_shaped_array_operations_match_numpy(expr):
    source = (f'def kernel(x: Integer[Array, "6"], m: Integer[Array, "2 3"]):\n'
              f'    y = x[x > 0]\n    return {expr}\n')
    x = np.array([1, -1, 2, 7, -3, 9])
    m = np.array([[0, 1, 0], [0, 0, 1]])
    expected = np.asarray(numpy_kernel(source)(x.copy(), m.copy()), dtype=float).ravel()
    satisfied, outputs = run_mock(compile_kernel(source), x, m)
    assert satisfied
    np.testing.assert_allclose(runtime_values(outputs, expected.size), expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("source, args", [
    ('def kernel(x: Integer[Array, "6"], i: Integer):\n    y = x[x > 0]\n    return y[i]\n',
     (np.array([1, -1, 2, 7, -3, 9]), 4)),
    ('def kernel(a: Integer[Array, "4 3"], r: Integer, i: Integer):\n    s = a[0:r, :]\n    return s[i]\n',
     (np.arange(12).reshape(4, 3), 2, 3)),  # row 3 of a 2-row slice: was unchecked
])
def test_out_of_range_index_into_runtime_shaped_array_is_rejected(source, args):
    satisfied, _ = run_mock(compile_kernel(source), *args)
    assert not satisfied


# ── a runtime `if` assigning runtime-shaped arrays (perf-ops) ────────────
# The merged value had a flat buffer and strides [1, 1]: `result.sum()` of
# the crop below proved 27 (NumPy: 135), `result.max()` 10 (22), `result[1]`
# [3, 3, 9] (NumPy: [14, 15, 16]).

IM = np.arange(30).reshape(5, 6)
CROP_OR_EMPTY = ('def kernel(im: Integer[Array, "5 6"], a: Integer, b: Integer, c: Integer,'
                 ' d: Integer, k: Integer):\n'
                 '    if k == 0:\n        result = np.array([])\n'
                 '    else:\n        result = im[a:b, c:d]\n'
                 '    return {}\n')


@pytest.mark.parametrize("expr, expected", [
    ("result.sum()", 135), ("result.max()", 22), ("result.min()", 8), ("result[1, 1]", 15),
])
def test_merged_crop_reductions_and_reads_match_numpy(expr, expected):
    source = CROP_OR_EMPTY.format(expr)
    assert numpy_kernel(source)(IM.copy(), 1, 4, 2, 5, 1) == expected
    satisfied, outputs = run_mock(compile_kernel(source), IM, 1, 4, 2, 5, 1)
    assert satisfied and outputs == [expected]


def test_merged_crop_row_read_matches_numpy():
    satisfied, outputs = run_mock(compile_kernel(CROP_OR_EMPTY.format("result[1]")),
                                  IM, 1, 4, 2, 5, 1)
    assert satisfied and outputs[0] == 3 and outputs[2:5] == [14, 15, 16]


def test_merged_empty_arm_exposes_no_data():
    # The empty arm's padding must not carry the other arm's (private) cells.
    satisfied, outputs = run_mock(compile_kernel(CROP_OR_EMPTY.format("result")),
                                  IM, 1, 4, 2, 5, 0)
    assert satisfied and outputs[0] == 0
    assert all(v == 0 for v in outputs[4:])


def test_merged_compact_and_padded_arms_are_rejected():
    source = ('def kernel(x: Integer[Array, "3 4"], t: Integer, k: Integer):\n'
              '    if k == 0:\n        y = x[x > t]\n    else:\n        y = x[:t, :t]\n'
              '    return y.sum()\n')
    with pytest.raises(Exception, match="runtime `if`"):
        compile_kernel(source)


# ── reductions without an identity on an empty runtime-shaped array ──────
# NumPy raises ValueError; the circuit proved 0.

@pytest.mark.parametrize("expr", ["y.max()", "y.min()", "y.argmax()", "y.argmin()", "np.max(y)"])
def test_reduction_of_empty_filter_has_no_proof(expr):
    source = f'def kernel(x: Integer[Array, "6"], t: Integer):\n    y = x[x > t]\n    return {expr}\n'
    x = np.array([1, 5, 2, 7, 3, 0])
    with pytest.raises(ValueError):
        numpy_kernel(source)(x.copy(), 100)
    program = compile_kernel(source)
    assert not run_mock(program, x, 100)[0]
    satisfied, outputs = run_mock(program, x, 3)
    assert satisfied and outputs == [numpy_kernel(source)(x.copy(), 3)]


def test_axis_reduction_over_empty_runtime_axis_has_no_proof():
    source = ('def kernel(x: Float[Array, "4 3"], n: Integer):\n'
              '    return x[:n, :].max(axis=0)\n')
    x = np.arange(12.0).reshape(4, 3)
    assert_matches_numpy(source, x, 2)
    assert not run_mock(compile_kernel(source), x, 0)[0]


# ── runtime extents of slices, splits, concatenations, transposes ───────
# (core-2 handoff H3: V2, V3, V5, C1/C2; and a transpose found with them)

A2 = np.arange(6).reshape(2, 3)


def _proves(source, *args):
    return run_mock(compile_kernel(source), *args)[0]


def test_one_range_on_2d_bounded_array_slices_rows():
    # V2: `a[:]` was a flat 1-D slice; the honest sum was rejected.
    source = 'def kernel(a: Integer[Array, "N<=4 3"], e: Integer):\n    assert a[:].sum() == e\n'
    assert _proves(source, A2, 15)
    assert _proves(source, np.arange(12).reshape(4, 3), 66)
    assert not _proves(source, A2, 14)


@pytest.mark.parametrize("body", ["a[0:k, :]", "a[0:3, :]", "a[:, 0:k]"])
def test_slice_bounds_clamp_to_runtime_extent(body):
    # V3: `a[0:k, :]` with k = 10 on 2 rows proved shape[0] == 10.
    source = (f'def kernel(a: Integer[Array, "N<=4 M<=3"], k: Integer, r: Integer, c: Integer):\n'
              f'    b = {body}\n    assert b.shape[0] == r\n    assert b.shape[1] == c\n'
              f'    assert b.sum() == 15\n')
    expected = numpy_kernel(f'def kernel(a, k):\n    return {body}\n')(A2.copy(), 10)
    assert _proves(source, A2, 10, *expected.shape)
    assert not _proves(source, A2, 10, expected.shape[0] + 1, expected.shape[1])


@pytest.mark.parametrize("points, lengths", [([5, 2], [5, 0, 6]), ([2, 5], [2, 3, 3]), ([3, 3], [3, 0, 5])])
def test_split_points_follow_python_slices(points, lengths):
    # V5: decreasing points proved a length of -3 (NumPy: 0). (Runtime
    # points outside [0, len] have no proof, like runtime slice bounds:
    # test_slice_oob_witness.py.)
    source = ('def kernel(a: Integer[Array, "N<=8"], i: Integer, j: Integer, e0: Integer, '
              'e1: Integer, e2: Integer):\n'
              '    parts = np.split(a, [i, j])\n'
              '    assert len(parts[0]) == e0\n    assert len(parts[1]) == e1\n'
              '    assert len(parts[2]) == e2\n')
    a = np.arange(8)
    assert [len(p) for p in np.split(a, points)] == lengths
    assert _proves(source, a, *points, *lengths)
    assert not _proves(source, a, *points, lengths[0], lengths[1] + 1, lengths[2])


def test_split_of_runtime_length_array():
    source = ('def kernel(a: Integer[Array, "N<=8"], e: Integer):\n'
              '    assert len(np.split(a, [3])[1]) == e\n')
    assert _proves(source, np.arange(5), 2)
    assert not _proves(source, np.arange(5), 5)


def test_concatenate_runtime_shaped_rows():
    source = ('def kernel(a: Integer[Array, "N<=4 M<=3"], b: Integer[Array, "K<=4 L<=3"], '
              's: Integer, r: Integer):\n'
              '    c = np.concatenate([a, b])\n'
              '    assert c.sum() == s\n    assert c.shape[0] == r\n')
    narrow_a, narrow_b = np.array([[0, 1], [2, 3]]), np.array([[0, 1]])
    assert _proves(source, narrow_a, narrow_b, 7, 3)
    assert not _proves(source, narrow_a, narrow_b, 6, 3)
    # Different column counts: NumPy raises.
    assert not _proves(source, A2, narrow_b, 16, 3)


@pytest.mark.parametrize("body", ["a.T.sum()", "a.T[0].sum()", "np.transpose(a)[1].sum()",
                                  "a.T[:, 1:2].sum()"])
def test_transpose_of_padded_runtime_shaped_array(body):
    # A 2x2 input in a 4x3 envelope: `a.T.sum()` proved 0, `a.T[0]` read a row of `a`.
    a = np.array([[0, 1], [2, 3]])
    source = f'def kernel(a: Integer[Array, "N<=4 M<=3"]):\n    return {body}\n'
    expected = numpy_kernel(f'def kernel(a):\n    return {body}\n')(a.copy())
    satisfied, outputs = run_mock(compile_kernel(source), a)
    assert satisfied and outputs == [expected]


@pytest.mark.parametrize("fn, x, sums", [
    ("split", np.arange(4.0), [1.0, 5.0]),
    ("array_split", np.arange(5.0), [3.0, 7.0]),
    ("array_split", np.arange(3.0), [1.0, 2.0]),
])
def test_split_into_sections_of_runtime_length(fn, x, sums):
    # The cut points follow the runtime length, not the 6-element envelope.
    source = (f'def kernel(x: Float[Array, "N<=6"], s0: Float, s1: Float):\n'
              f'    parts = np.{fn}(x, 2)\n'
              f'    assert parts[0].sum() == s0\n    assert parts[1].sum() == s1\n')
    assert [p.sum() for p in getattr(np, fn)(x, 2)] == sums
    assert _proves(source, x, *sums)
    assert not _proves(source, x, sums[0] + 1, sums[1])


def test_split_into_unequal_sections_has_no_proof():
    source = 'def kernel(x: Float[Array, "N<=6"]):\n    parts = np.split(x, 2)\n    assert len(parts) == 2\n'
    assert _proves(source, np.arange(4.0))
    assert not _proves(source, np.arange(5.0))  # NumPy raises ValueError
