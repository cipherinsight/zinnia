"""Cheaper lowerings of the second performance round (perf-2, 2026-10-06).

Each section covers one change: the outputs equal NumPy position by
position on the mock backend, the halo2-base proof's public values equal the
mock's, inputs outside the program's domain have no proof, and the IR shape
that makes the lowering cheaper is pinned.
"""
import numpy as np
import pytest

from testing.lang.mock_outputs import assert_matches_numpy, compile_kernel
from testing.lang.test_cost_lowerings import HALO2_PARAMS, assert_backends_agree, ir_classes


def proves(source, *args):
    """Whether halo2-base produces a proof that verifies."""
    program = compile_kernel(source)
    try:
        result = program.prove(*args, backend="halo2-base", params=HALO2_PARAMS)
    except Exception:  # noqa: BLE001 - an unsatisfiable witness raises
        return False
    return program.verify(result)


# ── domain preconditions are checked before the operation ────────────────

DOMAIN_OPS = [
    ('def kernel(a: Float[Array, "4"]):\n    return np.sqrt(a)\n', "SqrtFIR",
     np.array([0.0, 2.0, 1e-6, 1.5e6]), np.array([0.0, -1e-9, 4.0, 9.0])),
    ('def kernel(a: Float[Array, "3"]):\n    return np.arcsin(a)\n', "ArcSinFIR",
     np.array([0.5, -1.0, 0.0]), np.array([1.5, 0.0, 0.0])),
]


@pytest.mark.parametrize("source, op, good, bad", DOMAIN_OPS)
def test_precondition_precedes_the_operation(source, op, good, bad):
    classes = ir_classes(compile_kernel(source))
    first_op = classes.index(op)
    # The precondition's assertion (after the input-domain checks) comes
    # before the operation, so the lowering can use the range it proves.
    assert classes[:first_op].count("AssertIR") > 2 * len(good), classes[:first_op + 1]


@pytest.mark.parametrize("source, op, good, bad", DOMAIN_OPS)
def test_domain_ops_match_numpy_and_reject_outside_inputs(source, op, good, bad):
    assert_matches_numpy(source, good)
    assert_backends_agree(source, good)
    assert proves(source, good)
    assert not proves(source, bad)


# ── bounded inputs with per-axis bounds: padded buffer, padded strides ───

BOUNDED = 'def kernel(x: Integer[Array, "N<=4 M<=3"]):\n    return {}\n'
X22 = np.array([[1, 2], [3, 4]])
X32 = np.array([[5, -1], [0, 7], [2, 2]])


@pytest.mark.parametrize("expr", ["x", "x + 1", "x * x", "x.T", "x.sum(axis=0)", "np.argmax(x)", "np.argmin(x)"])
@pytest.mark.parametrize("x", [X22, X32])
def test_bounded_input_reads_its_padded_buffer(expr, x):
    """The input parser fills `N<=4 M<=3` padded (element (i, j) at 3i + j);
    the array records those strides, so returning it (decoded by its
    strides) and flat argextrema (C-order positions) equal NumPy. Before,
    `x` returned [1, 2, 0, 3] for [[1, 2], [3, 4]] and argmax 4."""
    from testing.lang.mock_outputs import run_mock_layout, numpy_kernel
    source = BOUNDED.format(expr)
    satisfied, values, shapes = run_mock_layout(compile_kernel(source), x)
    expected = np.asarray(numpy_kernel(source)(x))
    assert satisfied
    assert list(shapes[0]) == list(expected.shape)
    assert list(values[: expected.size]) == expected.ravel().tolist()


def test_argmax_of_a_runtime_box_is_its_c_order_position():
    from testing.lang.mock_outputs import run_mock_layout
    source = ('def kernel(x: Integer[Array, "4 3"], r: Integer, c: Integer):\n'
              '    return np.argmax(x[0:r, 0:c])\n')
    x = np.array([[1, 9, 0], [3, 4, 8], [0, 0, 0], [7, 7, 7]])
    satisfied, values, _ = run_mock_layout(compile_kernel(source), x, 2, 2)
    assert satisfied and values[0] == int(np.argmax(x[0:2, 0:2]))


@pytest.mark.parametrize("source", [
    BOUNDED.format("x.reshape(-1)"),
    BOUNDED.format("x.reshape(1, -1)"),
    'def kernel(x: Integer[Array, "4 3"], r: Integer, c: Integer):\n    return x[0:r, 0:c].reshape(-1)\n',
])
def test_reshape_of_a_padded_buffer_with_holes_is_refused(source):
    """Reshape reads its source as a C-order sequence; a padded buffer whose
    inner extents are runtime values has holes, so the compiler refuses it
    (it returned [1, 2, 0, 3] for [[1, 2], [3, 4]] before)."""
    with pytest.raises(Exception, match="padded buffer"):
        compile_kernel(source)


def test_reshape_of_a_row_prefix_still_compiles():
    source = 'def kernel(x: Integer[Array, "4 3"], r: Integer):\n    return x[0:r, :].reshape(-1)\n'
    from testing.lang.mock_outputs import run_mock_layout
    x = np.arange(12).reshape(4, 3)
    satisfied, values, shapes = run_mock_layout(compile_kernel(source), x, 2)
    assert satisfied and list(shapes[0]) == [6] and list(values[:6]) == list(range(6))


@pytest.mark.parametrize("expr", ["x[:, 1]", "x[:, -1]", "x[0:2, 1]", "x[1, :]", "x[-1, :]", "x[:, 1] * 2", "x[:, -2].sum()"])
@pytest.mark.parametrize("x", [X22, X32])
def test_mixed_subscripts_of_a_bounded_input_follow_its_runtime_shape(expr, x):
    """A whole-axis slice keeps the axis' runtime extent and a single index
    wraps against it (before: `x[:, 1]` had the buffer's 4 rows and
    `x[:, -1]` read column 2 of the 3-wide buffer)."""
    from testing.lang.mock_outputs import run_mock_layout, numpy_kernel
    source = BOUNDED.format(expr)
    satisfied, values, shapes = run_mock_layout(compile_kernel(source), x)
    expected = np.asarray(numpy_kernel(source)(x))
    assert satisfied
    assert list(shapes[0]) == list(expected.shape)
    assert list(values[: expected.size]) == expected.ravel().tolist()


@pytest.mark.parametrize("expr", ["x[2, :]", "x[:, 2]", "x[:, -3]"])
def test_single_index_past_the_runtime_extent_has_no_proof(expr):
    """NumPy raises IndexError for row 2 of a 2-row array, even though the
    buffer has 4 rows."""
    from testing.lang.mock_outputs import run_mock
    satisfied, _ = run_mock(compile_kernel(BOUNDED.format(expr)), X22)
    assert not satisfied


# ── read after write at a bounded runtime index ──────────────────────────

ROW_UPDATE = ('def kernel(a: Float[Array, "8 6"], row: Integer, m: Float):\n'
              '    a[row-1, :] *= m\n'
              '    return np.cumsum(a[row-1, :])\n')
COL_UPDATE = ('def kernel(a: Float[Array, "5 4"], col: Integer, m: Float):\n'
              '    a[:, col] = a[:, col] / m\n'
              '    return a.sum(axis=0) + a[:, col].sum()\n')


@pytest.mark.parametrize("source, args", [
    (ROW_UPDATE, (np.arange(48.0).reshape(8, 6) / 7, 3, 2.5)),
    (ROW_UPDATE, (np.arange(48.0).reshape(8, 6) / 7, 0, -1.5)),  # row -1: the last row
    (ROW_UPDATE, (np.arange(48.0).reshape(8, 6) / 7, 8, 0.5)),
    (COL_UPDATE, (np.arange(20.0).reshape(5, 4) - 3, 2, 4.0)),
    (COL_UPDATE, (np.arange(20.0).reshape(5, 4) - 3, -1, 2.0)),
])
def test_read_after_write_at_a_runtime_index_matches_numpy(source, args):
    assert_matches_numpy(source, *args)
    assert_backends_agree(source, *args)


def test_read_after_write_reads_the_written_values():
    """The second read of `a[row-1, :]` is the written row itself: the
    program costs no more than reading the row once (a second gather over
    the updated rows would add several hundred cells). Compared by halo2-base
    cells, since the layer may build the one read differently in the two
    programs (the write's position one-hot, or the pinned sign's tests)."""
    import json
    from zinnia._zinnia_core import halo2_base_cell_stats

    def cells(source):
        stats = json.loads(halo2_base_cell_stats(compile_kernel(source)._compiled_ir))
        return stats["advice"] + stats["lookup"]

    read_once = ('def kernel(a: Float[Array, "8 6"], row: Integer, m: Float):\n'
                 '    return np.cumsum(a[row-1, :] * m)\n')
    assert cells(ROW_UPDATE) <= cells(read_once)
