"""Domains NumPy defines and the circuits must follow (integ-1).

* A runtime row or column index in a slice assignment (`a[i, :] = v`,
  `a[:, j] = v`) must satisfy `-n <= i < n`; NumPy raises `IndexError`
  otherwise. Before the fix an out-of-range index proved the unchanged array
  (the written cells are selected by `i == coord`, which matched nothing).
* `np.insert(a, pos, v)` accepts `-n <= pos <= n`; a negative `pos` counts
  from the end. Before the fix a negative runtime `pos` had no proof.
* `abs(i64::MIN)` overflows int64 (NumPy wraps it to `i64::MIN`); like every
  int64 overflow it has no proof, on the mock and on every backend.

Each kernel asserts its result equal to a claimed `expected`: NumPy's result
is accepted, a wrong claim is rejected, and an input outside NumPy's domain
is rejected whatever the claim.
"""
import numpy as np
import pytest

from testing.lang.mock_outputs import assert_matches_numpy, compile_kernel
from testing.lang.test_dtype_conversions import halo2_accepts, mock_accepts, noir_accepts

I64_MIN = -(2**63)
I64_MAX = 2**63 - 1


def accepted_everywhere(program, *args) -> bool:
    return mock_accepts(program, *args) and halo2_accepts(program, *args) and noir_accepts(program, *args)


def rejected_everywhere(program, *args) -> bool:
    return not (mock_accepts(program, *args) or halo2_accepts(program, *args) or noir_accepts(program, *args))


# ── slice assignment at a runtime row / column ───────────────────────────

ZERO_ROW_AND_COLUMN = (
    'def kernel(a: Integer[Array, "4 4"], i: Integer, j: Integer, expected: Integer[Array, "4 4"]):\n'
    '    a[i, :] = 0\n'
    '    a[:, j] = 0\n'
    '    assert np.all(a == expected)\n'
)
A44 = np.arange(1, 17, dtype=np.int64).reshape(4, 4)


def zeroed(i, j):
    a = A44.copy()
    a[i, :] = 0
    a[:, j] = 0
    return a


@pytest.fixture(scope="module")
def zero_row_and_column():
    return compile_kernel(ZERO_ROW_AND_COLUMN)


@pytest.mark.parametrize("i, j", [(1, 2), (-1, -4), (3, 0), (-4, 3)])
def test_slice_assignment_at_a_runtime_index_proves_numpys_result(zero_row_and_column, i, j):
    good = zeroed(i, j)
    assert accepted_everywhere(zero_row_and_column, A44, i, j, good)
    wrong = good.copy()
    wrong[0, 0] += 1
    assert rejected_everywhere(zero_row_and_column, A44, i, j, wrong)


@pytest.mark.parametrize("i, j", [(7, 1), (4, 1), (-5, 1), (1, 4), (1, -5)])
def test_slice_assignment_out_of_range_has_no_proof(zero_row_and_column, i, j):
    """NumPy raises IndexError. The claim the circuit used to prove (the
    out-of-range axis left unchanged) and the unchanged array are both
    rejected."""
    with pytest.raises(IndexError):
        zeroed(i, j)
    unchanged_axis = A44.copy()
    if 0 <= j < 4 or -4 <= j < 0:
        unchanged_axis[:, j] = 0
    if 0 <= i < 4 or -4 <= i < 0:
        unchanged_axis[i, :] = 0
    for claim in (unchanged_axis, A44):
        assert rejected_everywhere(zero_row_and_column, A44, i, j, claim)


def test_slice_assignment_returns_numpys_array():
    source = 'def kernel(a: Float[Array, "3 5"], i: Integer):\n    a[:, i] = 2.5\n    return a\n'
    for i in (-5, -1, 0, 4):
        assert_matches_numpy(source, np.linspace(-1, 1, 15).reshape(3, 5), i)


# ── np.insert at a runtime position ──────────────────────────────────────

INSERT = (
    'def kernel(a: Integer[Array, "5"], pos: Integer, e: Integer, expected: Integer[Array, "6"]):\n'
    '    assert np.all(np.insert(a, pos, e) == expected)\n'
)
A5 = np.array([10, 20, 30, 40, 50], dtype=np.int64)


@pytest.fixture(scope="module")
def insert_program():
    return compile_kernel(INSERT)


@pytest.mark.parametrize("pos", [-5, -3, -1, 0, 2, 5])
def test_insert_at_a_runtime_position_proves_numpys_result(insert_program, pos):
    good = np.insert(A5, pos, 99)
    assert accepted_everywhere(insert_program, A5, pos, 99, good)
    # The other reading of a negative position: counted from the start.
    wrong = np.insert(A5, abs(pos) if pos < 0 else (pos + 1) % 6, 99)
    assert rejected_everywhere(insert_program, A5, pos, 99, wrong)


@pytest.mark.parametrize("pos", [-6, 6, -100])
def test_insert_out_of_range_has_no_proof(insert_program, pos):
    with pytest.raises(IndexError):
        np.insert(A5, pos, 99)
    for claim in (np.insert(A5, 0, 99), np.insert(A5, 5, 99)):
        assert rejected_everywhere(insert_program, A5, pos, 99, claim)


def test_insert_rows_at_negative_positions_match_numpy():
    runtime = ('def kernel(a: Float[Array, "4 3"], pos: Integer, r: Float[Array, "3"]):\n'
               '    return np.insert(a, pos, r, axis=0)\n')
    a = np.linspace(-2, 2, 12).reshape(4, 3)
    row = np.array([0.5, -0.25, 7.0])
    for pos in (-4, -1, 0, 4):
        assert_matches_numpy(runtime, a, pos, row)
    literal = 'def kernel(a: Integer[Array, "5"], e: Integer):\n    return np.insert(a, -2, e)\n'
    assert_matches_numpy(literal, A5, 7)
