"""Cheaper first-True lowerings (cost-gap round, sorting area, 2026-10-06).

`argmax` of a Boolean array is a select chain from the last index down to
the first (no found flag), and the `searchsorted` count of a strict
comparison with a sort's outputs counts the Trues and subtracts them from
the length (no negation per element). Each lowering must equal NumPy
position by position on the mock backend, the halo2-base proof's public
values must equal the mock's, and the IR shape that makes it cheaper is
pinned.
"""
import numpy as np
import pytest

from testing.lang.mock_outputs import assert_matches_numpy, compile_kernel
from testing.lang.test_cost_lowerings import assert_backends_agree, ir_classes

FIRST_MATCH = (
    'def kernel(x: Integer[Array, "6"], y: Integer[Array, "6"], a: Integer, b: Integer):\n'
    '    return ((x == a) & (y == b)).argmax()\n'
)

# (x, y, a, b): the first match at index 0, in the middle, at the end,
# several matches, and none (argmax 0).
MATCH_INPUTS = [
    (np.array([1, 2, 1, 4, 1, 6]), np.array([7, 8, 7, 0, 7, 0]), 1, 7),
    (np.array([0, 2, 3, 3, 5, 3]), np.array([0, 0, 9, 9, 0, 9]), 3, 9),
    (np.array([0, 2, 3, 4, 5, 3]), np.array([0, 0, 0, 0, 0, 9]), 3, 9),
    (np.array([5, 5, 5, 5, 5, 5]), np.array([1, 2, 3, 4, 5, 6]), 5, 4),
    (np.array([1, 2, 3, 4, 5, 6]), np.array([1, 2, 3, 4, 5, 6]), 1, 2),
]


@pytest.mark.parametrize("x, y, a, b", MATCH_INPUTS)
def test_boolean_argmax_is_the_first_true(x, y, a, b):
    assert_matches_numpy(FIRST_MATCH, x, y, a, b)
    assert_backends_agree(FIRST_MATCH, x, y, a, b)


def test_boolean_argmax_needs_no_found_flag():
    # One select per cell: no OR/NOT chain carrying "found so far".
    classes = ir_classes(compile_kernel(FIRST_MATCH))
    assert "LogicalOrIR" not in classes and "LogicalNotIR" not in classes, classes
    assert classes.count("SelectIIR") <= 6, classes


SEARCHSORTED = (
    'def kernel(a: Float[Array, "6"], e: Float):\n'
    '    return (np.sort(a) > e).argmax()\n'
)

# Ties with e, e below every element, e above every element (argmax 0).
SEARCH_INPUTS = [
    (np.array([0.5, -1.0, 2.0, 0.5, 3.5, 0.5]), 0.5),
    (np.array([0.5, -1.0, 2.0, 0.5, 3.5, 0.5]), -2.0),
    (np.array([0.5, -1.0, 2.0, 0.5, 3.5, 0.5]), 9.0),
    (np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 1.0),
]


@pytest.mark.parametrize("a, e", SEARCH_INPUTS)
def test_searchsorted_count_matches_numpy(a, e):
    assert_matches_numpy(SEARCHSORTED, a, e)
    assert_backends_agree(SEARCHSORTED, a, e)


def test_searchsorted_count_needs_no_negation():
    classes = ir_classes(compile_kernel(SEARCHSORTED))
    assert "LogicalNotIR" not in classes, classes
    assert "HintIR" not in classes, classes  # the sort itself is gone
