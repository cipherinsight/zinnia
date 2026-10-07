"""Cheaper first-True lowerings and runtime reads (cost-gap round, sorting area, 2026-10-06).

`argmax` of a Boolean array is a select chain from the last index down to
the first (no found flag), and the `searchsorted` count of a strict
comparison with a sort's outputs counts the Trues and subtracts them from
the length (no negation per element). Each lowering must equal NumPy
position by position on the mock backend, the halo2-base proof's public
values must equal the mock's, and the IR shape that makes it cheaper is
pinned.
"""
import json

import numpy as np
import pytest

from testing.lang.mock_outputs import assert_matches_numpy, compile_kernel
from testing.lang.test_cost_lowerings import HALO2_PARAMS, assert_backends_agree, ir_classes

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


# ── halo2-base runtime reads by a multiplexer ────────────────────────────


def proves(source, *args):
    """Whether halo2-base produces a proof that verifies."""
    program = compile_kernel(source)
    try:
        result = program.prove(*args, backend="halo2-base", params=HALO2_PARAMS)
    except Exception:  # noqa: BLE001 - an unsatisfiable witness raises
        return False
    return program.verify(result)


def program_cells(source):
    """halo2-base cells (advice + lookup, 8-bit lookups) of the program."""
    from zinnia._zinnia_core import halo2_base_cell_stats

    program = compile_kernel(source)
    params = json.dumps({"k": 12, "precision_bits": 32, "lookup_bits": 8})
    stats = json.loads(halo2_base_cell_stats(program._compiled_ir, params))
    return stats["advice"] + stats["lookup"]


ELEMENT_READS = ["memory: element by index bits", "gather: element by a select chain", "gather: element by indicators"]


def forced_cells(monkeypatch, source, construction):
    """`program_cells` with the reads built by `construction` alone."""
    monkeypatch.setenv("ZINNIA_FORCE_CONSTRUCTION", construction)
    cells = program_cells(source)
    monkeypatch.delenv("ZINNIA_FORCE_CONSTRUCTION")
    return cells


ONE_READ = 'def kernel(a: Float[Array, "7"], i: Integer):\n    return a[i]\n'
ONE_READ_SHIFTED = 'def kernel(a: Float[Array, "7"], i: Integer):\n    return a[i + 2]\n'
CONSTANT_TABLE = 'def kernel(i: Integer):\n    return (np.arange(1, 8) / 7.0)[i]\n'
A7 = np.array([0.5, -1.0, 2.25, 3.0, -4.5, 6.0, 7.75])


@pytest.mark.parametrize("i", range(-7, 7))
def test_one_runtime_read_matches_numpy(i):
    assert_matches_numpy(ONE_READ, A7, i)
    assert_backends_agree(ONE_READ, A7, i)
    assert_backends_agree(CONSTANT_TABLE, i)


@pytest.mark.parametrize("i", range(-2, 5))
def test_shifted_runtime_read_matches_numpy(i):
    assert_matches_numpy(ONE_READ_SHIFTED, A7, i)
    assert_backends_agree(ONE_READ_SHIFTED, A7, i)


@pytest.mark.parametrize("source, i", [(ONE_READ, 7), (ONE_READ, -8), (ONE_READ_SHIFTED, 5),
                                       (ONE_READ_SHIFTED, -10), (CONSTANT_TABLE, 7)])
def test_runtime_read_out_of_bounds_has_no_proof(source, i):
    args = (i,) if source == CONSTANT_TABLE else (A7, i)
    assert not proves(source, *args)


def test_one_runtime_read_is_a_multiplexer(monkeypatch):
    # A single read of 64 slots: a multiplexer over the index's bits, no
    # dearer than any other element read (the one-hot form pays ~17 cells
    # per slot).
    source = 'def kernel(a: Float[Array, "64"], i: Integer):\n    return a[i]\n'
    chosen = program_cells(source)
    assert chosen == forced_cells(monkeypatch, source, "memory: element by index bits")
    assert all(chosen <= forced_cells(monkeypatch, source, c) for c in ELEMENT_READS)


def test_reads_sharing_a_root_keep_the_one_hot_form(monkeypatch):
    # Four reads on one index share its equality bits, cheaper than four
    # multiplexers.
    arrays = ", ".join(f'{n}: Float[Array, "64"]' for n in "abcd")
    source = f'def kernel({arrays}, i: Integer):\n    return a[i] + b[i] + c[i] + d[i]\n'
    chosen = program_cells(source)
    assert chosen < forced_cells(monkeypatch, source, "memory: element by index bits")
    assert all(chosen <= forced_cells(monkeypatch, source, c) for c in ELEMENT_READS)
