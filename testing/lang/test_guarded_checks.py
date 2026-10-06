"""Checks hold only where Python executes them (fix-frontend handoff H1).

Python raises `IndexError` or `ZeroDivisionError` only on a path it runs. The
core used to emit its index-bound and divisor checks unconditionally, so a
check inside an untaken branch, an inactive iteration of a loop unrolled to its
proven maximum, or a filtered-out comprehension element rejected valid inputs.
Every check is now conditional on the statement's execution predicate.

Each test shows three things: Python's result is accepted where the old circuit
rejected it, a wrong result is rejected, and a check on a path that does run
still rejects (a malicious witness cannot use the guard to skip it).

The last tests cover facts learned from such conditional checks: a fact learned
where the code runs must not be used by code that runs elsewhere.
"""

import numpy as np
import pytest

from zinnia import ZKChip, ZKCircuit


def accepts(program, *args, backend="mock"):
    if backend == "mock":
        return program.prove(*args, backend=backend).proof_bytes_hex == "mock_satisfied"
    try:
        result = program.prove(*args, backend=backend)
    except RuntimeError as error:  # halo2-base refuses an unsatisfiable witness
        assert "does not satisfy" in str(error), error
        return False
    return program.verify(result)


def compile_source(source, chips=None):
    return ZKCircuit.from_source("kernel", source, chips=chips).compile()


# ── Runtime-bounded loops (the H1 probe) ────────────────────────────────

SUM_BY_INDEX = (
    'def kernel(x: Integer[Array, "N<=4"], expected: Integer):\n'
    "    total = 0\n"
    "    for i in range(len(x)):\n"
    "        total += x[i]\n"
    "    assert total == expected\n"
)


@pytest.mark.parametrize("values", [[], [7], [1, 2], [1, 2, 3], [1, 2, 3, 4]])
def test_loop_over_shorter_array_proves(values):
    program = compile_source(SUM_BY_INDEX)
    x = np.array(values, dtype=int)
    assert accepts(program, x, sum(values))
    assert not accepts(program, x, sum(values) + 1)


def test_loop_over_shorter_array_on_halo2_base():
    program = compile_source(SUM_BY_INDEX)
    x = np.array([1, 2, 3])
    assert accepts(program, x, 6, backend="halo2-base")
    assert not accepts(program, x, 7, backend="halo2-base")


def test_loop_index_check_on_executed_iteration_still_rejects():
    # `y` has 2 elements: iterations 2 and 3 read past it. Python raises
    # when they run (len(x) > 2) and is fine when they do not.
    program = compile_source(
        'def kernel(x: Integer[Array, "N<=4"], y: Integer[Array, "2"], expected: Integer):\n'
        "    total = 0\n"
        "    for i in range(len(x)):\n"
        "        total += y[i]\n"
        "    assert total == expected\n"
    )
    y = np.array([10, 20])
    assert accepts(program, np.array([0, 0]), y, 30)
    assert not accepts(program, np.array([0, 0]), y, 0)
    for claimed in (30, 0):
        assert not accepts(program, np.array([0, 0, 0]), y, claimed)


# ── Conditional branches ────────────────────────────────────────────────

GUARDED_READ = (
    'def kernel(x: Integer[Array, "4"], j: Integer, expected: Integer):\n'
    "    y = 0\n"
    "    if j < 4:\n"
    "        y = x[j]\n"
    "    assert y == expected\n"
)


@pytest.mark.parametrize("backend", ["mock", "halo2-base"])
def test_index_in_untaken_branch(backend):
    program = compile_source(GUARDED_READ)
    x = np.array([5, 6, 7, 8])
    assert accepts(program, x, 9, 0, backend=backend)
    assert not accepts(program, x, 9, 5, backend=backend)
    assert accepts(program, x, 2, 7, backend=backend)
    assert not accepts(program, x, 2, 0, backend=backend)


def test_index_in_taken_branch_is_still_checked():
    program = compile_source(GUARDED_READ)
    x = np.array([5, 6, 7, 8])
    assert accepts(program, x, -1, 8)  # NumPy's x[-1]
    for claimed in (0, 5, 8):
        assert not accepts(program, x, -5, claimed)  # IndexError in Python


def test_index_past_guard_bound_rejects_where_it_runs():
    program = compile_source(
        'def kernel(x: Integer[Array, "4"], j: Integer, expected: Integer):\n'
        "    y = 0\n"
        "    if j < 6:\n"
        "        y = x[j]\n"
        "    assert y == expected\n"
    )
    x = np.array([5, 6, 7, 8])
    assert accepts(program, x, 7, 0)
    assert accepts(program, x, 3, 8)
    for claimed in (0, 5, 8):
        assert not accepts(program, x, 5, claimed)


def test_dynamic_array_index_in_untaken_branch():
    program = compile_source(
        'def kernel(x: Integer[Array, "N<=4"], j: Integer, expected: Integer):\n'
        "    y = -1\n"
        "    if j < len(x):\n"
        "        y = x[j]\n"
        "    assert y == expected\n"
    )
    x = np.array([5, 6])
    assert accepts(program, x, 3, -1)
    assert not accepts(program, x, 3, 0)
    assert accepts(program, x, 1, 6)
    assert not accepts(program, x, 1, -1)


GUARDED_DIVISION = (
    'def kernel(x: Integer[Array, "2"], expected: Integer):\n'
    "    y = 0\n"
    "    if x[1] != 0:\n"
    "        y = 12 // x[0]\n"
    "    assert y == expected\n"
)


@pytest.mark.parametrize("backend", ["mock", "halo2-base"])
def test_division_in_untaken_branch(backend):
    program = compile_source(GUARDED_DIVISION)
    assert accepts(program, np.array([0, 0]), 0, backend=backend)
    assert not accepts(program, np.array([0, 0]), 1, backend=backend)
    assert accepts(program, np.array([3, 1]), 4, backend=backend)
    assert not accepts(program, np.array([3, 1]), 0, backend=backend)


def test_division_by_zero_in_taken_branch_rejects():
    program = compile_source(GUARDED_DIVISION)
    for claimed in (0, 12):
        assert not accepts(program, np.array([0, 1]), claimed)


def test_sqrt_domain_in_untaken_arm():
    program = compile_source(
        "def kernel(v: Float, expected: Float):\n"
        "    y = math.sqrt(v) if v >= 0 else 0.0\n"
        "    assert y == expected\n"
    )
    assert accepts(program, -4.0, 0.0)
    assert not accepts(program, -4.0, 2.0)
    assert accepts(program, 4.0, 2.0)


@pytest.mark.parametrize("backend", ["mock", "halo2-base"])
def test_power_exponent_in_untaken_arm(backend):
    # halo2-base decomposes a runtime exponent into 6 bits; 100 has no such
    # decomposition, which must not matter where the power is not computed.
    program = compile_source(
        "def kernel(e: Integer, expected: Integer):\n"
        "    y = 2 ** e if e < 10 else 0\n"
        "    assert y == expected\n"
    )
    assert accepts(program, 100, 0, backend=backend)
    assert not accepts(program, 100, 1, backend=backend)
    assert accepts(program, 3, 8, backend=backend)
    assert not accepts(program, 3, 0, backend=backend)


def test_index_in_unselected_conditional_expression_arm():
    program = compile_source(
        'def kernel(x: Integer[Array, "4"], j: Integer, expected: Integer):\n'
        "    y = x[j] if j < 4 else -1\n"
        "    assert y == expected\n"
    )
    x = np.array([5, 6, 7, 8])
    assert accepts(program, x, 9, -1)
    assert not accepts(program, x, 9, 5)
    assert accepts(program, x, 1, 6)


# ── Comprehension filters ───────────────────────────────────────────────


@pytest.mark.parametrize("backend", ["mock", "halo2-base"])
def test_index_in_filtered_out_element(backend):
    program = compile_source(
        'def kernel(x: Integer[Array, "4"], idx: Integer[Array, "3"], expected: Integer):\n'
        "    vals = [x[i] for i in idx if i < 4]\n"
        "    assert sum(vals) == expected\n"
    )
    x = np.array([5, 6, 7, 8])
    assert accepts(program, x, np.array([1, 9, 3]), 14, backend=backend)
    assert not accepts(program, x, np.array([1, 9, 3]), 6, backend=backend)
    # -9 passes the filter and is out of range: Python raises.
    assert not accepts(program, x, np.array([1, -9, 3]), 14, backend=backend)


def test_division_in_filtered_out_element():
    program = compile_source(
        'def kernel(x: Integer[Array, "3"], expected: Integer):\n'
        "    vals = [12 // v for v in x if v != 0]\n"
        "    assert sum(vals) == expected\n"
    )
    assert accepts(program, np.array([3, 0, 4]), 7)
    assert not accepts(program, np.array([3, 0, 4]), 6)


# ── Facts learned where the code runs ───────────────────────────────────
#
# Before this fix the facts `0 <= j < 4` learned from these asserts were used
# after the code that asserted them, on paths where it never ran: the read
# `x[j]` then had no bound check, and `j = 7` read the (zero) padding.

X4 = np.array([5, 6, 7, 8])


def test_fact_from_chip_called_in_branch_does_not_escape():
    check = ZKChip.from_source(
        "check", "def check(j: Integer) -> Integer:\n    assert j < 4\n    assert j >= 0\n    return j\n"
    )
    program = compile_source(
        'def kernel(x: Integer[Array, "4"], j: Integer, c: Integer, expected: Integer):\n'
        "    if c > 0:\n"
        "        t = check(j)\n"
        "    assert x[j] == expected\n",
        chips=[check],
    )
    assert accepts(program, X4, 1, 0, 6)
    for claimed in (0, 5):
        assert not accepts(program, X4, 7, 0, claimed)


def test_fact_from_chip_after_early_return_does_not_escape():
    check = ZKChip.from_source(
        "check",
        "def check(j: Integer, c: Integer) -> Integer:\n"
        "    if c > 0:\n        return 0\n"
        "    assert j < 4\n    assert j >= 0\n    return j\n",
    )
    program = compile_source(
        'def kernel(x: Integer[Array, "4"], j: Integer, c: Integer, expected: Integer):\n'
        "    t = check(j, c)\n"
        "    assert x[j] == expected\n",
        chips=[check],
    )
    assert accepts(program, X4, 1, 1, 6)
    for claimed in (0, 5):
        assert not accepts(program, X4, 7, 1, claimed)


def test_fact_from_iteration_after_continue_does_not_reach_next_iteration():
    program = compile_source(
        'def kernel(x: Integer[Array, "2"], y: Integer[Array, "4"], j: Integer, expected: Integer):\n'
        "    t = 0\n"
        "    for i in range(2):\n"
        "        if i == 1:\n"
        "            t = y[j]\n"
        "        if x[i] < 0:\n"
        "            continue\n"
        "        assert j >= 0\n"
        "        assert j < 4\n"
        "    assert t == expected\n"
    )
    assert accepts(program, np.array([-1, -1]), X4, 1, 6)
    for claimed in (0, 3):
        assert not accepts(program, np.array([-1, -1]), X4, 7, claimed)
