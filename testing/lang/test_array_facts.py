"""Facts about whole arrays (`docs/array-facts.md`).

An array fact describes one array value: a list of cells, or an IR version
after a write at a runtime position. Every name of an array (an alias, a view)
reads its storage's current value, so a write through any of them gives the
next reader a new value, about which the old facts say nothing; a copy NumPy
would share is linked, and a write makes it stale. A fact derived from what
is known where some code runs is used only there.

Each test shows either that a fact a program or an operation states is used
(a check is not emitted) and justified (an input violating it has no proof;
the outputs equal NumPy's on the mock, and halo2-base and Noir prove the
mock's values), or that a fact that no longer holds, or does not hold on
that path, is not used.

AF-1: a `forall` bound was keyed by the array value's identity and the
input's name, both kept by an in-place write, so after `a[0] = x` the
reduction `np.min(a)` still got `0 <= min(a) <= 4`: the index `c[np.min(a)]`
was neither wrapped nor checked, and `x = -1` proved `c[-1] == 0` where NumPy
gives 50.
"""
import re

import numpy as np
import pytest

from testing.lang.all_backends import assert_all_backends_match_numpy, nargo_available, noir_field_publics
from testing.lang.mock_outputs import numpy_outputs, run_mock, run_mock_layout
from zinnia import ZKCircuit


def compile_source(source):
    return ZKCircuit.from_source("kernel", source).compile()


def classes(program):
    return [s["ir_instance"]["__class__"] for s in program.get_ir_stmts()]


def index_checks(program):
    """The checks a runtime index adds, in the forms the lowering gives them:
    its bound (`AssertU128LtIR`), the hinted sign of a possibly negative
    index (a `negative` hint), or the hinted one-hot of its position in
    NumPy's window `[-n, n)` (a `position_one_hot` hint, which checks that
    window as it wraps). (`AssertU128LtBitIR` is not one: it is the
    index-bits construction of a read, which a known range makes possible;
    nor is an `AssertBitIR` of a Boolean input's domain; nor an
    `AssertU128LtIR` of an input cell (less a constant) against a constant,
    the form the lowering layer gives a `forall(a, in_range(lo, hi))`
    precondition: one range check `0 <= a[i] - lo < hi - lo + 1` per element,
    `docs/array-facts-lowering.md`.)"""
    stmts = program.get_ir_stmts()
    constants = {s["stmt_id"] for s in stmts if s["ir_instance"]["__class__"] == "ConstantIntIR"}
    inputs = {s["stmt_id"] for s in stmts if s["ir_instance"]["__class__"].startswith("Read")}
    inputs |= {s["stmt_id"] for s in stmts if s["ir_instance"]["__class__"] == "SubIIR"
               and s["arguments"][0] in inputs and s["arguments"][1] in constants}

    def precondition_range(s):
        return s["arguments"][0] in inputs and s["arguments"][1] in constants

    out = []
    for s in stmts:
        inst = s["ir_instance"]
        if inst["__class__"] == "AssertU128LtIR" and not precondition_range(s):
            out.append("bound")
        elif inst["__class__"] == "HintIR" and inst["ir_data"]["kind"] == "negative":
            out.append("sign")
        elif inst["__class__"] == "HintIR" and inst["ir_data"]["kind"] == "position_one_hot":
            out.append("position")
    return out


def without_requires(source):
    return re.sub(r"@requires\(.*\)\n", "", source)


def has_no_proof(source, *args):
    satisfied, _ = run_mock(compile_source(source), *args)
    return not satisfied


# ── Sources stated by the program: `@requires` ──────────────────────────

FORALL_INDEX = """
@requires(lambda a: forall(a, in_range(0, 4)))
def kernel(a: Integer[Array, '4']):
    c = np.array([10, 20, 30, 40, 50])
    return c[np.max(a)]
"""


def test_a_forall_bound_discharges_the_index_of_a_reduction():
    # The maximum is one of the elements, in [0, 4]: no wrap, no bound check.
    assert index_checks(compile_source(FORALL_INDEX)) == []
    assert index_checks(compile_source(without_requires(FORALL_INDEX))) != []
    assert_all_backends_match_numpy(FORALL_INDEX, np.array([3, 0, 4, 1]))


def test_an_input_violating_the_forall_has_no_proof():
    assert has_no_proof(FORALL_INDEX, np.array([3, 0, 5, 1]))


FORALL_SYMBOLIC = """
@requires(lambda a, k: forall(a, in_range(0, k)))
@requires(lambda a, k: 0 <= k <= 4)
def kernel(a: Integer[Array, '4'], k: Integer):
    c = np.array([10, 20, 30, 40, 50])
    return c[np.min(a)]
"""


@pytest.mark.parametrize("a, k", [([3, 1, 2, 1], 3), ([0, 0, 0, 0], 0), ([4, 4, 2, 3], 4)])
def test_a_forall_bound_by_an_input(a, k):
    # `0 <= a[i] <= k <= 4`: the bound is the statement reading `k`.
    assert index_checks(compile_source(FORALL_SYMBOLIC)) == []
    assert_all_backends_match_numpy(FORALL_SYMBOLIC, np.array(a), k)


def test_a_forall_bound_by_an_input_is_checked_against_it():
    assert has_no_proof(FORALL_SYMBOLIC, np.array([3, 1, 2, 4]), 3)


FORALL_KINDS = {
    "lt_bound": ("forall(a, lt_bound(5)))\n@requires(lambda a: forall(a, is_nonneg)", [4, 0, 3, 1], [5, 0, 3, 1]),
    "is_nonneg": ("forall(a, is_nonneg)", [7, 0, 3, 1], [7, -1, 3, 1]),
    "eq_const": ("forall(a, eq_const(2))", [2, 2, 2, 2], [2, 2, 3, 2]),
}


@pytest.mark.parametrize("name", sorted(FORALL_KINDS))
def test_the_other_forall_predicates(name):
    predicate, good, bad = FORALL_KINDS[name]
    source = (f"@requires(lambda a: {predicate})\n"
              "def kernel(a: Integer[Array, '4']):\n"
              "    c = np.arange(8) * 10\n"
              "    return c[np.min(a)]\n")
    if name != "is_nonneg":
        # Both ends known: no check at all (is_nonneg leaves the upper one).
        assert index_checks(compile_source(source)) == []
    assert "sign" not in index_checks(compile_source(source)), "a non-negative index needs no wrap"
    assert_all_backends_match_numpy(source, np.array(good))
    assert has_no_proof(source, np.array(bad))


# ── AF-1 and its relatives: writes through any name expire the facts ──────

STALE_FORALL = """
@requires(lambda a, x: forall(a, in_range(0, 4)))
def kernel(a: Integer[Array, '4'], x: Integer):
    {write}
    c = np.array([10, 20, 30, 40, 50])
    return c[np.min(a)]
"""

WRITES = {
    "direct": "a[0] = x",
    "alias": "b = a\n    b[0] = x",
    "view": "v = a[1:]\n    v[2] = x",
    "strided view": "v = a[::2]\n    v[1] = x",
    "reshaped view": "v = a.reshape(2, 2)\n    v[1, 0] = x",
    "container": "l = [a]\n    l[0][0] = x",
    "augmented": "a[0] -= 5 - x",
    "runtime position": "a[x % 4] = x",
    "slice": "a[1:3] = x",
    "mask": "a[a == a[0]] = x",
    "in place op through alias": "b = a\n    b -= 2 - x",
    "out=": "np.add(a, x - 4, out=a)",
}


@pytest.mark.parametrize("x", [2, -1])
@pytest.mark.parametrize("name", sorted(WRITES))
def test_a_forall_bound_does_not_outlive_a_write(name, x):
    source = STALE_FORALL.format(write=WRITES[name])
    assert_all_backends_match_numpy(source, np.array([1, 2, 3, 0]), x)


@pytest.mark.parametrize("name", ["direct", "alias", "view", "container"])
def test_a_written_value_out_of_numpy_range_has_no_proof(name):
    # min(a) = -6: c[-6] raises in NumPy.
    assert has_no_proof(STALE_FORALL.format(write=WRITES[name]), np.array([1, 2, 3, 0]), -6)


CONDITIONAL_STALE = """
@requires(lambda a, x, c: forall(a, in_range(0, 4)))
def kernel(a: Integer[Array, '4'], x: Integer, c: Integer):
    if c > 0:
        a[2] = x
    d = np.array([10, 20, 30, 40, 50])
    return d[np.max(a)]
"""


@pytest.mark.parametrize("x, c", [(1, 1), (-2, 1), (-2, 0), (9, 0)])
def test_a_write_under_a_branch_widens_the_array_where_it_may_run(x, c):
    # Where `c > 0` the array holds `x` and NumPy wraps `d[-2]`; elsewhere
    # the precondition's range still bounds it.
    assert_all_backends_match_numpy(CONDITIONAL_STALE, np.array([1, 2, 3, 0]), x, c)


def test_a_write_under_a_branch_out_of_range_where_it_runs_has_no_proof():
    assert has_no_proof(CONDITIONAL_STALE, np.array([1, 2, 3, 0]), 7, 1)


LOOP_STALE = """
@requires(lambda a, x, n: forall(a, in_range(0, 4)))
@requires(lambda a, x, n: 0 <= n <= 4)
def kernel(a: Integer[Array, '4'], x: Integer, n: Integer):
    for k in range(n):
        if a[k] == 0:
            break
        a[k] = x
    d = np.array([10, 20, 30, 40, 50])
    return d[np.max(a)], d[np.min(a)]
"""


@pytest.mark.parametrize("a, x, n", [
    ([1, 2, 3, 4], -1, 2),
    ([1, 0, 3, 4], -1, 4),
    ([1, 2, 3, 4], -3, 0),
    ([2, 2, 2, 2], 4, 4),
])
def test_writes_in_a_loop_with_an_early_exit(a, x, n):
    assert_all_backends_match_numpy(LOOP_STALE, np.array(a), x, n)


# ── Facts derived where some code runs are used only there ──────────────

BRANCH_LOCAL = """
def kernel(a: Integer[Array, '4'], c: Integer, i: Integer):
    b = a + 0
    if c > 0:
        assert a[0] >= 0 and a[0] <= 3
        b = a[0] + np.zeros(4, dtype=int)
    d = np.array([10, 20, 30, 40])
    return d[b[i]]
"""


@pytest.mark.parametrize("a, c, i", [
    ([2, 9, 1, -1], 1, 3),
    ([2, -1, 1, 3], 0, 1),
    ([2, 9, 1, -1], 0, 3),
])
def test_a_fact_from_a_branch_is_not_used_after_it(a, c, i):
    # Inside the branch `b`'s cells are `a[0]`, known in [0, 3] there only;
    # after the merge an index of `b` comes from `a` on the other path.
    assert_all_backends_match_numpy(BRANCH_LOCAL, np.array(a), c, i)


def test_a_fact_from_a_branch_does_not_drop_a_check_outside_it():
    # `a[i] = 9` is out of range for `d` where `c <= 0`.
    assert has_no_proof(BRANCH_LOCAL, np.array([2, 9, 1, -1]), 0, 1)


# ── Operation sources: each output's range reaches a later index ────────

SOURCES = {
    # A clipped value is in [0, 4]: `d[c[i]]` needs no bound check.
    "clip": ("def kernel(a: Integer[Array, '4'], i: Integer):\n"
             "    c = np.clip(a, 0, 4)\n    d = np.array([10, 20, 30, 40, 50])\n    return d[c[i]]\n",
             (np.array([-3, 2, 9, 4]), 2)),
    "minimum": ("@requires(lambda a, i: forall(a, is_nonneg))\n"
                "def kernel(a: Integer[Array, '4'], i: Integer):\n"
                "    c = np.minimum(a, 3)\n    d = np.array([10, 20, 30, 40])\n    return d[c[i]]\n",
                (np.array([0, 7, 2, 3]), 1)),
    "where": ("@requires(lambda a, b, m, i: forall(a, in_range(0, 2)))\n"
              "def kernel(a: Integer[Array, '4'], b: Integer[Array, '4'], m: Bool[Array, '4'], i: Integer):\n"
              "    w = np.where(m, a, np.clip(b, 1, 3))\n    d = np.array([10, 20, 30, 40])\n    return d[w[i]]\n",
              (np.array([0, 2, 1, 1]), np.array([5, -5, 2, 3]), np.array([True, False, True, False]), 1)),
    "elementwise": ("@requires(lambda a, i: forall(a, in_range(0, 3)))\n"
                    "def kernel(a: Integer[Array, '4'], i: Integer):\n"
                    "    e = 2 * a - 1\n    d = np.arange(7) * 10\n    return d[e[i] + 1]\n",
                    (np.array([3, 0, 2, 1]), 0)),
    "cumsum": ("@requires(lambda a: forall(a, in_range(0, 3)))\n"
               "def kernel(a: Integer[Array, '4']):\n"
               "    s = np.cumsum(a)\n    d = np.arange(13) * 10\n    return d[s[3]]\n",
               (np.array([3, 0, 2, 3]),)),
    "take": ("@requires(lambda a, k, i: forall(a, in_range(0, 4)))\n"
             "def kernel(a: Integer[Array, '6'], k: Integer[Array, '3'], i: Integer):\n"
             "    t = np.take(a, k)\n    d = np.array([10, 20, 30, 40, 50])\n    return d[t[i]]\n",
             (np.array([4, 0, 1, 2, 3, 4]), np.array([5, 0, 3]), 2)),
    "sort": ("@requires(lambda a, i: forall(a, in_range(0, 4)))\n"
             "def kernel(a: Integer[Array, '5'], i: Integer):\n"
             "    s = np.sort(a)\n    d = np.array([10, 20, 30, 40, 50])\n    return d[s[i]]\n",
             (np.array([4, 0, 1, 2, 3]), 4)),
    "runtime write": ("@requires(lambda a, i, j: forall(a, in_range(0, 3)))\n"
                      "def kernel(a: Integer[Array, '4'], i: Integer, j: Integer):\n"
                      "    a[i] = 4\n    d = np.array([10, 20, 30, 40, 50])\n    return d[a[j]]\n",
                      (np.array([3, 0, 2, 1]), 1, 1)),
    "mask write": ("@requires(lambda a, i: forall(a, in_range(0, 3)))\n"
                   "def kernel(a: Integer[Array, '4'], i: Integer):\n"
                   "    a[a > 1] = 4\n    d = np.array([10, 20, 30, 40, 50])\n    return d[a[i]]\n",
                   (np.array([3, 0, 2, 1]), 0)),
}


def without_facts(source):
    """The program without its precondition (or, for `clip`, without the
    clip): the same operations with nothing known about the inputs."""
    if "@requires" in source:
        return without_requires(source)
    return source.replace("np.clip(a, 0, 4)", "a")


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_an_operation_range_discharges_index_checks(name):
    source, args = SOURCES[name]
    program = compile_source(source)
    # The operation's range reaches the reads of its result: fewer index
    # checks than the same program with nothing known about the inputs.
    assert len(index_checks(program)) < len(index_checks(compile_source(without_facts(source))))
    assert_all_backends_match_numpy(source, *args)


def test_the_operation_facts_are_not_a_precondition():
    # A violating input of the program's own precondition has no proof; the
    # operation facts themselves hold for every input (clip of any value).
    source, _ = SOURCES["minimum"]
    assert has_no_proof(source, np.array([0, -1, 2, 3]), 1)
    source, _ = SOURCES["clip"]
    for a in ([-100, 2, 100, 4], [5, 5, 5, 5]):
        assert_all_backends_match_numpy(source, np.array(a), 0)


# ── Facts across writes: conditional writes, reads before and after ────

CONDITIONAL_WRITE = """
def kernel(a: Integer[Array, '4'], c: Integer, i: Integer):
    b = np.clip(a, 0, 3)
    b[i] = 1
    if c > 0:
        b[1] = 12
    d = np.arange(20)
    return d[b[1]] + d[b[2]]
"""


@pytest.mark.parametrize("c, i", [(1, 0), (0, 0), (1, 1), (0, 1), (-3, 2)])
def test_a_conditional_write_widens_the_cell_it_may_change(c, i):
    assert_all_backends_match_numpy(CONDITIONAL_WRITE, np.array([7, -2, 5, 1]), c, i)


READ_BEFORE_WRITE = """
def kernel(a: Integer[Array, '6'], i: Integer, j: Integer):
    b = np.clip(a, 0, 3)
    b[i] = 2
    x = b[0]
    b[0] = 50
    d = np.arange(60)
    return d[x], d[b[0]], d[b[j]]
"""


@pytest.mark.parametrize("i, j", [(1, 0), (0, 0), (5, 3), (3, 5)])
def test_a_read_keeps_what_it_read_and_a_later_read_sees_the_write(i, j):
    assert_all_backends_match_numpy(READ_BEFORE_WRITE, np.array([9, -4, 1, 2, 3, 0]), i, j)


# ── np.unique: the inverse is bounded by the runtime count ───────────────

UNIQUE_ONE_HOT = """
def kernel(a: Float[Array, '5']):
    vals, idx = np.unique(a, return_inverse=True)
    b = np.zeros((a.size, vals.size))
    b[np.arange(a.size), idx] = 1
    return b
"""


def test_unique_inverse_needs_no_bound_check_against_the_unique_count():
    # `idx[i] <= count - 1` holds by the definition of the inverse (computed
    # in-circuit from comparisons, no hint), and the one-hot's second axis
    # has length `count`: no `idx < count` check is emitted.
    assert "AssertU128LtIR" not in classes(compile_source(UNIQUE_ONE_HOT))


@pytest.mark.parametrize("a", [
    [3.0, 1.0, 3.0, 2.0, 1.0],
    [5.0, 5.0, 5.0, 5.0, 5.0],
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [-1.5, 2.25, -1.5, 0.0, 9.0],
])
def test_unique_one_hot_equals_numpy(a):
    assert_all_backends_match_numpy(UNIQUE_ONE_HOT, np.array(a))


CLAIMED_UNIQUE_ONE_HOT = """
def kernel(a: Float[Array, '5'], claimed: Integer[Array, '5']):
    vals, idx = np.unique(a, return_inverse=True)
    assert np.all(idx == claimed)
    b = np.zeros((a.size, vals.size))
    b[np.arange(a.size), idx] = 1
    return b
"""


def test_the_unique_inverse_without_its_bound_checks_is_still_pinned():
    # The `idx < len(vals)` checks are dropped (the inverse is bounded by
    # definition); the inverse itself stays computed in-circuit: an honest
    # claim of it is accepted and any other is rejected.
    program = compile_source(CLAIMED_UNIQUE_ONE_HOT)
    assert "AssertU128LtIR" not in classes(program)
    a = np.array([3.0, 1.0, 3.0, 2.0, 1.0])
    _, honest = np.unique(a, return_inverse=True)
    satisfied, _ = run_mock(program, a, honest)
    assert satisfied
    for wrong in ([2, 0, 2, 1, 1], [2, 0, 2, 1, 3], [3, 0, 2, 1, 0]):
        satisfied, _ = run_mock(program, a, np.array(wrong))
        assert not satisfied, wrong


UNIQUE_WRITTEN_INVERSE = """
def kernel(a: Float[Array, '5'], k: Integer):
    vals, idx = np.unique(a, return_inverse=True)
    idx[0] = k
    b = np.zeros((a.size, vals.size))
    b[np.arange(a.size), idx] = 1
    return b
"""


def test_a_written_unique_inverse_is_checked():
    # After `idx[0] = k` the inverse's cells are others: the bound must be
    # checked again (`k = 3` is past the 3 distinct values).
    assert "AssertU128LtIR" in classes(compile_source(UNIQUE_WRITTEN_INVERSE))
    a = np.array([3.0, 1.0, 3.0, 2.0, 1.0])
    assert_all_backends_match_numpy(UNIQUE_WRITTEN_INVERSE, a, 2)
    assert has_no_proof(UNIQUE_WRITTEN_INVERSE, a, 3)


# ── Boolean inputs ───────────────────────────────────────────────────────

BOOL_INDEX = """
@requires(lambda m, i: 0 <= i <= 3)
def kernel(m: Bool[Array, '4'], i: Integer):
    d = np.array([10, 20])
    return d[m[i] + 0]
"""


def test_a_boolean_input_element_is_an_index_in_0_1():
    # `m[i]` is read at a runtime position of cells known in [0, 1].
    assert index_checks(compile_source(BOOL_INDEX)) == []
    for i in range(4):
        assert_all_backends_match_numpy(BOOL_INDEX, np.array([True, False, True, False]), i)


def test_a_non_boolean_value_for_a_boolean_input_has_no_proof():
    assert has_no_proof(BOOL_INDEX, np.array([2, 0, 1, 1]), 0)


# ── Permutations: argsort lanes and `is_permutation` ─────────────────────

INVERSE_PERMUTATION = """
def kernel(a: Float[Array, '8']):
    p = np.argsort(a, kind="stable")
    inv = np.empty(8, dtype=int)
    inv[p] = np.arange(8)
    return inv
"""


@pytest.mark.parametrize("a", [
    [3.0, 1.0, 2.0, 8.0, 5.0, 7.0, 4.0, 6.0],
    [1.0, 1.0, 0.0, 2.0, 2.0, 0.0, 1.0, 3.0],
    [0.5] * 8,
])
def test_scatter_by_an_argsort_is_a_permutation_scatter(a):
    # The argsort lanes of one line are a permutation of 0..8 (every sort
    # construction yields NumPy's stable argsort), so `inv[p] = arange` is a
    # permutation scatter with no permutation check of its own; ties included.
    assert_all_backends_match_numpy(INVERSE_PERMUTATION, np.array(a))


WRITTEN_ARGSORT = """
def kernel(a: Float[Array, '4'], k: Integer):
    p = np.argsort(a, kind="stable")
    p[0] = k
    inv = np.zeros(4, dtype=int)
    inv[p] = np.arange(4)
    return inv
"""


@pytest.mark.parametrize("k", [0, 1, 3])
def test_a_written_argsort_is_no_permutation(k):
    # `p[0] = k` may repeat a position: NumPy's scatter then keeps the last
    # write to it, which a permutation scatter would not give.
    assert_all_backends_match_numpy(WRITTEN_ARGSORT, np.array([3.0, 1.0, 2.0, 0.5]), k)


PERMUTATION_REQUIRES = """
@requires(lambda p: is_permutation(p))
def kernel(p: Integer[Array, '4'], x: Float[Array, '4']):
    y = np.zeros(4)
    y[p] = x
    return y
"""


def test_an_asserted_permutation_is_a_permutation_scatter():
    assert_all_backends_match_numpy(PERMUTATION_REQUIRES, np.array([2, 0, 3, 1]), np.array([1.0, 2.0, 3.0, 4.0]))
    assert has_no_proof(PERMUTATION_REQUIRES, np.array([2, 0, 2, 1]), np.array([1.0, 2.0, 3.0, 4.0]))


CLAIMED_INVERSE_PERMUTATION = """
def kernel(a: Float[Array, '8'], claimed: Integer[Array, '8']):
    p = np.argsort(a, kind="stable")
    inv = np.empty(8, dtype=int)
    inv[p] = np.arange(8)
    assert np.all(inv == claimed)
"""


def test_the_inverse_permutation_without_its_permutation_check_is_still_pinned():
    # The scatter by the argsort lanes has no permutation check of its own
    # (the lanes are a permutation by construction); the inverse stays
    # computed in-circuit from them: an honest claim of it is accepted on
    # every backend, and any other claim (a swap of two tied elements, a
    # repeated position, a position out of range) is rejected.
    program = compile_source(CLAIMED_INVERSE_PERMUTATION)
    a = np.array([1.0, 1.0, 0.0, 2.0, 2.0, 0.0, 1.0, 3.0])
    honest = np.empty(8, dtype=int)
    honest[np.argsort(a, kind="stable")] = np.arange(8)
    satisfied, _ = run_mock(program, a, honest)
    assert satisfied
    program.prove(a, honest, backend="halo2-base")
    if nargo_available():
        noir_field_publics(program, a, honest)
    swapped, repeated, out_of_range = honest.copy(), honest.copy(), honest.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    repeated[0] = repeated[1]
    out_of_range[7] = 8
    for wrong in (swapped, repeated, out_of_range):
        satisfied, _ = run_mock(program, a, wrong)
        assert not satisfied, wrong
    with pytest.raises(Exception):
        program.prove(a, swapped, backend="halo2-base")
    if nargo_available():
        with pytest.raises(BaseException):
            noir_field_publics(program, a, swapped)


# ── nonzero / flatnonzero: coordinates in the array, padding 0 ──────────

CROP = """
def kernel(im: NDArray[Integer, 5, 6]):
    rows = np.flatnonzero(im.any(axis=1))
    cols = np.flatnonzero(im.any(axis=0))
    return im[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
"""


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_crop_by_flatnonzero_bounds(seed):
    im = np.zeros((5, 6), dtype=int)
    rng = np.random.default_rng(seed)
    r0, c0 = rng.integers(0, 4), rng.integers(0, 5)
    im[r0:r0 + 2, c0:c0 + 2] = rng.integers(1, 9, size=(2, 2))
    assert_all_backends_match_numpy(CROP, im)


NONZERO_INDEX = """
@requires(lambda m, i: 0 <= i <= 5)
def kernel(m: Integer[Array, '6'], i: Integer):
    nz = np.flatnonzero(m)
    d = np.array([10, 20, 30, 40, 50, 60])
    return d[nz[i]]
"""


def test_a_nonzero_coordinate_is_an_index_of_the_array():
    # Coordinates (and the padding 0) are in [0, 5]: the only check is
    # NumPy's `i < len(nz)`, none against `d`'s 6 elements.
    assert index_checks(compile_source(NONZERO_INDEX)) == ["bound"]
    for i in (0, 2):
        assert_all_backends_match_numpy(NONZERO_INDEX, np.array([0, 3, 0, 1, 7, 0]), i)


WRITTEN_NONZERO = """
def kernel(m: Integer[Array, '6'], x: Integer):
    nz = np.flatnonzero(m)
    nz[0] = x
    d = np.array([10, 20, 30, 40, 50, 60])
    return d[nz[0]]
"""


def test_a_written_nonzero_array_is_checked():
    assert_all_backends_match_numpy(WRITTEN_NONZERO, np.array([0, 3, 0, 1, 7, 0]), 4)
    assert has_no_proof(WRITTEN_NONZERO, np.array([0, 3, 0, 1, 7, 0]), 6)


# ── Merges at branch joins ───────────────────────────────────────────────

MERGED_SORT = """
@requires(lambda a, b, c, i: forall(a, in_range(0, 2)))
@requires(lambda a, b, c, i: forall(b, in_range(1, 4)))
@requires(lambda a, b, c, i: 0 <= i <= 3)
def kernel(a: Integer[Array, '4'], b: Integer[Array, '4'], c: Integer, i: Integer):
    if c > 0:
        s = np.sort(a)
    else:
        s = np.sort(b)
    d = np.array([10, 20, 30, 40, 50])
    return d[s[i]]
"""


@pytest.mark.parametrize("c", [1, 0])
def test_a_merge_of_two_ranges_is_their_join(c):
    # Either side's sorted cells are in [0, 4]: `d[s[i]]` needs no check.
    assert index_checks(compile_source(MERGED_SORT)) == []
    assert index_checks(compile_source(without_requires(MERGED_SORT))) != []
    assert_all_backends_match_numpy(MERGED_SORT, np.array([2, 0, 1, 1]), np.array([4, 1, 3, 2]), c, 3)


# ── Sorted arrays with a runtime length (`test_sorted_reductions.py` has the
# ties and alias writes of the boundary reads) ─────────────────────────────

SORTED_CONDITIONAL_WRITE = """
@requires(lambda n, c: 2 <= n <= 8)
def kernel(n: Integer, c: Integer):
    x = np.arange(n)
    if c > 0:
        x[0] = 100
    return np.max(x), np.argmax(x)
"""


@pytest.mark.parametrize("n, c", [(5, 1), (5, 0), (2, 1)])
def test_a_sorted_array_written_under_a_branch_is_not_sorted(n, c):
    assert_all_backends_match_numpy(SORTED_CONDITIONAL_WRITE, n, c)


# ── Operator preconditions: the divisor of an integer division ────────────

DIVISOR = """
@requires(lambda a, x, i: forall(a, in_range(1, 5)))
def kernel(a: Integer[Array, '4'], x: Integer, i: Integer):
    {write}
    return 60 // a[i]
"""


def divisor_checks(program):
    """The `!= 0` checks of a division's divisor."""
    return [c for c in classes(program) if c == "NotEqualIIR"]


def test_an_array_fact_discharges_a_divisor_check():
    # `a[i]` is read from cells in [1, 5]: the division needs no `!= 0` check.
    assert divisor_checks(compile_source(DIVISOR.format(write="pass"))) == []
    assert divisor_checks(compile_source(without_requires(DIVISOR.format(write="pass")))) != []
    assert_all_backends_match_numpy(DIVISOR.format(write="pass"), np.array([1, 5, 2, 3]), 0, 2)


@pytest.mark.parametrize("write", [
    "a[0] = x",
    "b = a\n    b[1] = x",
    "a[x % 4] = x",
    "if x > 3:\n        a[2] = x - 4",
])
def test_a_divisor_fact_does_not_outlive_a_write(write):
    source = DIVISOR.format(write=write)
    assert divisor_checks(compile_source(source)) != []
    assert_all_backends_match_numpy(source, np.array([1, 5, 2, 3]), 3, 3)
    # A zero divisor has no proof (Zinnia's integer division requires a
    # non-zero divisor; NumPy warns and gives 0), whether or not a fact
    # was known before the write.
    zero = {"a[0] = x": (0, 0), "b = a\n    b[1] = x": (0, 1), "a[x % 4] = x": (0, 0),
            "if x > 3:\n        a[2] = x - 4": (4, 2)}[write]
    assert has_no_proof(source, np.array([1, 5, 2, 3]), *zero)


# ── The ablation switch ──────────────────────────────────────────────────

def test_the_ablation_turns_array_facts_off(monkeypatch):
    # `ZINNIA_ABLATE_ARRAY_FACTS=1` (docs/ablations.md): no query sees an
    # array fact, so the index of `FORALL_INDEX` is checked again (and the
    # program stays correct).
    monkeypatch.setenv("ZINNIA_ABLATE_ARRAY_FACTS", "1")
    assert index_checks(compile_source(FORALL_INDEX)) != []
    assert_all_backends_match_numpy(FORALL_INDEX, np.array([3, 0, 4, 1]))


# ── More sources and consumers ────────────────────────────────────────────

MORE = {
    # argwhere: every packed coordinate is in [0, max extent - 1].
    "argwhere": ("@requires(lambda m, i: 0 <= i <= 1)\n"
                 "def kernel(m: NDArray[Integer, 2, 3], i: Integer):\n"
                 "    w = np.argwhere(m)\n    d = np.array([10, 20, 30])\n    return d[w[0, i]]\n",
                 (np.array([[0, 2, 0], [1, 0, 0]]), 1)),
    # A boolean-mask read: its cells are the input's or 0.
    "mask read": ("@requires(lambda a, i: forall(a, in_range(0, 4)))\n"
                  "@requires(lambda a, i: 0 <= i <= 1)\n"
                  "def kernel(a: Integer[Array, '6'], i: Integer):\n"
                  "    x = a[a > 1]\n    d = np.array([10, 20, 30, 40, 50])\n    return d[x[i]]\n",
                  (np.array([0, 3, 1, 4, 2, 1]), 1)),
    # `array op scalar` on a runtime-shaped array.
    "runtime-shaped op scalar": ("@requires(lambda a, i: forall(a, in_range(0, 3)))\n"
                                 "@requires(lambda a, i: 0 <= i <= 1)\n"
                                 "def kernel(a: Integer[Array, '6'], i: Integer):\n"
                                 "    x = a[a > 0] + 1\n    d = np.array([10, 20, 30, 40, 50])\n    return d[x[i]]\n",
                                 (np.array([0, 3, 1, 0, 2, 1]), 1)),
    # The exponent of an integer power: no `>= 0` check.
    "integer power": ("@requires(lambda a, i: forall(a, in_range(0, 3)))\n"
                      "def kernel(a: Integer[Array, '4'], i: Integer):\n"
                      "    return 2 ** a[i]\n",
                      (np.array([3, 0, 2, 1]), 2)),
}


def sign_checks(program):
    """Statements that only a possibly negative value needs: the hinted wrap
    of an index, and the `>= 0` check of an integer power's exponent."""
    return [s for s in index_checks(program) if s == "sign"] + \
        [c for c in classes(program) if c == "AssertNonNegIIR"]


@pytest.mark.parametrize("name", sorted(MORE))
def test_more_sources_and_consumers(name):
    source, args = MORE[name]
    program = compile_source(source)
    if name == "integer power":
        assert len(sign_checks(program)) < len(sign_checks(compile_source(without_requires(source))))
    elif name == "argwhere":
        assert "bound" not in index_checks(program)
    else:
        assert len(index_checks(program)) <= len(index_checks(compile_source(without_requires(source)))) - 1
    # The Noir backend has no integer power at a runtime exponent.
    assert_all_backends_match_numpy(source, *args, noir=name != "integer power")


BOOL_AND = """
def kernel(m: Bool[Array, '4'], i: Integer, j: Integer):
    return (m[i] + 0) & (m[j] + 0)
"""


def test_boolean_elements_are_combined_as_booleans():
    # `m[i] + 0` is read from cells in [0, 1]: `&` is a logical and, not a
    # bitwise one over 64-bit integers.
    assert "BitAndIIR" not in classes(compile_source(BOOL_AND))
    m = np.array([True, False, True, True])
    for i, j in ((0, 2), (0, 1), (1, 1)):
        assert_all_backends_match_numpy(BOOL_AND, m, i, j)


BOOL_INPUTS_AND = """
def kernel(m: Bool[Array, '4'], n: Bool[Array, '4']):
    return m & n
"""


def test_boolean_input_arrays_are_combined_as_booleans():
    # Every element of a Boolean input is in [0, 1] (its domain is checked):
    # `m & n` is a logical and, and a non-Boolean element has no proof.
    assert "BitAndIIR" not in classes(compile_source(BOOL_INPUTS_AND))
    assert_all_backends_match_numpy(BOOL_INPUTS_AND, np.array([True, False, True, True]),
                                    np.array([True, True, False, True]))
    assert has_no_proof(BOOL_INPUTS_AND, np.array([2, 0, 1, 1]), np.array([1, 1, 0, 1]))


SIZED_BY_A_FACT = """
@requires(lambda a: forall(a, in_range(0, 5)))
def kernel(a: Integer[Array, '4']):
    z = np.zeros(np.max(a) + 1, dtype=int)
    z[a[0]] = 7
    return np.sum(z)
"""


def test_a_capacity_bounded_by_an_array_fact():
    # `np.max(a) + 1 <= 6` sizes the runtime-shaped array.
    assert_all_backends_match_numpy(SIZED_BY_A_FACT, np.array([2, 5, 0, 1]))
    assert has_no_proof(SIZED_BY_A_FACT, np.array([2, 6, 0, 1]))


WRITTEN_PERMUTATION = """
@requires(lambda p, k: is_permutation(p))
def kernel(p: Integer[Array, '4'], x: Float[Array, '4'], k: Integer):
    q = p
    q[0] = k
    y = np.zeros(4)
    y[p] = x
    return y
"""


@pytest.mark.parametrize("k", [0, 2, 3])
def test_a_permutation_written_through_another_name_is_no_permutation(k):
    # After `q[0] = k` the array `p` may repeat a position: the scatter keeps
    # NumPy's last write per position instead of assuming a permutation.
    assert_all_backends_match_numpy(WRITTEN_PERMUTATION, np.array([2, 0, 3, 1]), np.array([1.0, 2.0, 3.0, 4.0]), k)
