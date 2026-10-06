"""Facts the optimizer assumes must be enforced by the circuit (audit M10, M2).

Each case used to compile to a circuit that proved a different result from
NumPy, or accepted a witness violating a fact the compiler relied on:

- `IntCast` (float -> int) got the interval [0, 1] in range analysis, so
  `np.arange(int(x))` was sized for one element. It is now unbounded, and a
  runtime `arange` stop with no provable bound is a compile error.
- Bounded `np.arange` used the raw span `stop - start` as the length (negative
  for `stop < start`) while its contract claimed a non-negative length; the
  3-argument form and bounded `np.repeat` divided with a field inverse
  (`ir_div_i`), which is not a division when the divisor does not divide.
- Bounded constructors never checked `0 <= runtime length <= envelope`.
- The `nnz` / `popcount` / `fixed_point_count` precondition checks cast each
  boolean indicator with the float->int `IntCast`, which maps 1 to 0: the
  honest `nnz(x) == 3` was rejected and the false `nnz(x) == 0` accepted.
- A precondition the circuit cannot enforce (`cycle_count`, an unsupported
  operator) was kept as a fact without any check; it is now a compile error.
"""
import re

import numpy as np
import pytest

from zinnia import NDArray, Float, Integer, ZinniaException, requires, zk_circuit
from zinnia.api.zk_circuit import ZKCircuit
from zinnia.spec.predicates import cycle_count, fixed_point_count, forall, in_range, max_run, nnz


def prove(fn, *args):
    """(satisfied, publics in exposure order) on the mock backend."""
    proof = ZKCircuit.from_method(fn).compile().prove(*args, backend="mock")
    entries = sorted(proof.public_values, key=lambda e: int(re.search(r"_(\d+)=", e).group(1)))
    return proof.proof_bytes_hex == "mock_satisfied", [int(e.split("=")[1]) for e in entries]


def dynamic_output(fn, *args):
    """Satisfied flag and the runtime prefix of a 1-D dynamic output
    (layout: runtime length, which is also its only dim, then the padded
    payload)."""
    satisfied, publics = prove(fn, *args)
    length = publics[0]
    return satisfied, length, publics[1:1 + max(length, 0)]


@zk_circuit
@requires(lambda r: r <= 3)
@requires(lambda r: r >= -2)
def repeat_bounded(a: NDArray[Integer, 3], r: int):
    return np.repeat(a, r)


@zk_circuit
@requires(lambda n: n <= 9)
@requires(lambda n: n >= -5)
def arange_step_two(n: int):
    return np.arange(0, n, 2)


@zk_circuit
@requires(lambda n: n <= 5)
@requires(lambda n: n >= -5)
def arange_one_arg(n: int):
    return np.arange(n)


@zk_circuit
@requires(lambda n: n <= 5)
@requires(lambda n: n >= -5)
def zeros_bounded(n: int):
    return np.zeros(n, dtype=int)


@pytest.mark.parametrize("fn, arg, expected", [
    (repeat_bounded, 2, np.repeat([1, 2, 3], 2)),
    (repeat_bounded, 0, np.repeat([1, 2, 3], 0)),
    (arange_step_two, 7, np.arange(0, 7, 2)),
    (arange_step_two, 2, np.arange(0, 2, 2)),
    (arange_step_two, -3, np.arange(0, -3, 2)),
    (arange_one_arg, 4, np.arange(4)),
    (arange_one_arg, -2, np.arange(-2)),
    (zeros_bounded, 3, np.zeros(3, dtype=int)),
])
def test_bounded_constructor_matches_numpy(fn, arg, expected):
    args = (np.array([1, 2, 3]), arg) if fn is repeat_bounded else (arg,)
    satisfied, length, values = dynamic_output(fn, *args)
    assert satisfied
    assert length == len(expected)
    assert values == list(expected)


@pytest.mark.parametrize("fn, args", [
    (repeat_bounded, (np.array([1, 2, 3]), -1)),  # NumPy: negative dimensions
    (zeros_bounded, (-1,)),
])
def test_negative_size_is_rejected(fn, args):
    satisfied, _ = prove(fn, *args)
    assert not satisfied


def test_arange_of_int_cast_is_rejected_not_miscompiled():
    @zk_circuit
    def kernel(x: Float):
        return np.arange(int(x)).sum()

    # Used to compile and prove 0 for x = 5.0 (NumPy: 10).
    with pytest.raises(ZinniaException, match="compile-time constant"):
        ZKCircuit.from_method(kernel).compile()


@zk_circuit
@requires(lambda x, k: nnz(x) == k)
def nnz_eq(x: NDArray[Integer, 4], k: int):
    return k


@zk_circuit
@requires(lambda p, k: fixed_point_count(p) == k)
def fixed_points_eq(p: NDArray[Integer, 4], k: int):
    return k


@zk_circuit
@requires(lambda x, k: max_run(x) <= k)
def float_max_run(x: NDArray[Float, 2], k: int):
    return k


@zk_circuit
@requires(lambda b: forall(b, in_range(0, 1)))
def forall_2d(b: NDArray[Integer, 2, 2]):
    return b.sum()


@pytest.mark.parametrize("fn, args, honest", [
    # A multi-dimensional argument used to be checked row by row, and the
    # rows (not numbers) were skipped: nothing was enforced.
    (forall_2d, (np.array([[0, 1], [1, 0]]),), True),
    (forall_2d, (np.array([[0, 1], [2, 0]]),), False),
    (nnz_eq, (np.array([1, 0, 2, 3]), 3), True),
    (nnz_eq, (np.array([1, 0, 2, 3]), 0), False),
    (nnz_eq, (np.array([0, 0, 0, 0]), 0), True),
    (fixed_points_eq, (np.array([0, 2, 1, 3]), 2), True),
    (fixed_points_eq, (np.array([0, 2, 1, 3]), 0), False),
    (float_max_run, (np.array([0.0, 2.5]), 2), False),  # gap 2.5 > 2
    (float_max_run, (np.array([0.0, 2.0]), 2), True),
])
def test_structural_precondition_is_enforced(fn, args, honest):
    satisfied, _ = prove(fn, *args)
    assert satisfied == honest


def test_unenforceable_precondition_is_a_compile_error():
    @zk_circuit
    @requires(lambda p, k: cycle_count(p) == k)
    def kernel(p: NDArray[Integer, 3], k: int):
        return k

    with pytest.raises(ZinniaException, match="cannot be enforced"):
        ZKCircuit.from_method(kernel).compile()


@zk_circuit
@requires(lambda n: n >= 0)
@requires(lambda n: n <= 5)
def write_then_sum_1d(n: int, expected: int):
    a = np.zeros(n, dtype=int)
    a[1] = 5
    assert a.sum() == expected


@zk_circuit
@requires(lambda n: n >= 0)
@requires(lambda n: n <= 5)
def write_then_sum_2d(n: int, expected: int):
    a = np.zeros((3, n), dtype=int)
    a[1, 0] = 5
    assert a.sum() == expected


@pytest.mark.parametrize("fn", [write_then_sum_1d, write_then_sum_2d])
@pytest.mark.parametrize("n", [3, 5])
def test_content_facts_do_not_survive_a_write(fn, n):
    # The "all zeros" fact of `np.zeros` stayed attached to the array after
    # the in-place write (same identity), so `a.sum()` folded to 0: the
    # false claim 0 was accepted and the true sum 5 rejected.
    assert prove(fn, n, 5)[0]
    assert not prove(fn, n, 0)[0]
