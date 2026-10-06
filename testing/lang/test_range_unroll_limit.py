"""A `range` with a runtime bound is unrolled to its proven maximum trip
count.

- A maximum too large to unroll (an `int64` value's full range) is no usable
  bound: collecting that many items aborted with Rust's "capacity overflow";
  it is rejected as an unbounded loop.
- The bound of `n - 1` ignored the precondition on `n`: the range pass bounds
  an input by its dtype only, and that answer was taken as it was, so
  `range(n - 1, 0, -1)` under `@requires(0 <= n <= 8)` was unbounded
  (pythran guerre) while `range(n + 1, 0, -1)` compiled. The bound is now
  narrowed by the facts."""
import numpy as np
import pytest

from testing.lang.mock_outputs import run_mock_layout
from zinnia import NDArray, ZKCircuit, requires, zk_circuit
from zinnia.debug.exception.base import ZinniaException


@zk_circuit
@requires(lambda n: 0 <= n <= 4611686018427387904)
def huge_bound(n: int):
    s = 0
    for i in range(n):
        s += i
    return s


@zk_circuit
@requires(lambda n: 0 <= n <= 6)
def small_bound(n: int):
    s = 0
    for i in range(n):
        s += i
    return s


def test_a_bound_too_large_to_unroll_is_rejected_as_unbounded():
    with pytest.raises(ZinniaException, match="loop bound depends on a private value with no "
                                              "provable maximum: the proven bound .* is "
                                              "4611686018427387904"):
        ZKCircuit.from_method(huge_bound).compile()


def test_a_small_bound_still_unrolls():
    program = ZKCircuit.from_method(small_bound).compile()
    assert run_mock_layout(program, 4)[:2] == (True, [6])
    assert not run_mock_layout(program, 7)[0]


@zk_circuit
@requires(lambda n: 0 <= n <= 8)
def countdown_from_n_minus_one(a: NDArray[float, 8], n: int):
    s = 0.0
    for i in range(n - 1, 0, -1):
        s = s + a[i] * i
    for i in range(0, n - 2):
        s = s - a[i]
    return s


@pytest.mark.parametrize("n", [0, 1, 2, 5, 8])
def test_a_range_from_n_minus_one_is_bounded_by_the_precondition(n):
    program = ZKCircuit.from_method(countdown_from_n_minus_one).compile()
    a = np.arange(8.0) / 4
    expected = sum(a[i] * i for i in range(n - 1, 0, -1)) - sum(a[i] for i in range(0, n - 2))
    satisfied, values, _ = run_mock_layout(program, a, n)
    assert satisfied
    np.testing.assert_allclose(values, [expected], atol=1e-6)


def test_the_precondition_still_holds():
    program = ZKCircuit.from_method(countdown_from_n_minus_one).compile()
    assert not run_mock_layout(program, np.zeros(8), 9)[0]
