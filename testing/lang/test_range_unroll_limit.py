"""A `range` with a runtime bound is unrolled to its proven maximum trip
count. A maximum too large to unroll (an `int64` value's full range) is no
usable bound: collecting that many items aborted with Rust's "capacity
overflow" (pythran guerre); it is rejected as an unbounded loop."""
import numpy as np
import pytest

from testing.lang.mock_outputs import run_mock_layout
from zinnia import ZKCircuit, requires, zk_circuit
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
