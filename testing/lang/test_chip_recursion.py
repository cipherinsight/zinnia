"""Regression tests for @zk_chip recursion handling.

Each @zk_chip call is inlined at compile time. A recursive call is
unrolled while the condition under which it is reached may hold; once the
compiler proves that path condition false (from constants, the bounds of
the compared values, or the SMT solver under the program's preconditions)
the call is not inlined. A recursive call that is not proved unreachable
within ZinniaConfig.recursion_limit nested chip calls (default 16) is
rejected with a RecursionLimitExceededError naming the chip.
"""
import pytest

from zinnia import *
from zinnia.api.zk_circuit import ZKCircuit
from zinnia.config.zinnia_config import ZinniaConfig


def test_bounded_recursion_compiles_within_limit():
    """A recursive chip whose unroll depth fits inside the configured
    recursion_limit must compile and execute correctly. We use a
    statically-known argument (factorial(5)) so the compiler can fold
    branches and the unroll terminates at the base case rather than
    descending forever (chip arguments are fully unrolled even when the
    base-case if-cond is statically true at runtime)."""

    @zk_chip
    def factorial(n) -> Integer:
        if n <= 1:
            return 1
        return n * factorial(n - 1)

    @zk_circuit
    def fact5_check(witness: Public[Integer]):
        # `factorial(5)` is statically resolvable to 120.
        assert factorial(5) == witness

    cfg = ZinniaConfig(recursion_limit=8)
    circuit = ZKCircuit.from_method(fact5_check, chips=[factorial], config=cfg)
    # Compiling proves the chip unrolls without overflowing the limit.
    circuit.compile()
    # And the satisfied witness matches 5! = 120.
    assert circuit(120)


def test_recursion_beyond_limit_raises_clear_error():
    """A branching-recursive chip (fibonacci) blows past the limit; the
    compiler must fail fast with a message naming the chip and the limit
    rather than silently unrolling 2**depth nodes."""

    @zk_chip
    def fibo(n) -> Integer:
        if n < 2:
            return n
        return fibo(n - 1) + fibo(n - 2)

    @zk_circuit
    def fibo_test(n: Public[Integer]):
        assert fibo(n) >= 0

    cfg = ZinniaConfig(recursion_limit=4)
    with pytest.raises(Exception) as exc_info:
        ZKCircuit.from_method(fibo_test, chips=[fibo], config=cfg).compile()

    msg = str(exc_info.value)
    assert "RecursionLimitExceeded" in msg
    assert "fibo" in msg
    assert "4" in msg  # the configured limit appears in the message


def test_default_recursion_limit_is_small_enough_to_fail_fast():
    """The default recursion_limit must be small enough that exponential
    recursion fails-fast rather than hanging. We don't pin the exact
    value (it may evolve) — just enforce an upper bound."""
    cfg = ZinniaConfig()
    assert cfg.recursion_limit() <= 32, (
        "Default recursion_limit too high; exponential @zk_chip recursion "
        "will hang the compiler instead of failing fast."
    )


def _compile(circuit, chips, recursion_limit=16):
    cfg = ZinniaConfig(recursion_limit=recursion_limit)
    compiled = ZKCircuit.from_method(circuit, chips=chips, config=cfg)
    compiled.compile()
    return compiled


def test_recursion_depth_bounded_by_precondition():
    """`fibo(n)` with a private `n`: the depth depends on the input, and the
    precondition `n <= 6` lets the compiler prove that the call reached
    after six nested calls (which needs `n - 6 >= 2`) never runs."""

    @zk_chip
    def fibo(n: Integer) -> Integer:
        if n < 2:
            return n
        return fibo(n - 1) + fibo(n - 2)

    @zk_circuit(requires=[lambda n: 0 <= n <= 6])
    def fibo_check(n: Integer, out: Public[Integer]):
        assert fibo(n) == out

    circuit = _compile(fibo_check, [fibo])
    for n, expected in enumerate([0, 1, 1, 2, 3, 5, 8]):
        assert circuit(n, expected).satisfied
    assert not circuit(6, 7).satisfied


def test_recursion_depth_bounded_by_shrinking_arguments():
    """Euclid's algorithm by subtraction: every call lowers `a + b`, so with
    `1 <= a, b <= 5` the solver proves the call after enough nested calls
    unreachable although neither argument is a constant."""

    @zk_chip
    def gcd(a: Integer, b: Integer) -> Integer:
        if a == b:
            return a
        if a > b:
            return gcd(a - b, b)
        return gcd(a, b - a)

    @zk_circuit(requires=[lambda a: 1 <= a <= 5, lambda b: 1 <= b <= 5])
    def gcd_check(a: Integer, b: Integer, out: Public[Integer]):
        assert gcd(a, b) == out

    circuit = _compile(gcd_check, [gcd])
    assert circuit(4, 2, 2).satisfied
    assert circuit(5, 3, 1).satisfied
    assert circuit(5, 5, 5).satisfied
    assert not circuit(4, 2, 4).satisfied


def test_recursive_binary_search_over_a_fixed_array():
    """Binary search halves a constant index range, so its depth is a
    compile-time number while the comparisons depend on private data."""

    @zk_chip
    def search(xs: NDArray[Integer, 8], x: Integer, lo: Integer, hi: Integer) -> Integer:
        if lo > hi:
            return -1
        mid = (lo + hi) // 2
        if xs[mid] == x:
            return mid
        if xs[mid] < x:
            return search(xs, x, mid + 1, hi)
        return search(xs, x, lo, mid - 1)

    @zk_circuit
    def search_check(xs: NDArray[Integer, 8], x: Integer, out: Public[Integer]):
        assert search(xs, x, 0, 7) == out

    circuit = _compile(search_check, [search])
    xs = [1, 3, 5, 7, 9, 11, 13, 15]
    assert circuit(xs, 11, 5).satisfied
    assert circuit(xs, 1, 0).satisfied
    assert circuit(xs, 4, -1).satisfied
    assert not circuit(xs, 11, 4).satisfied


def test_unannotated_recursive_chip_bounded_by_precondition():
    """Without a return annotation, the value standing in for a call that
    never runs takes the type of the values the chip already returned."""

    @zk_chip
    def sum_to(n):
        if n <= 0:
            return 0
        return n + sum_to(n - 1)

    @zk_circuit(requires=[lambda n: n <= 5])
    def sum_check(n: Integer, out: Public[Integer]):
        assert sum_to(n) == out

    circuit = _compile(sum_check, [sum_to])
    assert circuit(5, 15).satisfied
    assert circuit(-3, 0).satisfied
    assert not circuit(5, 14).satisfied


def test_recursion_without_a_provable_bound_is_rejected():
    """The same countdown without a maximum for `n`: the call is reachable at
    every depth, so the program is rejected at the limit, with a message
    naming the chip, rather than truncated."""

    @zk_chip
    def sum_to(n: Integer) -> Integer:
        if n <= 0:
            return 0
        return n + sum_to(n - 1)

    @zk_circuit
    def sum_check(n: Integer, out: Public[Integer]):
        assert sum_to(n) == out

    with pytest.raises(Exception) as exc_info:
        _compile(sum_check, [sum_to], recursion_limit=8)
    msg = str(exc_info.value)
    assert "RecursionLimitExceeded" in msg
    assert "`sum_to`" in msg
    assert "not proven unreachable" in msg


def test_precondition_bound_above_the_limit_is_rejected():
    """A bound exists (`n <= 10`) but exceeds `recursion_limit`: the call is
    still reachable at the limit, so the program is rejected."""

    @zk_chip
    def sum_to(n: Integer) -> Integer:
        if n <= 0:
            return 0
        return n + sum_to(n - 1)

    @zk_circuit(requires=[lambda n: n <= 10])
    def sum_check(n: Integer, out: Public[Integer]):
        assert sum_to(n) == out

    with pytest.raises(Exception, match="RecursionLimitExceeded"):
        _compile(sum_check, [sum_to], recursion_limit=6)
    assert _compile(sum_check, [sum_to], recursion_limit=12)(10, 55).satisfied
