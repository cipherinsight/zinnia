"""Regression tests for @zk_chip recursion handling.

Each recursive @zk_chip call is unrolled at compile time. Branching
recursions (e.g. fibo(n-1) + fibo(n-2)) explode as 2**depth and used to
silently hang the unroller. The compiler now caps unroll depth at
ZinniaConfig.recursion_limit (default 16) and raises a clear
RecursionLimitExceededError pointing at the offending chip.
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
