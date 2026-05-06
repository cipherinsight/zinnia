"""Regression tests for the kanban ticket
``compiler.module-const-and-chip-default-args``.

Two related symptoms produced ``Variable X not found`` errors:

1. Module-level int/float/bool literal constants referenced inside a
   ``@zk_circuit`` or ``@zk_chip`` body could not be resolved because the
   IR generator only sees the function source captured by
   ``inspect.getsource``, not the surrounding module globals.
2. Default-argument values declared on a ``@zk_chip`` (e.g. ``epsilon=1e-6``)
   were dropped, so calling the chip without that argument raised
   ``Variable epsilon not found`` instead of substituting the default.

Both are now handled in
``zinnia.compile.module_constants`` (constants) and the chip transformer +
Rust ``visit_chip_call`` (defaults).
"""
from zinnia import *


# ── Module-level literal constants ────────────────────────────────────────
# These are the kind of declarations the fix must resolve via
# ``method.__globals__`` — a single literal int / float / bool.

MOD_THRESHOLD = 7         # int
MOD_SCALE = 0.5           # float
MOD_FLAG = True           # bool


def test_module_int_constant_in_circuit_body():
    @zk_circuit
    def cmp_threshold(x: Public[Integer]):
        assert x >= MOD_THRESHOLD

    assert cmp_threshold(10)


def test_module_float_constant_in_circuit_body():
    @zk_circuit
    def scale_and_check(x: Public[Float]):
        # MOD_SCALE = 0.5 — referenced via module globals.
        assert x * MOD_SCALE >= 1.0

    assert scale_and_check(4.0)


def test_module_bool_constant_in_circuit_body():
    @zk_circuit
    def flag_check(x: Public[Integer]):
        if MOD_FLAG:
            assert x > 0
        else:
            assert x < 0

    assert flag_check(3)


# ── Module-level constant referenced inside a @zk_chip ───────────────────

CHIP_BIAS = 100


def test_module_int_constant_in_chip_body():
    @zk_chip
    def add_bias(x) -> Integer:
        return x + CHIP_BIAS

    @zk_circuit
    def biased_sum(a: Public[Integer]):
        assert add_bias(a) == 105

    assert biased_sum(5)


# ── Chip default-argument values ─────────────────────────────────────────


def test_chip_default_arg_int():
    @zk_chip
    def offset(x, k=10) -> Integer:
        return x + k

    @zk_circuit
    def use_default(a: Public[Integer]):
        # Call with no `k`, expect default 10 to be used.
        assert offset(a) == 15

    assert use_default(5)


def test_chip_default_arg_float():
    @zk_chip
    def near_zero(x, epsilon=1e-6) -> Integer:
        # Returns 1 if |x| < epsilon, else 0.
        if x < epsilon and x > -epsilon:
            return 1
        else:
            return 0

    @zk_circuit
    def check(a: Public[Float]):
        # 1e-9 < 1e-6 (the default), so near_zero(a) should be 1.
        assert near_zero(a) == 1

    assert check(1e-9)


def test_chip_default_arg_overridden_by_positional():
    @zk_chip
    def offset(x, k=10) -> Integer:
        return x + k

    @zk_circuit
    def use_explicit(a: Public[Integer]):
        # Explicit positional overrides the default.
        assert offset(a, 100) == 105

    assert use_explicit(5)


def test_chip_default_arg_overridden_by_kwarg():
    @zk_chip
    def offset(x, k=10) -> Integer:
        return x + k

    @zk_circuit
    def use_kwarg(a: Public[Integer]):
        assert offset(a, k=2) == 7

    assert use_kwarg(5)
