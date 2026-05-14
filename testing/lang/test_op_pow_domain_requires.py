"""Smoke tests for `pow` `requires(base != 0 OR exp >= 0)`.

The `good` functions pair a satisfying `@requires` annotation with a
`**` call; either branch of the OR-form precondition discharges Proved
and compilation proceeds without a witness check. The `bad` function
omits the annotation; under the lenient default the discharge is
Unknown and a witness check is emitted, so compile succeeds and the
prover would refuse to forge a witness at proof time.
"""
import os

from zinnia import zk_circuit, requires
from zinnia.api.zk_circuit import ZKCircuit


def test_pow_good_compiles_with_nonzero_base_requires():
    """`@requires(b != 0)` discharges `pow`'s domain via the first branch."""

    @zk_circuit
    @requires(lambda b, e: b != 0)
    def good(b: int, e: int):
        y = b ** e
        _zinnia_result = y

    _ = ZKCircuit.from_method(good).compile()


def test_pow_good_compiles_with_nonneg_exp_requires():
    """`@requires(e >= 0)` discharges `pow`'s domain via the second branch."""

    @zk_circuit
    @requires(lambda b, e: e >= 0)
    def good(b: int, e: int):
        y = b ** e
        _zinnia_result = y

    _ = ZKCircuit.from_method(good).compile()


def test_pow_bad_compiles_in_lenient_default():
    """Without an annotation, lenient mode emits a witness check and
    compilation still succeeds.
    """

    @zk_circuit
    def bad(b: int, e: int):
        y = b ** e
        _zinnia_result = y

    prev = os.environ.pop("ZINNIA_OP_REQUIRES_STRICT", None)
    try:
        _ = ZKCircuit.from_method(bad).compile()
    finally:
        if prev is not None:
            os.environ["ZINNIA_OP_REQUIRES_STRICT"] = prev
