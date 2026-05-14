"""Smoke tests for div/mod/floor_div `requires(rhs != 0)`.

The `good` function pairs a satisfying `@requires` annotation with a
`//` call; the precondition discharges Proved and compilation
proceeds without a witness check. The `bad` function omits the
annotation; under the lenient default the discharge is Unknown and a
witness check is emitted, so compile succeeds and the prover would
refuse to forge a witness at proof time.
"""
import os

from zinnia import zk_circuit, requires
from zinnia.api.zk_circuit import ZKCircuit


def test_floor_div_good_compiles_with_satisfying_requires():
    """`@requires(b != 0)` discharges `floor_div`'s domain Proved."""

    @zk_circuit
    @requires(lambda a, b: b != 0)
    def good(a: int, b: int):
        y = a // b
        _zinnia_result = y

    _ = ZKCircuit.from_method(good).compile()


def test_floor_div_bad_compiles_in_lenient_default():
    """Without an annotation, lenient mode emits a witness check and
    compilation still succeeds.
    """

    @zk_circuit
    def bad(a: int, b: int):
        y = a // b
        _zinnia_result = y

    prev = os.environ.pop("ZINNIA_OP_REQUIRES_STRICT", None)
    try:
        _ = ZKCircuit.from_method(bad).compile()
    finally:
        if prev is not None:
            os.environ["ZINNIA_OP_REQUIRES_STRICT"] = prev
