"""Group 1 smoke tests for `sqrt_f`'s domain `requires(x >= 0)`.

The `good` function pairs a satisfying `@requires` annotation with a
`sqrt` call; the precondition discharges Proved and compilation
proceeds without a witness check. The `bad` function omits the
annotation; under the lenient default the discharge is Unknown and a
witness check is emitted, so compile succeeds and the prover would
refuse to forge a witness at proof time.
"""
import os
import pytest

from zinnia import zk_circuit, requires, NDArray, Float
from zinnia.api.zk_circuit import ZKCircuit


def test_sqrt_good_compiles_with_satisfying_requires():
    """`@requires(x >= 0)` discharges `sqrt`'s domain Proved. The int
    operand is implicitly lowered to float before `SqrtF` fires.
    """
    import numpy as np

    @zk_circuit
    @requires(lambda x: x >= 0)
    def good(x: int):
        y = np.sqrt(x)
        _zinnia_result = y

    _ = ZKCircuit.from_method(good).compile()


def test_sqrt_bad_compiles_in_lenient_default():
    """Without an annotation, lenient mode emits a witness check and
    compilation still succeeds.
    """
    import numpy as np

    @zk_circuit
    def bad(x: int):
        y = np.sqrt(x)
        _zinnia_result = y

    prev = os.environ.pop("ZINNIA_OP_REQUIRES_STRICT", None)
    try:
        _ = ZKCircuit.from_method(bad).compile()
    finally:
        if prev is not None:
            os.environ["ZINNIA_OP_REQUIRES_STRICT"] = prev
