"""Group 1 smoke tests for `log_f`'s domain `requires(x > 0)`.

The `good` function uses `@requires(x >= 1)`, which transitively proves
the strict-positive precondition. The `bad` function omits the
annotation; lenient mode emits a witness check and compile succeeds.
"""
import os
import pytest

from zinnia import zk_circuit, requires, NDArray, Float
from zinnia.api.zk_circuit import ZKCircuit


def test_log_good_compiles_with_satisfying_requires():
    """`@requires(x >= 1)` transitively proves `x > 0`. The int operand
    is implicitly lowered to float before `LogF` fires.
    """
    import numpy as np

    @zk_circuit
    @requires(lambda x: x >= 1)
    def good(x: int):
        y = np.log(x)
        _zinnia_result = y

    _ = ZKCircuit.from_method(good).compile()


def test_log_bad_compiles_in_lenient_default():
    """Without an annotation, lenient mode emits a witness check and
    compilation still succeeds.
    """
    import numpy as np

    @zk_circuit
    def bad(x: int):
        y = np.log(x)
        _zinnia_result = y

    prev = os.environ.pop("ZINNIA_OP_REQUIRES_STRICT", None)
    try:
        _ = ZKCircuit.from_method(bad).compile()
    finally:
        if prev is not None:
            os.environ["ZINNIA_OP_REQUIRES_STRICT"] = prev
