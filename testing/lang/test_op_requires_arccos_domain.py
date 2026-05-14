"""Group 1 smoke tests for `arccos_f`'s domain `requires(-1 <= x <= 1)`.

The `good` function annotates the closed-interval bound as a chained
comparison; the precondition discharges Proved. The `bad` function
omits the annotation; lenient mode emits a witness check and compile
succeeds.
"""
import os
import pytest

from zinnia import zk_circuit, requires, NDArray, Float
from zinnia.api.zk_circuit import ZKCircuit


def test_arccos_good_compiles_with_satisfying_requires():
    """`@requires(-1 <= x <= 1)` proves the closed-interval domain. The
    int operand is implicitly lowered to float before `ArcCosF` fires.
    """
    import numpy as np

    @zk_circuit
    @requires(lambda x: -1 <= x <= 1)
    def good(x: int):
        y = np.arccos(x)
        _zinnia_result = y

    _ = ZKCircuit.from_method(good).compile()


def test_arccos_bad_compiles_in_lenient_default():
    """Without an annotation, lenient mode emits a witness check and
    compilation still succeeds.
    """
    import numpy as np

    @zk_circuit
    def bad(x: int):
        y = np.arccos(x)
        _zinnia_result = y

    prev = os.environ.pop("ZINNIA_OP_REQUIRES_STRICT", None)
    try:
        _ = ZKCircuit.from_method(bad).compile()
    finally:
        if prev is not None:
            os.environ["ZINNIA_OP_REQUIRES_STRICT"] = prev
