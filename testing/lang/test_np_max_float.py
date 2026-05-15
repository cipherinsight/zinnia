"""Regression tests for select_value float-typing fix.

Card: compiler.fix-prover-kernel-float-sqrt (actual fix lives in
`helpers::value_ops::select_value`, not sqrt; see card README for the
diagnosis walk-through).

Before the fix, `select_value`'s scalar fallback routed two float operands
through `ir_select_i`, which retyped the result as `Value::Integer` while
the cell still carried Q32 float bits. A subsequent op against another float
(`out - 4.0`) would then `ensure_float` the mislabeled side, multiplying by
2^32 a second time and yielding ~2^32 garbage. The asserts below would have
failed pre-fix.
"""
import numpy as np
from zinnia import zk_circuit
from zinnia.lang.operator import NDArray, Float


@zk_circuit
def f_max(x: NDArray[Float, 4]):
    out = np.max(x)
    diff = out - 4.0
    assert diff < 1e-6
    assert diff > -1e-6


def test_np_max_float():
    result = f_max(np.array([1.0, 2.0, 3.0, 4.0]))
    assert result.satisfied


@zk_circuit
def f_min(x: NDArray[Float, 4]):
    out = np.min(x)
    diff = out - 1.0
    assert diff < 1e-6
    assert diff > -1e-6


def test_np_min_float():
    result = f_min(np.array([1.0, 2.0, 3.0, 4.0]))
    assert result.satisfied
