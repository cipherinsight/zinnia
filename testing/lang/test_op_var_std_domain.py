"""Group 3e smoke tests for `var`/`std`'s domain `requires(len(arr) >= 1)`.

Both ops bind the multi-formal `len_arr` to the array length at the call
site — static-array path uses `flatten_composite(arr).len()` as an IR
constant, dyn-array path uses `runtime_length`. A non-empty static array
makes the substituted requires `<lit-N> >= 1` (literal length, planted
ge-zero by `ir_constant_int`), which the resolver discharges Proved.
The ensures `Output >= 0.0` lands on the variance / std output's
fact bucket.
"""

from zinnia import zk_circuit
from zinnia.api.zk_circuit import ZKCircuit


def test_var_nonempty_static_array_compiles():
    """`np.var` on a length-8 static array: `len_arr == 8` literal, the
    `len_arr >= 1` requires discharges Proved without a witness check.
    """
    import numpy as np

    @zk_circuit
    def good():
        a = np.asarray([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        v = np.var(a)
        _zinnia_result = v

    _ = ZKCircuit.from_method(good).compile()


def test_std_nonempty_static_array_compiles():
    """`np.std` on a length-8 static array: same shape as `var`, plus
    the inner `sqrt` ensures non-negativity in its own right.
    """
    import numpy as np

    @zk_circuit
    def good():
        a = np.asarray([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        s = np.std(a)
        _zinnia_result = s

    _ = ZKCircuit.from_method(good).compile()
