"""Lang smoke for `compiler.fuzz-finding-v2-slice-oob-witness-miss`.

Slice `arr[i:j]` with dynamic `i`/`j` and no `@requires` annotation must
refuse an out-of-bounds witness — the discharge for `start` and `stop`
fires Phase E in lenient mode, the prover then refuses to forge a
witness for an OOB input.

Mirrors the scalar baseline in `test_op_index_out_of_range_witness.py`:
the scalar arm already emits the discharge (Group 5a); this test pins
the parallel behavior on the slice arm. The annotated variant proves
that a satisfying `@requires` still discharges Proved and the program
runs.
"""
import numpy as np

from zinnia import requires, zk_circuit
from zinnia.lang.operator import Integer, NDArray


def test_scalar_oob_witness_refused_baseline():
    """Scalar `arr[i]` with `i = -1`: discharge_index_in_range fires on
    the dyn-read chokepoint, witness refused.
    """
    @zk_circuit
    def g(x: NDArray[Integer, 8], i: int):
        out = x[i]
        assert out >= -1000000

    r = g(np.array([-1, 10, 8, 4, -1, 2, 1, -10]), -1)
    assert r.satisfied is False


def test_slice_negative_start_witness_refused():
    """Slice `arr[i:j]` with `i = -1`: slice-bound discharge rejects."""
    @zk_circuit
    def f(x: NDArray[Integer, 8], i: int, j: int):
        sub = x[i:j]
        out = np.sum(sub)
        assert out >= -1000000

    r = f(np.array([-1, 10, 8, 4, -1, 2, 1, -10]), -1, 1)
    assert r.satisfied is False


def test_slice_start_past_len_witness_refused():
    """Slice `arr[i:j]` with `i = 7 > len`: slice-bound discharge
    rejects (valid range for `i` is `[0, len]`, i.e. `[0, 6]` here).
    """
    @zk_circuit
    def f(x: NDArray[Integer, 6], i: int, j: int):
        sub = x[i:j]
        out = np.sum(sub)
        assert out >= -1000000

    r = f(np.array([1, 2, 3, 4, 5, 6]), 7, 8)
    assert r.satisfied is False


def test_slice_stop_past_len_witness_refused():
    """Slice `arr[i:j]` with `i == len, j > len`: discharge on `j`
    rejects (`j = 7` against `[0, 7)` for len-6 array).
    """
    @zk_circuit
    def f(x: NDArray[Integer, 6], i: int, j: int):
        sub = x[i:j]
        out = np.sum(sub)
        assert out >= -1000000

    # i == len is allowed by slice semantics; j > len is not.
    r = f(np.array([9, 6, -4, 1, 6, -10]), 6, 7)
    assert r.satisfied is False


def test_slice_in_range_with_requires_satisfied():
    """Annotated slice with a satisfying `@requires` discharges Proved;
    no witness check fires, program compiles and the assertion holds.
    """
    @zk_circuit
    @requires(lambda x, i, j: 0 <= i)
    @requires(lambda x, i, j: i < 8)
    @requires(lambda x, i, j: 0 < j)
    @requires(lambda x, i, j: j <= 8)
    @requires(lambda x, i, j: i < j)
    def f(x: NDArray[Integer, 8], i: int, j: int):
        sub = x[i:j]
        out = np.sum(sub)
        assert out >= -1000000

    r = f(np.array([1, 2, 3, 4, 5, 6, 7, 8]), 2, 5)
    assert r.satisfied is True


def test_slice_start_equals_len_with_requires_satisfied():
    """Slice semantics allow `i == len` (empty trailing slice). With an
    annotated `i <= len`, the discharge passes Proved.
    """
    @zk_circuit
    @requires(lambda x, i: 0 <= i)
    @requires(lambda x, i: i <= 8)
    def f(x: NDArray[Integer, 8], i: int):
        sub = x[i:8]
        out = np.sum(sub)
        assert out >= -1000000

    r = f(np.array([1, 2, 3, 4, 5, 6, 7, 8]), 8)
    assert r.satisfied is True
