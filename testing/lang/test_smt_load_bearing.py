"""Smoke tests for the SMT-invocation load-bearing wiring.

These programs exercise patterns where the layered resolver actually has to
fire beyond the `static_val` fast path: the integer obligation needs range
analysis or SMT to discharge. They aren't strictly assertions about which
layer resolves them (range vs SMT) — that's an implementation detail — but
they DO assert end-to-end compilation success on patterns where a static_val-
only resolver would reject the program.

Card: compiler.smt-invocation-load-bearing
"""
import numpy as np
from zinnia import *


def test_shape_axis_via_smt_required_obligation():
    """`np.zeros(bound)` where `bound = (x - x) + 16`.

    The IR builder doesn't constant-fold `x - x` (both operands are runtime
    ScalarValues), so the resulting `bound` has no `static_val`. The
    `SiteKind::ShapeAxis(0)` chokepoint must escalate to range / SMT to
    prove `bound == 16`. Without that escalation, this program would be
    rejected with `shape element at axis 0 must be a compile-time constant
    int`.
    """
    @zk_circuit
    def shape_via_smt(x: int):
        bound = (x - x) + 16
        arr = np.zeros(bound, dtype=Integer)
        arr[0] = 7
        _zinnia_result = arr[0]

    assert shape_via_smt(42)


def test_binary_search_dyn_index_bound():
    """Classic binary search on a fixed-size sorted array.

    `arr[mid]` is read with a runtime-computed `mid = (lo + hi) // 2`. The
    `dyn_index_bound` chokepoint asks the resolver to prove
    `mid ∈ [0, 16)` per iteration. Range analysis pins it tightly because
    `lo`, `hi` themselves carry tight intervals. The program compiles to a
    bounded-unroll circuit and returns the index of the target on each
    iteration where the guard matches.
    """
    @zk_circuit
    def binsearch(arr: NDArray[Integer, 16], target: int):
        lo = 0
        hi = 16
        found = -1
        while lo < hi:
            mid = (lo + hi) // 2
            v = arr[mid]
            if v == target:
                found = mid
                break
            elif v < target:
                lo = mid + 1
            else:
                hi = mid
        _zinnia_result = found

    a = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31])
    # target=13 → expected index 6
    assert binsearch(a, 13)


def test_dyn_index_via_smt_required_obligation():
    """`arr[idx]` where `idx = (x - x) + 3` is provably 3.

    The IR builder doesn't constant-fold `x - x` (both operands are
    runtime), so the index has no `static_val`. Interval analysis on
    integer-typed `x` gives `[-∞, ∞] - [-∞, ∞] = [-∞, ∞]`, so the range
    layer can't tighten the bound either. The `dyn_index_bound` chokepoint
    must escalate to SMT to prove the read is in-range. The probe is
    informational — the memory-trace argument enforces soundness at prove
    time — but this is the canonical pattern that exercises the SMT path
    in the resolver.
    """
    @zk_circuit
    def dyn_idx_smt(arr: NDArray[Integer, 16], x: int):
        idx = (x - x) + 3
        _zinnia_result = arr[idx]

    assert dyn_idx_smt(np.arange(16, dtype=np.int64), 99)


def test_argmin_lookup():
    """`symbols[np.argmin(...)]` — paper's make_decision pattern.

    `np.argmin` returns an Integer whose range is exactly `[0, M)` by
    construction (the accumulator only ever holds an iterator index).
    The `dyn_index_bound` chokepoint at `symbols[im]` discharges via the
    range layer.
    """
    @zk_circuit
    def md(E: NDArray[Float, 8], symbols: NDArray[Float, 4]):
        out = np.zeros(8, dtype=Float)
        for i in range(8):
            im = np.argmin(np.abs(E[i] - symbols) ** 2)
            out[i] = symbols[im]
        _zinnia_result = out

    e = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    s = np.array([0.0, 0.33, 0.66, 1.0])
    assert md(e, s)
