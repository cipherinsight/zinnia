"""Regression tests for fancy indexing (`x[idx_array]`) followed by a
reduction. These were divergences found by the Phase 3a fuzzer (Shape D,
seed 31415). The root cause was that `visit_subscript` routed
`StaticArray[StaticArray idx]` through `static_array_subscript`, whose
single-Single arm has no fancy-index path and treated the composite index
as a dynamic-scalar address. See the bug card at
`kanban/cards/compiler/fuzz-finding-grammar-ext-fancy-index-reduce/`.
"""

from zinnia import *


# ---------------------------------------------------------------------------
# Parameter-input (StaticArray) source — the original divergence shape.
# ---------------------------------------------------------------------------

def test_fancy_index_sum_int_param():
    @zk_circuit
    def foo(x: NDArray[Integer, 12]):
        idx = np.asarray([0, 4])
        sub = x[idx]
        out = np.sum(sub)
        assert out == 6

    assert foo(np.asarray([-2, 0, -5, 7, 8, 9, 9, 5, 5, 0, 0, 7]))


def test_fancy_index_min_int_param():
    @zk_circuit
    def foo(x: NDArray[Integer, 12]):
        idx = np.asarray([2, 1, 0])
        sub = x[idx]
        out = np.min(sub)
        assert out == 1

    assert foo(np.asarray([1, 6, 7, 9, 8, 0, 9, -6, -3, 6, -4, 5]))


def test_fancy_index_max_float_param():
    @zk_circuit
    def foo(x: NDArray[Float, 8]):
        idx = np.asarray([1, 2])
        sub = x[idx]
        out = np.max(sub)
        _d = out - (-1.0838787598542103)
        assert _d < 0.001
        assert _d > -0.001

    assert foo(np.asarray([
        0.8142681375312222, -2.665779239079715, -1.0838787598542103,
        2.8485236592096355, 0.6253279694629983, 0.6555172512365224,
        -0.39705700545884515, -0.8300835534005966,
    ]))


# ---------------------------------------------------------------------------
# Literal-source variant: `a = np.asarray([...])` inside the circuit also
# produces a StaticArray today, so the same divergence path applies.
# ---------------------------------------------------------------------------

def test_fancy_index_sum_int_literal():
    @zk_circuit
    def foo():
        a = np.asarray([-2, 0, -5, 7, 8])
        idx = np.asarray([0, 4])
        out = np.sum(a[idx])
        assert out == 6

    assert foo()


def test_fancy_index_matches_take():
    """`x[idx]` and `np.take(x, idx)` must agree element-wise."""
    @zk_circuit
    def foo(x: NDArray[Integer, 12]):
        idx = np.asarray([0, 4, 7, 2])
        a = x[idx]
        b = np.take(x, idx)
        assert (a == b).all()

    assert foo(np.asarray([-2, 0, -5, 7, 8, 9, 9, 5, 5, 0, 0, 7]))
