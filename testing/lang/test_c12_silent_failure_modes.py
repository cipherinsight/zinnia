"""C12 — silent failure modes.

Three independent "compiles fine, computes the wrong thing" paths, each of
which must now either compute the numpy-correct result or fail loudly:

* 12a — chained subscript write `a[i][j] = v` on a dynamic array was a silent
  no-op; it must now raise a clear error.
* 12b — `ensure_ptr` silently coerced None/Class -> const 0 and List/Tuple ->
  first element; passing a composite to a scalar op must now surface loudly.
* 12c — mixed `Single+Range` reads did not numpy-wrap a negative dynamic
  single index, so `arr[i, 2:5]` with `i == -1` returned the wrong row. It
  must now match numpy. (Fails at 4b69c08 / HEAD-before-fix.)
"""
import pytest

from zinnia import *  # noqa: F401,F403  (np, zk_circuit, NDArray, Integer, ...)
from zinnia.api.zk_circuit import ZKCircuit


# ── 12a ──────────────────────────────────────────────────────────────────
def test_dyn_chained_subscript_write_raises():
    """`a[i][j] = v` on a dynamic array used to drop the write silently.

    It must raise a clear error rather than miscompile. (Chained writes into
    static arrays are flattened to `a[i, j] = v`; dynamic arrays are not,
    because their single-step writes are themselves pending a core fix.)
    """
    source = """
@requires(lambda n: 0 <= n <= 5)
def chained_dyn_write(n: int):
    a = np.zeros((3, n), dtype=int)  # runtime axis -> DynamicNDArray
    a[1][0] = 5
    _zinnia_result = a.sum()
"""
    with pytest.raises(Exception) as excinfo:
        ZKCircuit.from_source("chained_dyn_write", source).compile()
    msg = str(excinfo.value).lower()
    assert "chained subscript" in msg


# ── 12c ──────────────────────────────────────────────────────────────────
def test_mixed_single_range_read_wraps_negative_index():
    """`arr[i, 2:5]` with runtime `i == -1` must read the LAST row (numpy
    semantics), not row 0 / garbage.

    At HEAD-before-fix the read-side `index_equals_coord` compared the raw
    (un-wrapped) index, so `i == -1` matched no candidate row and the select
    chain returned the first candidate — the wrong row.
    """
    @zk_circuit
    def foo(arr: NDArray[Integer, 4, 5], i: int):
        # numpy: arr[-1, 2:5] == [32, 33, 34] for the grid below (last row).
        row = arr[i, 2:5]
        assert row[0] == 32
        assert row[1] == 33
        assert row[2] == 34

    # 0,1,2,3,4 / 10,11,... / 20,21,... / 30,31,32,33,34
    grid = np.asarray([[r * 10 + c for c in range(5)] for r in range(4)])
    assert foo(grid, -1)


def test_mixed_single_range_read_positive_index_still_correct():
    """Control: a non-negative dynamic single index still reads the right row."""
    @zk_circuit
    def foo(arr: NDArray[Integer, 4, 5], i: int):
        row = arr[i, 2:5]
        assert row[0] == 12
        assert row[1] == 13
        assert row[2] == 14

    grid = np.asarray([[r * 10 + c for c in range(5)] for r in range(4)])
    assert foo(grid, 1)


# ── 12b ──────────────────────────────────────────────────────────────────
def test_none_to_scalar_op_surfaces_loudly():
    """Feeding `None` into a scalar IR op must fail loudly rather than being
    silently coerced to constant 0.

    At HEAD-before-fix `ensure_ptr` turned `Value::None` into
    `ir_constant_int(0)`, so `x + None` quietly computed `x + 0`.
    """
    @zk_circuit
    def foo(x: Integer):
        y = None
        z = x + y  # type: ignore[operator]
        _zinnia_result = z

    with pytest.raises(Exception) as excinfo:
        ZKCircuit.from_method(foo).compile()
    assert "ensure_ptr" in str(excinfo.value) or "None" in str(excinfo.value)
