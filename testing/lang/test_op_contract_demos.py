"""Demo corpus for `compiler.op-contract-corpus-demos`.

Each test compiles a small `@zk_circuit` that exercises one or more op
contracts (from `compiler.op-contract-content`) and shows the framework
pipeline closes end-to-end:

    op contract template (in registry)
       ↓ instantiate at call site
    Fact landed in `IRBuilder.facts.per_stmt[output_ptr]`
       ↓ chokepoint consults facts when resolver is stumped
    `require_static_or_bounded_int` returns Bounded { min, max }
       ↓ ShapeAxis / RangeBound / ... accepts the bound
    compilation proceeds without a user-written `@requires`

Most demos are pure smoke tests at this stage: they confirm a program
that uses a contract'd op compiles cleanly under SMT-on. The
fact-fallback chokepoint test in `src/optim/resolver.rs` covers the
unit-level "facts unblock the chokepoint" claim end-to-end with no
external state.

Demos that are *load-bearing* — i.e., that would fail to compile
without op contracts — are marked with the `LOAD-BEARING:` prefix in
their docstrings. Currently most use of op-contract facts is mediated
by ops whose ptr-anchored quantities are also accessible via other
resolver paths, so the load-bearing demo set is small for v1.
"""
import os
import pytest

from zinnia import zk_circuit, NDArray, Integer
from zinnia.api.zk_circuit import ZKCircuit


# ── 1. dyn-ndarray filter compiles (smoke test for `dyn_filter` contract) ─

def test_dyn_filter_contract_fires_during_compile():
    """Compiling a circuit that uses `.filter(...)` exercises `dyn_filter`'s
    contract registration + firing. Smoke test: no assertion on the fact
    content (the IR doesn't surface FactStack to Python today); the
    Rust-side `op_contract_registry_has_dyn_filter_entry` test covers the
    template side, and the chokepoint-fallback unit test covers the
    consumer side.
    """
    import numpy as np

    @zk_circuit
    def foo():
        a = np.asarray([10, 20, 30, 40, 50])
        dyn_a = np.promote_to_dynamic(a)
        mask = np.asarray([True, False, True, False, True])
        result = dyn_a.filter(mask)
        # Just exercise the output; we don't care about specific values.
        _zinnia_result = result.sum()

    # Just compiling without error is the success condition.
    _ = ZKCircuit.from_method(foo).compile()


# ── 2. dyn-ndarray concatenate compiles (smoke test for `dyn_concatenate`) ─

def test_dyn_concatenate_contract_fires_during_compile():
    """Compiling a circuit that uses `np.concatenate` on dyn-ndarrays
    exercises `dyn_concatenate`'s contract. Same smoke-test caveat as the
    filter demo.
    """
    import numpy as np

    @zk_circuit
    def foo():
        a = np.asarray([1, 2, 3])
        b = np.asarray([4, 5, 6])
        dyn_a = np.promote_to_dynamic(a)
        dyn_b = np.promote_to_dynamic(b)
        c = np.concatenate((dyn_a, dyn_b))
        _zinnia_result = c.sum()

    _ = ZKCircuit.from_method(foo).compile()


# ── 3. Pure scalar bound unlocks np.zeros(k) — moved here from
#       test_scalar_precondition.py as a corpus exemplar. The
#       canonical home is still in that file; this is a pointer.

def test_scalar_bound_unlocks_zeros_canonical():
    """LOAD-BEARING: this is the canonical "user @requires + op contract"
    demo. Compiles `np.zeros(k)` with `0 <= k <= 16` and the dyn-ndarray
    path falls out via `dyn_fill_with_active`, which fires the
    `Var(Output) >= 0` contract.

    The full test (with SMT-on/off flip) lives at
    `test_scalar_precondition.py::test_pure_scalar_bound_unlocks_dyn_ndarray_construction`
    — this stub is a corpus pointer so the demo file lists every
    contract-relevant load-bearing scenario in one place.
    """
    from zinnia import requires
    import numpy as np

    @zk_circuit
    @requires(lambda k: 0 <= k <= 16)
    def foo(k: int):
        out = np.zeros(k, dtype=Integer)
        _zinnia_result = out

    prev = os.environ.get("ZINNIA_SMT_ENABLE")
    os.environ["ZINNIA_SMT_ENABLE"] = "1"
    try:
        _ = ZKCircuit.from_method(foo).compile()
    finally:
        if prev is None:
            os.environ.pop("ZINNIA_SMT_ENABLE", None)
        else:
            os.environ["ZINNIA_SMT_ENABLE"] = prev
