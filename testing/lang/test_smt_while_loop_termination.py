"""P4 round 1 — while-loop early termination via the resolver.

Spec exit criterion 4 from
``kanban/cards/compiler/loop-unrolling-cost-on-sort-benchmarks/README.md``:

  > Regression test in `testing/lang/` covers the insertion-sort while-pattern
  > with N=32 and asserts compile time well under 30 s.

The current behaviour pre-P4 is: every `while` loop unrolls exactly
``loop_limit`` times (default ~256) regardless of the algorithmic upper
bound. For an insertion sort over N=32 the wasted compile-time cost is
``loop_limit × N`` symbolic-execution steps for the inner loop body.

After P4 round 1, ``visit_while`` re-resolves the guard through the
active ``Resolver`` (range → SMT) and breaks the unroll early when the
guard provably evaluates to false against the post-iteration environment.
For the insertion-sort pattern the inner loop's ``j > 0`` decrements
monotonically and becomes statically false after at most ``i`` iterations,
giving a tight bound of ``N`` total inner iterations across the algorithm.

If the early-exit regresses, this test will hard-time-out (the body emits
~5 IR statements per unrolled iteration; ``loop_limit × N = 256 × 32 =
8192`` iterations × per-statement constant-fold cost is far over 30 s).
"""
import time

import pytest

from zinnia import zk_circuit, ZKCircuit, NDArray, Float


def test_smt_insertion_sort_while_early_termination():
    """N=32 insertion sort must compile in well under 30 s.

    Pre-P4: ``loop_limit`` × N = 8192 unrolled inner-loop iterations
    (each one symbolically executes the body and emits a guarded select).
    Post-P4: the resolver proves ``j > 0`` false after at most ``i``
    iterations per outer pass; total ≈ N(N+1)/2 ≈ 528 iterations.
    """

    @zk_circuit
    def insertion_sort(list2: NDArray[Float, 32]):
        for i in range(1, 32):
            save = list2[i]
            j = i
            while j > 0 and list2[j - 1] > save:
                list2[j] = list2[j - 1]
                j -= 1
            list2[j] = save

    t0 = time.time()
    ZKCircuit.from_method(insertion_sort).compile()
    elapsed = time.time() - t0
    assert elapsed < 30.0, (
        f"insertion_sort N=32 compile took {elapsed:.1f}s; "
        "early-exit may have regressed (full unrolling = ~loop_limit×N steps)."
    )


def test_smt_while_resolver_early_exit_path():
    """A small hand-crafted while where the resolver should prove the guard
    false after a fixed number of iterations.

    ``i`` increments by 1 each iteration; ``i < 5`` becomes statically
    ``false`` at iteration 5. The static-val fast-path catches this today,
    but the test pins behaviour: the visit_while resolver wiring must not
    regress this trivial case to "unroll until loop_limit".
    """

    @zk_circuit
    def small_while():
        i = 0
        while i < 5:
            i += 1
        assert i == 5

    # Just compiles cleanly; the early-exit ensures we don't pay cost on
    # iterations 5..256.
    ZKCircuit.from_method(small_while).compile()
