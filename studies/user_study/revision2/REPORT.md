# Revision 2 compatibility and study readiness

The local site now assigns the six composed array tasks documented in
[tasks.md](tasks.md). Original task IDs remain supported; the two completed
self-pilots retain their original submissions, results, time allowances and task
contracts. New preview links are separate assignments excluded from research
exports.

## Validation coverage

| Check | Coverage |
|---|---:|
| Required Zinnia implementations | 60 |
| Implementation/shape combinations | 180 |
| Compiled correct-output checks | 1,440 |
| Incorrect-output controls | 180 |
| Actual participant-engine checks | 1,584: 1,440 Zinnia, 144 RISC Zero |
| Real Halo2 proof smoke checks | 6, each verified |
| Original task compatibility gate | 172 combinations; 1,204 correct-output checks |
| Study API, timing, worker, manual and versioning tests | 24 |
| Focused compiler regressions | 63 |
| Broader language/array regressions | 1,273 passed; 1 xpassed |
| Browser flow | Both conditions; 12 tasks, 16 example runs, 12 hidden grades |

Machine-readable results: [compiler gate](validation.json),
[participant engines](site-validation.json), [proof smoke checks](proofs.json),
and [original task regressions](legacy-regression.json). Reports identify the
working source and extension; the Git commit alone does not identify this dirty
checkout. The broader language/array run passed 1,273 tests, with one non-strict
expected-failure test also passing. Its one warning is NumPy deprecation of
`row_stack`. The final focused run additionally covers individual shape and
len-derived while bounds. A final loop/scope/branch selection passed all 39
selected tests. All four machine-readable compatibility reports match the final
compiler source and native-extension fingerprints.

The per-task implementation counts are 11, 11, 10, 10, 9, and 9, respectively.
Every inventoried alternative is required to pass. None is marked expected-fail,
removed, or replaced with a preferred solution to obtain a green result.

## Problems found and repaired

1. Batched `@` products were rejected. Native lowering now handles leading-axis
   broadcasting, vector promotion, and empty intermediates. `np.matmul` uses
   the same path.
2. Empty interior arrays lost trailing dimensions or triggered zero-size chunk
   panics. Preserve their metadata through min/max, clipping, where, and slice
   assignment.
3. Canonical nested while counters inside masked branches caused excessive
   symbolic unrolling. Normalize invariant integer counters while preserving
   ordinary post-loop behavior. Direct shape, unpacked shape and len-derived
   dimension bindings are covered.
4. An existing loop target could change even when its enclosing branch was
   inactive. Restore the prior value on that path; dedicated tests vary the
   private branch condition.
5. The participant output wrapper allowed at most four dimensions. It now
   accepts five-dimensional integer-valued output, checked against the task
   oracle.
6. The worker repeatedly compiled the same source and shape for each private
   input. Caching the compiled program by shape removed needless work without
   increasing time or memory limits. All 24 inputs per inventoried implementation
   still execute through compiled Zinnia or RISC Zero.

## Remaining study decisions

Compatibility evidence is finite. It does not establish that every possible
valid NumPy program compiles, nor that arbitrary external libraries are supported.
New natural implementations must become required tests. Continue to record
compiler failures encountered by participants.

The 93-minute total allowance is provisional. Conduct a fresh human pilot to
check difficulty, instruction clarity and session length before freezing budgets
and recruiting. No usability advantage or effect size is inferred from this
compatibility work.
