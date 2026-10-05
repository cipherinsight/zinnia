> Coordinated final validation: **100/100 profile checks and 25/25 Halo2 proofs pass**, and all 25 task programs pass the actual study worker. The task review below preserves the baseline and repair history. See [the final audit report](../README.md) and [result manifest](../final-manifest.json) for the verified build and complete evidence.

# SMOOTH adversarial audit

This audit tests 25 distinct ordinary Python/NumPy implementations of three synchronous smoothing rounds. Inputs are positive H×W integer grids with values in [0, 4096]. Each round preserves the outer edge and replaces each interior value with `(4*center + north + south + west + east) // 8`. A grid with no interior is unchanged. Every implementation uses fixed R=3, including profiles whose historical task metadata lists a different R.

## Preserved baseline

Command: `.venv/bin/python -m studies.user_study.adversarial.check --task smooth --directory studies/user_study/adversarial/smooth --output studies/user_study/adversarial/smooth/baseline-results.json`

The checked source files are unmodified candidates, with `np` supplied by the task environment. Each original candidate is compiled before the assertion-instrumented variant. The checker tests profiles 4×5, 3×3, 2×3, and 8×8, with 19 independent-oracle fixtures each (including 12 seeded random grids), followed by a deliberately incorrect-answer mock proof. All 100 source hashes in the saved results were compared with the corpus and matched.

**Result: 69/100 program/profile runs pass.** All 1,900 Python/oracle comparisons pass. Original and checked compilation succeed for 71 runs. There are 1,313 successful compiled fixture checks and 69 successful incorrect-answer negative controls. Of the 31 failing runs, 29 reject during original compilation; two compile but reject a correct answer. No timeout occurred. These are mock backend results, not Halo2 proof results.

The environment hash and dirty-checkout status are recorded in `baseline-results.json`. Other task agents may edit the shared compiler concurrently; the environment record is taken at report completion rather than being a per-worker compiler snapshot. Candidate source hashes are per run. This audit agent did not edit or rebuild the compiler.

## Candidate inventory

| Candidate | Strategy | Passing profiles |
|---|---|---|
| `slice_copy.py` | Vectorized five slices with a fresh synchronous copy. | 4/4 |
| `double_buffer.py` | Reuse two arrays and swap after each complete round. | 4/4 |
| `nested_loops.py` | Scalar indexed loops and fresh copy each round. | 4/4 |
| `while_loops.py` | Three nested while loops express fixed iteration and interior traversal. | 4/4 |
| `row_vectorized.py` | Iterate rows while vectorizing columns. | 4/4 |
| `column_vectorized.py` | Iterate columns while vectorizing rows. | 4/4 |
| `offset_accumulation.py` | Accumulate a reusable four-neighbor offset stencil. | 4/4 |
| `ndindex_guard.py` | Traverse all coordinates with ndindex and skip fixed boundaries. | 0/4 |
| `flat_indices.py` | Flatten arrays and use row-major neighbor offsets. | 4/4 |
| `list_grid.py` | Nested Python lists with scalar traversal and array conversion. | 1/4 |
| `comprehension_grid.py` | Whole-grid nested comprehension with boundary conditional. | 4/4 |
| `row_append.py` | Build output as appended rows and cells. | 4/4 |
| `local_helper.py` | Local helper implements one synchronous round. | 0/4 |
| `global_helper.py` | Top-level pure helper and explicit three-round composition. | 0/4 |
| `roll_restore.py` | Compute full-grid rolled neighbors then restore boundary. | 0/4 |
| `pad_neighbors.py` | Edge padding supplies neighbor arrays; only interior is replaced. | 0/4 |
| `sliding_windows.py` | Use sliding_window_view and a 3x3 cross stencil. | 1/4 |
| `stack_sum.py` | Stack weighted neighbor planes and sum along leading axis. | 4/4 |
| `ufunc_reduce.py` | Use add.reduce over weighted neighbor stack. | 2/4 |
| `einsum_windows.py` | Contract sliding windows with an integer stencil using einsum. | 1/4 |
| `separable_cross.py` | Sum horizontal and vertical 1D filters, sharing central weight. | 4/4 |
| `take_neighbors.py` | Gather interior neighbor planes with take and index arrays. | 4/4 |
| `transpose_round.py` | Transpose axes to reuse row-oriented processing. | 4/4 |
| `concatenate_edges.py` | Assemble interior and unchanged edges using concatenation. | 4/4 |
| `enumerate_rows.py` | Enumerate rows and cells while preserving boundary values. | 4/4 |

## Rejections and repair directions

- **Local helper, 0/4:** a pure nested `step(a)` is rejected as capturing outer values, although it captures none. Support statically resolvable local helpers by inlining or ordinary function lowering, preserving synchronous array allocation and returns. Requiring decorators or banning helpers would narrow the intended ordinary-Python contract.
- **Global helper, 0/4:** `from_source("kernel", source)` treats the first definition, `step(a)`, as the entry function and reports missing circuit annotations on its parameter. Resolve the requested entry name and retain helper definitions in the compilation environment.
- **Nested list grid, 1/4:** `a=x.tolist(); b=[row[:] for row in a]; b[i][j]=...` is valid Python but rejected by the chained-subscript guard in `src/ir_gen/visitors.rs`. That guard handles static-array views only. List mutation needs recursive parent writeback or shared list identity. The 2×3 pass never reaches an interior write, so it does not demonstrate this feature works.
- **ndindex traversal, 0/4:** `np.ndindex(a.shape)` is not implemented. These dimensions are public and static, so producing a Cartesian-product tuple iterator is sufficient for this candidate.
- **Rolled neighbors, 0/4:** `np.roll` is not implemented. Static shifts and axis selection map directly to wrapped coordinate reads; the candidate restores every boundary afterward.
- **Padded neighbors, 0/4:** `np.pad(..., mode="edge")` rejects because only constant mode is supported. Edge mode requires clamping source coordinates. The candidate subsequently restores the original boundary, so the padded values only support full-grid neighbor expressions.
- **Sliding windows and einsum windows, 1/4 each:** `np.lib` namespace resolution fails before reaching `np.lib.stride_tricks.sliding_window_view`. Support the namespace chain and window shape/stride mapping, then validate the downstream arithmetic and einsum independently. Passing 2×3 runs return early and do not exercise windows.
- **ufunc reduction, 2/4:** profiles 4×5 and 8×8 compile but reject the second fixture after accepting the first. This is a semantic failure, not a syntax limitation. The 3×3 profile has only one interior cell and the 2×3 profile none. This pattern is consistent with reducing all axes and broadcasting instead of NumPy `np.add.reduce` defaulting to axis 0. `stack_sum.py`, which explicitly sums the same five planes along axis 0, passes all profiles. The coordinating agent confirmed the Python transformer already handled the default axis before this audit. The remaining failure is in the list-of-ndarrays reduction path; its planned repair converts the list to a StaticArray before reducing axis 0. A separate post-repair rerun must establish success.

An exploratory import-prefixed version caused universal entry parsing errors (`The program must be a function`). Redundant imports were removed because this task explicitly supplies `np`; this exploratory batch is not included in the final 100-run baseline. The top-level-helper candidate is retained because ordinary helper functions are within the intended scope.

## Scope and limitations

These 25 strategies are empirical coverage, not a claim about all possible correct programs. They deliberately include array slicing, buffer swaps, scalar and while loops, row/column vectorization, offset stencils, flat indexing, comprehensions, nested lists, append construction, local/global helpers, roll, padding, window views, reductions, einsum, separable filters, gathers, transposition, and concatenation. The four checked shapes include a no-interior case, but do not cover every positive dimension or dtype. All candidates passed the supplied Python oracle fixtures; this is strong test evidence and is not a formal equivalence proof.

Compiler repairs and their post-repair validation belong to the coordinating agent. Preserve `baseline-results.json` and use a distinct output filename for later reruns.
