# Task B: cost of bounds

## Final compiler (2569c9ed, remote), 2026-10-12

All stages were rerun on the final compiler, commit `2569c9edb8a2bf0b88e9351ca5752f164d2fad24`
(checkout `/home/zhantong/zinnia-eval-run/zinnia-sizefix`, clean, extension SHA-256 `637a5c9a…1324` =
the epoch-fz build), on the remote host shuaicpu6 (container zhantong-dev), program set = the
final compiler's `accepted_correct` account (`final/accept-merged.json`, same 214 entries).
Scripts, logs and all outputs are in `final/`; everything below this section is the older
3ebe0558 run, kept as it was.

Command (remote, `$RUN=/home/zhantong/zinnia-eval-run`; logs `$RUN/logs/pnB.log`, `pnB2.log`,
`final/reproduce.log`):

```sh
source $RUN/env.sh; cd $RUN/paper-numbers-final/B && ./reproduce.sh
```

Changes to the scripts: remote paths, 16 workers per stage, `ulimit -v` 32 GiB per process
instead of `systemd-run`, and one added stage, `final/static_audit_final.py`. On 2569c9ed two
observational probe rewrites (not the original programs) no longer compile:
`npbench_polybench_durbin_durbin/kernel` (the probe keeps views `y[:k]` that the loop later
modifies, which the compiler now rejects) and `pythran_multitype/check2` (the probe returns lists
of different structure from the two branches; the compiler panics with an internal error,
`IRBuilder::create_ir(SelectI): arg[1] has no stmt_id`, instead of rejecting it cleanly). Both had
no bounded arrays in the 3ebe0558 run; the stage records them as fixed-shape after checking that
their original layouts have no dynamic part and that Task A's instrumentation on 2569c9ed
(`../A/final/classification.json`) finds no runtime-shaped value in them. The first run stopped at
`report.py` on these two issues; the pipeline was resumed from that stage (`pnB2`).

**Result: unchanged.** `final/arrays.csv` is byte-identical to `arrays.csv` (36 programs, 141
arrays, same capacities and lengths on A and B). With the definitions of the paper
(`fig/data/rq1-bounds.py`):

| Quantity | 3ebe0558 | 2569c9ed |
|---|---:|---:|
| Programs with data-dependent shapes / arrays | 36 / 141 | 36 / 141 |
| Mean fill (elements/capacity, summed per program, mean of A and B, mean over 36 programs) | 53.6% | 53.6% |
| Median program ratio capacity/elements (all 36, ∞ included) | 2.0 | 2.0 |
| Geometric mean / median over the 34 finite programs | 2.215 / 1.869 | 2.215 / 1.869 |
| Shape-control specialization, Halo2 cells GM (not in the paper) | 3.665 | 3.703 |
| Shape-control specialization, Noir gates GM (not in the paper) | 1.586 | 1.585 |

Worst cases (2569c9ed):

- Empty arrays: 26 arrays are empty on both inputs, in two programs. `pythran_periodic_dist/dist`:
  all 12 of its arrays are empty (fill 0%, ratio ∞). `pythran_tax/tax`: 14 of its arrays are empty
  (fill 17.6%, ratio ∞).
- Largest finite program ratio: `ds1000 case0508` 14.67 (fill 6.8%), then `case0383` 14.29 (fill
  7.0%), `case0481` 9.0 (11.1%), `case0392` 6.67 (15.0%).
- Largest finite single-array ratio: `case0490`, `a[np.logical_and(temp >= t1, temp < t2)]`,
  capacity 50 with 8 elements on A and 2 on B (6.25 and 25).

The full report of the rerun is `final/REPORT.md` (per-program and per-array tables, cost tables).


Compiler `3ebe0558f25592a0c1f8b6cf9bf7ad09f3ddb1d1`; acceptance membership is the supplied remote `accepted_correct` account (older compiler), not a new acceptance run. All 214 accepted entry functions compile on the final compiler.

**Result:** 36 programs contain bounded runtime-shaped source array values (141 expression-result array occurrences). The equal-program capacity/actual geometric mean is **∞**, because two programs contain empty arrays on both fixtures. Over the 34 programs with no empty measured array it is **2.21509×**, median **1.86898×**. Sound checked shape-control specialization was measured for **17 programs**: default/specialized Halo2 cells have geometric mean **3.66497×** (range 1.04579–11.3009×); Noir gates **1.5858×** (range 1.01358–10.4513×). These circuit ratios include savings from knowing shape-control values; they are not an isolated padding-only causal effect.

## Definitions and scope

An array is a NumPy ndarray-valued source expression result (including intermediate arithmetic results, views, tuple/list elements, and each unrolled occurrence). Plain references to variables are not counted again. Boolean-mask subscript augmented assignments are desugared to a read, arithmetic result, and write, matching the compiler transformer; their implicit array temporaries count too. The shape is runtime-dependent when the compiler exports a witness-dependent length, rank, or dimension. Fixed-total reshapes count, even when capacity/length is one. Scalar arrays use total element count (`size`), not Python `len` (first-axis size). Complex payload pairs count as one element. Fixed-shape Boolean masks and scalar values do not count.

Capacity is the number of payload element slots in the compiler’s exported dynamic layout, including constant padding. For internal values, an observational source rewrite lifts each expression to a temporary and returns the collected values; the same unmodified compiler then exports their layouts. The resulting sources are in `probes/`. A/B actual lengths and shapes come from the harness fixture loader and independent NumPy execution. Mock evaluation checks the exported shape metadata. This measures the compiler’s **logical bounded buffer envelope**, not distinct allocated storages or live resident memory: views can share storage, and lowering can fuse or eliminate buffers. Exposing intermediates can prevent such elimination. Consequently 36 is a source-value census, not a claim that all 36 optimized circuits physically allocate padding. The compiler has no public original-program census of internal buffer identity/shape; an exact unique-storage cost cannot be inferred from these results.

Probe handling: branch-local crop results in 0507/0509/0510 use the original compiled output layout; their row/column index arrays use a prefix probe. In periodic_dist the three periodic branches are exposed unconditionally for inspection; both fixtures enable all three. Window helper results are enumerated individually. Static-only probe failures/oversized probes are source-audited, with reasons recorded in `audit.json`. The trace container itself is excluded. `nonzero()[0]` is a same-array alias of the ndarray inside `nonzero()`; the alias-excluded alternative below removes this one extra occurrence. Other view expressions remain logical values.

For program p and fixture f, r[p,f] is the geometric mean of C/L over its arrays. The program’s combined ratio is sqrt(r[p,A] r[p,B]); the headline geometric mean and median weight programs equally. C>0,L=0 is +∞, never silently omitted or replaced by one. “Finite complete programs” drops the entire program if any measured array is empty. The finite-observation alternative drops only empty array/fixture pairs. A/B are the supplied fixtures, not a representative input distribution.

## (a) Capacity divided by actual elements (older run, 3ebe0558)

| Population / weighting | A GM | B GM | A+B GM | A+B median |
|---|---:|---:|---:|---:|
| All programs, equal program weight | ∞ | ∞ | ∞ | 2 |
| Finite complete programs, equal program weight | 2.21685 | 2.21333 | 2.21509 | 1.86898 |
| All array/fixture pairs, equal array weight | ∞ | ∞ | ∞ | 2.66667 |
| Finite array/fixture pairs, equal array weight | 2.36904 | 2.43987 | 2.40419 | 2 |

There are 26 empty array values, on both fixtures (52 array/fixture pairs), all in `pythran_periodic_dist/dist` and `pythran_tax/tax`. The finite-array summary uses 230 array/fixture pairs. The all-program median (including infinite programs) is 2×.

**Alternative: original outputs only.** 32 programs, 62 arrays: program-weighted GM 2.44784×, median 2.02566×; array-weighted GM 2.55139×, median 2.66667×. These use unmodified original circuits, and all lengths are positive. Per-array data: `output_arrays.json`. This excludes internal-only case0490, periodic_dist, projection_simplex, and tax.

**Alternative: omit the explicit tuple extraction alias in case0481.** 140 logical array values; finite array-weighted GM 2.37652×, median 2×. Equal-program figures are unchanged.

**Alternative: literal Python len.** Dividing element capacity by first-axis length instead of total elements gives finite-complete-program GM 4.60655×, median 3.37974× (the full GM is still ∞). This mixes element and row units for higher-rank arrays and is not recommended. All per-program values are in `summary.json`; dimensions are in `arrays.json`.

### Per-program list

| Program | Array values | GM A | GM B | GM A+B | Median array ratio A+B |
|---|---:|---:|---:|---:|---:|
| ds1000_numpy__case0298__kernel | 2 | 2.5 | 1.66667 | 2.04124 | 2.08333 |
| ds1000_numpy__case0301__kernel | 1 | 1 | 1 | 1 | 1 |
| ds1000_numpy__case0302__kernel | 1 | 1 | 1 | 1 | 1 |
| ds1000_numpy__case0303__kernel | 2 | 1.15 | 1.15 | 1.15 | 1.15 |
| ds1000_numpy__case0304__kernel | 3 | 1.15 | 1.15 | 1.15 | 1.15 |
| ds1000_numpy__case0322__kernel | 1 | 2 | 1.33333 | 1.63299 | 1.66667 |
| ds1000_numpy__case0353__kernel | 1 | 2 | 2 | 2 | 2 |
| ds1000_numpy__case0383__kernel | 1 | 14.2857 | 14.2857 | 14.2857 | 14.2857 |
| ds1000_numpy__case0389__kernel | 3 | 1 | 1 | 1 | 1 |
| ds1000_numpy__case0391__kernel | 1 | 3.33333 | 3.33333 | 3.33333 | 3.33333 |
| ds1000_numpy__case0392__kernel | 1 | 6.66667 | 6.66667 | 6.66667 | 6.66667 |
| ds1000_numpy__case0393__kernel | 1 | 3.33333 | 3.33333 | 3.33333 | 3.33333 |
| ds1000_numpy__case0412__kernel | 1 | 2 | 5 | 3.16228 | 3.5 |
| ds1000_numpy__case0413__kernel | 1 | 2.5 | 3.33333 | 2.88675 | 2.91667 |
| ds1000_numpy__case0414__kernel | 3 | 1.5874 | 1.5874 | 1.5874 | 1 |
| ds1000_numpy__case0415__kernel | 3 | 1.5874 | 1.5874 | 1.5874 | 1 |
| ds1000_numpy__case0416__kernel | 3 | 1.25992 | 1.25992 | 1.25992 | 1 |
| ds1000_numpy__case0417__kernel | 3 | 1.5874 | 1.5874 | 1.5874 | 1 |
| ds1000_numpy__case0418__kernel | 3 | 1.25992 | 1.25992 | 1.25992 | 1 |
| ds1000_numpy__case0419__kernel | 4 | 1.41421 | 1.41421 | 1.41421 | 1.5 |
| ds1000_numpy__case0442__kernel | 1 | 1.66667 | 1.17647 | 1.40028 | 1.42157 |
| ds1000_numpy__case0443__kernel | 1 | 4 | 2.85714 | 3.38062 | 3.42857 |
| ds1000_numpy__case0444__kernel | 1 | 1.66667 | 1.17647 | 1.40028 | 1.42157 |
| ds1000_numpy__case0462__kernel | 1 | 2 | 2.5 | 2.23607 | 2.25 |
| ds1000_numpy__case0463__kernel | 2 | 2.22222 | 1.81818 | 2.01008 | 2.0202 |
| ds1000_numpy__case0467__kernel | 16 | 2.66667 | 2.66667 | 2.66667 | 2.66667 |
| ds1000_numpy__case0468__kernel | 16 | 2.66667 | 2.66667 | 2.66667 | 2.66667 |
| ds1000_numpy__case0481__kernel | 2 | 9 | 9 | 9 | 9 |
| ds1000_numpy__case0490__kernel | 20 | 4.44472 | 5.48719 | 4.93853 | 4.58333 |
| ds1000_numpy__case0507__kernel | 3 | 1.31037 | 1.31037 | 1.31037 | 1.25 |
| ds1000_numpy__case0508__kernel | 2 | 10.5409 | 20.4124 | 14.6685 | 16.6667 |
| ds1000_numpy__case0509__kernel | 3 | 2.0274 | 1.48984 | 1.73796 | 1.83333 |
| ds1000_numpy__case0510__kernel | 3 | 1.15083 | 1.15083 | 1.15083 | 1.11111 |
| pythran__pythran_periodic_dist__dist | 12 | ∞ | ∞ | ∞ | ∞ |
| pythran__pythran_projection_simplex__projection_simplex | 2 | 2 | 2 | 2 | 2 |
| pythran__pythran_tax__tax | 17 | ∞ | ∞ | ∞ | ∞ |

## (b) Circuit sizes

General mechanism: retain the private input and assert equality of each shape-control value to its fixture value, then bind that local variable to the asserted constant before executing the unchanged kernel body. An array control (`size`) is checked element by element. Payload inputs remain private and unconstrained beyond the original domain. Every variant is exactly the original function on this explicitly restricted input domain; substituting an asserted equal constant is sound. Source for each default/A/B variant and emitted Noir are in `costs/<program>/`. All 17 programs have identical shape-control values on A and B, so their A/B sizes coincide.

This makes the relevant source arrays statically shaped with element counts equal to the fixture’s actual lengths. Static windows can be exported as lists of rows; validation groups them by original window. The exact output capacities are checked in `cost_validation.json`; source-level internal slices/reshapes/reductions have constant fixture dimensions as well. The comparison is useful as the cost of retaining general shape controls, but it also removes shape arithmetic, dynamic indexing/routing, and metadata. It must not be called a pure cost of over-reservation.

Halo2 uses Zinnia’s witness-free `halo2_base_cell_stats`: advice + lookup cells, at the **same lookup width per pair**, the maximum of the two auto-selected widths (k may differ; counts depend on width). Fixed cells are excluded, as in the harness cost definition. Raw own-width stats, k, fixed cells and common-width stats remain in each result JSON. Noir is compiled for the Noir target, emitted with `emit_structural`, then `nargo compile` and `bb gates --scheme ultra_honk --verifier_target noir-recursive`; the metric is `functions[0].circuit_size`, not padded powers of two. Gate counting reuses `benchmarking.noir_gates.gate_count`, with temporary packages under B/tmp. No keys, real proofs, or timing runs were made.

| Program | Controls fixed | Common lookup bits | Default cells | Exact-shape cells | Growth | Default gates | Exact-shape gates | Growth |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ds1000_numpy__case0301__kernel | ncol | 6 | 1142 | 1092 | 1.04579 | 2986 | 2946 | 1.01358 |
| ds1000_numpy__case0302__kernel | nrow | 6 | 1142 | 1092 | 1.04579 | 2986 | 2946 | 1.01358 |
| ds1000_numpy__case0303__kernel | ncol | 8 | 2337 | 735 | 3.17959 | 3118 | 2946 | 1.05838 |
| ds1000_numpy__case0304__kernel | ncol | 7 | 2887 | 987 | 2.92503 | 3176 | 2946 | 1.07807 |
| ds1000_numpy__case0383__kernel | N | 8 | 16900 | 9722 | 1.73833 | 5288 | 4322 | 1.22351 |
| ds1000_numpy__case0389__kernel | h,w | 7 | 7294 | 3478 | 2.09718 | 34249 | 3277 | 10.4513 |
| ds1000_numpy__case0391__kernel | low,high | 8 | 11167 | 2170 | 5.14608 | 6456 | 3202 | 2.01624 |
| ds1000_numpy__case0392__kernel | low,high | 8 | 12657 | 1120 | 11.3009 | 6586 | 3014 | 2.18514 |
| ds1000_numpy__case0393__kernel | low,high | 8 | 11238 | 2170 | 5.1788 | 6516 | 3202 | 2.03498 |
| ds1000_numpy__case0414__kernel | bin_size | 7 | 3206 | 555 | 5.77658 | 3662 | 2880 | 1.27153 |
| ds1000_numpy__case0415__kernel | bin_size | 7 | 6103 | 753 | 8.10491 | 4368 | 2899 | 1.50673 |
| ds1000_numpy__case0416__kernel | bin_size | 7 | 7769 | 3719 | 2.089 | 4450 | 3223 | 1.3807 |
| ds1000_numpy__case0417__kernel | bin_size | 7 | 3206 | 555 | 5.77658 | 3662 | 2880 | 1.27153 |
| ds1000_numpy__case0418__kernel | bin_size | 7 | 7769 | 3719 | 2.089 | 4450 | 3223 | 1.3807 |
| ds1000_numpy__case0419__kernel | bin_size | 7 | 8695 | 3719 | 2.33799 | 5033 | 3223 | 1.56159 |
| ds1000_numpy__case0467__kernel | size | 8 | 7072 | 630 | 11.2254 | 4993 | 3095 | 1.61325 |
| ds1000_numpy__case0468__kernel | size | 8 | 7072 | 630 | 11.2254 | 4993 | 3095 | 1.61325 |

Across 17 programs: Halo2 GM 3.66497×, median 3.17959×, range 1.04579–11.3009×; Noir GM 1.5858×, median 1.3807×, range 1.01358–10.4513×. A/B rows are retained separately in `costs.csv`.

Excluding the zero-slack fixed-total reshape programs 0301, 0302, 0389 leaves 14 programs: Halo2 GM 4.56237×, range 1.73833–11.3009×; Noir GM 1.47749×, range 1.05838–2.18514×. This remains specialization, not isolated padding cost.

### Programs without a sound exact-capacity comparison

No general capacity override exists in the Python API. `requirements_probe.json` records representative attempts: assertions and scalar `@requires` leave the exported capacities unchanged (0303: 23; 0383: 100; 0391: 200); an asserted mask count in 0412 leaves capacity 10, and asserted unique count in 0298 leaves output capacity 25. `@requires` rejects the tested aggregate/unique-count predicates. Filtering/nonzero compaction is sized from the input envelope in `src/ops/dyn_ndarray/memory_ops.rs`, before ordinary scalar optimization. Tightening the result afterwards does not shrink that construction. A truncated output or fixed fixture payload would not be a sound capacity-only comparison. Implementing a different compactor/unique algorithm could be sound, but would measure another algorithm, so none is presented here.

| Program | Reason exact-capacity comparison is unavailable in this method |
|---|---|
| ds1000_numpy__case0298__kernel | Unique cardinality is data-dependent; both unique values and the zero matrix depend on it. |
| ds1000_numpy__case0322__kernel | Boolean filtering or nonzero/argwhere cardinality depends on payload; count preconditions do not resize the current compaction construction. |
| ds1000_numpy__case0353__kernel | Boolean filtering or nonzero/argwhere cardinality depends on payload; count preconditions do not resize the current compaction construction. |
| ds1000_numpy__case0412__kernel | Boolean filtering or nonzero/argwhere cardinality depends on payload; count preconditions do not resize the current compaction construction. |
| ds1000_numpy__case0413__kernel | Boolean filtering or nonzero/argwhere cardinality depends on payload; count preconditions do not resize the current compaction construction. |
| ds1000_numpy__case0442__kernel | Boolean filtering or nonzero/argwhere cardinality depends on payload; count preconditions do not resize the current compaction construction. |
| ds1000_numpy__case0443__kernel | Boolean filtering or nonzero/argwhere cardinality depends on payload; count preconditions do not resize the current compaction construction. |
| ds1000_numpy__case0444__kernel | Boolean filtering or nonzero/argwhere cardinality depends on payload; count preconditions do not resize the current compaction construction. |
| ds1000_numpy__case0462__kernel | Boolean filtering or nonzero/argwhere cardinality depends on payload; count preconditions do not resize the current compaction construction. |
| ds1000_numpy__case0463__kernel | Boolean filtering or nonzero/argwhere cardinality depends on payload; count preconditions do not resize the current compaction construction. |
| ds1000_numpy__case0481__kernel | Boolean filtering or nonzero/argwhere cardinality depends on payload; count preconditions do not resize the current compaction construction. |
| ds1000_numpy__case0490__kernel | Boolean filtering or nonzero/argwhere cardinality depends on payload; count preconditions do not resize the current compaction construction. |
| ds1000_numpy__case0507__kernel | Nonzero/argwhere intermediate cardinality and crop extrema depend on payload; fixing crop endpoints alone would leave the index buffer oversized. |
| ds1000_numpy__case0508__kernel | Nonzero/argwhere intermediate cardinality and crop extrema depend on payload; fixing crop endpoints alone would leave the index buffer oversized. |
| ds1000_numpy__case0509__kernel | Nonzero/argwhere intermediate cardinality and crop extrema depend on payload; fixing crop endpoints alone would leave the index buffer oversized. |
| ds1000_numpy__case0510__kernel | Nonzero/argwhere intermediate cardinality and crop extrema depend on payload; fixing crop endpoints alone would leave the index buffer oversized. |
| pythran__pythran_periodic_dist__dist | Masked intermediate arrays are empty on both fixtures; scalar L specialization does not constrain payload-dependent masks or remove their envelope. |
| pythran__pythran_projection_simplex__projection_simplex | Two filtered intermediates depend on a predicate over sorted payload values; fixing z does not fix their lengths. |
| pythran__pythran_tax__tax | Several mask results and dependent arithmetic arrays are empty; a count assertion does not change mask-compaction capacity. |

## Validation and reproduction

`validation/*.json` contains 276 successful A/B length-and-shape checks for 138 exposed internal/source array occurrences. The other three occurrences are the original branch-merged crop outputs. In addition, `original_validation/` checks all 36 original programs on A and B: mock constraints are satisfied, every logical output matches NumPy, and all 62 dynamic output shapes/lengths match. `cost_validation.json` contains 34 successful valid-domain mock checks, exact output-capacity checks, and rejection checks after changing a constrained shape control. Cost result files also contain NumPy logical-output comparisons for original A/B and each specialized fixture. Mock satisfaction is checked using `proof_bytes_hex == "mock_satisfied"`; no cryptographic proof is generated.

`manifest.json` records the compiler/extension hash, tool versions and hashes, acceptance hash, and every kernel/fixture hash. All artifacts and generated scripts are under this directory. Heavy Python processes run inside `systemd-run --user --scope -q -p MemoryMax=16G`, with four workers per stage. Stages must run sequentially. Resource deadlines are failure guards, not performance measurements.

From this folder, run:

```sh
bash reproduce.sh
```

The scripts reuse existing census/probe files. For an independent fresh run, move the generated `census/`, `probes/`, `validation/`, and `costs/` directories aside within B before running. To regenerate just this report from the saved measurements:

```sh
systemd-run --user --scope -q -p MemoryMax=16G /home/zhantong/Projects/zinnia-project/wt/npb-hw/.venv/bin/python -B report.py
```

## Per-array list

Array identifiers are source span `line:column:end_line:end_column:AST-kind`, invocation ordinal, and tuple/list element path. For windows, the element path distinguishes each returned window. Capacity and length are element counts; ratios are also in `arrays.csv`. The full compiler layout and NumPy shapes are retained in `arrays.json`.

| Program | Array ID | Expression | Capacity | Actual A | Actual B | C/A | C/B |
|---|---|---|---:|---:|---:|---:|---:|
| ds1000_numpy__case0298__kernel | 11:16:11:49:Call#0[0] | `np.unique(a, return_inverse=True)` | 5 | 2 | 3 | 2.5 | 1.66667 |
| ds1000_numpy__case0298__kernel | 12:8:12:37:Call#0[] | `np.zeros((a.size, vals.size))` | 25 | 10 | 15 | 2.5 | 1.66667 |
| ds1000_numpy__case0301__kernel | 11:8:11:33:Call#0[] | `np.reshape(A, (-1, ncol))` | 20 | 20 | 20 | 1 | 1 |
| ds1000_numpy__case0302__kernel | 11:8:11:33:Call#0[] | `np.reshape(A, (nrow, -1))` | 20 | 20 | 20 | 1 | 1 |
| ds1000_numpy__case0303__kernel | 12:8:12:15:Subscript#0[] | `A[:col]` | 23 | 20 | 20 | 1.15 | 1.15 |
| ds1000_numpy__case0303__kernel | 13:7:13:32:Call#0[] | `np.reshape(B, (-1, ncol))` | 23 | 20 | 20 | 1.15 | 1.15 |
| ds1000_numpy__case0304__kernel | 12:8:12:22:Subscript#0[] | `A[len(A) - col:]` | 23 | 20 | 20 | 1.15 | 1.15 |
| ds1000_numpy__case0304__kernel | 12:8:12:28:Subscript#0[] | `A[len(A) - col:][::-1]` | 23 | 20 | 20 | 1.15 | 1.15 |
| ds1000_numpy__case0304__kernel | 13:8:13:33:Call#0[] | `np.reshape(B, (-1, ncol))` | 23 | 20 | 20 | 1.15 | 1.15 |
| ds1000_numpy__case0322__kernel | 11:13:11:40:Call#0[] | `np.argwhere(a == np.min(a))` | 8 | 4 | 6 | 2 | 1.33333 |
| ds1000_numpy__case0353__kernel | 12:13:12:87:Subscript#0[] | `A[~np.in1d(np.ravel_multi_index(A.T, dims), np.ravel_multi_index(B.T, dims))]` | 6 | 3 | 3 | 2 | 2 |
| ds1000_numpy__case0383__kernel | 11:13:11:36:Subscript#0[] | `np.argsort(a)[::-1][:N]` | 100 | 7 | 7 | 14.2857 | 14.2857 |
| ds1000_numpy__case0389__kernel | 12:13:12:50:Call#0[] | `a.reshape(h // nrows, -1, nrows, ncols)` | 72 | 72 | 72 | 1 | 1 |
| ds1000_numpy__case0389__kernel | 12:13:12:64:Call#0[] | `a.reshape(h // nrows, -1, nrows, ncols).swapaxes(1, 2)` | 72 | 72 | 72 | 1 | 1 |
| ds1000_numpy__case0389__kernel | 12:13:12:78:Call#0[] | `a.reshape(h // nrows, -1, nrows, ncols).swapaxes(1, 2).reshape(h, w)` | 72 | 72 | 72 | 1 | 1 |
| ds1000_numpy__case0391__kernel | 11:13:11:27:Subscript#0[] | `a[:, low:high]` | 200 | 60 | 60 | 3.33333 | 3.33333 |
| ds1000_numpy__case0392__kernel | 11:13:11:27:Subscript#0[] | `a[low:high, :]` | 200 | 30 | 30 | 6.66667 | 6.66667 |
| ds1000_numpy__case0393__kernel | 12:13:12:27:Subscript#0[] | `a[:, low:high]` | 200 | 60 | 60 | 3.33333 | 3.33333 |
| ds1000_numpy__case0412__kernel | 11:13:11:21:Subscript#0[] | `x[x >= 0]` | 10 | 5 | 2 | 2 | 5 |
| ds1000_numpy__case0413__kernel | 11:13:11:26:Subscript#0[] | `x[x.imag != 0]` | 10 | 4 | 3 | 2.5 | 3.33333 |
| ds1000_numpy__case0414__kernel | 11:20:11:61:Subscript#0[] | `data[:data.size // bin_size * bin_size]` | 8 | 8 | 8 | 1 | 1 |
| ds1000_numpy__case0414__kernel | 11:20:11:83:Call#0[] | `data[:data.size // bin_size * bin_size].reshape(-1, bin_size)` | 8 | 8 | 8 | 1 | 1 |
| ds1000_numpy__case0414__kernel | 11:20:11:96:Call#0[] | `data[:data.size // bin_size * bin_size].reshape(-1, bin_size).mean(axis=1)` | 8 | 2 | 2 | 4 | 4 |
| ds1000_numpy__case0415__kernel | 11:19:11:60:Subscript#0[] | `data[:data.size // bin_size * bin_size]` | 8 | 8 | 8 | 1 | 1 |
| ds1000_numpy__case0415__kernel | 11:19:11:82:Call#0[] | `data[:data.size // bin_size * bin_size].reshape(-1, bin_size)` | 8 | 8 | 8 | 1 | 1 |
| ds1000_numpy__case0415__kernel | 11:19:11:94:Call#0[] | `data[:data.size // bin_size * bin_size].reshape(-1, bin_size).max(axis=1)` | 8 | 2 | 2 | 4 | 4 |
| ds1000_numpy__case0416__kernel | 11:20:11:67:Subscript#0[] | `data[:, :data.shape[1] // bin_size * bin_size]` | 48 | 48 | 48 | 1 | 1 |
| ds1000_numpy__case0416__kernel | 11:20:11:104:Call#0[] | `data[:, :data.shape[1] // bin_size * bin_size].reshape(data.shape[0], -1, bin_size)` | 48 | 48 | 48 | 1 | 1 |
| ds1000_numpy__case0416__kernel | 11:20:11:118:Call#0[] | `data[:, :data.shape[1] // bin_size * bin_size].reshape(data.shape[0], -1, bin_size).mean(axis=-1)` | 48 | 24 | 24 | 2 | 2 |
| ds1000_numpy__case0417__kernel | 12:20:12:65:Subscript#0[] | `new_data[:data.size // bin_size * bin_size]` | 8 | 8 | 8 | 1 | 1 |
| ds1000_numpy__case0417__kernel | 12:20:12:87:Call#0[] | `new_data[:data.size // bin_size * bin_size].reshape(-1, bin_size)` | 8 | 8 | 8 | 1 | 1 |
| ds1000_numpy__case0417__kernel | 12:20:12:100:Call#0[] | `new_data[:data.size // bin_size * bin_size].reshape(-1, bin_size).mean(axis=1)` | 8 | 2 | 2 | 4 | 4 |
| ds1000_numpy__case0418__kernel | 12:20:12:71:Subscript#0[] | `new_data[:, :data.shape[1] // bin_size * bin_size]` | 48 | 48 | 48 | 1 | 1 |
| ds1000_numpy__case0418__kernel | 12:20:12:108:Call#0[] | `new_data[:, :data.shape[1] // bin_size * bin_size].reshape(data.shape[0], -1, bin_size)` | 48 | 48 | 48 | 1 | 1 |
| ds1000_numpy__case0418__kernel | 12:20:12:122:Call#0[] | `new_data[:, :data.shape[1] // bin_size * bin_size].reshape(data.shape[0], -1, bin_size).mean(axis=-1)` | 48 | 24 | 24 | 2 | 2 |
| ds1000_numpy__case0419__kernel | 12:20:12:71:Subscript#0[] | `new_data[:, :data.shape[1] // bin_size * bin_size]` | 48 | 48 | 48 | 1 | 1 |
| ds1000_numpy__case0419__kernel | 12:20:12:108:Call#0[] | `new_data[:, :data.shape[1] // bin_size * bin_size].reshape(data.shape[0], -1, bin_size)` | 48 | 48 | 48 | 1 | 1 |
| ds1000_numpy__case0419__kernel | 12:20:12:122:Call#0[] | `new_data[:, :data.shape[1] // bin_size * bin_size].reshape(data.shape[0], -1, bin_size).mean(axis=-1)` | 48 | 24 | 24 | 2 | 2 |
| ds1000_numpy__case0419__kernel | 12:20:12:130:Subscript#0[] | `new_data[:, :data.shape[1] // bin_size * bin_size].reshape(data.shape[0], -1, bin_size).mean(axis=-1)[:, ::-1]` | 48 | 24 | 24 | 2 | 2 |
| ds1000_numpy__case0442__kernel | 11:8:11:24:Subscript#0[] | `A[~np.in1d(A, B)]` | 20 | 12 | 17 | 1.66667 | 1.17647 |
| ds1000_numpy__case0443__kernel | 11:8:11:23:Subscript#0[] | `A[np.in1d(A, B)]` | 20 | 5 | 7 | 4 | 2.85714 |
| ds1000_numpy__case0444__kernel | 11:8:11:82:Subscript#0[] | `A[np.logical_and(A > B[0], A < B[1]) &#124; np.logical_and(A > B[1], A < B[2])]` | 20 | 12 | 17 | 1.66667 | 1.17647 |
| ds1000_numpy__case0462__kernel | 14:13:14:25:Subscript#0[] | `a[selection]` | 20 | 10 | 8 | 2 | 2.5 |
| ds1000_numpy__case0463__kernel | 14:13:14:25:Subscript#0[] | `a[selection]` | 20 | 9 | 11 | 2.22222 | 1.81818 |
| ds1000_numpy__case0463__kernel | 14:13:14:40:Call#0[] | `a[selection].reshape(-1, 1)` | 20 | 9 | 11 | 2.22222 | 1.81818 |
| ds1000_numpy__case0467__kernel | 27:13:27:28:Call#0[0] | `window(a, size)` | 16 | 4 | 4 | 4 | 4 |
| ds1000_numpy__case0467__kernel | 27:13:27:28:Call#0[1] | `window(a, size)` | 16 | 6 | 6 | 2.66667 | 2.66667 |
| ds1000_numpy__case0467__kernel | 27:13:27:28:Call#0[2] | `window(a, size)` | 16 | 6 | 6 | 2.66667 | 2.66667 |
| ds1000_numpy__case0467__kernel | 27:13:27:28:Call#0[3] | `window(a, size)` | 16 | 4 | 4 | 4 | 4 |
| ds1000_numpy__case0467__kernel | 27:13:27:28:Call#0[4] | `window(a, size)` | 16 | 6 | 6 | 2.66667 | 2.66667 |
| ds1000_numpy__case0467__kernel | 27:13:27:28:Call#0[5] | `window(a, size)` | 16 | 9 | 9 | 1.77778 | 1.77778 |
| ds1000_numpy__case0467__kernel | 27:13:27:28:Call#0[6] | `window(a, size)` | 16 | 9 | 9 | 1.77778 | 1.77778 |
| ds1000_numpy__case0467__kernel | 27:13:27:28:Call#0[7] | `window(a, size)` | 16 | 6 | 6 | 2.66667 | 2.66667 |
| ds1000_numpy__case0467__kernel | 27:13:27:28:Call#0[8] | `window(a, size)` | 16 | 6 | 6 | 2.66667 | 2.66667 |
| ds1000_numpy__case0467__kernel | 27:13:27:28:Call#0[9] | `window(a, size)` | 16 | 9 | 9 | 1.77778 | 1.77778 |
| ds1000_numpy__case0467__kernel | 27:13:27:28:Call#0[10] | `window(a, size)` | 16 | 9 | 9 | 1.77778 | 1.77778 |
| ds1000_numpy__case0467__kernel | 27:13:27:28:Call#0[11] | `window(a, size)` | 16 | 6 | 6 | 2.66667 | 2.66667 |
| ds1000_numpy__case0467__kernel | 27:13:27:28:Call#0[12] | `window(a, size)` | 16 | 4 | 4 | 4 | 4 |
| ds1000_numpy__case0467__kernel | 27:13:27:28:Call#0[13] | `window(a, size)` | 16 | 6 | 6 | 2.66667 | 2.66667 |
| ds1000_numpy__case0467__kernel | 27:13:27:28:Call#0[14] | `window(a, size)` | 16 | 6 | 6 | 2.66667 | 2.66667 |
| ds1000_numpy__case0467__kernel | 27:13:27:28:Call#0[15] | `window(a, size)` | 16 | 4 | 4 | 4 | 4 |
| ds1000_numpy__case0468__kernel | 28:13:28:28:Call#0[0] | `window(a, size)` | 16 | 4 | 4 | 4 | 4 |
| ds1000_numpy__case0468__kernel | 28:13:28:28:Call#0[1] | `window(a, size)` | 16 | 6 | 6 | 2.66667 | 2.66667 |
| ds1000_numpy__case0468__kernel | 28:13:28:28:Call#0[2] | `window(a, size)` | 16 | 6 | 6 | 2.66667 | 2.66667 |
| ds1000_numpy__case0468__kernel | 28:13:28:28:Call#0[3] | `window(a, size)` | 16 | 4 | 4 | 4 | 4 |
| ds1000_numpy__case0468__kernel | 28:13:28:28:Call#0[4] | `window(a, size)` | 16 | 6 | 6 | 2.66667 | 2.66667 |
| ds1000_numpy__case0468__kernel | 28:13:28:28:Call#0[5] | `window(a, size)` | 16 | 9 | 9 | 1.77778 | 1.77778 |
| ds1000_numpy__case0468__kernel | 28:13:28:28:Call#0[6] | `window(a, size)` | 16 | 9 | 9 | 1.77778 | 1.77778 |
| ds1000_numpy__case0468__kernel | 28:13:28:28:Call#0[7] | `window(a, size)` | 16 | 6 | 6 | 2.66667 | 2.66667 |
| ds1000_numpy__case0468__kernel | 28:13:28:28:Call#0[8] | `window(a, size)` | 16 | 6 | 6 | 2.66667 | 2.66667 |
| ds1000_numpy__case0468__kernel | 28:13:28:28:Call#0[9] | `window(a, size)` | 16 | 9 | 9 | 1.77778 | 1.77778 |
| ds1000_numpy__case0468__kernel | 28:13:28:28:Call#0[10] | `window(a, size)` | 16 | 9 | 9 | 1.77778 | 1.77778 |
| ds1000_numpy__case0468__kernel | 28:13:28:28:Call#0[11] | `window(a, size)` | 16 | 6 | 6 | 2.66667 | 2.66667 |
| ds1000_numpy__case0468__kernel | 28:13:28:28:Call#0[12] | `window(a, size)` | 16 | 4 | 4 | 4 | 4 |
| ds1000_numpy__case0468__kernel | 28:13:28:28:Call#0[13] | `window(a, size)` | 16 | 6 | 6 | 2.66667 | 2.66667 |
| ds1000_numpy__case0468__kernel | 28:13:28:28:Call#0[14] | `window(a, size)` | 16 | 6 | 6 | 2.66667 | 2.66667 |
| ds1000_numpy__case0468__kernel | 28:13:28:28:Call#0[15] | `window(a, size)` | 16 | 4 | 4 | 4 | 4 |
| ds1000_numpy__case0481__kernel | 12:13:12:31:Call#0[0] | `idx_list.nonzero()` | 9 | 1 | 1 | 9 | 9 |
| ds1000_numpy__case0481__kernel | 12:13:12:34:Subscript#0[] | `idx_list.nonzero()[0]` | 9 | 1 | 1 | 9 | 9 |
| ds1000_numpy__case0490__kernel | 13:10:13:29:Call#0[0] | `np.where(temp < t1)` | 50 | 21 | 28 | 2.38095 | 1.78571 |
| ds1000_numpy__case0490__kernel | 14:10:14:30:Call#0[0] | `np.where(temp >= t2)` | 50 | 19 | 16 | 2.63158 | 3.125 |
| ds1000_numpy__case0490__kernel | 15:8:15:48:Subscript#0[] | `a[np.logical_and(temp >= t1, temp < t2)]` | 50 | 10 | 6 | 5 | 8.33333 |
| ds1000_numpy__case0490__kernel | 15:8:15:53:BinOp#0[] | `a[np.logical_and(temp >= t1, temp < t2)] + 5` | 50 | 10 | 6 | 5 | 8.33333 |
| ds1000_numpy__case0490__kernel | 13:10:13:29:Call#1[0] | `np.where(temp < t1)` | 50 | 34 | 23 | 1.47059 | 2.17391 |
| ds1000_numpy__case0490__kernel | 14:10:14:30:Call#1[0] | `np.where(temp >= t2)` | 50 | 12 | 24 | 4.16667 | 2.08333 |
| ds1000_numpy__case0490__kernel | 15:8:15:48:Subscript#1[] | `a[np.logical_and(temp >= t1, temp < t2)]` | 50 | 4 | 3 | 12.5 | 16.6667 |
| ds1000_numpy__case0490__kernel | 15:8:15:53:BinOp#1[] | `a[np.logical_and(temp >= t1, temp < t2)] + 5` | 50 | 4 | 3 | 12.5 | 16.6667 |
| ds1000_numpy__case0490__kernel | 13:10:13:29:Call#2[0] | `np.where(temp < t1)` | 50 | 18 | 23 | 2.77778 | 2.17391 |
| ds1000_numpy__case0490__kernel | 14:10:14:30:Call#2[0] | `np.where(temp >= t2)` | 50 | 27 | 24 | 1.85185 | 2.08333 |
| ds1000_numpy__case0490__kernel | 15:8:15:48:Subscript#2[] | `a[np.logical_and(temp >= t1, temp < t2)]` | 50 | 5 | 3 | 10 | 16.6667 |
| ds1000_numpy__case0490__kernel | 15:8:15:53:BinOp#2[] | `a[np.logical_and(temp >= t1, temp < t2)] + 5` | 50 | 5 | 3 | 10 | 16.6667 |
| ds1000_numpy__case0490__kernel | 13:10:13:29:Call#3[0] | `np.where(temp < t1)` | 50 | 26 | 26 | 1.92308 | 1.92308 |
| ds1000_numpy__case0490__kernel | 14:10:14:30:Call#3[0] | `np.where(temp >= t2)` | 50 | 16 | 22 | 3.125 | 2.27273 |
| ds1000_numpy__case0490__kernel | 15:8:15:48:Subscript#3[] | `a[np.logical_and(temp >= t1, temp < t2)]` | 50 | 8 | 2 | 6.25 | 25 |
| ds1000_numpy__case0490__kernel | 15:8:15:53:BinOp#3[] | `a[np.logical_and(temp >= t1, temp < t2)] + 5` | 50 | 8 | 2 | 6.25 | 25 |
| ds1000_numpy__case0490__kernel | 13:10:13:29:Call#4[0] | `np.where(temp < t1)` | 50 | 25 | 19 | 2 | 2.63158 |
| ds1000_numpy__case0490__kernel | 14:10:14:30:Call#4[0] | `np.where(temp >= t2)` | 50 | 20 | 25 | 2.5 | 2 |
| ds1000_numpy__case0490__kernel | 15:8:15:48:Subscript#4[] | `a[np.logical_and(temp >= t1, temp < t2)]` | 50 | 5 | 6 | 10 | 8.33333 |
| ds1000_numpy__case0490__kernel | 15:8:15:53:BinOp#4[] | `a[np.logical_and(temp >= t1, temp < t2)] + 5` | 50 | 5 | 6 | 10 | 8.33333 |
| ds1000_numpy__case0507__kernel | 12:11:12:46:Call#0[] | `np.flatnonzero((~mask).sum(axis=1))` | 5 | 4 | 4 | 1.25 | 1.25 |
| ds1000_numpy__case0507__kernel | 13:11:13:46:Call#0[] | `np.flatnonzero((~mask).sum(axis=0))` | 6 | 5 | 5 | 1.2 | 1.2 |
| ds1000_numpy__case0507__kernel | return:crop | `result (branch-merged crop)` | 30 | 20 | 20 | 1.5 | 1.5 |
| ds1000_numpy__case0508__kernel | 11:8:11:22:Call#0[] | `np.argwhere(A)` | 200 | 12 | 8 | 16.6667 | 25 |
| ds1000_numpy__case0508__kernel | 13:13:13:42:Subscript#0[] | `A[ystart:ystop, xstart:xstop]` | 100 | 15 | 6 | 6.66667 | 16.6667 |
| ds1000_numpy__case0509__kernel | 12:11:12:45:Call#0[] | `np.flatnonzero(mask.sum(axis=1))` | 10 | 5 | 7 | 2 | 1.42857 |
| ds1000_numpy__case0509__kernel | 13:11:13:45:Call#0[] | `np.flatnonzero(mask.sum(axis=0))` | 10 | 5 | 6 | 2 | 1.66667 |
| ds1000_numpy__case0509__kernel | return:crop | `result (branch-merged crop)` | 100 | 48 | 72 | 2.08333 | 1.38889 |
| ds1000_numpy__case0510__kernel | 12:11:12:46:Call#0[] | `np.flatnonzero((~mask).sum(axis=1))` | 10 | 9 | 9 | 1.11111 | 1.11111 |
| ds1000_numpy__case0510__kernel | 13:11:13:46:Call#0[] | `np.flatnonzero((~mask).sum(axis=0))` | 10 | 9 | 9 | 1.11111 | 1.11111 |
| ds1000_numpy__case0510__kernel | return:crop | `result (branch-merged crop)` | 100 | 81 | 81 | 1.23457 | 1.23457 |
| pythran__pythran_periodic_dist__dist | 19:25:19:39:Subscript#0[] | `dx[dx > L / 2]` | 256 | 0 | 0 | ∞ | ∞ |
| pythran__pythran_periodic_dist__dist | 19:25:19:43:BinOp#0[] | `dx[dx > L / 2] - L` | 256 | 0 | 0 | ∞ | ∞ |
| pythran__pythran_periodic_dist__dist | 20:26:20:41:Subscript#0[] | `dx[dx < -L / 2]` | 256 | 0 | 0 | ∞ | ∞ |
| pythran__pythran_periodic_dist__dist | 20:26:20:45:BinOp#0[] | `dx[dx < -L / 2] + L` | 256 | 0 | 0 | ∞ | ∞ |
| pythran__pythran_periodic_dist__dist | 22:25:22:39:Subscript#0[] | `dy[dy > L / 2]` | 256 | 0 | 0 | ∞ | ∞ |
| pythran__pythran_periodic_dist__dist | 22:25:22:43:BinOp#0[] | `dy[dy > L / 2] - L` | 256 | 0 | 0 | ∞ | ∞ |
| pythran__pythran_periodic_dist__dist | 23:26:23:41:Subscript#0[] | `dy[dy < -L / 2]` | 256 | 0 | 0 | ∞ | ∞ |
| pythran__pythran_periodic_dist__dist | 23:26:23:45:BinOp#0[] | `dy[dy < -L / 2] + L` | 256 | 0 | 0 | ∞ | ∞ |
| pythran__pythran_periodic_dist__dist | 25:25:25:39:Subscript#0[] | `dz[dz > L / 2]` | 256 | 0 | 0 | ∞ | ∞ |
| pythran__pythran_periodic_dist__dist | 25:25:25:43:BinOp#0[] | `dz[dz > L / 2] - L` | 256 | 0 | 0 | ∞ | ∞ |
| pythran__pythran_periodic_dist__dist | 26:26:26:41:Subscript#0[] | `dz[dz < -L / 2]` | 256 | 0 | 0 | ∞ | ∞ |
| pythran__pythran_periodic_dist__dist | 26:26:26:45:BinOp#0[] | `dz[dz < -L / 2] + L` | 256 | 0 | 0 | ∞ | ∞ |
| pythran__pythran_projection_simplex__projection_simplex | 19:10:19:19:Subscript#0[] | `ind[cond]` | 10 | 5 | 5 | 2 | 2 |
| pythran__pythran_projection_simplex__projection_simplex | 20:12:20:22:Subscript#0[] | `cssv[cond]` | 10 | 5 | 5 | 2 | 2 |
| pythran__pythran_tax__tax | 10:15:10:28:Subscript#0[] | `d[d > 256303]` | 10 | 0 | 0 | ∞ | ∞ |
| pythran__pythran_tax__tax | 10:15:10:35:BinOp#0[] | `d[d > 256303] * 0.45` | 10 | 0 | 0 | ∞ | ∞ |
| pythran__pythran_tax__tax | 10:15:10:46:BinOp#0[] | `d[d > 256303] * 0.45 - 16164.53` | 10 | 0 | 0 | ∞ | ∞ |
| pythran__pythran_tax__tax | 11:15:11:45:Subscript#0[] | `d[(d > 54057) & (d <= 256303)]` | 10 | 10 | 10 | 1 | 1 |
| pythran__pythran_tax__tax | 11:15:11:52:BinOp#0[] | `d[(d > 54057) & (d <= 256303)] * 0.42` | 10 | 10 | 10 | 1 | 1 |
| pythran__pythran_tax__tax | 11:15:11:62:BinOp#0[] | `d[(d > 54057) & (d <= 256303)] * 0.42 - 8475.44` | 10 | 10 | 10 | 1 | 1 |
| pythran__pythran_tax__tax | 12:11:12:40:Subscript#0[] | `d[(d > 13769) & (d <= 54057)]` | 10 | 0 | 0 | ∞ | ∞ |
| pythran__pythran_tax__tax | 12:11:12:48:BinOp#0[] | `d[(d > 13769) & (d <= 54057)] - 13769` | 10 | 0 | 0 | ∞ | ∞ |
| pythran__pythran_tax__tax | 13:11:13:39:Subscript#0[] | `d[(d > 8820) & (d <= 13769)]` | 10 | 0 | 0 | ∞ | ∞ |
| pythran__pythran_tax__tax | 13:11:13:46:BinOp#0[] | `d[(d > 8820) & (d <= 13769)] - 8820` | 10 | 0 | 0 | ∞ | ∞ |
| pythran__pythran_tax__tax | 14:16:14:33:BinOp#0[] | `seg3 * 2.2376e-06` | 10 | 0 | 0 | ∞ | ∞ |
| pythran__pythran_tax__tax | 14:16:14:42:BinOp#0[] | `seg3 * 2.2376e-06 + 0.2397` | 10 | 0 | 0 | ∞ | ∞ |
| pythran__pythran_tax__tax | 15:16:15:34:BinOp#0[] | `seg4 * 1.00727e-05` | 10 | 0 | 0 | ∞ | ∞ |
| pythran__pythran_tax__tax | 15:16:15:41:BinOp#0[] | `seg4 * 1.00727e-05 + 0.14` | 10 | 0 | 0 | ∞ | ∞ |
| pythran__pythran_tax__tax | 16:57:16:73:BinOp#0[] | `seg3 * prog_seg3` | 10 | 0 | 0 | ∞ | ∞ |
| pythran__pythran_tax__tax | 16:57:16:82:BinOp#0[] | `seg3 * prog_seg3 + 939.57` | 10 | 0 | 0 | ∞ | ∞ |
| pythran__pythran_tax__tax | 16:93:16:109:BinOp#0[] | `seg4 * prog_seg4` | 10 | 0 | 0 | ∞ | ∞ |

## Full accepted-program audit

| Program | Bounded array values | Note |
|---|---:|---|
| ds1000_numpy__case0291__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0298__kernel | 2 | compiled; expression probe |
| ds1000_numpy__case0300__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0301__kernel | 1 | compiled; expression probe |
| ds1000_numpy__case0302__kernel | 1 | compiled; expression probe |
| ds1000_numpy__case0303__kernel | 2 | compiled; expression probe |
| ds1000_numpy__case0304__kernel | 3 | compiled; expression probe |
| ds1000_numpy__case0309__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0310__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0311__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0312__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0313__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0314__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0317__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0318__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0319__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0320__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0321__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0322__kernel | 1 | compiled; expression probe |
| ds1000_numpy__case0323__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0324__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0325__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0326__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0329__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0330__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0334__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0335__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0336__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0337__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0338__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0339__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0340__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0341__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0342__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0343__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0345__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0346__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0347__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0349__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0353__kernel | 1 | compiled; expression probe |
| ds1000_numpy__case0355__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0356__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0357__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0358__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0359__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0360__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0361__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0363__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0364__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0365__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0368__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0369__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0370__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0373__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0374__kernel | 0 | observational prefix / unconditional branch probe; see source |
| ds1000_numpy__case0375__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0381__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0382__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0383__kernel | 1 | compiled; expression probe |
| ds1000_numpy__case0385__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0386__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0387__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0389__kernel | 3 | compiled; expression probe |
| ds1000_numpy__case0391__kernel | 1 | compiled; expression probe |
| ds1000_numpy__case0392__kernel | 1 | compiled; expression probe |
| ds1000_numpy__case0393__kernel | 1 | compiled; expression probe |
| ds1000_numpy__case0398__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0399__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0400__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0401__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0402__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0403__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0409__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0412__kernel | 1 | compiled; expression probe |
| ds1000_numpy__case0413__kernel | 1 | compiled; expression probe |
| ds1000_numpy__case0414__kernel | 3 | compiled; expression probe |
| ds1000_numpy__case0415__kernel | 3 | compiled; expression probe |
| ds1000_numpy__case0416__kernel | 3 | compiled; expression probe |
| ds1000_numpy__case0417__kernel | 3 | compiled; expression probe |
| ds1000_numpy__case0418__kernel | 3 | compiled; expression probe |
| ds1000_numpy__case0419__kernel | 4 | compiled; expression probe |
| ds1000_numpy__case0420__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0422__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0428__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0429__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0430__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0431__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0433__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0434__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0435__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0436__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0437__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0438__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0439__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0440__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0441__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0442__kernel | 1 | compiled; expression probe |
| ds1000_numpy__case0443__kernel | 1 | compiled; expression probe |
| ds1000_numpy__case0444__kernel | 1 | compiled; expression probe |
| ds1000_numpy__case0448__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0449__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0450__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0451__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0452__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0453__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0454__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0456__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0457__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0458__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0462__kernel | 1 | compiled; expression probe |
| ds1000_numpy__case0463__kernel | 2 | compiled; expression probe |
| ds1000_numpy__case0467__kernel | 16 | compiled; expression probe |
| ds1000_numpy__case0468__kernel | 16 | compiled; expression probe |
| ds1000_numpy__case0471__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0472__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0477__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0478__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0480__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0481__kernel | 2 | compiled; expression probe |
| ds1000_numpy__case0485__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0490__kernel | 20 | compiled; expression probe |
| ds1000_numpy__case0491__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0500__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0501__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0502__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0503__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0504__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0505__kernel | 0 | compiled; expression probe |
| ds1000_numpy__case0507__kernel | 3 | observational prefix / unconditional branch probe; see source |
| ds1000_numpy__case0508__kernel | 2 | compiled; expression probe |
| ds1000_numpy__case0509__kernel | 3 | observational prefix / unconditional branch probe; see source |
| ds1000_numpy__case0510__kernel | 3 | observational prefix / unconditional branch probe; see source |
| npbench__npbench_compute_compute__compute | 0 | compiled; expression probe |
| npbench__npbench_crc16_crc16__crc16 | 0 | compiled; expression probe |
| npbench__npbench_deep_learning_conv2d_bias_conv2d__conv2d_bias | 0 | compiled; expression probe |
| npbench__npbench_deep_learning_mlp_mlp__mlp | 0 | compiled; expression probe |
| npbench__npbench_deep_learning_resnet_resnet__resnet_basicblock | 0 | compiled; expression probe |
| npbench__npbench_deep_learning_softmax_softmax__softmax | 0 | compiled; expression probe |
| npbench__npbench_go_fast_go_fast__go_fast | 0 | compiled; expression probe |
| npbench__npbench_polybench_atax_atax__kernel | 0 | compiled; expression probe |
| npbench__npbench_polybench_bicg_bicg__kernel | 0 | compiled; expression probe |
| npbench__npbench_polybench_cholesky_cholesky__kernel | 0 | compiled; expression probe |
| npbench__npbench_polybench_covariance2_covariance2__kernel | 0 | compiled; expression probe |
| npbench__npbench_polybench_deriche_deriche__kernel | 0 | compiled; expression probe |
| npbench__npbench_polybench_durbin_durbin__kernel | 0 | compiled; expression probe |
| npbench__npbench_polybench_floyd_warshall_floyd_warshall__kernel | 0 | compiled; expression probe |
| npbench__npbench_polybench_gemm_gemm__kernel | 0 | compiled; expression probe |
| npbench__npbench_polybench_gemver_gemver__kernel | 0 | compiled; expression probe |
| npbench__npbench_polybench_gesummv_gesummv__kernel | 0 | compiled; expression probe |
| npbench__npbench_polybench_gramschmidt_gramschmidt__kernel | 0 | compiled; expression probe |
| npbench__npbench_polybench_k2mm_k2mm__kernel | 0 | compiled; expression probe |
| npbench__npbench_polybench_k3mm_k3mm__kernel | 0 | compiled; expression probe |
| npbench__npbench_polybench_lu_lu__kernel | 0 | compiled; expression probe |
| npbench__npbench_polybench_ludcmp_ludcmp__kernel | 0 | compiled; expression probe |
| npbench__npbench_polybench_mvt_mvt__kernel | 0 | compiled; expression probe |
| npbench__npbench_polybench_symm_symm__kernel | 0 | compiled; expression probe |
| npbench__npbench_polybench_syr2k_syr2k__kernel | 0 | compiled; expression probe |
| npbench__npbench_polybench_syrk_syrk__kernel | 0 | compiled; expression probe |
| npbench__npbench_polybench_trisolv_trisolv__kernel | 0 | compiled; expression probe |
| npbench__npbench_polybench_trmm_trmm__kernel | 0 | compiled; expression probe |
| npbench__npbench_pythran_arc_distance_arc_distance__arc_distance | 0 | compiled; expression probe |
| npbench__npbench_scattering_self_energies_scattering_self_energies__scattering_self_energies | 0 | All array extents are input shape dimensions or literals. Loop indices range over fixed input shape extents; indexed/subarray temporary shapes are fixed. |
| npbench__npbench_weather_stencils_hdiff_hdiff__hdiff | 0 | compiled; expression probe |
| npbench__npbench_weather_stencils_vadv_vadv__vadv | 0 | All dimensions derive from fixed input shapes; original census compile succeeds. BET_M is harness-captured global. |
| pythran__pythran_allpairs__sqr_dists_loops | 0 | compiled; expression probe |
| pythran__pythran_allpairs__sqr_dists | 0 | compiled; expression probe |
| pythran__pythran_arc_distance__arc_distance | 0 | compiled; expression probe |
| pythran__pythran_bubble_sort__bubble_sort | 0 | compiled; expression probe |
| pythran__pythran_ccopy__CCOPY | 0 | No runtime-shaped source arrays: only fixed inputs/indexing and scalar loop state; original census compile succeeds. |
| pythran__pythran_clip__clip | 0 | compiled; expression probe |
| pythran__pythran_create_grid__create_grid | 0 | compiled; expression probe |
| pythran__pythran_cronbach__cronbach | 0 | compiled; expression probe |
| pythran__pythran_crotg__CROTG | 0 | compiled; expression probe |
| pythran__pythran_d2q9_nxnyns__one_time_step | 0 | compiled; expression probe |
| pythran__pythran_deuxd_convolution__conv | 0 | compiled; expression probe |
| pythran__pythran_euclidean_distance_square__euclidean_distance_square | 0 | compiled; expression probe |
| pythran__pythran_fbcorr__fbcorr | 0 | compiled; expression probe |
| pythran__pythran_fibo_seq__fibo | 0 | No runtime-shaped source arrays: only fixed inputs/indexing and scalar loop state; original census compile succeeds. |
| pythran__pythran_goodexpomeasure__goodExpoMeasure | 0 | observational prefix / unconditional branch probe; see source |
| pythran__pythran_harris__harris | 0 | compiled; expression probe |
| pythran__pythran_hasting__fweb | 0 | compiled; expression probe |
| pythran__pythran_insertion_sort__insertion_sort | 0 | compiled; expression probe |
| pythran__pythran_ising__ising_step | 0 | compiled; expression probe |
| pythran__pythran_l2norm__l2_norm | 0 | compiled; expression probe |
| pythran__pythran_lap__laplacian | 0 | observational prefix / unconditional branch probe; see source |
| pythran__pythran_lapl3d__laplacien | 0 | compiled; expression probe |
| pythran__pythran_ln__ln | 0 | compiled; expression probe |
| pythran__pythran_log_likelihood__log_likelihood | 0 | observational prefix / unconditional branch probe; see source |
| pythran__pythran_lu__factorMatrix1 | 0 | compiled; expression probe |
| pythran__pythran_lu__factorMatrix0 | 0 | compiled; expression probe |
| pythran__pythran_make_decision__md | 0 | compiled; expression probe |
| pythran__pythran_matmul__matrix_multiply | 0 | zero(len(m0), len(m1[0])) constructs a static nested list; all loop bounds are fixed input lengths and all arithmetic is scalar indexing. No value-dependent array shape. |
| pythran__pythran_multi_export__a | 0 | compiled; expression probe |
| pythran__pythran_multitype__check2 | 0 | compiled; expression probe |
| pythran__pythran_multitype__check | 0 | compiled; expression probe |
| pythran__pythran_multitype__times | 0 | compiled; expression probe |
| pythran__pythran_pairwise__pairwise | 0 | compiled; expression probe |
| pythran__pythran_periodic_bc__periodic_bc | 0 | compiled; expression probe |
| pythran__pythran_periodic_dist__dist | 12 | observational prefix / unconditional branch probe; see source |
| pythran__pythran_perm__permutations | 0 | compiled; expression probe |
| pythran__pythran_projection_simplex__projection_simplex | 2 | compiled; expression probe |
| pythran__pythran_pythagorean_triples__next_pythagorean_triples | 0 | compiled; expression probe |
| pythran__pythran_reverse_cumsum__reverse_cumsum | 0 | compiled; expression probe |
| pythran__pythran_rosen__rosen | 0 | compiled; expression probe |
| pythran__pythran_smoothing__smoothing | 0 | compiled; expression probe |
| pythran__pythran_specialconvolve__specialconvolve | 0 | compiled; expression probe |
| pythran__pythran_spectraldns__add_pressure_diffusion | 0 | compiled; expression probe |
| pythran__pythran_spectraldns__cross2 | 0 | compiled; expression probe |
| pythran__pythran_sumarray3d__summation | 0 | compiled; expression probe |
| pythran__pythran_tax__tax | 17 | compiled; expression probe |
| pythran__pythran_value_index__tri | 0 | compiled; expression probe |
| pythran__pythran_vibr_energy__calculate_vibr_energy | 0 | observational prefix / unconditional branch probe; see source |
| pythran__pythran_wdist__slow_wdist | 0 | compiled; expression probe |
| pythran__pythran_weights__weights | 0 | compiled; expression probe |
