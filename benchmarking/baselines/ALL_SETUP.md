# Setup batch: all accepted DS-1000 programs

Branch: `all-setup`. Original shapes; hand-authored from `kernel.py` and shared
Q32 gadgets. No compiler IR, emitted code or compiler sizes were consulted.
No proofs, key generation, timing or scaling runs are authorized or performed.

## Why the five folders were missing

`datasets/FILTERING.md`, "Rules removed on 2026-10-06", records audit DB-09:
`excluded_numpy_family` wrongly excluded programs using `np.linalg`, `np.lib`
or `np.polyfit`. The nine restored programs included all five in this batch.
The current manifest marks all five `in_scope`, and the supplied RQ1 acceptance
list includes them. The performance inventory still enumerated only its 162
pre-existing folders (`ds1000_perf.case_dirs`); it never discovered the newly
accepted programs from `ds1000_numpy/generated`. Thus these were missing
entries, not five entries with `zinnia_rejects` status. In 0477, SVD is in the
DS-1000 input prelude: its generated kernel only puts the supplied vector on a
diagonal.

## Kernel and fixture construction

Each new `kernel.py` comes from `ds1000_perf.expected_kernel`, which calls
`generate_ds1000_kernels.generate` on the original DS-1000 record with
`SHAPE_CONSTANTS`. Each `fixture.json` comes from `perf_fixtures.build_fixture`.
Only these five cases' fixture files are written by this batch.

- 0386: `a: Float[100,200]`; 99 x 199 overlapping 2 x 2 windows, flattened in
  window-row, window-column, local-row, local-column order (78,804 values).
- 0456: `a: Float[8,3]`; all 8 x 8 pairwise distances.
- 0457/0458: `a: Float[8,6]`; `dim = 6` is a compile-time shape constant.
  Interpretation: `dim` fixes the input array width in DS-1000's construction,
  although the kernel never reads it. The accepted runtime signatures remain
  unchanged. 0458 returns the upper triangle of the distance matrix.
- 0477: `i: Float[4]`; the 4 x 4 diagonal matrix. Its fixture draws singular
  values from random 4 x 4 integer matrices; SVD remains fixture preprocessing,
  not part of any circuit. No input restriction to singular values is imposed.

## Validation

In progress. Evidence directory:
`/home/zhantong/Projects/zinnia-project/eval-redo/all-work/setup/`.
All checks use `hw_check.check --no-prove`, `HW_CHECK_LOCK=.../all-work/lock-a`,
8 GiB memory limits and `dev-fast` builds with three Cargo jobs. Optimized and
literal binaries use separate `target` and `target-literal` directories.

| Case | Optimized Halo2 advice + lookup = total | Literal Halo2 advice + lookup = total | Optimized Noir gates | Literal Noir gates | Four check verdicts |
|---|---:|---:|---:|---:|---|
| 0386 | pending | pending | pending | pending | pending |
| 0456 | 5,045 + 1,228 = 6,273 | 11,656 + 2,560 = 14,216 | 3,727 | 4,392 | pass / pass / pass / pass |
| 0457 | 6,281 + 1,420 = 7,701 | 14,608 + 2,752 = 17,360 | 3,997 | 4,830 | pass / pass / pass / pass |
| 0458 | 6,281 + 1,420 = 7,701 | 14,609 + 2,752 = 17,361 | 3,941 | 4,774 | pass / pass / pass / pass |
| 0477 | 184 + 56 = 240 | 185 + 56 = 241 | 2,841 | 2,841 | pass / pass / pass / pass |

The four files for each row are
`datasets/ds1000/ds1000_caseNNNN/{target.rs,main.nr,target.literal.rs,main.literal.nr}`.

## Per-case construction notes

0386 reuses only the input/domain/output plumbing of 0385. It changes
non-overlapping, transposed 2 x 2 blocks into every overlapping window in
row-major order. Literal code constructs the full window array, then reshapes
it without changing its wires. The optimized form is already direct; equal
sizes are expected, with no artificial work added to the literal form.

0456 has no pre-existing twin. Only house-style loading and output plumbing is
reused. Optimized computes the 28 unordered row pairs and mirrors each norm;
literal materializes all 192 differences and computes all 64 norms. Each norm
forms the exact sum of Q32-code squares, rescales once using the shared gadget,
then takes the shared square root. Interpretation: this one-call norm reduction
uses the fused sum-of-squares arithmetic permitted for reductions; the literal
variant uses the identical rounding point. The rescale constrains the squared
norm to the signed Q32 window, identically in both variants. The diagonal is
computed in literal code but declared constant in both output layouts.

0457 adapts the new 0456 baselines from width 3 to width 6, preserving its
input/output order and arithmetic. Its full literal difference array has 384 cells.

0458 adapts 0457, emitting only the 28 strict-upper-triangle values as public
cells. Literal code computes the complete difference and norm arrays and then
forms the full `triu` matrix. Optimized skips the discarded lower triangle.
Their domains still agree: reversed pairs have identical squared norms, and
diagonal differences are zero. No guard on discarded work adds a restriction.

0477 uses standard loaders and constant-output layout plumbing, with no
pre-existing twin computation. Literal constructs all 16 matrix cells, optimized
exposes the four diagonal input wires directly. Noir sizes are equal because
constant-index array construction is wiring. Halo2 literal has one extra
constant cell for zero initialization (not additional arithmetic).
