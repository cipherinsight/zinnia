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

All five cases pass all four circuits. Evidence directory:
`/home/zhantong/Projects/zinnia-project/eval-redo/all-work/setup/`.
All checks use `hw_check.check --no-prove`, `HW_CHECK_LOCK=.../all-work/lock-a`,
8 GiB memory limits and `dev-fast` builds with three Cargo jobs. Optimized and
literal binaries use separate `target` and `target-literal` directories.

| Case | Optimized Halo2 advice + lookup = total | Literal Halo2 advice + lookup = total | Optimized Noir gates | Literal Noir gates | Four check verdicts |
|---|---:|---:|---:|---:|---|
| 0386 | 540,000 + 160,000 = 700,000 | 540,000 + 160,000 = 700,000 | 245,424 | 245,424 | pass / pass / pass / pass |
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

0386 initially exceeded the 8 GiB cap in Noir compilation with a flat array of
78,804 successive updates. The superseded literal compilation was stopped
while running the same construction. Both final files build patches and rows
as nested arrays; the ABI flattens in the identical row-major order. Both final
Noir checks pass. The Halo2 files were unchanged and passed both variants.

All five cases pass both variants' static lint (zero errors). The A/B outputs
of 0456--0458 also match an independent integer-Q32 sum-of-squares, floor-rescale
and integer-square-root oracle exactly across all four circuits, beyond the
checker’s NumPy-tolerance test. No full-domain equivalence certification is
claimed. 0386 is equal-sized on both backends; 0477 is equal-sized on Noir.
No literal baseline is smaller than its optimized counterpart.

## Case-set bookkeeping

`ds1000_perf.RQ1_ACCEPTED` freezes the 132 names from the supplied
`eval-redo/all-work/rq1-ds-accepted.txt`. The final DS inventory has 167 entries:
140 perf, 13 historical rejections, 6 unaccepted duplicates, 8 out of scope.
The accepted perf entries comprise 92 formerly perf, 27 formerly duplicate,
6 formerly degenerate, 0413/0467 formerly rejected, and the five formerly
absent folders. The eight legacy perf cases rejected in RQ1 remain unchanged
apart from the added former-decision fields. All entries retain
`former_status`, `former_duplicate_of`, `former_reason` and
`former_decision_source`; missing folders are recorded as `former_status:
"absent"`, not as a fabricated compiler rejection.

The data-movement decision is true for 38 perf entries, including new 0386
and 0477; it is false for 0456--0458 and the constant-output cases. The latter
have no output dependence on private inputs. CCOPY remains false: its private
N and strides control comparisons, index arithmetic and which cells are copied.

NPBench/Pythran changes are limited to the requested resnet_basicblock,
floyd_warshall kernel and CCOPY entries, plus counts and selection provenance.
All three are now perf with former statuses and duplicate links retained.
Counts are NPBench 32 perf / 22 rejected, Pythran 50 perf / 60 rejected. Their
old resource-limit acceptance observations remain explicitly historical;
this batch does not claim new validation of their circuits or ports.

The former `perf_case_set.py --write` ran Zinnia compilation and mock-proving,
which the batch brief forbids. Regeneration now uses the frozen author
selection and NumPy fixture validation only, preserving existing compiler
observations rather than running or fabricating new ones. Plain `--write`
updates only the case-set JSON; `--write --write-fixtures` additionally
materializes fixtures. `--reuse-zinnia` remains a compatibility flag; retaining
historical evidence is always the policy. New performance variants carry
`compiles: null` rather than inheriting an unperformed compiler check.

Added reproducible draws for ten restored textual twins and six degenerate
cases. Twin draws use each new case's own seed. Empty A/B inputs are permitted
only when the kernel has no parameters, and constant outputs are allowed only
by explicit draw metadata. All 140 perf draws run successfully under NumPy,
match their recorded fixture hashes, and satisfy the data-movement checks.

The initial `--check` exposed an existing inconsistency for 0318 and 0319:
their checked-in kernels already carried the permutation precondition required
by `FILTERING.md` (author decision 2026-10-06, commit 2561803d), but
`expected_kernel` omitted it. The generation policy now reproduces that exact
existing decorator. Neither kernel nor either case's circuits were edited.
The acceptance generator is unchanged.

### Integration boundary: other batches' fixtures

This batch's instructions prohibit writing other cases' fixture files. The
35 newly promoted, pre-existing DS folders still have no `fixture.json` in
this worktree; their complete draws and hashes are now reproducible. No
other batch's circuit, kernel or fixture was edited. After integrating the
batches, the owner can materialize the missing fixtures with
`perf_case_set.py --write --write-fixtures` (NumPy only). The exact list is in
`eval-redo/all-work/setup/inventory-verification.txt`. This is a file-ownership
boundary, not an unimplemented fixture draw or a failed owned baseline.

## Final verification and evidence

- All 20 owned circuits pass: Halo2 MockProver A/B and counts; Noir compile,
  unpadded UltraHonk gates and witness execution A/B; both reject the checker's
  out-of-domain input. Constant outputs are decoded but not public cells.
- Final optimized and literal lint: zero errors.
- `perf_case_set.py --check`: `0 kernel(s) differ`.
- Inventory assertions: exact 132 accepted + 8 legacy selection; every former
  status and duplicate link preserved; legacy cases' other fields unchanged;
  fixture generation/hash checks for all 140 perf entries; correct movement
  count; exactly the three requested NPBench/Pythran promotions.
- Regeneration is byte-identical on a second run, even with subprocess calls
  disabled by the verification script. Python syntax compilation passes.
- No proofs, keys, compiler invocations, compiler circuit sizes, timing runs
  or scaling runs. Other batches' circuits were not checked by this batch.

Halo2 sizing is automatic, with identical settings for each optimized/literal
pair: 0386 k=17 / lookup_bits=8; 0456--0458 k=9 / lookup_bits=8;
0477 k=6 / lookup_bits=5. The table reports advice and lookup separately and
summed; it excludes fixed and public instance cells. Noir uses the pinned
noir-recursive target and the checker's unpadded gate count.

Evidence files under `eval-redo/all-work/setup/`:
`0386-opt.json` and `0386-lit.json` hold the passing Halo2 checks (their Noir
results belong to the superseded flat-array attempt);
`0386-opt-nested.json` and `0386-lit-nested.json` hold the final passing Noir
checks. `rest-opt.json` and `rest-lit.json` hold all four checks for 0456,
0457, 0458 and 0477. `q32-oracle.txt`, `inventory-verification.txt` and
`verify_inventory.py` record the additional arithmetic and bookkeeping checks.

All five owned cases are finished. There are no unfinished owned circuits and
no literal-smaller-than-optimized cases. The resolved Noir memory failure and
pre-existing generator mismatch are the only substantive tool issues found.
