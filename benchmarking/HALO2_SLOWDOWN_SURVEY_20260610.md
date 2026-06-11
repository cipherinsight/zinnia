# Zinnia halo2-base vs handwritten halo2-graph — slowdown survey (2026-06-10)

Inputs: `zinnia_halo2_base_groundtruth_20260610.json` + `hw_halo2_walk_20260610.json`
(both swept back-to-back on the same machine; full table in
`HALO2_FULL_COMPARISON_v20_groundtruth_k.md`).

## Methodology change: ground-truth k selection (both sides)

The old Zinnia `estimate_params` used IR-op-count thresholds ("comparisons
>= 50 -> k=10" etc.). Replaced with a two-stage ground-truth scheme:

1. **Seed (Rust, `src/prove/halo2_base/mod.rs`)** — one witness-free dry
   synthesis gives exact advice/lookup/fixed cell counts and the public
   instance count. For every feasible k (instance column fits; range table
   `2^lb <= usable`, `lb = min(8, k-1)`), the real halo2 `ConstraintSystem`
   is configured for that exact column layout and the number of committed
   polynomials is read off it (advice + lookup-advice columns, 2 polys per
   mv_lookup argument, permutation grand products `ceil(cols/(degree-2))`,
   quotient pieces, vanishing blinder). Seed = argmin of `polys * 2^k`.
   No tuned constants; everything is queried from the library.
2. **Measured walk (runners)** — from the seed, probe k±1 with real timed
   proves and keep the measured argmin (3% stop-rule noise margin). Probes
   are recorded per case in the new `k_probes` field.

Rationale: prove time is **U-shaped in k**. Any k "fits" in halo2-base
(`calculate_params` just adds columns), but each column is an extra committed
polynomial + lookup argument + permutation share. The smallest *fitting* k is
catastrophic for cell-heavy circuits (0319 fits at k=9 → 1200 columns →
61.8s vs 10.5s at k=13). The packing margin also changed from
`calculate_params(Some(8))` to `Some(20)` to match the halo2-graph scaffold.

**Fairness**: the HW runner previously recorded its smallest *viable* k,
which has the same pathology. It now does the same upward measured walk.
This collapsed several apparent Zinnia "wins" that were k-selection
artifacts: HW 0329/0330 went 90s@k7 → 6.0s@k13, 0381 21s@k8 → 3.4s@k13,
0373-0375 ~15-18s@k7 → ~2.9s@k10.

## Headline (120 fair, both-verified cases)

- Geomean Zinnia/HW prove ratio: **0.467×** (median 0.52×)
- Zinnia faster: **99/120 (82%)** — was 78/124 before the sizing fix
- Zinnia slower: 21 cases; 18 above 1.05× (listed below)
- All 124 Zinnia cases prove + verify; 0 errors.
- Tiny-circuit floor eliminated: e.g. 0336 115ms@k8 → 35ms@k5,
  0400 115ms@k8 → 29ms@k4 (HW floor is k=7 because its fixed
  `LOOKUP_BITS=6` table is always materialized — Zinnia drops the table
  when a kernel needs no range checks, a genuine compiler advantage).

## Remaining slowdown causes, ranked

### 1. zkRAM dynamic-address scans (12 of 18 slower cases; up to 27.8×)

`read_memory` with a dynamic address lowers to a full O(len) slot scan
(eq + select per slot) **per read**, plus an overlay of one eq+select per
*pending dynamic write* per subsequent read
(`src/prove/halo2_base/synthesizer.rs:681-740`). Nothing is reused across
reads of the same segment.

- **0319 (27.8×, 11.8s vs 425ms)** — permutation-inverse + row gather
  (`a[c,:,:]`): 384 dynamic reads × 384-slot segment. HW computes 8 row
  indicators once and reuses them across each 48-cell plane (~4k cells
  total); Zinnia rebuilds the indicator set per element.
  **⚠ Regression on top**: yesterday's binary proved this at 2.75s@k12;
  today's probe at the same k=12 is 25.5s. The uncommitted
  `src/helpers/static_array/{read,write}.rs` changes (11:27/12:00 Jun 9,
  after yesterday's 11:07 sweep build) are the prime suspect — plausibly the
  single-`ir_write_memory` setitem path turned static result-materialization
  writes into *dynamic* writes, whose overlay cost is O(#writes) on every
  later read (quadratic). Worth bisecting before anything else.
- **argwhere family 0507/0508/0509/0510 (1.6-3.3×)** and
  **0442/0304/0358/0415/0480 (1.1-2.0×)** — same generic-scan lowering on
  smaller segments.

Optimization directions: cache the one-hot indicator set per (segment,
address cell) and reuse across reads; decompose affine addresses
(`addr = i*stride + const`) to scan only the dynamic axis (8 planes instead
of 384 slots for 0319); batch row/plane gathers.

### 2. Select-mask lowering density (0381/0382, 0364, 0300, 0365, 0434; 1.1-1.6×)

No memory ops — these are 2k-20k `Select*` one-hot masking circuits where
HW's hand-fused gadgets simply use fewer cells per selected element.
Moderate, broad-based gap; per-op cost parity would need fused
select-accumulate gates or indicator reuse at the IR level.

### 3. Public-output-bound circuits (0385/0387; 1.06-1.16×)

40k IR ops are pure `ExposePublic`; both sides sit at k=15 pinned by the
20k-row instance column. Near parity; residual gap is synthesis-side copy
overhead inside `create_proof`. Low priority.

## Measurement caveats

- Run-to-run prove variance on this box is real: HW cases at *identical*
  k/config moved 1.4-1.7× between yesterday and today; some Zinnia warm
  medians differ ~40% from their own walk probe at the same k. Cross-side
  conclusions use same-day back-to-back numbers; treat <1.2× ratios as
  parity.
- The walk is greedy (stops on first non-improving probe per direction);
  with the observed noise it can land ±1 k from the true optimum.

---

## Post-fix update (2026-06-10, evening sweep)

Three fixes landed after the morning survey (commits 8e08dbe, cbf8a78,
8b92c8e): single-root affine zkRAM decomposition with shared indicators,
factored two-root gathers for windowed reads, and range-window fact
propagation for comparisons. Fresh full Zinnia sweep
(`zinnia_halo2_base_postfixes_20260610.json`): 124/124 prove + verify,
0 errors. Full table: `HALO2_FULL_COMPARISON_v21_postfixes.md`.

**Epoch caveat**: the evening box runs Zinnia tiny-circuit proves ~1.4-1.6x
slower than the morning epoch (verified on structurally untouched cases at
identical k), while HW cargo proves are stable (1.02x re-measured). The
mixed-epoch geomean of 0.509 is therefore a pessimistic bound; the
epoch-corrected estimate min(morning, evening)/HW is **0.422** (pre-fix
morning baseline: 0.467). The fixes only remove constraints, so the true
same-epoch number lies in [0.42, 0.51], most likely near 0.42.

**Same-epoch slower set** (HW re-measured in the evening epoch alongside
the Zinnia numbers; everything else is parity at 0.98-1.15x):

| Case | ZN | HW | Ratio | Residual cause |
|------|-----:|-----:|------:|----------------|
| 0442 | 307ms | 170ms | 1.8x | in1d membership: per-(write,read) overlay eq pairs, all distinct |
| 0304 | 400ms | 229ms | 1.7x | small dyn-memory reshape; overlay + per-read scans on 23-slot segment |
| 0319 | 580ms | 426ms | 1.4x | inverse-permutation writes still overlay-priced on the gather |
| 0434 | 214ms | 162ms | 1.3x | row/col zeroing select density vs HW's fused or-chains |
| 0385 | 997ms | 823ms | 1.2x | 20k public instances; both sides k=15, copy overhead |
| 0387 | 981ms | 810ms | 1.2x | same as 0385 |

Worst case is now 1.8x (was 27.8x in the morning survey). All residual
absolute deltas are <= 180ms.

## Output-binding corrections (2026-06-10, late)

While chasing 0304's overhead, two correctness holes surfaced: (1) 1-D
dynamic-array slices used positive-step defaults for all steps and padded-
length-relative bounds, so `[::-1]` returned an empty array; (2) branch
merges between an empty `np.array([])` arm and a computed array silently
discarded the computed side. Net effect: **0304, 0507, 0509 and 0510 bound
0-2 public values where their handwritten counterparts bind 20-30** — all
prior comparison rows for these cases overstate Zinnia.

Both fixed (commits d804a2e, 535bdf7; suites 54→53 pre-existing failures —
the dynamic negative_step semantics test was failing because of exactly
this bug). Honest same-epoch numbers with full payload binding:

| Case | ZN (honest) | HW | Ratio | prior (invalid) ratio |
|------|------------:|-----:|------:|------:|
| 0507 | 294ms | 309ms | 0.95x | (1.6x) |
| 0509 | 558ms | 727ms | 0.77x | (2.2x) |
| 0510 | 548ms | 757ms | 0.72x | (3.4x) |
| 0304 | 574ms | 229ms | 2.50x | (1.7x) |

0304 is now the genuine worst case: ~16k cells from the variable-length
slice machinery (two materialization rounds + per-element masks). The
clean optimization is dynamic-slice *view composition* (compose affine
index maps across chained slices; the backend's lineage then folds
cancelling dynamic terms — for 0304 the reversal cancels the dynamic
start entirely, ~500 cells like HW). Deferred along with the
compacted-vs-masked output-encoding decision for 0442.

## Final dual sweep (2026-06-10 evening, same-epoch, all fixes)

`zinnia_halo2_base_final_20260610.json` + `hw_halo2_final_20260610.json`,
run back-to-back. 124/124 prove + verify on both sides, 0 errors. Full
table: `HALO2_FULL_COMPARISON_v22_final.md`.

- **Geomean ZN/HW prove ratio: 0.481x** (median 0.69x); Zinnia faster on
  102/120 fair cases; 14 above 1.05x, of which 12 are <=1.16x (parity
  band, deltas 5-127ms).
- The day's fix targets hold under the full sweep: 0304 absent from the
  slower list (was 2.5x), argwhere family 0507-0510 absent (0507-0510 all
  faster than HW), 0442 at 1.32x (was 1.8x), 0319 at 1.15x (was 27.8x).
- Two findings exposed by honest walking:
  - **0451 (2.65x, 212ms vs 80ms)**: the auto-seed picked k=12 where
    yesterday it sat at k=8 — one of today's IR changes altered this
    case's circuit; under investigation.
  - **0318 (1.97x, 530ms vs 269ms)**: the morning HW baseline had
    stopped its k-walk prematurely at k=7/812ms (greedy stop on a noisy
    probe); today's walk found k=10/269ms. 0318 also runs ~1.6x slower
    than its own yesterday — consistent with the in-flight fancy-index
    static_array lowering committed in the checkpoint (same family that
    originally regressed 0319 before the affine zkRAM fix rescued it).

## Circuit-size audit + bitwidth fairness (2026-06-11)

Per-case advice-cell comparison (`zn_halo2_cells_20260611.json` vs the HW
final sweep): geomean **0.351x**, totals 1.48M vs 5.09M; five Zinnia
circuits are zero-cell (fully folded). Zinnia larger on 16 cases, six at
>=2x — all in the small-kernel select/eq-density family where prove time
is floor-dominated (time ratios ~1.0-1.26x despite 2-4x cells).

Root cause of the largest gaps: discharge/comparison **bitwidth**. Zinnia
enforces its 64-bit integer domain (range-check both operands + 64-bit
compare, ~100 cells per bound discharge); the handwritten 0402/0403 used a
bare 32-bit check_less_than whose input assumption was never enforced. A
width audit across all 134 target.rs files found 0402/0403 to be the only
unjustified narrow-width integer sites; they now enforce 64-bit (commit
4fc2c4f). With equal rules: 0402/0403 at cell parity (1.04x/1.07x) and
Zinnia faster in time (0.74-0.77x). The remaining cell-larger cases
(0442 4.0x, 0415 2.5x, 0412 2.3x, 0353 1.65x, ...) are genuine per-gadget
density differences (select/eq chains, floor_div machinery) with prove
times at or near parity; candidates if circuit size ever becomes a
headline metric.

Residual prove-time gap on 0385/0387 (1.06-1.16x) was traced past circuit
structure (byte-identical configs) and library version (A/B-tested
halo2_proofs v0.3.0 vs v2023_02_02: no difference) to the execution
environment: Zinnia proves inside a Python cdylib (PIC + unwind) vs HW's
panic=abort standalone binary. Fix would be a standalone zinnia-prove
benchmark binary; documented as accepted parity-band for now.

## Circuit-size families 2/3/4 closed (2026-06-11, commit 45d7853)

- **Discharge elision** (family 2): min/max relational facts + gated
  add/sub equality facts + per-conjunct discharge with the new cheap
  AssertNonNegI residual. 0353: 4,334 -> 2,839 cells (1.08x), Noir
  6,809 -> 4,794 gates (0.70x).
- **0415 excluded** (family 3): HW uses host-provided bin_size as a
  compile-time loop bound, unconstrained in-circuit — Category C.
- **Select/eq density** (family 4): halo2 zero-branch select fusion and
  the Noir normalized-index dual-equality peephole. 0433: Noir 2,903 ->
  113 (1.15x), halo2 0.94x; 0434: Noir 2,982 -> 134 (**below** the
  handwritten 139), halo2 0.98x.

Remaining circuit-size gaps are all family 1 (compacted-prefix vs
masked-in-place dynamic-output encoding, deferred): 0442 3.8x, 0412
2.2x, 0353's last 8%. Everything else is at or below handwritten size.

## Noir slowdown survey (2026-06-11, fresh sweep zinnia_noir_fresh_20260611.json)

Geomean 0.533x over 126 pairs; 26 cases >1.0x, only 3 >1.1x, none >1.5x.
Causes, by band:

1. **Dyn-metadata exposure** (4 cases: 0301/0302/0303/0304, +2 publics
   each): Zinnia binds the runtime shape dims; HW didn't. FIXED
   2026-06-11: those four baselines now expose (rows, cols) -> 0301/0302
   land at exactly 1.000x (80 = 80 gates), 0303 1.005x, 0304 1.038x
   (residual = reversal/slice machinery, class 2). CORRECTION to the
   first draft of this survey: the other dust cases (0309/0311/0312/
   0313/0321/0326/0329/0330/0431/0438/0452) have EQUAL output contracts;
   their +1..+44 gate deltas are emitter micro-differences, not metadata.
   0322 is already Category-C flagged. 0364's +277 is class 2 (insert
   position), like 0363.
2. **Negative-index semantics** — FIXED 2026-06-11 (commit fa09acb):
   all six baselines (0345/0346/0347 select, 0363/0364/0365 insert),
   BOTH Noir and halo2 flavors, now implement python's negative-index
   wrapping (select cases via dual-equality indicators + membership-sum
   bound; insert cases via one in-circuit sign test + normalized
   comparisons). The halo2 select targets previously had NO bound assert
   and silently selected zero on negative indices. Post-fix Noir ratios:
   1.016-1.065x; the residual is ZN's signed bound-discharge vs the
   baselines' near-free membership-sum bound — emitter optimization
   opportunity (replace the residual discharge with assert(sum(ind)==1)
   when the index feeds only dual-eq indicator sets). 0508's similar
   band is argwhere-bbox dyn-slice machinery, unrelated.
3. **Runtime-reshape window hoisting** (0416/0418, 1.12x, +9k gates):
   the mask-formulation reduce tests windows per (row, slot) combo; the
   leading static row axis means tests could hoist across rows (HW
   does). Same root as the halo2-side 19x on these two (258k vs 13.5k
   cells — halo2 pays ~60 cells per window test vs Noir's ~4 gates):
   slice-view composition through reshape is the principled fix.
4. **0433** (1.15x): +15 gates absolute; eq/select dust, known.
