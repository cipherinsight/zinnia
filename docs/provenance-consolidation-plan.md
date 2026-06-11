# Provenance Consolidation & Quality-Hardening Plan

Date: 2026-06-11. Status: Step 0 (a34555e..f4cb2bb — safety nets caught and
fixed ~20 compiler bugs; suites 78→0 failures; audit green 131 + 3 xfails
deferred to Step 3), Step 1 (60455b1 — table + slice_views pilot,
bit-identical), Step 2 (28aa4c9 halo2 + 99d15ce Noir — both backends
read the table live; derivation happens once in create_ir; affine algebra
single-definition; halo2 0391 dropped 533,943→14,040 advice cells, 38x;
Noir emissions byte-identical on the probe set), and Step 3 (56241da +
0e2760f + 89e77ed — compaction records + argwhere generic consumers +
runtime extents/ClampView/BlockView; audit 134/134 with EXPECTED_FAILURES
deleted; 0416/0418 halo2 258,157→~8,000 advice cells; full Noir sweep
zero regressions, geomean 0.948 vs pre-Step-3 HEAD) are DONE.

Step 3 measured notes: (a) 0507 0.852 / 0509 0.751 / 0510 0.745 vs fresh
same-bb HW — targets met; 0508 landed 1.394 vs the 1.25 target. The
residual is the packed-output 2-D gather's ~900-mul floor: Zinnia's
contract exposes the bbox top-left-packed (strictly more information),
while the handwritten baseline only masks in place. Not pursued further —
closing it would mean changing the output contract. (b) The plan's
"±1% of zinnia_noir_ucmp_20260611 / geomean ≤0.59" Step 2/3 gates were
written against pre-Step-0c numbers; the ~20 correctness fixes invalidated
that baseline (0389's old 65 gates was a witness-specialized mis-compile).
Verification used wip-vs-HEAD sweeps instead, per the re-baseline rule.
(c) A pre-existing bug found during validation (bisected to 28aa4c9, the
Step 2 halo2 half): synthesis consumes the provenance table but verify
re-synthesized from artifact IR with an empty table — bigger circuit,
"break points don't match" panic on the provenance-heavy bbox kernels.
FIXED in debdaf1: the artifact carries a stmt-keyed ProvenanceWire
(import_stmts mints fresh ValueIds, so vid keys would dangle); 13/13
spot set proves + verifies.

Step 4 DONE (a240525): compile runs on a dedicated 512 MiB-stack thread;
5 consecutive combined-suite runs at an 8 MiB limit, zero segfaults.
Next: Step 5 (fresh four-way re-baseline). Owner: this session.

## Context

The 2026-06 benchmark campaign (DS-1000, halo2-base + Noir vs handwritten
baselines) repeatedly hit the same meta-problem: *a lowering knows something
structural at emit time; a later consumer needs it; no general channel
exists*. Each instance was solved with an ad-hoc cache, and the worst bugs of
the campaign were provenance-semantics bugs:

- `runtime_strides` meant "compact view" for some producers and "buffer
  layout" for others → 0415/0416/0418 bound wrong outputs (fixed a871f91).
- The Noir emitter's `value_cache` mixed structural constants with
  witness-folded values → witness-specialized circuits, a soundness hole
  across 31 dynamic-address kernels (fixed 71e3048).
- `logical_shape` is silently a padding skeleton after runtime reshape
  (consumers documented to "consult runtime_shape" — but each consumer must
  rediscover this).

This plan consolidates the structural-provenance channels into one layer with
written semantics, hardens the validation that caught these bugs into
permanent tests, and clears known hygiene debt. Performance follow-ups
(0507–0510's residual 1.3–1.5x Noir band) are expected to fall out of the
consolidation rather than needing more special cases.

## Inventory: today's ad-hoc provenance channels

Builder-side (`src/builder.rs`):
- `slice_views: HashMap<u32, SliceView>` — affine provenance of materialized
  slice segments; invalidated in `ir_write_memory`.
- `argwhere_keep_indicators: HashMap<u32, (Vec<usize>, Vec<Value>)>` —
  keep-bit cells per argwhere source; consumed only by
  `argwhere_axis0_bbox_bounds`.
- `FactStack` (`src/optim/predicates/facts.rs`) — the LOGICAL layer: scoped
  ContractTerm facts per ValueId, Z3-backed `prove()`. Well-engineered; keep
  as-is and feed it.

halo2-base synthesizer (`src/prove/halo2_base/synthesizer.rs`):
- `addr_lineage: HashMap<u64, AffineLin>` (≤2-root affine views)
- `value_interval: HashMap<u64, (i64, i64)>`
- `indicator_cache`, `membership_asserted`, `plane_cache`, `write_version`
- `signed64_cells` / `unsigned64_cells` (range-window facts)

Noir emitter (`src/prove/noir/emit.rs`) — duplicates of the above plus more:
- `affine_of: HashMap<String, AffLin>`, `interval_of`, `segment_iv`,
  `ind_cache`, `membership_done`, `plane_cache`, `write_version`
- `lt_pair_of` (min/max select recognition), `lt_zero_of`, `add_const_of`,
  `norm_index_of` (dual-eq peephole), `literal_cells` (structural-constant
  marker — load-bearing for soundness; see s_int_of)

The halo2 and Noir copies of (affine, interval, indicator, plane, version)
are the same design implemented twice — the consolidation target.

## Invariants to codify (write these into the new layer's doc comment)

1. **Layout**: a padded buffer requires `runtime_shape[i] <= logical_shape[i]`
   on every axis. An axis whose bound exceeds its logical size is a
   runtime-reshape skeleton axis ⇒ the buffer is compact and addressing uses
   runtime strides. (`src/ops/dyn_ndarray/aggregation.rs` encodes this today.)
2. **Structural vs witness values**: any value cache that sees witness data
   must never drive emission decisions. Decision sites use structural-only
   accessors (`s_int_of` / `literal_cells` pattern). New rule for all
   backends.
3. **Runtime-axis bound**: each runtime axis of a reshape is bounded by
   `total / product(static dims)`; exceeding it is unsatisfiable via the
   divisibility assert (commit a587f8d).
4. **Invalidation**: any structural record derived from segment contents is
   keyed by (segment, write_version); writes bump the version
   unconditionally.
5. **Membership soundness**: a shared-indicator gather must one-hot-assert
   its enumerated value set once per (root, set); silently gathering
   init/zero on an out-of-set witness is the bug class that motivated this.

---

## Step 0 — Safety nets (do first; everything later is judged by these)

### 0a. Witness-independence regression test

The check that exposed the specialization bug, institutionalized.

Instructions:
- New test (suggested: `testing/lang/test_witness_independence.py`).
- For a representative kernel set — minimum: 0433 (select zero-path), 0391
  (dyn 2-D slice / gathers), 0415 (runtime reshape), 0442 (compaction),
  0345 (negative index) — compile ONCE, then emit/prove with two different
  witnesses via `backend='noir'` with `ZINNIA_NOIR_KEEP_TEMP=1`, and diff the
  two staged `src/main.nr` byte-for-byte. Any difference = failure.
- For halo2-base: prove the same compiled circuit with two witnesses and
  assert the keygen artifacts (vk bytes or `ZINNIA_HALO2_BASE_DEBUG_CELLS`
  cell counts) are identical.

Verification criteria:
- Test fails if you revert commit 71e3048's `s_int_of` gating (do this check
  once, locally, to prove the test has teeth), passes on HEAD.
- Runs in CI/pytest under 2 minutes.

### 0b. Output-binding audit (publics vs numpy, all 134 cases)

Five cases were found binding wrong/missing outputs, all incidentally
(0304/0507/0509/0510 empty-slice; 0415/0416/0418 wrong reshape reduce).
Make it systematic.

Instructions:
- New script `benchmarking/audit_bindings.py`:
  - For each case: execute the kernel in Python/numpy on the
    `INPUT_BUILDERS` inputs to get expected outputs; quantize floats with
    `round(v * 2**32)` (fpc convention).
  - Compile + `prove(backend='mock')`; parse publics
    (`name=decimal` strings).
  - Compare as multisets AND check the payload count matches the expected
    element count + dyn metadata (length/dims). Empty or truncated payload
    = failure even if values present match.
- Wire into pytest as a slow marker or a standalone CI step.

Verification criteria:
- All 134 cases pass on HEAD.
- Deliberately re-introducing the 0415 stride bug (or any one historical
  binding bug) makes exactly the affected cases fail.

### 0c. Triage the 53 pre-existing test failures

`/tmp/fail_0415final.txt` (and the suites) carry 53 failures predating the
campaign; every validation for two days has been "diff vs the 53".

Instructions:
- Classify each: (a) stale expectation (update test), (b) real regression
  from some earlier change (fix), (c) intentionally unsupported (mark
  xfail with reason).
- Burn the list down to zero unexplained failures.

Verification criteria:
- `pytest testing/operator testing/lang testing/semantics` exits with 0
  failures (xfails allowed, each with a reason string).
- Note: run with `ulimit -s unlimited` until Step 4 lands; `RUST_MIN_STACK`
  does NOT cover the main thread.

---

## Step 1 — Provenance layer, phase 1: one table, written semantics

Instructions:
- New module (suggested: `src/provenance.rs` or `src/builder/provenance.rs`)
  owned by `IRBuilder`:
  - `ProvenanceTable` keyed by `ValueId` (scalar records) and `u32` segment
    id (segment records), with explicit record types:
    - `AffineView { terms: Vec<(ValueId, i64)>, offset: i64 }` (≤2 roots)
    - `Interval { lo: i64, hi: i64 }`
    - `Layout { kind: Padded | Compact, runtime_strides: ... }` — subsumes
      the logical/runtime strides confusion; every dyn-array producer must
      state which layout it built.
    - `CompactionInfo { keep: Vec<ValueId>, prefix: Vec<ValueId>, count: ValueId, sorted: bool }`
    - `SliceView` (move the existing struct here unchanged)
  - One `write_version: HashMap<u32, u64>`; every `ir_write_memory` bumps it;
    all segment-derived records carry the version they were built at.
  - Doc comment at module top = the five invariants above, verbatim.
- Migrate ONE existing cache as the pilot: `slice_views` (smallest blast
  radius, has existing consumers and invalidation). Keep behavior
  bit-identical.
- **The hard part**: records referencing `ValueId`s must survive optimization
  passes. Add a remap hook to the passes that rewrite value identity
  (`duplicate_code_elimination` at minimum): when vid A is replaced by B,
  provenance for A re-keys to B (and term lists rewrite). DCE: records whose
  referenced vids die are dropped, never left dangling. Add a debug-mode
  consistency check (every vid referenced by a record exists in the graph)
  run at the end of the pipeline.

Verification criteria:
- Full suites: zero new failures (post-0c: zero failures total).
- Binding audit (0b) and witness-independence (0a) pass.
- Benchmark spot set unchanged within noise: 0304 halo2 cells (slice_views
  consumer) identical before/after; full-sweep gate/cell counts for
  0303/0304/0388/0391 identical.
- Debug consistency check clean across the full DS-1000 compile sweep.

## Step 2 — Phase 2: backends consume, don't re-derive

Instructions:
- Move affine + interval derivation to the IR/builder level (they're
  backend-independent: both backends derive them from the same IR ops).
  Lowering passes/`create_ir` populate `AffineView`/`Interval` records.
- halo2 synthesizer: replace `addr_lineage`/`value_interval` construction
  with reads of the table (mapping ValueId → its cell). Keep
  backend-specific state (indicator/plane caches, signed64 windows) local.
- Noir emitter: same for `affine_of`/`interval_of`/`segment_iv`/`lt_pair_of`.
  KEEP `literal_cells`/`s_int_of` (structural-vs-witness is emitter-local
  and load-bearing).
- Delete the duplicated `AffineLin`/`AffLin` (one definition, in the
  provenance module).

Verification criteria:
- Full Noir sweep: per-case gates within ±1% of
  `zinnia_noir_ucmp_20260611.json` (geomean 0.590x); no case crosses 1.5x.
- halo2 cells spot set (0318, 0319, 0353, 0415, 0416, 0442, 0507, 0508)
  within ±1% of current values (0416/0418 are expected to IMPROVE only in
  Step 3 / slice-view-through-reshape work, not here).
- Suites + 0a + 0b green.
- `grep -c 'AffineLin\|AffLin' src/` shows one definition site.

## Step 3 — Phase 3: compaction artifacts + generic consumers

Instructions:
- Producers: `dyn_filter`, boolean-mask read, `flatnonzero`-shaped lowerings
  (mask→sum→nonzero), `argwhere` deposit `CompactionInfo` + facts:
  `is_sorted(out)` (PredicateApp — kind already registered),
  element-range facts, `count == Σ keep` relational fact.
- Consumers, in order of value:
  1. `dyn_aggregate` / reductions: `min()`/`max()` over a CompactionInfo
     array with `sorted: true` lower to first/last-set-bit scans over
     `keep` (generalizes `argwhere_axis0_bbox_bounds`; afterwards delete the
     special case and route argwhere through the same rule).
  2. Slice-bound computation over compacted index arrays (the
     `im[rows.min():rows.max()+1]` pattern): reuse the same keep scans.
  3. Gathers indexed by compacted arrays: share the `keep`/`prefix`
     indicator family instead of minting fresh indicators.
- Also fold in the deferred reshape item: extend `slice_views` composition
  through `reshape` (the 0416/0418 halo2 fix — target structure: static row
  windows, masks hoisted across rows).

Verification criteria (the performance gates):
- `benchmarking/audit_bindings.py`: the three EXPECTED_FAILURES entries
  (0507/0509/0510 — bbox slice metadata pins static envelope dims instead
  of runtime extents) are deleted and the audit passes 134/134 with no
  xfails. This falls out of slice-view provenance carrying runtime
  extents through to the output contract.
- Noir: 0509/0510 ≤ 1.15x (from 1.33x); 0507/0508 ≤ 1.25x (from 1.33/1.47);
  no case regresses by more than +2% gates; geomean ≤ 0.59x.
- halo2: 0416/0418 cells from 258k to ≤ 25k (≤ 2x of HW's 13,545); mock ==
  halo2-base publics multiset on the full dyn regression set
  (0303/0304/0318/0319/0353/0391–0393/0415/0416/0418/0442/0507–0510).
- The argwhere bbox special case is deleted and 0508–0510 still hit the
  fast path (assert via IR: no full-scan reductions in their compiled IR).
- Suites + 0a + 0b green.

## Step 4 — Compile-time main-thread stack overflow

The flaky segfault in `compile` is main-thread stack exhaustion
(`RUST_MIN_STACK` doesn't apply to the main thread; only `ulimit -s` works).

Instructions:
- Find the deep recursion (likely AST visit or IR topological walk) via the
  crash backtrace under a debugger or by `ulimit -s 8192` to make it
  deterministic, then either convert to an explicit work-list or run
  `compile` on a dedicated `std::thread::Builder::new().stack_size(...)`
  thread (the localized, low-risk fix).

Verification criteria:
- Full suites pass repeatedly (5 consecutive runs) WITHOUT `ulimit -s`
  adjustments, default 8 MiB stack.
- Remove the ulimit notes from the test-running docs/recipes afterward.

## Step 5 — Re-baseline and record

After Steps 1–4:
- Fresh four-way sweep (halo2 time+cells, Noir gates+time, both sides same
  epoch — re-baseline HW with the same bb build per the versioning rule).
- Update `FOUR_WAY_FINAL` doc + memory with the new canonical numbers and
  JSON filenames.

Verification criteria:
- A single doc lists: geomeans, slower-case tables, and the JSON ground-truth
  files, all from the same epoch.

---

## Standing validation recipes (use at every step)

- Build: `RUSTUP_TOOLCHAIN=nightly-2025-12-01 .venv/bin/maturin develop
  --release --features halo2-base-backend,noir-backend` (vendored-lib
  rebuild cycle applies; ZINNIA_SKIP_RELEASE_BUILD=1 for python runs).
- Suites: `ulimit -s unlimited` (until Step 4) + `RUST_MIN_STACK=536870912`;
  run semantics/lang/operator separately if the combined run segfaults.
- Publics equality: mock vs halo2-base as sorted integer multisets
  (mock `name=decimal`, halo2 hex LE); Noir correctness rides on
  `bb verify` against honestly-evaluated `public_decimal`.
- Witness independence: two-witness emit + diff (Step 0a test).
- Never compare bb gate counts across bb versions; re-baseline HW and ZN
  with the same binary.

## Current ground truth (2026-06-11, for regression comparison)

- Noir: `zinnia_noir_ucmp_20260611.json` vs `noir_baseline_final_20260610.json`
  + overrides (`noir_hw_compact*/meta/negidx*/wrap_20260611.json`):
  134/134 ok, geomean 0.590x, max ratio 0508 = 1.47x.
- halo2: HW overrides for 0353/0412/0415/0416/0418/0433/0434/0442 in
  `hw_*_20260611.json`; ZN cells spot values in survey doc; known open gap:
  0416/0418 ZN 258,157 cells vs HW 13,545 (reshape composition, Step 3).
- Suite baseline after Step 0c: zero failures (currently 53 pre-existing).
