# DS4 literal baselines

Original DS-1000 shapes, 2026-10-09. All 25 cases have both literal source files.
23 cases pass both checkers. Cases 0450 and 0490 pass witness/output/domain checks
but cannot receive a clean Halo2 checker verdict under its current repeated-bound policy.
No case is missing a circuit or blocked by the memory cap. Final lint: **0 errors**.

## Validation and environment

Sources were derived from each `kernel.py` and audited against L1-L7, including full
intermediates, source order, per-operation bounds, and the absence of local hints.
The only inherited files were 0450's unverified drafts from 75d711f7. Its custom
witness-based `index_long` helper was removed and replaced with the required
`q.index`; its Noir statement plumbing and axis-by-axis gather were retained and reviewed.
All other literal files were authored in this batch. Optimized sources, fixtures,
shared gadgets, case sets, and checking tools were not changed.

Builds used `HW_VARIANT=literal`, `CARGO_TARGET_DIR=target-literal`,
`CARGO_BUILD_JOBS=3`, `--profile dev-fast`, and `memrun.sh` with `MEMRUN_MAX=8G`.
The optimized build used the same settings with `HW_VARIANT` unset and target `target`.
Nargo/noirc: 1.0.0-beta.20, commit b4236c1957d0c26cb65d82adc9e5447b6ff1d629;
Barretenberg: 5.0.0-nightly.20260324, the driver's `noir-recursive` target.
Every checker invocation used `--no-prove` and
`HW_CHECK_LOCK=/home/zhantong/Projects/zinnia-project/eval-redo/literal-work/lock-b`.
TMPDIR, all staging packages, results, and logs are under
`/home/zhantong/Projects/zinnia-project/eval-redo/literal-work/ds4/`.

Checks cover Halo2 MockProver A/B, Noir witness execution A/B, NumPy output
comparison under the case-set tolerance, and rejection of the checker's selected
out-of-domain input. Counts are advice + lookup cells and unpadded Noir gates.
The checker's automatic Halo2 sizing is reported, including lookup width; these
are not proving-time measurements. No key generation, proving, verification,
Zinnia runs, IR inspection, timing runs, or scaling runs were performed.
Fixture checks do not constitute exhaustive testing of all valid inputs.

## Rule interpretations and algorithm/arithmetic audit

- 0450: a multidimensional advanced-index gather is implemented axis by axis.
  The second-axis selected row is materialized for each `(i,j)`, then the third-axis
  reads produce its two outputs. Shapes/reshapes only change static metadata.
  The standard indexing gadgets retain negative-index wrapping and bounds.
- 0452/0454: `np.abs` uses the shared `abs_int64` gadget in both systems. This is
  treated as the shared elementwise arithmetic implementation, not a local hint.
  The entire absolute-value array precedes row sum/max; each division gets a
  fresh denominator bound. Raw integer numerator and denominator in the Q32
  division gadget implement NumPy true division without changing its quotient.
- 0453: integer products are materialized in full, then row sums, then Q32 square
  roots, then broadcast divisions. No fused product-sum or shared norm proof.
  The optimized baseline's extra 32-bit input checks are unnecessary once inputs
  are int64: four integer squares total at most 2^128, and the twice-scaled root
  target at most 2^192, below the field modulus. The shared root constraints force
  a representable norm. No extra narrower input checks were added to the literal.
- 0462/0481: full masks precede running-count/equality compaction. Every input and
  output slot pair receives an equality test; public lengths and zero padding
  match the existing statement. No triangular shift/one-hot compaction gadget.
- 0471/0472: static slices read and domain-check only the selected input cells,
  as L7 requires. Constant wire selection is the complete computation.
- 0478: `einsum('ii->i')` is a writable diagonal view. Copy its diagonal, zero the
  full array, then write the saved diagonal through the view. Only diagonal input
  elements are read. Constant outputs remain absent from the public instance.
- 0480: argmax is a strict-improvement linear scan with first-occurrence ties.
  Both source indexed comparisons are evaluated, then their OR selects -1.
- 0485: tensordot uses its defining sum, in contraction-index order, with a single
  final rescale per output under L5. No per-product rescale or arithmetic rewrite.
- 0490: each explicit `where` constructs its full compact index array before
  assignments scan every destination. The final boolean-index read uses full
  equality compaction before adding five and scattering back. Private lengths
  use the fixed bound 50; active flags gate writes. Padded arithmetic has no effect.
  Integer thresholds are converted at each mixed element comparison. The four
  comparison arrays are distinct; no complement-based reuse or merged piecewise formula.
- 0491: `isclose` uses `abs(a-b) <= atol + rtol*abs(b)` with Q32 constants
  43 and 42950. The full close array, its negation, and running sum remain separate.
- 0495/0496/0497: pad materializes the full output array. A scalar fill is converted
  once to the array dtype, as a pad-call argument; static constant outputs are
  omitted only in statement serialization, following L7.
- 0500: reshape preserves wire order and changes only static shape metadata.
- 0501/0503/0504/0505: take_along_axis performs one standard indexed read per row;
  newaxis and the final singleton-axis slice are metadata. Full gathered arrays
  precede reductions. In 0505, sum(a) is computed before sum(arr), after arr exists.
- 0507/0509/0510: build the complete mask and both reduction arrays, independently
  computing each source `~mask`; construct flatnonzero arrays with equality
  compaction, then compute each min and max by a separate active scan. Empty-branch
  dummy reduction values have no observable effect. For the rectangular slice,
  enumerate output slots in row-major order using quotient/remainder by private
  width and standard indexed reads. Inactive slots select a safe zero index before
  the read; empty branches use divisor one. This implements L4 without imposing
  errors from an unexecuted branch.
- 0508: construct the full two-column argwhere array with equality compaction,
  reduce its minima before its maxima, add one, then use the same direct slice
  implementation. Empty argwhere explicitly rejects, matching NumPy's reduction error.

Shared fixed-point arithmetic and its overflow policy are retained; this is not
an implementation of NumPy integer wraparound outside that arithmetic contract.
No manual CSE, range-proof reuse, local witness hints, hinted sorts, or forbidden
compaction/indexing helpers are used.

## Checker/tool issues

The checker unconditionally fails any nonzero `redundant_input_checks`, a rule
appropriate to optimized baselines but inconsistent with literal L5's fresh bounds
and L6's ban on reusing input I64 proofs. This blocks a clean checker verdict for
0450 (28 duplicate bounds inside required `q.index` calls) and 0490 (1,000 bounds
for four comparisons over 250 input cells). The files are complete translations,
not partial drafts; the policy failure is retained honestly rather than bypassed.
No shared code or checker was modified.

An intermediate restricted build was replaced while a queued check was still
running, causing 0481 and 0485 to report “not compiled”. Both were rerun against
the complete ds4 build; only the rerun Halo2 results are authoritative. 0497's
final Halo2 rerun likewise supersedes its intermediate measurement.
The virtual environment emits harmless `sys.prefix` warnings when invoked via
`../../.venv/bin/python`; these did not stop the tools.

## Results

All optimized checker verdicts are **pass/pass** (Halo2/Noir).
Literal verdict **policy FAIL/pass** means only the repeated-bound policy failed;
A/B outputs, witness satisfaction, and out-of-domain rejection all passed.
Cell columns show **advice + lookup = total**, excluding fixed/instance cells.
These are the original automatic-sizing checker measurements.

| Case | Literal Halo2 cells | Literal Noir gates | Optimized Halo2 cells | Optimized Noir gates | Literal verdict | Notes |
|---|---:|---:|---:|---:|---|---|
| ds1000_case0450 | 45,650 + 4,596 = 50,246 | 7,294 | 17,930 + 4,064 = 21,994 | 6,673 | policy FAIL/pass | Reused and repaired draft; 28 repeated bounds. |
| ds1000_case0452 | 4,721 + 1,160 = 5,881 | 3,607 | 3,180 + 840 = 4,020 | 3,347 | pass/pass | Full abs, row sums, broadcast division. |
| ds1000_case0453 | 4,596 + 1,105 = 5,701 | 3,511 | 3,220 + 845 = 4,065 | 3,378 | pass/pass | Full integer squares, row sums, sqrt, division. |
| ds1000_case0454 | 6,260 + 1,535 = 7,795 | 3,844 | 3,825 + 1,040 = 4,865 | 3,422 | pass/pass | Full abs then pairwise max scans. |
| ds1000_case0462 | 11,126 + 200 = 11,326 | 5,096 | 3,589 + 200 = 3,789 | 3,650 | pass/pass | Equality compaction. |
| ds1000_case0471 | 61,236 + 18,144 = 79,380 | 16,991 | 61,236 + 18,144 = 79,380 | 16,991 | pass/pass | **H=N ties**: static slice already direct. |
| ds1000_case0472 | 40,824 + 12,096 = 52,920 | 12,266 | 40,824 + 12,096 = 52,920 | 12,266 | pass/pass | **H=N ties**: static slice already direct. |
| ds1000_case0478 | 185 + 56 = 241 | 2,841 | 184 + 56 = 240 | 2,841 | pass/pass | **N tie**: diagonal view and constant output plumbing. |
| ds1000_case0480 | 2,291 + 397 = 2,688 | 3,134 | 1,009 + 216 = 1,225 | 2,963 | pass/pass | Linear argmax and both indexed rechecks. |
| ds1000_case0481 | 2,973 + 200 = 3,173 | 3,274 | 1,534 + 240 = 1,774 | 3,103 | pass/pass | Equality compaction; auto lookup widths differ. |
| ds1000_case0485 | 8,434 + 1,872 = 10,306 | 4,234 | 7,173 + 1,872 = 9,045 | 4,234 | pass/pass | **N tie**: compiler lowers defining sum like shared dot. |
| ds1000_case0490 | 1,958,021 + 39,580 = 1,997,601 | 346,761 | 31,320 + 6,660 = 37,980 | 8,089 | policy FAIL/pass | 1,000 repeated bounds; full where arrays and scatter. |
| ds1000_case0491 | 2,095 + 510 = 2,605 | 3,077 | 1,846 + 450 = 2,296 | 3,115 | pass/pass | **N smaller**: shared abs_int64 for the difference. |
| ds1000_case0495 | 14,392 + 4,264 = 18,656 | 6,147 | 14,391 + 4,264 = 18,655 | 6,147 | pass/pass | **N tie**: pad is wiring plus constants. |
| ds1000_case0496 | 11,341 + 3,360 = 14,701 | 5,441 | 11,340 + 3,360 = 14,700 | 5,441 | pass/pass | **N tie**: pad is wiring plus constants. |
| ds1000_case0497 | 11,371 + 3,368 = 14,739 | 7,023 | 11,371 + 3,368 = 14,739 | 7,023 | pass/pass | **H=N ties**: scalar fill conversion and wiring. |
| ds1000_case0500 | 1,458 + 432 = 1,890 | 3,154 | 1,458 + 432 = 1,890 | 3,154 | pass/pass | **H=N ties**: reshape is metadata. |
| ds1000_case0501 | 22,308 + 4,056 = 26,364 | 7,759 | 22,308 + 4,056 = 26,364 | 6,661 | pass/pass | **H tie**: optimized Halo2 already uses q.index. |
| ds1000_case0503 | 49,179 + 8,112 = 57,291 | 10,931 | 49,179 + 8,112 = 57,291 | 11,013 | pass/pass | **H tie; N smaller**: required native Noir indexing. |
| ds1000_case0504 | 49,856 + 8,112 = 57,968 | 10,651 | 49,684 + 8,112 = 57,796 | 10,845 | pass/pass | **N smaller**: native Noir indexing and compiler lowering. |
| ds1000_case0505 | 53,240 + 8,112 = 61,352 | 10,932 | 52,221 + 8,112 = 60,333 | 10,845 | pass/pass | Full gathered array before both ordered sums. |
| ds1000_case0507 | 30,290 + 3,420 = 33,710 | 4,808 | 2,834 + 360 = 3,194 | 3,468 | pass/pass | Two flatnonzero arrays, four reductions, direct crop. |
| ds1000_case0508 | 589,814 + 28,200 = 618,014 | 65,221 | 9,656 + 800 = 10,456 | 5,750 | pass/pass | Full argwhere, axis minima/maxima, direct crop. |
| ds1000_case0509 | 199,597 + 10,200 = 209,797 | 8,752 | 9,670 + 800 = 10,470 | 5,617 | pass/pass | Full zero mask, coordinate arrays, direct crop. |
| ds1000_case0510 | 200,397 + 10,200 = 210,597 | 8,752 | 9,662 + 800 = 10,462 | 5,749 | pass/pass | Full mask and two independent negations, direct crop. |

H means Halo2, N means Noir. Every not-larger result is explicitly flagged above.
None of the Halo2 literal totals is smaller, at automatic or controlled sizing.
Equal Halo2 cases are 0471, 0472, 0497, 0500, 0501, and 0503.
Equal Noir cases are 0471, 0472, 0478, 0485, 0495, 0496, 0497, and 0500.

The three smaller Noir results are:

- **0491: 3,077 vs 3,115.** Literal uses the documented shared `abs_int64` for
  both absolute values. Optimized Noir uses `abs` for the difference; its sign
  comparison is more expensive here. This is shared-gadget implementation cost,
  not a source-level rewrite, and means the optimized file is not minimal here.
- **0503: 10,931 vs 11,013.** L5 requires native checked indexing, whereas the
  optimized file uses `pick`. The compiler/backend makes native indexing cheaper
  for this five-element gather at these shapes.
- **0504: 10,651 vs 10,845.** The same native indexing choice and compiler
  treatment of the final sum make the literal Noir circuit smaller. The full
  gathered array is present in source before the reduction; no manual fusion.

## Controlled Halo2 counts

Both variants were also counted with **lookup_bits=8** and identical k for each
case: `max(9, literal_auto_k, optimized_auto_k)`. The CLI requires `--k` when
`--lookup-bits` is supplied. These are `stats --input A` counts only; they do not
rerun proofs or claim a separate witness-validation run at this sizing. All
count calls succeeded under the same 8G cap and lock. The auto `(k, lookup_bits)`
pairs are included to make the preceding automatic measurements reproducible.

| Case | Literal auto (k, lb) | Optimized auto (k, lb) | Common k | Literal cells at lb=8 | Optimized cells at lb=8 |
|---|---|---|---:|---:|---:|
| ds1000_case0450 | (9, 8) | (9, 8) | 9 | 45,650 + 4,596 = 50,246 | 17,930 + 4,064 = 21,994 |
| ds1000_case0452 | (9, 8) | (9, 8) | 9 | 4,721 + 1,160 = 5,881 | 3,180 + 840 = 4,020 |
| ds1000_case0453 | (9, 8) | (9, 8) | 9 | 4,596 + 1,105 = 5,701 | 3,220 + 845 = 4,065 |
| ds1000_case0454 | (9, 8) | (9, 8) | 9 | 6,260 + 1,535 = 7,795 | 3,825 + 1,040 = 4,865 |
| ds1000_case0462 | (8, 7) | (8, 7) | 9 | 10,926 + 160 = 11,086 | 3,389 + 160 = 3,549 |
| ds1000_case0471 | (12, 8) | (12, 8) | 12 | 61,236 + 18,144 = 79,380 | 61,236 + 18,144 = 79,380 |
| ds1000_case0472 | (11, 8) | (11, 8) | 11 | 40,824 + 12,096 = 52,920 | 40,824 + 12,096 = 52,920 |
| ds1000_case0478 | (6, 5) | (6, 5) | 9 | 109 + 32 = 141 | 108 + 32 = 140 |
| ds1000_case0480 | (8, 7) | (7, 6) | 9 | 1,929 + 319 = 2,248 | 775 + 144 = 919 |
| ds1000_case0481 | (8, 7) | (7, 6) | 9 | 2,773 + 160 = 2,933 | 1,274 + 160 = 1,434 |
| ds1000_case0485 | (9, 8) | (9, 8) | 9 | 8,434 + 1,872 = 10,306 | 7,173 + 1,872 = 9,045 |
| ds1000_case0490 | (11, 8) | (9, 8) | 11 | 1,958,021 + 39,580 = 1,997,601 | 31,320 + 6,660 = 37,980 |
| ds1000_case0491 | (8, 7) | (8, 7) | 9 | 1,729 + 426 = 2,155 | 1,540 + 378 = 1,918 |
| ds1000_case0495 | (10, 8) | (10, 8) | 10 | 14,392 + 4,264 = 18,656 | 14,391 + 4,264 = 18,655 |
| ds1000_case0496 | (9, 8) | (9, 8) | 9 | 11,341 + 3,360 = 14,701 | 11,340 + 3,360 = 14,700 |
| ds1000_case0497 | (11, 8) | (11, 8) | 11 | 11,371 + 3,368 = 14,739 | 11,371 + 3,368 = 14,739 |
| ds1000_case0500 | (9, 8) | (9, 8) | 9 | 1,458 + 432 = 1,890 | 1,458 + 432 = 1,890 |
| ds1000_case0501 | (9, 8) | (9, 8) | 9 | 22,308 + 4,056 = 26,364 | 22,308 + 4,056 = 26,364 |
| ds1000_case0503 | (9, 8) | (9, 8) | 9 | 49,179 + 8,112 = 57,291 | 49,179 + 8,112 = 57,291 |
| ds1000_case0504 | (9, 8) | (9, 8) | 9 | 49,856 + 8,112 = 57,968 | 49,684 + 8,112 = 57,796 |
| ds1000_case0505 | (9, 8) | (9, 8) | 9 | 53,240 + 8,112 = 61,352 | 52,221 + 8,112 = 60,333 |
| ds1000_case0507 | (9, 8) | (7, 6) | 9 | 30,290 + 3,420 = 33,710 | 2,444 + 240 = 2,684 |
| ds1000_case0508 | (11, 8) | (9, 8) | 11 | 589,814 + 28,200 = 618,014 | 9,656 + 800 = 10,456 |
| ds1000_case0509 | (10, 8) | (9, 8) | 10 | 199,597 + 10,200 = 209,797 | 9,670 + 800 = 10,470 |
| ds1000_case0510 | (10, 8) | (9, 8) | 10 | 200,397 + 10,200 = 210,597 | 9,662 + 800 = 10,462 |

## Evidence and source fingerprints

Raw results under the work directory stated above:
`0500.json`, `group1.json`, `group2.json`, `group3.json`, `recheck.json`,
`recheck0497.json`, `optimized.json`, and `common-width.json`.
`literal-final.json` merges literal results in that order, replacing earlier
Halo2 measurements with final reruns. `lint.log` records zero errors.
Build logs are `build-final.log` and `build-optimized.log`. No measured memory-cap
failure occurred. The two cases not finishable as clean checker passes are
0450 and 0490, for the policy conflict described above; no implementation is missing.

Each fingerprint is SHA-256 over the following ordered source bundle:
`kernel.py`, `target.literal.rs`, `main.literal.nr`, `target.rs`, `main.nr`.
For each file append its UTF-8 filename, a NUL, its exact bytes, and a NUL.
This identifies both compared implementations and the source kernel, including
headers and output-layout declarations. Shapes are the unchanged original fixtures.

| Case | SHA-256 source bundle |
|---|---|
| ds1000_case0450 | `9ee63cc295ad2eb0ee50b747818decdc5f25dde3ac5661238c7150a58b9a3957` |
| ds1000_case0452 | `6bf616f34b9b6b59f2edf6553ecfb7a51fd13afb01c281c48013914209b36561` |
| ds1000_case0453 | `190d5e7e714288e134faba2f83313c2580b36c3a7eb63805684087b0a7fdca7e` |
| ds1000_case0454 | `7788c5e19d340942ef75f97b183af5e58a697a28467dcc03ab0eb1a2bf8198dd` |
| ds1000_case0462 | `413b8a0231a04d614879bb85d748fdc3a31d83f3e945b22b365d6a672717c873` |
| ds1000_case0471 | `72c810b18a8b94d32fc266cfea94dff89e0e7b466a03935e3378bdeab98002d9` |
| ds1000_case0472 | `84b52d8de581b31947615d55a09ee5e1173edf4cd751d2652f83002e9654f4e3` |
| ds1000_case0478 | `483b55fb1b907f56632123fa833b449d4ac35b41efd5d14c4bcc354de385f723` |
| ds1000_case0480 | `1e51db147249a4ba89bd0d6cc00560416ac0a0fcd2c9de5bcf1ee47ecab3edcd` |
| ds1000_case0481 | `ccf38edf319072c94746afeb871f17655c77a50dec5bc8693a26f10e392a4510` |
| ds1000_case0485 | `75060a6908dd4e280f998e85a064f6b5fdd0579b3fc63360405b0fe2c883fd36` |
| ds1000_case0490 | `4a8007bf7aa0963a9ff6ea0753889d2fab294bf82f3ba646ae08d9dda1853065` |
| ds1000_case0491 | `c58529b2c58f8dbe9cb2ce5653a35ef97ac2a8509030ba0b4777b9125e9c57c6` |
| ds1000_case0495 | `34edbc81f6040ee93012df6a59c5a9889e3ac623c0cac55de3425246f14e8163` |
| ds1000_case0496 | `6277f3ef7477e8a03a175fc6a3bfd6e87e86009300945bd6bc7695a1163790b6` |
| ds1000_case0497 | `7d95a73777f45cae3fc5637ed9d68e4046654331c0e34f6003bfd47d959fc738` |
| ds1000_case0500 | `510c824894cc142bda6989cbf7dc56822d2a7a01b765f1ffbb993d9bbe710856` |
| ds1000_case0501 | `0d03d7c13277f8de7fafb1ef4586a4e88eccad3611b4903c65675192d0c8b6e2` |
| ds1000_case0503 | `0888ca12afe2f0a26b4e1f71860c4e51d118d6afe15e318bbfd33ac664beb3a9` |
| ds1000_case0504 | `225f233a1ae647d9f13757c02ef17d46f2d389ce0c9b44fe74b26b731cfc833a` |
| ds1000_case0505 | `1aba44f19d9c98643cbded4508e1fe9e99515ebbd0627a70777831062b6d8694` |
| ds1000_case0507 | `d7f83aa8fbcc92524d517f5015000d732ec1efb17d7dd77fe7df1df9c30d5f2a` |
| ds1000_case0508 | `dbbed643755831c88d5350b8ea5f3521ee047bdebec0cc2e427fc897b5fea016` |
| ds1000_case0509 | `99da4d9e30a9871b205ff2f32566cc688fe601247912c436046df6d0874ce458` |
| ds1000_case0510 | `58ba87d270ea29da80c84b9381c4c0714f1de74a0c77e1d864a54e536e9480fe` |
