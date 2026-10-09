# DS-1000 literal arithmetic audit (2026-10-09)

All 100 DS-1000 cases (both literal implementations, their optimized counterparts and kernel.py), including 0420, were statically audited for rounding, quantized constants, power method, input/intermediate/output guards, and failing inactive branches. Eight cases changed. The six reviewer findings are repaired. No known literal-versus-optimized arithmetic mismatch remains. One intentional acceptance difference remains: the optimized 0420 inactive-comparison defect described below. This is a source-level audit plus targeted circuit executions, not an exhaustive proof over all inputs.

Only the eight cases' literal files and this report were edited. Optimized files, shared gadgets, tools, and repository fixtures were not changed. All edge fixtures and execution evidence are outside the repository under `/home/zhantong/Projects/zinnia-project/eval-redo/literal-work/arith-ds/`. The initial source revision was `875c269b23f6fe5ab3f8aac1150a05a51467bf74`. `audit-sha256.json` there records kernel and both baseline source hashes for all 100 cases.

## Changed cases and measured counts

Every row passed `HW_VARIANT=literal hw_check.check --no-prove`: Halo2 mocks A/B, cell statistics and out-of-domain rejection; Noir compilation/gates, A/B witness execution and out-of-domain rejection. Optimized counterparts passed the same no-prove check with HW_VARIANT unset. No keys, proofs, timing/scaling benchmarks or Zinnia runs were performed.

Counts below are automatic-sizing Halo2 advice + lookup cells (fixed/instance cells excluded), and unpadded Noir gates. Each count was freshly measured in this session; unchanged cases were audited statically and were not remeasured. Before values for 0420 and 0453 come from `first.json`, before their edits; the other before values come from `before.json`. After values come from `first.json`, `second.json`, and `l4.json` as appropriate. Optimized values come from `optimized.json`.

| Case | Literal Halo2 before -> after (advice + lookup = total) | Noir before -> after | Optimized Halo2 (advice + lookup = total) | Optimized Noir |
|---|---|---|---|---|
| 0300 | 22123 + 5023 = **27146** -> 22045 + 5003 = **27048** | 6895 -> **6880** | 3331 + 692 = **4023** | 3368 |
| 0326 | 1232 + 336 = **1568** -> 1232 + 336 = **1568** | 2986 -> **2996** | 1228 + 336 = **1564** | 2974 |
| 0420 | 635 + 165 = **800** -> 643 + 165 = **808** | 2877 -> **2877** | 529 + 142 = **671** | 2880 |
| 0428 | 4844 + 1070 = **5914** -> 2956 + 462 = **3418** | 3270 -> **3086** | 1804 + 443 = **2247** | 3094 |
| 0429 | 4844 + 1070 = **5914** -> 2956 + 462 = **3418** | 3270 -> **3086** | 1804 + 443 = **2247** | 3094 |
| 0431 | 3128 + 766 = **3894** -> 3196 + 738 = **3934** | 3145 -> **3101** | 1624 + 395 = **2019** | 3053 |
| 0438 | 4034 + 930 = **4964** -> 2750 + 624 = **3374** | 3264 -> **3057** | 742 + 192 = **934** | 2930 |
| 0453 | 4596 + 1105 = **5701** -> 4876 + 1185 = **6061** | 3511 -> **3573** | 3220 + 845 = **4065** | 3378 |

| Case | Halo2 (k, lookup bits), before -> after; optimized |
|---|---|
| 0300 | (9, 8) -> (9, 8); (8, 7) |
| 0326 | (7, 6) -> (7, 6); (7, 6) |
| 0420 | (7, 6) -> (7, 6); (6, 5) |
| 0428 | (9, 8) -> (8, 7); (7, 6) |
| 0429 | (9, 8) -> (8, 7); (7, 6) |
| 0431 | (9, 8) -> (8, 7); (7, 6) |
| 0438 | (8, 7) -> (8, 7); (7, 6) |
| 0453 | (9, 8) -> (9, 8); (9, 8) |

0431 has fewer rescale operations but a slightly higher automatic cell total because sizing changes from lookup width 8 to 7; these are not common-width circuit comparisons. Every changed Halo2 literal remains larger than optimized. Noir 0420 (2877 vs 2880), 0428 and 0429 (3086 vs 3094) are slightly smaller: the authored code retains independent source calls; Noir may eliminate repeated pure expressions, and the literal std helpers use `floor_div_const` whereas optimized uses general `divmod_pos`, with identical arithmetic and domain. No authored CSE was introduced.

## Arithmetic and structure choices

- **0300:** Percentile retains the full stable bubble sort and both indexed reads; interpolate with floor(rem * difference / 100), without first quantizing the weight.
- **0326:** Keep the arcsin call and its materialized radians; degrees uses floor(radians * Q(360/pi) / (2*SCALE)), exactly the optimized constant and rounding.
- **0420:** Retain the existing inactive-polynomial guard and independently computed powers. Guard x_max before converting/comparing it in the inactive inner where branch (L4).
- **0428:** Each of the two std calls still independently forms every deviation and every raw square in index order; rescale the sum once, then divide by 30 and take sqrt.
- **0429:** Same independent-call squared-sum correction as 0428; retain the source multiplier 2.
- **0431:** Same independent-call squared-sum correction, divide by 8; preserve both comparison arrays, conjunction and negation.
- **0438:** Form the full 2x2 centred-moment matrix and all four normalized/clipped entries; use optimized integer Pearson arithmetic and its root/division domain.
- **0453:** Restore the optimized signed-32 operand guards before forming every integer square and row sum; retain all source intermediates and separate divisions.

For 0326, the radians code is exactly twice the shared atan result. Thus division by `2*SCALE` after multiplication by `Q(360/pi)` equals the optimized result without changing the separately materialized source call. No rounded half-constant is introduced.

For 0438, `np.corrcoef` is one library call. Its arithmetic is the optimized Pearson contract, but the literal library expansion still builds the full matrix: every pair of rows forms all products and their ordered sum; both off-diagonal entries are computed independently; all four entries are normalized and clipped. The centred moment is `Mij = 4*sum(xi*xj) - sum(xi)*sum(xj)`. Diagonal moments B and C have the same unsigned64 guards as optimized, and the off-diagonal denominator is `floor(sqrt(B*C)*SCALE)` with the same signed63 root restriction and nonzero division. For diagonal entries, `sqrt(B*B)` is computed by the shared integer-root gadget, followed by integer-unit Q32 division. This is exact (the radicand is a perfect square), and does not impose the spurious `B*SCALE < 2^63` restriction. On the optimized accepted domain, B,C are positive integers and B*C < 2^62, so each diagonal root fits the shared root window. This representation choice preserves the full matrix without adding a stricter domain.

0453's restored operand checks are the optimized signed32 guards. With already checked signed64 inputs, the accepted norm domain implies them mathematically; the boundary test rejects in literal and optimized. This is explicit guard alignment, not a claimed new counterexample or a new accepted-input restriction.

## Reviewer counterexamples (both backends, both variants)

`edges.json` records every Halo2 `mock --fixture` and Noir `execute --fixture` result. All seven DS reviewer examples (including both percentile magnitudes) now agree. Noir returned field values were also compared exactly, not only as decoded floating-point values.

| Case / input | Literal and optimized result in both systems |
|---|---|
| 0300: a=[0]+[1e9]*19, p=1 | Accept, 190000000 exactly. |
| 0300: a=[0]+[100]*19, p=1 | Accept, 19 exactly (the old result was 24 Q32 units lower). |
| 0326: value=0.5 | Accept, 29.999999900814146 (same Q32 code). |
| 0428: a codes=[365092,0,...,0], N=30 | Accept, interval codes [-184439,208777]. |
| 0429: same input | Accept, interval codes [-118903,143241]. |
| 0431: a codes=[198163,0,...,0], N=8 | Accept, [true,false,false,false,false,false,false,false]. |
| 0438: post=distance=[0,30000,0,30000] | Both reject the root-window overflow. |

Additional `edges.json` tests: 0438 post=[0,1000000,0,1000000], distance=[0,1,0,1] accepts with r=1 (guards against accidentally narrowing the diagonal domain); constant post rejects zero variance; 0420 x=+/-10000 with bounds 0,1 accepts the selected clamp; 0453 rows [2^31,0,0,0] reject in both variants.

## Remaining difference: optimized 0420 defect

With **x=-1, x_min=0, x_max=2^40**, the selected outer branch returns 0. Both optimized backends nevertheless reject because the unused inner comparison bounds `float(x_max)` in Q32. Literal now selects zero for that comparison's x_max operand when the outer branch is selected. Both literal backends accept and return 0. This is the explicitly required L4 exception to domain matching; optimized files were left untouched. `l4-edges.json` contains all four executions. A control with x=0.5 and the same bounds rejects in both variants because the inner comparison is active and retains the optimized guard.

The restored polynomial guard already existed in the starting 0420 files; the historical withdrawal in LITERAL_DS3.md is not the current implementation. No other failing inactive branch was found. No other unresolved literal-versus-optimized rounding, constant, power-method or acceptance difference was identified within the audited contracts. Explicit optimized intermediate-window restrictions are treated as the arithmetic contract, rather than silently broadened to all real-valued NumPy inputs.

## Complete static audit ledger

Each unchanged row means both backends were compared statically; its counts are unchanged and were not rerun. Changed-case counts and witness results are above.

| Case | Status | Arithmetic/domain/L4 audit |
|---|---|---|
| 0298 | Unchanged; static audit | Exact unique ranks and inverse indices; full stable sort/compaction; same input window and dynamic shape. |
| 0300 | Changed; checked both | Percentile retains the full stable bubble sort and both indexed reads; interpolate with floor(rem * difference / 100), without first quantizing the weight. |
| 0301 | Unchanged; static audit | Only copy/reshape/transpose/flatten wiring; same checked input elements and output layout; no rounding or failing inactive operation. |
| 0309 | Unchanged; static audit | Same signed64 comparisons, first-tie extrema and exact index arithmetic; literal recomputes source extrema. |
| 0310 | Unchanged; static audit | Same signed64 comparisons, first-tie extrema and exact index arithmetic; literal recomputes source extrema. |
| 0311 | Unchanged; static audit | Same signed64 comparisons, first-tie extrema and exact index arithmetic; literal recomputes source extrema. |
| 0312 | Unchanged; static audit | Same signed64 comparisons, first-tie extrema and exact index arithmetic; literal recomputes source extrema. |
| 0314 | Unchanged; static audit | Same signed64 comparisons, first-tie extrema and exact index arithmetic; literal recomputes source extrema. |
| 0317 | Unchanged; static audit | Only copy/reshape/transpose/flatten wiring; same checked input elements and output layout; no rounding or failing inactive operation. |
| 0318 | Unchanged; static audit | Same explicitly documented permutation-of-0..N precondition; full inverse assignment then gather. Negative permutations remain excluded by that precondition. |
| 0319 | Unchanged; static audit | Same explicitly documented permutation-of-0..N precondition; full inverse assignment then gather. Negative permutations remain excluded by that precondition. |
| 0320 | Unchanged; static audit | Same signed64 comparisons, first-tie extrema and exact index arithmetic; literal recomputes source extrema. |
| 0321 | Unchanged; static audit | Same signed64 comparisons, first-tie extrema and exact index arithmetic; literal recomputes source extrema. |
| 0322 | Unchanged; static audit | Same minimum, all equal-minimum coordinates, and dynamic length; padded compaction terms cannot fail. |
| 0323 | Unchanged; static audit | Same Q(pi/180), exact integer scaling, shared trigonometric gadgets and their range guards; independent calls retained. |
| 0324 | Unchanged; static audit | Same Q(pi/180), exact integer scaling, shared trigonometric gadgets and their range guards; independent calls retained. |
| 0325 | Unchanged; static audit | Same Q(pi/180), exact integer scaling, shared trigonometric gadgets and their range guards; independent calls retained. |
| 0326 | Changed; checked both | Keep the arcsin call and its materialized radians; degrees uses floor(radians * Q(360/pi) / (2*SCALE)), exactly the optimized constant and rounding. |
| 0327 | Unchanged; static audit | Same selected/read input scope, constant outputs and wiring. No additional output bounds introduced. |
| 0328 | Unchanged; static audit | Same selected/read input scope, constant outputs and wiring. No additional output bounds introduced. |
| 0329 | Unchanged; static audit | Same exponent-six repeated-squaring method (x2, x4, x2*x4) and each multiplication rounding/domain. |
| 0334 | Unchanged; static audit | Same exact sums and floor division for means; equivalent constant-divisor gadgets do not bound the unaveraged sum. |
| 0335 | Unchanged; static audit | Same extrema over checked signed64 inputs; all literal pairwise comparisons are safe and preserve ties. |
| 0336 | Unchanged; static audit | Same selected/read input scope, constant outputs and wiring. No additional output bounds introduced. |
| 0338 | Unchanged; static audit | Same selected/read input scope, constant outputs and wiring. No additional output bounds introduced. |
| 0340 | Unchanged; static audit | Only copy/reshape/transpose/flatten wiring; same checked input elements and output layout; no rounding or failing inactive operation. |
| 0345 | Unchanged; static audit | Same negative-index wrap and index domain, selected-element Q32 products, then exact cumsum; full writes and second reads retained. |
| 0346 | Unchanged; static audit | Same negative-index wrap and index domain, selected-element Q32 products, then exact cumsum; full writes and second reads retained. |
| 0347 | Unchanged; static audit | Same divisor sign convention, quotient ranges and separately rounded running product in each backend; inactive writes only select already computed values. |
| 0353 | Unchanged; static audit | Same nonnegative coordinates and signed63 total dimension product; literal prefix-product checks are implied by positive dimensions and final product, with no field wrap. |
| 0355 | Unchanged; static audit | Same stable ordering/ties and checked keys (0358: exact plane sums); full bubble sort replaces optimized ordering without changing arithmetic/domain. |
| 0358 | Unchanged; static audit | Same stable ordering/ties and checked keys (0358: exact plane sums); full bubble sort replaces optimized ordering without changing arithmetic/domain. |
| 0359 | Unchanged; static audit | Same selected/read input scope, constant outputs and wiring. No additional output bounds introduced. |
| 0360 | Unchanged; static audit | Same selected/read input scope, constant outputs and wiring. No additional output bounds introduced. |
| 0361 | Unchanged; static audit | Same selected/read input scope, constant outputs and wiring. No additional output bounds introduced. |
| 0362 | Unchanged; static audit | Same selected/read input scope, constant outputs and wiring. No additional output bounds introduced. |
| 0363 | Unchanged; static audit | Same insertion domain [-100,100], negative wrap and endpoint; candidates use only static in-range reads and exact selects. |
| 0364 | Unchanged; static audit | Same insertion domain [-100,100], negative wrap and endpoint; candidates use only static in-range reads and exact selects. |
| 0368 | Unchanged; static audit | Same absolute-difference domain and quantized isclose constants. abs_int64 plus signed comparisons equals optimized abs/window domain, including rejection at magnitude 2^63. |
| 0369 | Unchanged; static audit | Same absolute-difference domain and quantized isclose constants. abs_int64 plus signed comparisons equals optimized abs/window domain, including rejection at magnitude 2^63. |
| 0374 | Unchanged; static audit | ECDF constants match floor((count*SCALE+25)/50). Inactive argmax-minus-one index is always in [-1,48], so the standard wrapped read cannot fail. |
| 0375 | Unchanged; static audit | Rounded ECDF constants give exactly optimized floor((50*threshold+24)/SCALE) with fallback rank zero; full sort/argmax retained. Rescaled optimized rank fits signed64 for every input code. |
| 0381 | Unchanged; static audit | Same stable ordering/ties and checked keys (0358: exact plane sums); full bubble sort replaces optimized ordering without changing arithmetic/domain. |
| 0383 | Unchanged; static audit | Same stable ordering/ties and checked keys (0358: exact plane sums); full bubble sort replaces optimized ordering without changing arithmetic/domain. |
| 0385 | Unchanged; static audit | Only copy/reshape/transpose/flatten wiring; same checked input elements and output layout; no rounding or failing inactive operation. |
| 0389 | Unchanged; static audit | Only copy/reshape/transpose/flatten wiring; same checked input elements and output layout; no rounding or failing inactive operation. |
| 0391 | Unchanged; static audit | Same selected/read input scope, constant outputs and wiring. No additional output bounds introduced. |
| 0398 | Unchanged; static audit | Same exact field additions/integer multiplications; no new intermediate/output window bounds, no added rounding; source recurrences/reductions remain separate. |
| 0399 | Unchanged; static audit | Same exact field additions/integer multiplications; no new intermediate/output window bounds, no added rounding; source recurrences/reductions remain separate. |
| 0402 | Unchanged; static audit | Same per-axis nonnegative index bounds and exact ravel strides; unused a is not checked. |
| 0403 | Unchanged; static audit | Same per-axis nonnegative index bounds and exact ravel strides; unused a is not checked. |
| 0409 | Unchanged; static audit | Same exact field additions/integer multiplications; no new intermediate/output window bounds, no added rounding; source recurrences/reductions remain separate. |
| 0412 | Unchanged; static audit | Same element predicates and checked inputs, length and zero-padded compact outputs. Equality/Boolean/selector operations cannot fail in unused slots. |
| 0414 | Unchanged; static audit | Same exact sums and floor division for means; equivalent constant-divisor gadgets do not bound the unaveraged sum. |
| 0415 | Unchanged; static audit | Same extrema over checked signed64 inputs; all literal pairwise comparisons are safe and preserve ties. |
| 0416 | Unchanged; static audit | Same exact sums and floor division for means; equivalent constant-divisor gadgets do not bound the unaveraged sum. |
| 0420 | Changed; checked both | Retain the existing inactive-polynomial guard and independently computed powers. Guard x_max before converting/comparing it in the inactive inner where branch (L4). |
| 0422 | Unchanged; static audit | Every product formed in source reduction order; one rescale per correlate/tensordot output, identical to optimized dot. |
| 0425 | Unchanged; static audit | Same low two-complement bits for every signed64 input, with explicit source mask and >0 comparisons; all shift counts are constants. |
| 0426 | Unchanged; static audit | Same low two-complement bits for every signed64 input, with explicit source mask and >0 comparisons; all shift counts are constants. |
| 0428 | Changed; checked both | Each of the two std calls still independently forms every deviation and every raw square in index order; rescale the sum once, then divide by 30 and take sqrt. |
| 0429 | Changed; checked both | Same independent-call squared-sum correction as 0428; retain the source multiplier 2. |
| 0431 | Changed; checked both | Same independent-call squared-sum correction, divide by 8; preserve both comparison arrays, conjunction and negation. |
| 0433 | Unchanged; static audit | Same wrapped write indices and bounds, repeated writes in source order, and checked full input; unselected assignments only select. |
| 0434 | Unchanged; static audit | Same wrapped write indices and bounds, repeated writes in source order, and checked full input; unselected assignments only select. |
| 0435 | Unchanged; static audit | Same selected/read input scope, constant outputs and wiring. No additional output bounds introduced. |
| 0436 | Unchanged; static audit | Same extrema over checked signed64 inputs; all literal pairwise comparisons are safe and preserve ties. |
| 0437 | Unchanged; static audit | Same extrema over checked signed64 inputs; all literal pairwise comparisons are safe and preserve ties. |
| 0438 | Changed; checked both | Form the full 2x2 centred-moment matrix and all four normalized/clipped entries; use optimized integer Pearson arithmetic and its root/division domain. |
| 0439 | Unchanged; static audit | All broadcast products computed, including symmetric entries; each has the same multiplication rounding/window as optimized. |
| 0440 | Unchanged; static audit | Same checked diagonal inputs and shared sqrt; no extra radicand or output bounds. Off-diagonal inputs remain unread. |
| 0441 | Unchanged; static audit | Equality tests/OR and optimized product-zero membership are equivalent over this prime field; checked signed64 inputs. |
| 0442 | Unchanged; static audit | Same element predicates and checked inputs, length and zero-padded compact outputs. Equality/Boolean/selector operations cannot fail in unused slots. |
| 0443 | Unchanged; static audit | Same element predicates and checked inputs, length and zero-padded compact outputs. Equality/Boolean/selector operations cannot fail in unused slots. |
| 0444 | Unchanged; static audit | Same element predicates and checked inputs, length and zero-padded compact outputs. Equality/Boolean/selector operations cannot fail in unused slots. |
| 0450 | Unchanged; static audit | Same checked inputs, NumPy negative-index wrap and bounds; native/standard reads replace optimized shared selectors. |
| 0452 | Unchanged; static audit | Same integer magnitudes, norm bounds, zero-divisor rejection and Q32 quotient. Any |INT64_MIN| exclusion is already implied by the optimized norm window. |
| 0453 | Changed; checked both | Restore the optimized signed-32 operand guards before forming every integer square and row sum; retain all source intermediates and separate divisions. |
| 0454 | Unchanged; static audit | Same integer magnitudes, norm bounds, zero-divisor rejection and Q32 quotient. Any |INT64_MIN| exclusion is already implied by the optimized norm window. |
| 0462 | Unchanged; static audit | Same element predicates and checked inputs, length and zero-padded compact outputs. Equality/Boolean/selector operations cannot fail in unused slots. |
| 0471 | Unchanged; static audit | Same selected/read input scope, constant outputs and wiring. No additional output bounds introduced. |
| 0472 | Unchanged; static audit | Same selected/read input scope, constant outputs and wiring. No additional output bounds introduced. |
| 0478 | Unchanged; static audit | Same selected/read input scope, constant outputs and wiring. No additional output bounds introduced. |
| 0480 | Unchanged; static audit | Same first match or -1; both extra source reads use argmax in [0,7], even when no match exists. |
| 0481 | Unchanged; static audit | Same element predicates and checked inputs, length and zero-padded compact outputs. Equality/Boolean/selector operations cannot fail in unused slots. |
| 0485 | Unchanged; static audit | Every product formed in source reduction order; one rescale per correlate/tensordot output, identical to optimized dot. |
| 0490 | Unchanged; static audit | Same checked thresholds and exact sequential masked assignments, including t1>=t2; inactive padded additions have no failing range operation. |
| 0491 | Unchanged; static audit | Same absolute-difference domain and quantized isclose constants. abs_int64 plus signed comparisons equals optimized abs/window domain, including rejection at magnitude 2^63. |
| 0495 | Unchanged; static audit | Same selected/read input scope, constant outputs and wiring. No additional output bounds introduced. |
| 0496 | Unchanged; static audit | Same selected/read input scope, constant outputs and wiring. No additional output bounds introduced. |
| 0497 | Unchanged; static audit | Same selected/read input scope, constant outputs and wiring. No additional output bounds introduced. |
| 0500 | Unchanged; static audit | Only copy/reshape/transpose/flatten wiring; same checked input elements and output layout; no rounding or failing inactive operation. |
| 0501 | Unchanged; static audit | Same checked inputs, NumPy negative-index wrap and bounds; native/standard reads replace optimized shared selectors. |
| 0503 | Unchanged; static audit | Same checked inputs, NumPy negative-index wrap and bounds; native/standard reads replace optimized shared selectors. |
| 0504 | Unchanged; static audit | Same exact field additions/integer multiplications; no new intermediate/output window bounds, no added rounding; source recurrences/reductions remain separate. |
| 0505 | Unchanged; static audit | Same exact field additions/integer multiplications; no new intermediate/output window bounds, no added rounding; source recurrences/reductions remain separate. |
| 0507 | Unchanged; static audit | Empty branch uses padded zero coordinate reductions, safe width 1 and safe index 0; nonempty crop bounds imply all active reads valid. Same dynamic shape/domain. |
| 0508 | Unchanged; static audit | Same required nonempty argwhere, coordinate extrema and crop; padded output reads select safe index 0. Empty input is deliberately rejected in both. |
| 0509 | Unchanged; static audit | Empty branch uses padded zero coordinate reductions, safe width 1 and safe index 0; nonempty crop bounds imply all active reads valid. Same dynamic shape/domain. |
| 0510 | Unchanged; static audit | Empty branch uses padded zero coordinate reductions, safe width 1 and safe index 0; nonempty crop bounds imply all active reads valid. Same dynamic shape/domain. |

## Reproduction and limits

Run from `benchmarking/baselines` with `../../.venv/bin/python`. Builds used `HALO2_HW_CASES=0300,0326,0420,0428,0429,0431,0438,0453`, `CARGO_TARGET_DIR=target-literal` (optimized: `target`), `CARGO_BUILD_JOBS=3`, `RUSTC_BOOTSTRAP=1`, `--profile dev-fast`, and the supplied memrun wrapper at 8G. All checks/edge jobs used `HW_CHECK_LOCK=/home/zhantong/Projects/zinnia-project/eval-redo/literal-work/lock-arith-ds` and TMPDIR/workroots below the external arith-ds work directory. Only `mock`, `stats`, and Noir `execute` were used.

`HW_VARIANT=literal python -m hw_check.lint` with all 100 case IDs finished with **0 errors**. Its only warnings are pre-existing constant shift/cast expressions in unchanged 0425 and 0426. `git diff --check` also passed.

No exhaustive input enumeration, malicious-witness soundness audit, proof validation, or new execution of the 92 unchanged cases is claimed. Corpus arithmetic/domain conclusions are static review findings supported by the targeted checks above. Shared gadget implementations and compiler behavior were not modified.
