# DS2 literal baselines

All 25 cases have halo2-base and Noir literal sources. Validation is non-proving only: A/B halo2 MockProver and Noir witness execution, NumPy output comparison, circuit counts, and one out-of-domain input rejection per case. No setup, key generation, proof, verification, timing benchmark, scaling, Zinnia execution, IR inspection, or Zinnia-size inspection was performed.

Builds use `HW_VARIANT=literal`, `HALO2_HW_CASES` restricted to this batch, `CARGO_TARGET_DIR=target-literal`, `--profile dev-fast`, three build jobs, and memrun with an 8 GiB cap. Optimized comparisons use the same checker with `HW_VARIANT` unset and `CARGO_TARGET_DIR=target`. Heavy checks use `lock-b`. Halo2 figures below are advice + lookup = total, at the checker's **automatic sizing** (lookup widths recorded below); they are not controlled-common-width or proving-time comparisons. Noir counts are the driver's unpadded UltraHonk gates.

| Case | Literal halo2 | Literal Noir | Optimized halo2 | Optimized Noir | Literal verdict H / N | Optimized verdict H / N | Notes |
|---|---:|---:|---:|---:|---|---|---|
| 0340 | 1200 + 360 = 1560 | 3004 | 1200 + 360 = 1560 | 3004 | pass / pass | pass / pass | Reused unchanged; flatten is a row-major wire copy. **Not larger: halo2, Noir.** |
| 0345 | 5446 + 628 = 6074 | 3515 | 2624 + 628 = 3252 | 3179 | pass / pass | pass / pass | Reused; fixed full-array multiplication to slice gather/multiply/scatter; gathered cumsum input fully before reduction. |
| 0346 | 4788 + 472 = 5260 | 3497 | 1930 + 472 = 2402 | 3167 | pass / pass | pass / pass | Reused; analogous row gather/multiply/scatter and separate cumsum input. |
| 0347 | 5720 + 706 = 6426 | 3566 | 3201 + 771 = 3972 | 3237 | policy FAIL* / pass | pass / pass | Reused; fixed full-array division to selected slice and removed shared bounded divisor. |
| 0353 | 7409 + 1788 = 9197 | 3610 | 3250 + 822 = 4072 | 3318 | policy FAIL* / pass | pass / pass | Separate axis maxima, elementwise maximum, ravel calls, membership scan, count/equality compaction. |
| 0355 | 6913 + 1107 = 8020 | 3731 | 3807 + 837 = 4644 | 3376 | policy FAIL* / pass | pass / pass | Nine stable length-3 argsorts, constant indices, standard gathers. |
| 0358 | 4626 + 633 = 5259 | 3330 | 2061 + 483 = 2544 | 3202 | pass / pass | pass / pass | Full axis sums, stable length-3 argsort, standard gathers. |
| 0359 | 276 + 84 = 360 | 2854 | 276 + 84 = 360 | 2854 | pass / pass | pass / pass | Reused draft; repaired Rust closure capture. Constant deletion. **Not larger: halo2, Noir.** |
| 0360 | 552 + 168 = 720 | 2891 | 552 + 168 = 720 | 2891 | pass / pass | pass / pass | Reused unchanged after audit. Constant deletion. **Not larger: halo2, Noir.** |
| 0361 | 960 + 288 = 1248 | 2966 | 960 + 288 = 1248 | 2966 | pass / pass | pass / pass | Reused draft; repaired Rust closure capture. Constant deletion. **Not larger: halo2, Noir.** |
| 0362 | 1600 + 480 = 2080 | 3066 | 1600 + 480 = 2080 | 3066 | pass / pass | pass / pass | Reused Rust draft; repaired capture and added Noir. Constant mask deletes column zero. **Not larger: halo2, Noir.** |
| 0363 | 15091 + 3368 = 18459 | 4800 | 5168 + 816 = 5984 | 3887 | policy FAIL* / pass | pass / pass | Insertion by normalized position and prefix/element/suffix selects. |
| 0364 | 54174 + 10640 = 64814 | 13180 | 40507 + 8088 = 48595 | 11368 | policy FAIL* / pass | pass / pass | Row insertion by normalized position and prefix/element/suffix selects. |
| 0368 | 2768 + 675 = 3443 | 3024 | 1335 + 345 = 1680 | 3025 | pass / pass | pass / pass | Full isclose broadcast including self comparisons, then running all. **Not larger: Noir.** |
| 0369 | 2768 + 675 = 3443 | 3024 | 1335 + 345 = 1680 | 3025 | pass / pass | pass / pass | Full column isclose broadcast including self comparisons, then running all. **Not larger: Noir.** |
| 0374 | 188521 + 43785 = 232306 | 34188 | 12580 + 2780 = 15360 | 4537 | policy FAIL* / pass | pass / pass | ECDF helper sorts all 50 values; full masks, first-occurrence argmax, both branches computed. |
| 0375 | 144444 + 33527 = 177971 | 29480 | 5967 + 1306 = 7273 | 4036 | policy FAIL* / pass | pass / pass | ECDF helper sorts all 50 values; full threshold mask, first-occurrence argmax, indexed read. |
| 0381 | 616600 + 124550 = 741150 | 125038 | 66135 + 12570 = 78705 | 5366 | policy FAIL* / pass | pass / pass | Stable 100-element argsort, full reversal and slice. |
| 0383 | 616600 + 124550 = 741150 | 107640 | 13759 + 1930 = 15689 | 5180 | policy FAIL* / pass | pass / pass | Stable 100-element argsort, full reversal, then seven-element slice. |
| 0385 | 540000 + 160000 = 700000 | 127816 | 540000 + 160000 = 700000 | 127816 | pass / pass | pass / pass | Full intermediate reshape/swapaxes/transpose arrays; only wire permutations. **Not larger: halo2, Noir.** |
| 0389 | 1944 + 576 = 2520 | 3266 | 1944 + 576 = 2520 | 3266 | pass / pass | pass / pass | Full intermediate reshape/swapaxes arrays; only wire permutations. **Not larger: halo2, Noir.** |
| 0391 | 1620 + 480 = 2100 | 3191 | 1620 + 480 = 2100 | 3191 | pass / pass | pass / pass | Constant slice; only read columns get domain checks. **Not larger: halo2, Noir.** |
| 0398 | 1632 + 384 = 2016 | 3070 | 1487 + 384 = 1871 | 3070 | pass / pass | pass / pass | Source recurrence with separate exact integer/Q32 products and additions. **Not larger: Noir.** |
| 0399 | 1896 + 396 = 2292 | 3102 | 1611 + 396 = 2007 | 3102 | pass / pass | pass / pass | Source recurrence with separate exact integer/Q32 products and additions. **Not larger: Noir.** |
| 0402 | 306 + 80 = 386 | 58 | 33 + 10 = 43 | 58 | pass / pass | pass / pass | Fortran ravel sum; a unread, coordinates range-checked. **Not larger: Noir.** |

## Checker-policy conflict and unfinished acceptance

`policy FAIL*` is an actual checker failure, not a claimed pass. In these rows, the only checker problem is `N inputs checked twice`: the A/B mocks satisfy constraints and match NumPy, and out-of-domain inputs are rejected. L5 explicitly requires fresh `q.bound` operands at every comparison; L6 prohibits `load_*_bounded` and sharing an `I64` across source operations. The unchanged checker (`hw_check/check.py`) unconditionally rejects any repeated bound of a loaded input. Likewise each plain variable division requires its own bounded divisor. Sources follow L5/L6, and do not evade the counter with extra identity arithmetic or change shared tools. Full green checker acceptance of these cases is not finishable under both requirements as written. These are complete sources with a reported tool-policy blocker, not unimplemented computations.

- 0347: 6 inputs checked twice.
- 0353: 54 inputs checked twice.
- 0355: 27 inputs checked twice.
- 0363: 1 inputs checked twice.
- 0364: 1 inputs checked twice.
- 0374: 50 inputs checked twice.
- 0375: 100 inputs checked twice.
- 0381: 100 inputs checked twice.
- 0383: 100 inputs checked twice.

## Interpretations for review

- Augmented assignment (0345–0347) reads the selected slice, computes only that slice, then scatters the result with equality selects over all positions. The earlier drafts computed arithmetic on unselected entries, which added work and could reject valid inputs. Slice read and following reduction input are separate full arrays; index subtraction is repeated at each source occurrence. The reused polynomial trick for wrapped-index matches was replaced with two equality tests and OR, matching Noir.
- Constant deletions, masks, shapes, index arrays, slices and reshape permutations fold at authoring/compile time under L3. Removed/unread input entries are not domain-checked (L7). Full wire arrays are retained at every reshape/transpose stage. A Python flat/list copy needs no arithmetic.
- `np.insert`, not explicitly listed in L5, uses its direct piecewise definition at each output position: prefix, inserted element/row, shifted suffix. Negative positions normalize by axis length, with the positive endpoint accepted. One row-position decision is used across columns of a row insertion.
- Sorts are stable adjacent bubble sorts with exactly n(n-1)/2 comparisons; argsort carries indices, while plain sort carries only values. Reversing argsort reverses ties as the source requests. Boolean argmax scans best value and best index, with strict improvement and first occurrence winning, including all-false input.
- ECDF ys is an entirely input-independent NumPy expression. It is constant-folded and quantized once (round(k/50 * 2^32)), matching optimized statement arithmetic. Helper calls stay at the kernel call sites; 0374 computes all branch results before selecting. Its inactive indexed lookup is still valid: argmax-1 lies in [-1,48]. Integer-to-float conversion is repeated for each source comparison expression.
- `isclose` uses abs(x-y) <= 0 + rtol*abs(y), with a shared Q32 product per broadcast element, including self-comparisons; no hand-hoisting of tolerances. `.all()` is an index-order Boolean accumulator.
- `ravel_multi_index` uses coordinate-times-stride sums in index order. Case 0353 independently validates both calls, including intermediate dimension products to prevent field-wrap acceptance. In 0402, the halo2 checked loader supplies input witnesses (L6 disallows baseline-owned witness loads), followed by coordinate bounds; Noir retains the optimized coordinate checks, which imply its input domain. The unused a parameter is unconstrained.
- All integer-times-Q32 products in 0398/0399 use exact field multiplication, as the shared arithmetic contract specifies, with source-order additions. Static loop-index branches fold; no fused inner products are authored.
- Each fresh comparison bounds both operands, including constant operands in the insertion scan. Shared gadgets may optimize internally, and Noir may eliminate duplicate work through its own SSA passes. No manual I64/window reuse, hinted sorts, custom hints, or forbidden indexing/compaction helpers are used.

## Cases not larger than optimized

- 0340 (halo2, Noir): The source is already a pure constant-index copy/permutation; the same read-input domain checks dominate both forms.
- 0359 (halo2, Noir): The source is already a pure constant-index copy/permutation; the same read-input domain checks dominate both forms.
- 0360 (halo2, Noir): The source is already a pure constant-index copy/permutation; the same read-input domain checks dominate both forms.
- 0361 (halo2, Noir): The source is already a pure constant-index copy/permutation; the same read-input domain checks dominate both forms.
- 0362 (halo2, Noir): The source is already a pure constant-index copy/permutation; the same read-input domain checks dominate both forms.
- 0368 (Noir): Literal Noir is one gate smaller (3024 versus 3025). It uses the recommended abs_int64 for differences, while optimized Noir uses abs followed by window. Noir can also fold self comparisons and merge identical broadcast tolerances. Thus the optimized source is not strictly minimal under this toolchain; the literal source still authors every broadcast operation (no exact attribution of the one gate to an individual compiler pass was attempted).
- 0369 (Noir): Literal Noir is one gate smaller (3024 versus 3025). It uses the recommended abs_int64 for differences, while optimized Noir uses abs followed by window. Noir can also fold self comparisons and merge identical broadcast tolerances. Thus the optimized source is not strictly minimal under this toolchain; the literal source still authors every broadcast operation (no exact attribution of the one gate to an individual compiler pass was attempted).
- 0385 (halo2, Noir): The source is already a pure constant-index copy/permutation; the same read-input domain checks dominate both forms.
- 0389 (halo2, Noir): The source is already a pure constant-index copy/permutation; the same read-input domain checks dominate both forms.
- 0391 (halo2, Noir): The source is already a pure constant-index copy/permutation; the same read-input domain checks dominate both forms.
- 0398 (Noir): Noir compiles the direct field recurrence to the same optimized arithmetic; the halo2 source retains separate multiply/add operations.
- 0399 (Noir): Noir compiles the direct field recurrence to the same optimized arithmetic; the halo2 source retains separate multiply/add operations.
- 0402 (Noir): Noir folds the constant stride arithmetic to the same expression and uses identical coordinate-domain constraints.

## Evidence and limits

`hw_check.lint` on all 25 cases with `HW_VARIANT=literal`: **0 errors**. Output headers were checked against all optimized Noir headers; sources were reviewed against kernel.py and L1–L7. Passing fixtures do not establish correctness for every possible input, and no proof/public-tamper verification was run. The checker was not changed. Optimized sources and shared libraries are unchanged from base commit `b1e353d5fd2b08c0a609e31da991a88f170a90eb`. The environment emits benign virtualenv prefix warnings. An initial build invoked outside halo2_hw missed its local RUSTC_BOOTSTRAP config; rebuilding from the required directory fixed that invocation error.

Evidence directory: `/home/zhantong/Projects/zinnia-project/eval-redo/literal-work/ds2/`. Logs: `build0340.log`, `build1.log`, `build2.log`, `build3.log`, `build4.log`, `build-opt.log`, `check0340.log`, `check1.log`, `check2.log`, `check3.log`, `check4.log`, `check-optimized.log`, `lint-final.log`. Machine-readable checks: `results1.json`, `results2.json`, `results3.json`, `results4.json` (later per-system results supersede earlier rows), `results-optimized.json`. Work packages: `check/`, `check-final/`, `check-four/`, `check-optimized/`. No generated Prover files or build output are committed.

| Case | Literal k / lookup bits | Optimized k / lookup bits | SHA256 kernel + literal Rust + literal Noir (concatenated in that order) |
|---|---|---|---|
| 0340 | 7 / 6 | 7 / 6 | `ba879a752f78c136d0bd7d8816f436bfe5377b0374f17455319ca9e98bd58b34` |
| 0345 | 8 / 7 | 8 / 7 | `8b4434b6505fd38cb772d060227e9a5752810266aa075ce9ab1cdcfd15c0218a` |
| 0346 | 9 / 8 | 9 / 8 | `223b9055c3bcd8757b1e44853dbf435ea2a9cee283bff653f89e6b8bea0fba4c` |
| 0347 | 9 / 8 | 8 / 7 | `0ef0d3574a1c7afe9f97ce11b10d193e342115c4620c671586b28ce24e32eada` |
| 0353 | 9 / 8 | 8 / 7 | `ed9d54c2f5ad128469bd2fb6fea6559c5b4dece19d62f14317a103529151f701` |
| 0355 | 9 / 8 | 8 / 7 | `9431a0b3a94434b9a39437a5cf06a8c7a96bc107329276dd2193898869d18b77` |
| 0358 | 8 / 7 | 9 / 8 | `b205ae50fd3e18349d1cb39f56a56d7372292e494d2043522089b5fcd64d5530` |
| 0359 | 6 / 5 | 6 / 5 | `abf03c7ac8395d40e17a76b7d826472a58233e0ba70b712525b646d04fb082c6` |
| 0360 | 6 / 5 | 6 / 5 | `201aa59e759ad7f946b19fab8cb8ab2334b7b7e63ff025d2291564bd7ac1b551` |
| 0361 | 7 / 6 | 7 / 6 | `fe4019c3b7e9a1a01f7cb39e5a4b7cb0d020651a1a6bad8ee4bd24103726061c` |
| 0362 | 7 / 6 | 7 / 6 | `76902b0a5ac5e2b8f893851dab91290db28e95f5c35a2c19fb88d430ca2a6f2f` |
| 0363 | 9 / 8 | 9 / 8 | `6da4af5d45823d291e90244b11839462fc9f39268ed80126ce42217b33d405ef` |
| 0364 | 11 / 8 | 11 / 8 | `23c3b1ff33fc02a5620295962b0a7cdf87d60e5e8dc84404be819aca35bea090` |
| 0368 | 8 / 7 | 7 / 6 | `abf8841143937ce80e9f8cfaf84d32516bc686d0156e6116101a411a1e441ae8` |
| 0369 | 8 / 7 | 7 / 6 | `d62726e62683050699b735dd0ea571c0b101e7ab882f89a2ee3807336c9919f2` |
| 0374 | 10 / 8 | 9 / 8 | `c11ff59baab5e8b3b07efbaed68dbdcbc3bcd28b5e96a3fccc368173c8a7c8f9` |
| 0375 | 10 / 8 | 9 / 8 | `b6aa2751cdf14958078071d79cea9a18b7348732d8530b888e2f4825ca1a8342` |
| 0381 | 11 / 8 | 9 / 8 | `e8beb7badd91ca4aa4b24a3543058867caa5d19c7a7aff8ab48f9fa6469f7bd1` |
| 0383 | 11 / 8 | 9 / 8 | `f12f2eac5cd15dd01ef57105c8fde5badee2c074c25865341a9bb0bad8fab258` |
| 0385 | 15 / 8 | 15 / 8 | `a8f489d5aaabf4bfce580f5e358b51bbca1c34764d488bc348f46339e6501abe` |
| 0389 | 9 / 8 | 9 / 8 | `8762bd4f0e0298a79f3a953bf57243a11a1fc376912fc8274201a791804f54dc` |
| 0391 | 9 / 8 | 9 / 8 | `ccf0d4862eeda0d855c693a3f0e77bcd3d54a8c28f638547db5e678564ced2b6` |
| 0398 | 7 / 6 | 7 / 6 | `853a301852382ae83d79a5919873411a292f60560a65dc97eca7d672d1a8f544` |
| 0399 | 7 / 6 | 7 / 6 | `5566e81268629c9f39e3cd75842e72329f56c53335832e9521232475d0a3dd1f` |
| 0402 | 6 / 5 | 6 / 5 | `23138c7e3226b4bea5d6863977c05a79e72a6cd80775020a8010e6a6a917b463` |
