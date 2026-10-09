# Literal NPBench/Pythran arithmetic audit — 2026-10-09

19 entries changed across the two backends. All 79 literal/optimized pairs were statically audited for rounding points, constants, range checks, input dependence and inactive branches. No unexplained arithmetic or acceptance-domain discrepancy remains from this audit. Literal intentionally accepts the four optimized L4 defects listed below; the static review and finite tests are not an exhaustive equivalence proof.

Only literal baseline files and this report changed. No optimized files, gadgets, tools or repository fixtures were edited. No proofs, key generation, Zinnia runs, IR inspection or scaling runs were performed.

## Counts and checks

Each numeric pair is **Halo2 advice + lookup / Noir gates** (fixed and instance columns excluded). “Before” is freshly measured with `stats` from literal sources at commit `875c269b`, copied under the external work directory; “after” and optimized counts come from fresh `hw_check.check --no-prove`. Existing historical reports are not used as current before-count evidence. Verdicts are Halo2/Noir.

| Entry | Before literal H / N | After literal H / N | Optimized H / N | Literal; optimized checks | Change |
|---|---:|---:|---:|---|---|
| `npbench_deep_learning_conv2d_bias_conv2d__conv2d_bias` | 3,559,001 / 443,833 | 938,201 / 127,033 | 657,400 / 127,033 | pass/pass; pass/pass | Materialize all raw broadcast products, then rescale each reduction once; keep conv2d helper and separate bias addition. |
| `pythran_cronbach__cronbach` | 30,331 / 6,868 | 17,071 / 5,253 | 15,048 / 5,248 | pass/pass; pass/pass | Both variance calls retain means, deviations and square arrays; rescale each square sum once before ddof division. |
| `pythran_euclidean_distance_square__euclidean_distance_square` | 2,086 / 3,091 | 2,098 / 3,009 | 1,988 / 3,009 | pass/pass; pass/pass | Rescale each squared-norm reduction once; retain full square arrays, dot call and source addition order. |
| `pythran_fibo_seq__fibo` | 107,769 / 17,405 | 107,735 / 17,400 | 70,790 / 17,400 | pass/pass; pass/pass | Remove extra final signed64 result check; retain all 1024 recurrence steps and active selects. |
| `pythran_crotg__CROTG` | 4,779 / 3,336 | 4,787 / 3,338 | 3,812 / 3,296 | pass/pass; pass/pass | Mask CB before inactive magnitude; preserve every repeated abs call. Optimized still has the L4 defect. |
| `pythran_ising__ising_step` | 5,800 / 3,578 | 5,816 / 3,564 | 5,559 / 3,575 | pass/pass; pass/pass | Mask the inactive exponential argument before bounds/exp. Noir condition now uses optimized canonical comparison. Optimized still has the L4 defect. |
| `pythran_periodic_dist__dist` | 17,784,409 / 2,365,682 | 17,793,125 / 2,361,022 | 245,682 / 32,202 | pass/pass; pass/pass | Mask L separately per inactive axis; retain complete compaction/scatter work. Restore Halo2 signed64 and Noir unsigned64 three-square radicand guards. Noir comparisons follow optimized canonical gaps. |
| `pythran_clip__clip` | 4,157 / 3,143 | 3,469 / 3,211 | 2,639 / 3,112 | pass/pass; pass/pass | Restore Halo2 signed64 radicand guard and mask inactive limit1 operands. New optimized Noir L4 defect remains. |
| `pythran_sumarray3d__summation` | 197,217 / 26,812 | 206,177 / 27,388 | 180,432 / 24,444 | pass/pass; pass/pass | Restore Halo2 signed63 and Noir unsigned64 radicand checks, preserving nested source helper calls and ordered accumulations. |
| `pythran_wdist__slow_wdist` | 6,589 / 3,497 | 6,904 / 3,497 | 5,523 / 3,535 | pass/pass; pass/pass | Restore Halo2 signed63 radicand check. Noir unchanged: two nonnegative signed64 square results already imply its optimized unsigned64 guard. |
| `npbench_weather_stencils_hdiff_hdiff__hdiff` | 742,981 / 90,238 | 661,981 / 75,429 | 539,300 / 75,429 | pass/pass; pass/pass | Exact integer-coefficient 4*x products. Input guards follow optimized dependence: Noir omits dead-only bounds; Halo2 reconstructs dead-only quantized inputs from shared integer limbs. Every source lap corner remains computed. |
| `pythran_harris__harris` | 7,052 / 3,735 | 7,138 / 3,731 | 6,985 / 3,731 | pass/pass; pass/pass | Retain both full difference arrays. Noir omits only the optimized-omitted I[0][0] bound; Halo2 encodes that actual quantized input in shared integer limbs without a signed64 bound. |
| `pythran_weights__weights` | 5,386 / 3,388 | 5,386 / 3,264 | 2,513 / 3,057 | pass/pass; pass/pass | Noir threshold comparisons use optimized canonical gaps; the full equality arrays, counts and source reciprocal calls remain. |
| `pythran_projection_simplex__projection_simplex` | 14,409 / 4,772 | 14,409 / 4,695 | 4,876 / 3,381 | pass/pass; pass/pass | Noir condition and final max use optimized canonical gaps; preserve bubble sort, two full independent compactions and separate theta division. |
| `npbench_deep_learning_mlp_mlp__mlp` | 15,312 / 4,761 | 16,472 / 4,717 | 13,302 / 4,527 | pass/pass; pass/pass | Match optimized dot+b window using a bias shift solely inside rescale, then subtract bias to restore the standalone rounded dot. Source bias addition remains separate; all source helper calls remain. |
| `pythran_d2q9_nxnyns__one_time_step` | 1,164,913 / 92,644 | 1,022,225 / 74,595 | 480,480 / 67,110 | pass/pass; pass/pass | Use exact integer-coefficient products for ±2 and ±4 moments, without extra signed64 intermediate windows; preserve all six helpers and repeated products. |
| `pythran_lapl3d__laplacien` | 293,973 / 35,492 | 248,109 / 34,844 | 140,606 / 27,654 | pass/pass; pass/pass | Use shared canonical nonzero division for max+epsilon beyond signed64, matching optimized divisor acceptance. Retain max scan and independent division for every element. |
| `pythran_spectraldns__cross2` | 119,490 / 14,181 | 75,810 / 11,241 | 62,930 / 11,241 | pass/pass; pass/pass | Still form all four real terms of multiplication by 1j, using exact 0/1 products to avoid bounds absent from optimized negation/wires. All cross1 source terms remain. |
| `pythran_spectraldns__add_pressure_diffusion` | 125,545 / 18,910 | 245,082 / 362,181 | 46,550 / 8,721 | pass/pass; pass/pass | Retain every discarded dU product and update. Dead-only inputs have no extra Q32 windows; small shared floor divisions establish rounding residues, allowing exact field reconstruction of wide discarded products. |

Across these 19 entries, summed Halo2 advice + lookup counts change from **24,191,199 to 21,381,995** and summed Noir gates from **3,145,366 to 3,129,412**. These are size totals, not a claim of domain equivalence for the four optimized L4 defects.

### Cases not larger than optimized

- `npbench_deep_learning_conv2d_bias_conv2d__conv2d_bias`: Noir 127,033 = 127,033.
- `pythran_euclidean_distance_square__euclidean_distance_square`: Noir 3,009 = 3,009.
- `pythran_fibo_seq__fibo`: Noir 17,400 = 17,400.
- `pythran_ising__ising_step`: Noir 3,564 < 3,575.
- `pythran_wdist__slow_wdist`: Noir 3,497 < 3,535.
- `npbench_weather_stencils_hdiff_hdiff__hdiff`: Noir 75,429 = 75,429.
- `pythran_harris__harris`: Noir 3,731 = 3,731.
- `pythran_spectraldns__cross2`: Noir 11,241 = 11,241.

Equal Noir counts can result from compiler elimination of source-level arrays and exact arithmetic identities. Wdist retains the general source divisions but has no separate redundant unsigned64 sum check; two nonnegative signed64 square outputs already imply it. Ising uses the same canonical comparison schedule plus a safe inactive operand, and compiler simplification can offset the added select. These are not reasons to drop literal source work.

## Reviewer findings resolved without arithmetic edits

- `pythran_ln__ln`: its separate MSB-first power calls already match optimized d2, d3, d4, d5, d6=d3*d3, d7=d6*d, d8, d9. The 2026-10-09 section 3 amendment explicitly permits the optimized power method; replacing it with the other shared power schedule would change rounding. Every source power call is still computed separately.
- `npbench_scattering_self_energies_scattering_self_energies__scattering_self_energies`: the current optimized complex products round each of the four real products separately, then form the real/imaginary pair and accumulate. Literal already uses that schedule for each source complex matmul. The amendment resolves the old single-rescale rule conflict; no fused complex-dot rewrite was made.

## Discarded source work: domain fixes and interpretation

Harris and hdiff preserve every full source array. Noir simply omits the input checks absent from optimized. Halo2 uses three base-2^63 integer input limbs for each discarded-only input and reconstructs its actual quantized i128 value. Each limb uses shared `q.load_int`; there is no local witness or hint. The extra per-limb checks do not restrict the original input because every quantized i128 has this fixed decomposition. Live inputs retain their ordinary Q32 checks.

Pressure diffusion needs wider discarded products as well. Its live P_hat computations are unchanged. Halo2 encodes dead-only inputs in four base-2^32 limbs; Noir derives signed low limbs with standard constrained field comparison and bit decomposition. For B=2^32, the shared floor-division gadget establishes r=(a_low*b_low) mod B. The field expression (a*b-r)/B is then the exact Q32 floor modulo the field, without a signed64 result bound. The low word needed by the following viscosity*U_hat product is derived from the low two input words and the first small-product quotient. This preserves the second rounding even if the first full product exceeds the field. Every original product, subtraction and assignment is still present; only the range/rounding plumbing changes.

These helpers are constrained input encoding and arithmetic composition, not local hints: the only quotient witnesses come from shared floor division. No shared gadgets were edited, no input was replaced by a harmless value for dead work, and no source call was eliminated. Source loops and arrays stay literal; compiler dead-code elimination remains the compiler’s decision. The encoding uses existing constrained APIs and needs no local witness/hint exception.

Validation includes the reviewer dead-input/product examples on both variants, a temporary external copy of the actual pressure circuit with an assertion on a discarded dU value for negative fractional wide inputs (Halo2 mock and Noir execute), and 1,000 independent big-integer trials of the two-product remainder identity including values beyond the field modulus. The assert-probe changes no repository file.

## Optimized defects intentionally not copied

- **CROTG, both:** CA=0 and CB=1,000,000+0j should select the zero-CA branch, but optimized computes overflowing inactive abs(CB).
- **Ising, both:** field=[[-1,1],[1,1]], beta=100 selects a direct flip at the first site; optimized nevertheless evaluates exp(1600).
- **Periodic distance, both:** false flags, zero coordinates, L=2^40 should skip periodic updates; optimized nevertheless requires representable L/2.
- **Clip, Noir:** z=40,000+40,000j and max=1,000,000 takes the unchanged branch. The initial Noir magnitude accepts its unsigned64 radicand, but optimized computes an unused squared magnitude in limit1 that fails signed64. Literal masks the inactive helper operands. Halo2 rejects this example at the required initial signed64 radicand guard, so it is specifically a Noir optimized defect.

## Edge executions

All fixtures live under the external work directory, not the repository. Each row records actual `halo2_hw mock --fixture ... --input A` and `hw_check.noir_driver execute --fixture ... --input A` for both variants. A/R means accept/reject. When both accept, decoded public outputs are compared exactly within each backend; small rounding fixtures preserve every Q32 bit in the decoded floats. A shared rejection is an acceptance-domain result, not an output comparison.

| Fixture | Halo2 literal / optimized | Noir literal / optimized | Equal outputs when both accept (H / N) |
|---|---|---|---|
| `crotg-inactive.json` | A / R | A / R | — / — |
| `ising-inactive.json` | A / R | A / R | — / — |
| `periodic-inactive.json` | A / R | A / R | — / — |
| `fibo-91.json` | A / A | A / A | yes / yes |
| `wdist-32768.json` | R / R | A / A | — / yes |
| `wdist-40000.json` | R / R | A / A | — / yes |
| `clip-root.json` | R / R | R / R | — / — |
| `clip-inactive.json` | R / R | A / R | — / — |
| `summation-root.json` | R / R | R / R | — / — |
| `periodic-root.json` | R / R | R / R | — / — |
| `harris-dead-input.json` | A / A | A / A | yes / yes |
| `hdiff-dead-input.json` | A / A | A / A | yes / yes |
| `hdiff-four-times.json` | A / A | A / A | yes / yes |
| `pressure-dead-input.json` | A / A | A / A | yes / yes |
| `pressure-dead-product.json` | A / A | A / A | yes / yes |
| `weights-wide-threshold.json` | R / R | A / A | — / yes |
| `mlp-bias-window.json` | A / A | A / A | yes / yes |
| `conv-rounding.json` | A / A | A / A | yes / yes |
| `euclidean-rounding.json` | A / A | A / A | yes / yes |
| `cronbach-rounding.json` | A / A | A / A | yes / yes |
| `simplex-wide-gap.json` | R / R | A / A | — / yes |
| `lapl-divisor.json` | A / A | A / A | yes / yes |
| `cross-wide-difference.json` | A / A | A / A | yes / yes |
| `moments-wide-coefficients.json` | A / A | A / A | yes / yes |
| `pressure-beyond-field.json` | A / A | A / A | yes / yes |

Rounding fixtures: convolution uses input=weights=0.1 and zero bias; Euclidean distance uses x1=0.1 and x2=0.3; Cronbach uses Random(43) integer scores in [0,1000) times 1e-6. `pressure-beyond-field` uses nu=K2=U_hat=2^80, making the discarded triple product exceed the field modulus. The remaining fixtures contain the reviewer inputs and targeted added domain cases. `fibo-91` returns 12200160415121876738. `lapl-divisor` constructs max Q32 code 2^63-1 so max+4 crosses signed64. `moments-wide-coefficients` puts 600,000,000 in population 0 and zero elsewhere, so ±4 moment products leave signed64 while later selected rounded results fit.

## Audit inventory

For every row, both languages were compared to their optimized counterpart for input/output plumbing, constant quantization, reduction rounding, power/division schedules, intermediate windows, and input-dependent branches. “No further mismatch found” means static review found no additional baseline-level discrepancy; unchanged cases were not all re-executed. Branch review included zero-divisor guards in CROTG/clip, active-loop handling in fibo/insertion_sort, padded compaction arithmetic in tax, and axis guards in periodic_dist.

| Entry | Audit result |
|---|---|
| `npbench_compute_compute__compute` | No further mismatch found. |
| `npbench_crc16_crc16__crc16` | No further mismatch found. |
| `npbench_deep_learning_conv2d_bias_conv2d__conv2d_bias` | Changed; see counts and edge results above. |
| `npbench_deep_learning_mlp_mlp__mlp` | Changed; see counts and edge results above. |
| `npbench_deep_learning_softmax_softmax__softmax` | No further mismatch found. |
| `npbench_go_fast_go_fast__go_fast` | No further mismatch found. |
| `npbench_polybench_atax_atax__kernel` | No further mismatch found. |
| `npbench_polybench_bicg_bicg__kernel` | No further mismatch found. |
| `npbench_polybench_cholesky_cholesky__kernel` | No further mismatch found. |
| `npbench_polybench_covariance2_covariance2__kernel` | No further mismatch found. |
| `npbench_polybench_deriche_deriche__kernel` | No further mismatch found. |
| `npbench_polybench_durbin_durbin__kernel` | No further mismatch found. |
| `npbench_polybench_gemm_gemm__kernel` | No further mismatch found. |
| `npbench_polybench_gemver_gemver__kernel` | No further mismatch found. |
| `npbench_polybench_gesummv_gesummv__kernel` | No further mismatch found. |
| `npbench_polybench_gramschmidt_gramschmidt__kernel` | No further mismatch found. |
| `npbench_polybench_k2mm_k2mm__kernel` | No further mismatch found. |
| `npbench_polybench_k3mm_k3mm__kernel` | No further mismatch found. |
| `npbench_polybench_lu_lu__kernel` | No further mismatch found. |
| `npbench_polybench_ludcmp_ludcmp__kernel` | No further mismatch found. |
| `npbench_polybench_mvt_mvt__kernel` | No further mismatch found. |
| `npbench_polybench_symm_symm__kernel` | No further mismatch found. |
| `npbench_polybench_syr2k_syr2k__kernel` | No further mismatch found. |
| `npbench_polybench_syrk_syrk__kernel` | No further mismatch found. |
| `npbench_polybench_trisolv_trisolv__kernel` | No further mismatch found. |
| `npbench_polybench_trmm_trmm__kernel` | No further mismatch found. |
| `npbench_pythran_arc_distance_arc_distance__arc_distance` | No further mismatch found. |
| `npbench_scattering_self_energies_scattering_self_energies__scattering_self_energies` | Already matches optimized arithmetic; amended contract resolves reviewer exception. |
| `npbench_weather_stencils_hdiff_hdiff__hdiff` | Changed; see counts and edge results above. |
| `npbench_weather_stencils_vadv_vadv__vadv` | No further mismatch found. |
| `pythran_allpairs__sqr_dists` | No further mismatch found. |
| `pythran_allpairs__sqr_dists_loops` | No further mismatch found. |
| `pythran_arc_distance__arc_distance` | No further mismatch found. |
| `pythran_bubble_sort__bubble_sort` | No further mismatch found. |
| `pythran_clip__clip` | Changed; see counts and edge results above. |
| `pythran_create_grid__create_grid` | No further mismatch found. |
| `pythran_cronbach__cronbach` | Changed; see counts and edge results above. |
| `pythran_crotg__CROTG` | Changed; see counts and edge results above. |
| `pythran_d2q9_nxnyns__one_time_step` | Changed; see counts and edge results above. |
| `pythran_deuxd_convolution__conv` | No further mismatch found. |
| `pythran_euclidean_distance_square__euclidean_distance_square` | Changed; see counts and edge results above. |
| `pythran_fbcorr__fbcorr` | No further mismatch found. |
| `pythran_fibo_seq__fibo` | Changed; see counts and edge results above. |
| `pythran_goodexpomeasure__goodExpoMeasure` | No further mismatch found. |
| `pythran_harris__harris` | Changed; see counts and edge results above. |
| `pythran_hasting__fweb` | No further mismatch found. |
| `pythran_insertion_sort__insertion_sort` | No further mismatch found. |
| `pythran_ising__ising_step` | Changed; see counts and edge results above. |
| `pythran_l2norm__l2_norm` | No further mismatch found. |
| `pythran_lap__laplacian` | No further mismatch found. |
| `pythran_lapl3d__laplacien` | Changed; see counts and edge results above. |
| `pythran_ln__ln` | Already matches optimized arithmetic; amended contract resolves reviewer exception. |
| `pythran_log_likelihood__log_likelihood` | No further mismatch found. |
| `pythran_lu__factorMatrix0` | No further mismatch found. |
| `pythran_lu__factorMatrix1` | No further mismatch found. |
| `pythran_make_decision__md` | No further mismatch found. |
| `pythran_matmul__matrix_multiply` | No further mismatch found. |
| `pythran_multi_export__a` | No further mismatch found. |
| `pythran_multitype__check` | No further mismatch found. |
| `pythran_multitype__check2` | No further mismatch found. |
| `pythran_multitype__times` | No further mismatch found. |
| `pythran_pairwise__pairwise` | No further mismatch found. |
| `pythran_periodic_bc__periodic_bc` | No further mismatch found. |
| `pythran_periodic_dist__dist` | Changed; see counts and edge results above. |
| `pythran_perm__permutations` | No further mismatch found. |
| `pythran_projection_simplex__projection_simplex` | Changed; see counts and edge results above. |
| `pythran_pythagorean_triples__next_pythagorean_triples` | No further mismatch found. |
| `pythran_reverse_cumsum__reverse_cumsum` | No further mismatch found. |
| `pythran_rosen__rosen` | No further mismatch found. |
| `pythran_smoothing__smoothing` | No further mismatch found. |
| `pythran_specialconvolve__specialconvolve` | No further mismatch found. |
| `pythran_spectraldns__add_pressure_diffusion` | Changed; see counts and edge results above. |
| `pythran_spectraldns__cross2` | Changed; see counts and edge results above. |
| `pythran_sumarray3d__summation` | Changed; see counts and edge results above. |
| `pythran_tax__tax` | No further mismatch found. |
| `pythran_value_index__tri` | No further mismatch found. |
| `pythran_vibr_energy__calculate_vibr_energy` | No further mismatch found. |
| `pythran_wdist__slow_wdist` | Changed; see counts and edge results above. |
| `pythran_weights__weights` | Changed; see counts and edge results above. |

## Reproduction and evidence

Work directory: `/home/zhantong/Projects/zinnia-project/eval-redo/literal-work/arith-npb/`.

- Environment: `HW_VARIANT=literal`, `CARGO_TARGET_DIR=target-literal`, `RUSTC_BOOTSTRAP=1`, `CARGO_BUILD_JOBS=3`; optimized runs unset HW_VARIANT and use target. External pre-fix sources use a separate `before/target-literal` target and the existing source-root override, without changing sizes.
- `HW_CHECK_LOCK=/home/zhantong/Projects/zinnia-project/eval-redo/literal-work/lock-arith-npb`; TMPDIR is `arith-npb/tmp`; memrun caps are 8G for builds and up to 40G for checks. All heavy execution steps use the lock.
- Current literal evidence: `fibo-literal.json`, `group-literal.json`, `audit-literal.json`, `harris-literal.json`, `pressure-literal.json`, `dead-literal-final.json` (the latter supersedes earlier Harris/hdiff/pressure results). Latest audit results supersede the early clip/Ising rows in group-literal.
- Optimized evidence: `optimized.json`, `dead-optimized.json`. Fresh pre-fix stats: `before-counts.json`, `before-pressure.json` (pressure Halo2 was measured before its arithmetic edit).
- Wide-arithmetic evidence: `wide-probe-expected.json`, `wide-probe-results.json`; external assert-probe sources are under `wide-probe/`.
- Edge evidence: `edges-{literal,optimized}.json`, `edges-{literal,optimized}-extra.json`, `edges-literal-dead-final.json` (supersedes earlier Harris/hdiff/pressure edges), `edges-{literal,optimized}-wide.json`, `edges-manifest.json`; external scripts retain fixture generation and command details.
- `HW_VARIANT=literal ../../.venv/bin/python -m hw_check.lint --npb`: zero errors over all 79 entries; heuristic cast/size warnings remain. `git diff --check` passes.
- Standard checks cover A/B outputs against NumPy, Halo2 mock satisfaction, Noir witness solving and rejection of a tampered read input. All 76 A/B decoded output comparisons (19 entries × two backends × two inputs) also match optimized exactly; see `ab-exact-comparison.json`. Proof verification was deliberately not run.

Interpretations: optimized range plumbing can shift the MLP rescale window without merging the source matmul/bias expressions; integer-valued moment and complex-unit coefficients use exact field products where optimized does; existing optimized MSB-first powers and per-real-product complex rounding take precedence under section 3. Discarded work uses constrained wide input/product encoding to remove only the optimized-absent range restrictions; these local helpers merit author review.
