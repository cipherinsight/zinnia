# Literal npb4

All 19 entries have complete Halo2 and Noir literal source files. **All 19 pass both systems, including A/B constraints or witness solving, numerical output comparisons and input-domain rejection.** The two sorting cases use the unmodified upstream checker fix described below; the other 17 use this worktree's checker. No implementations remain unfinished. The five arithmetic/domain comparison exclusions below remain unresolved because optimized files are outside this task's edit scope. Every checker invocation used `--no-prove`: Halo2 MockProver A/B, circuit counts and out-of-domain rejection; Noir compilation, UltraHonk gate counting, A/B witness execution and out-of-domain rejection. No keys, proofs, proof verification, timings study, scaling, Zinnia execution, IR or generated-code inspection.

## Counts and verdicts

Halo2 cells are advice + lookup (fixed/instance cells excluded). Noir counts are unpadded UltraHonk gates, pinned Nargo 1.0.0-beta.20 / bb 5.0.0-nightly.20260324, noir-recursive settings. Counts below are from fresh `hw_check.check --no-prove` runs at original fixture shapes. L/O denote literal/optimized; H/N denote Halo2/Noir.

| Case | Literal Halo2 | Literal Noir | Optimized Halo2 | Optimized Noir | L H/N; O H/N | Review / size flags |
|---|---:|---:|---:|---:|---|---|
| npbench_deep_learning_conv2d_bias_conv2d__conv2d_bias | 2850841 + 708160 = 3559001 | 443833 | 554040 + 103360 = 657400 | 127033 | pass/pass; pass/pass | Reused; corrected i/j loop order, restored conv2d helper and separate bias addition, materialized rounded broadcast products before sum. |
| npbench_polybench_atax_atax__kernel | 8485 + 2140 = 10625 | 4397 | 8485 + 2140 = 10625 | 4397 | pass/pass; pass/pass | Reused; two complete left-associated matrix products. Computation unchanged. **Literal <= optimized: Halo2, Noir.** |
| npbench_polybench_deriche_deriche__kernel | 98435 + 24548 = 122983 | 21648 | 90423 + 23129 = 113552 | 21634 | pass/pass; pass/pass | Independent repeated exp calls; all column/row recurrences in source order; zero terms retained. |
| npbench_polybench_gesummv_gesummv__kernel | 39750 + 10768 = 50518 | 9396 | 39750 + 10768 = 50518 | 9396 | pass/pass; pass/pass | Reused; full alpha*A before @x, then full beta*B before @x, then addition. **Literal <= optimized: Halo2, Noir.** |
| npbench_polybench_lu_lu__kernel | 29081 + 6672 = 35753 | 9402 | 28965 + 6672 = 35637 | 9402 | pass/pass; pass/pass | Reused; full dot slices; empty dots are constant zero but subtraction is retained; in-place source recurrence. **Literal <= optimized: Noir.** |
| npbench_polybench_syr2k_syr2k__kernel | 165635 + 43644 = 209279 | 19170 | 102312 + 26796 = 129108 | 19170 | pass/pass; pass/pass | Reused; all slice intermediates materialized per i/k; no hoisting of A*alpha or B*alpha. **Literal <= optimized: Noir.** |
| npbench_pythran_arc_distance_arc_distance__arc_distance | 56560 + 13168 = 69728 | 11276 | 52848 + 12368 = 65216 | 11276 | pass/pass; pass/pass | Full arrays in expression order; canonical sin/cos/arctan2; no reused I64 proofs. **Literal <= optimized: Noir.** |
| pythran_arc_distance__arc_distance | 17675 + 4115 = 21790 | 5483 | 16515 + 3865 = 20380 | 5483 | pass/pass; pass/pass | As NPBench arc_distance, at the 5-element fixture. **Literal <= optimized: Noir.** |
| pythran_cronbach__cronbach | 24643 + 5688 = 30331 | 6868 | 12420 + 2628 = 15048 | 5248 | pass/pass; pass/pass | Two-pass ddof=1 variance: mean, deviations, rounded squares, ordered sum, division; source statement order. |
| pythran_euclidean_distance_square__euclidean_distance_square | 1666 + 420 = 2086 | 3091 | 1563 + 425 = 1988 | 3009 | pass/pass; pass/pass | Only np.dot fuses rescaling; both square arrays round individually; left-associated additions. |
| pythran_harris__harris | 5600 + 1452 = 7052 | 3735 | 5541 + 1444 = 6985 | 3731 | pass/pass; pass/pass | Full differences before slicing, including dropped boundaries; full arrays per assignment. |
| pythran_l2norm__l2_norm | 31136 + 7968 = 39104 | 7696 | 20848 + 5664 = 26512 | 6096 | pass/pass; pass/pass | Retains full abs array, rounded square array, row sums, then roots. |
| pythran_log_likelihood__log_likelihood | 35675 + 8216 = 43891 | 8934 | 35936 + 8360 = 44296 | 9008 | pass/pass; pass/pass | Full square/divide/exp/density-divide/log/sum arrays; sigma precondition; no divisor-proof reuse. **Literal <= optimized: Halo2, Noir.** |
| pythran_pairwise__pairwise | 141185 + 34304 = 175489 | 23552 | 129936 + 32224 = 162160 | 23552 | pass/pass; pass/pass | All ordered pairs including diagonal, scalar k recurrence, square before accumulation; constant diagonal omitted only from public instances. **Literal <= optimized: Noir.** |
| pythran_projection_simplex__projection_simplex | 12697 + 1712 = 14409 | 4772 | 4072 + 804 = 4876 | 3381 | pass/pass*; pass/pass | Stable bubble sort; two independent full equality-test compactions; nonempty checks, last-element indexing, fresh theta division. |
| pythran_smoothing__smoothing | 2579 + 650 = 3229 | 3152 | 2523 + 650 = 3173 | 3152 | pass/pass; pass/pass | Full copy, scalar recurrence; recomputes 1-alpha inside each step. **Literal <= optimized: Noir.** |
| pythran_spectraldns__add_pressure_diffusion | 98937 + 26608 = 125545 | 18910 | 36470 + 10080 = 46550 | 8721 | pass/pass; pass/pass | Full P_hat expression and subsequent dU update, including its otherwise dead input reads. |
| pythran_spectraldns__cross2 | 94290 + 25200 = 119490 | 14181 | 49490 + 13440 = 62930 | 11241 | pass/pass; pass/pass | cross1 helper executes three assignments; repeated a0*b2 retained; subsequent complex multiplication by 1j. |
| pythran_value_index__tri | 30690 + 3825 = 34515 | 5113 | 7263 + 890 = 8153 | 4201 | pass/pass*; pass/pass | Stable bubble argsort carrying indices; standard private indexing; dead constant-index score_t write retained. |

## Checker issue and resolved rerun

`pythran_projection_simplex__projection_simplex` and `pythran_value_index__tri` originally reported **"10 inputs checked twice"** in Halo2. Those were their only problems: `mock_A` and `mock_B` were satisfied and matched NumPy; `mock_out_of_domain` was unsatisfied as required; Noir passed.

LITERAL_BASELINES.md L5 requires `q.bound` on each comparison operand at the comparison, and L6 forbids input bounded-loaders/proof reuse. The first bubble-sort pass necessarily compares the ten original input cells. The worktree's `hw_check/check.py:220` unconditionally treats those required bounds as an error, even with HW_VARIANT=literal.

While this batch was running, the shared `literal` branch advanced to **3155495c**, which fixes this exact diagnostic by exempting literal baselines. No worktree tool was edited or cherry-picked. The exact upstream check.py was read with `git show`, saved as the external artifact `upstream-check-3155495c.py`, and run with `__package__="hw_check"` against this worktree's unchanged helper modules and final binary. Both complete `--no-prove` reruns **pass Halo2 and Noir**, with identical counts. The starred verdicts above come from `upstream-sort.json`; original diagnostic failures remain in `last.json`. The upstream checker SHA-256 is `57c8d632bce5f41bdeabbd736d45cb9b2016db4d68001a46b979b10740c2c0d3`.

## Literal not larger than optimized

Every flagged table entry is explained here:

- **Atax and gesummv, both systems: equal.** Their optimized arithmetic is already the direct matrix-product form; array materialization/order changes introduce no additional primitive operations.
- **LU, Noir: equal.** The additional explicit subtractions of empty-dot zero are folded by Noir. Halo2 retains those gates.
- **Syr2k and smoothing, Noir: equal.** Source-level repeated A/B*alpha and 1-alpha remain in the literal files; Noir's allowed common-subexpression elimination removes their repetition. Halo2 grows as expected.
- **Both arc_distance entries, Noir: equal.** Full intermediate arrays lower to the same arithmetic; canonical_arctan2_nonneg is itself an alias for canonical_arctan2 in the shared Noir library. Halo2 additionally pays for fresh bounds and the general quadrant handling.
- **Pairwise, Noir: equal.** The source diagonal calculations are present, but the compiler folds them to constants. Halo2 computes those operations.
- **Log_likelihood, both systems: smaller** (Halo2 43,891 versus 44,296; Noir 8,934 versus 9,008). Its direct `/` operations use the permitted shared canonical_div_nonzero API. The optimized source mixes div_by_abs for the squared-sigma divisor and div for the normalization divisor. The canonical gadget's different sign/magnitude implementation is cheaper overall here, even after retaining the explicit sigma precondition. No log/exp identity, hoisting across source statements, CSE or reassociation was introduced; the older optimized gadget combination is not minimal. The normalization divisor is already bounded by its preceding multiplication, and the doubled squared-sigma divisor is nonnegative below 2^64, so this choice does not add a narrower divisor domain.

## Controlled Halo2 comparison

Lookup widths match within every pair except euclidean_distance_square. Its automatic checker settings were literal k=9 / 8 bits and optimized k=7 / 6 bits. A separate locked, memory-capped **stats-only** optimized recount at k=9 / 8 bits gives **1,076 advice + 276 lookup = 1,352 cells**, versus literal **1,666 + 420 = 2,086** at the same settings. Artifact: `euclidean-optimized-matched.json`. Use these matched counts for that pair, not the automatic optimized 1,988-cell count in the table. All other table comparisons use matched lookup widths (7 for smoothing, 8 otherwise). This does not resolve the arithmetic exclusions below; no fair aggregate is claimed.

## Interpretations and limits

- L2/L5 take precedence over historical fusion permissions: elementwise multiply or square followed by `sum` materializes and rounds each product. Only actual `@`/`dot` calls use a single rescale. The unchanged optimized conv2d, Cronbach and euclidean_distance_square files still fuse these separate operations. A/B numerical agreement does **not** establish their Q32 equivalence; exclude those pairs from any claim of identical arithmetic.
- Library variance uses the ordinary two-pass definition (mean, deviations, squares, sum, ddof divisor). The row-axis call completes all its intermediates. Reductions scan increasing indices. Empty LU dots are known zero; their surrounding source subtraction remains explicit.
- L1/L2 require source computations even when a result is discarded. Harris computes both complete difference arrays before slicing, so I[0,0] is loaded and checked. add_pressure_diffusion computes the dead dU write and loads/checks U_hat, K2, K and nu. Their optimized counterparts remove this work and its input checks. Thus these pairs are not certified to have identical acceptance domains; L2 and the domain rule for every source-read input were prioritized over copying optimized dead-input omissions. Noir may eliminate dead computation itself (allowed compiler optimization).
- Pairwise computes the diagonal recurrence and sqrt but preserves the optimized constant-output layout: diagonal zeros have no public cells. Constant-index assignments and slices are wire movement, not extra arithmetic gates. score_t in tri and c in cross2 are overwritten before their private values are read, so those private inputs receive no domain checks.
- Shape/literal arithmetic and log_likelihood's input-independent sqrt(2*pi) are folded to the same Q32 constants as the optimized plumbing. Deriche's private `2.0 ** (-alpha)` uses the shared canonical exp schedule with Q32 log(2)=2977044472; there is no separate dynamic-power API. Its float `2.0 * alpha` remains a rounded multiplication. Integer factors such as the arc-distance 2 and Deriche c1/c2 use exact integer-times-Q32 multiplication.
- All private transcendental calls use shared canonical APIs. Shared `abs_int64` implements the source abs in l2norm. Existing backend-specific sqrt acceptance guards are preserved for l2norm/pairwise (Halo2 nonnegative signed64 radicand, Noir unsigned64 radicand); these are inherited arithmetic-domain constraints, not fixture bounds. Canonical nonzero division is used where needed for the shared divisor domain; no own hints or reused divisor proofs.
- Stable adjacent bubble sort uses exactly N(N-1)/2 compare-and-swaps, with fresh comparison operand bounds every time. Argsort carries indices and gathers via q.index / Noir index_pos. Each projection mask is independently expanded into a prefix-count scan and N-by-N equality-test compaction; an explicit nonempty assertion makes [-1] on an empty selection unsatisfiable.
- Complex-times-real means two component Q32 products. cross2's later multiplication by 1j uses the defining complex product, without merging its negation into cross1's subtraction or sharing a0*b2. This retains source operation boundaries; the compiler may fold constant-zero/one operations.

## Tools and validation scope

Builds use RUSTC_BOOTSTRAP=1, dev-fast, CARGO_BUILD_JOBS=3, target-literal for literal and target for optimized, and memrun MemoryMax=8G. Stable Rust without RUSTC_BOOTSTRAP initially failed with dependency E0554; no dependency/tool changes were made. Heavy checker steps use HW_CHECK_LOCK=.../literal-work/lock-b. TMPDIR and all logs/packages are under the batch work directory. An initial duplicate manifest-path build directory was removed.

Final `HW_VARIANT=literal hw_check.lint` over all 19 entries: **0 errors**. Fixed the inherited syr2k formula comment. Remaining warnings were reviewed: Deriche's 3 is the source reverse-loop offset; spectral 3 is the source's three vector components and 4 is two components times complex interleaving; Cronbach's cast computes a public shape-derived Q32 constant. No private unchecked casts or own hints were introduced. Formatter-only changes after validation do not change operations; the final Halo2 binary was rebuilt with all 19 entries after formatting. No extra benchmark-specific tests were added.

Artifacts: `/home/zhantong/Projects/zinnia-project/eval-redo/literal-work/npb4/` contains `atax.json`, `reused.json`, `new1.json`, `cronbach-fix.json`, `last.json`, `upstream-sort.json`, `optimized.json`, `lint.log`, build/check logs and staged packages. Cronbach's first Noir run failed on generic-expression syntax; the corrected file passed (`cronbach-fix.json` supersedes that Noir result). NumPy warnings during domain-input probing are recorded in logs and are not fixture failures.

## Backend settings

| Case | Literal k / lookup bits | Optimized k / lookup bits |
|---|---:|---:|
| npbench_deep_learning_conv2d_bias_conv2d__conv2d_bias | 13 / 8 | 13 / 8 |
| npbench_polybench_atax_atax__kernel | 9 / 8 | 9 / 8 |
| npbench_polybench_deriche_deriche__kernel | 9 / 8 | 9 / 8 |
| npbench_polybench_gesummv_gesummv__kernel | 9 / 8 | 9 / 8 |
| npbench_polybench_lu_lu__kernel | 9 / 8 | 9 / 8 |
| npbench_polybench_syr2k_syr2k__kernel | 10 / 8 | 10 / 8 |
| npbench_pythran_arc_distance_arc_distance__arc_distance | 9 / 8 | 9 / 8 |
| pythran_arc_distance__arc_distance | 9 / 8 | 9 / 8 |
| pythran_cronbach__cronbach | 9 / 8 | 9 / 8 |
| pythran_euclidean_distance_square__euclidean_distance_square | 9 / 8 | 7 / 6 |
| pythran_harris__harris | 9 / 8 | 9 / 8 |
| pythran_l2norm__l2_norm | 9 / 8 | 9 / 8 |
| pythran_log_likelihood__log_likelihood | 9 / 8 | 9 / 8 |
| pythran_pairwise__pairwise | 10 / 8 | 10 / 8 |
| pythran_projection_simplex__projection_simplex | 9 / 8 | 9 / 8 |
| pythran_smoothing__smoothing | 8 / 7 | 8 / 7 |
| pythran_spectraldns__add_pressure_diffusion | 10 / 8 | 9 / 8 |
| pythran_spectraldns__cross2 | 10 / 8 | 9 / 8 |
| pythran_value_index__tri | 9 / 8 | 9 / 8 |

## Source hashes

SHA-256 of final literal files, unchanged optimized files, kernels and fixtures are recorded in the batch `source-hashes.json` manifest.
Manifest SHA-256: `d1fdd71685785a572f0f9d9dc69ae830376f3b9271a6d0cec3dfd79f3201d141`.

| Case | Literal Rust SHA-256 | Literal Noir SHA-256 |
|---|---|---|
| npbench_deep_learning_conv2d_bias_conv2d__conv2d_bias | `bbf0948f650ca1bfeb9c561086118fcd7598b5a2f5aa313685ba66d0415de515` | `f79456cd6a6754c891e4f2f2734fc91a010c54c1a6b0e9b02e711ed3481b53b2` |
| npbench_polybench_atax_atax__kernel | `38aacefb5ed79d2bca9342c35641e43a16a0c98f314c2bae138a7d992fe9b0ce` | `4a41edefec1db6b69f6e89543d3b2a220d04043d6897ee36b525e17218f7ad42` |
| npbench_polybench_deriche_deriche__kernel | `50775226157b2290b636a01491535b14dc77f250f1899da1f24ec82dc13adbd5` | `d89321f1edfd812213fb467ba1acc38bc8a2c90f89783f2a5875ed2778b04f16` |
| npbench_polybench_gesummv_gesummv__kernel | `da4204a9aa6a6381fe8fe446fcc6ad00abba4f9710471b2afede132850fd456f` | `b0764efee787f154aa17318b8888c7f1da5d6784d11e58dbc4935267ab2cab9c` |
| npbench_polybench_lu_lu__kernel | `f8008f2e4b722e348223f5983153bcae861e1245d7958b9688cd5872dfe88865` | `78381e47f5b3a652ff06ac705525e079e1853c113937889e3020bc99d9069502` |
| npbench_polybench_syr2k_syr2k__kernel | `ed007cdfcdab124db60523ce2d37db22c77e646cb448d827f863b8baf0310f66` | `e2a781ed51548f92db8100b8667990b4c8e745f4f8a592576337f7fa37d22f2f` |
| npbench_pythran_arc_distance_arc_distance__arc_distance | `1396dcf48b65822276fcb47ad1c38d2259cbfc47e419d7c8265172ffebbbd66a` | `f095fdfd2aecdd4805f3ac6edd9302d172b7b42b33aa05e13f6169efd828ea37` |
| pythran_arc_distance__arc_distance | `4052ba6da7ecca96d355159d170a5ad48a2a657235a2b0003b07370149c5db5e` | `07edd8c72059d159444e8b66c99b41f115599f237b2fa0f3139ee0c097612dc2` |
| pythran_cronbach__cronbach | `b300c39495813f51af024fce5050b102b61e3febecf2e546f83ad79acd5a739d` | `3d0c40165992f3b80eee8030eef33b095e39f5fd04a6248217de3b8fafd17a16` |
| pythran_euclidean_distance_square__euclidean_distance_square | `9ec3669bdbc4c4bf0b24d61b72f1d17698fbd4a364367b01fd83008e699914d3` | `9dc4533511c5bdd4bade49395aef79b1c4bbb0264cf9c4699bc76ef406a6e975` |
| pythran_harris__harris | `e1770724c3bce9564a53dec909a200cac07c9460802c194439b016b5a0eadbbf` | `dda8b2b928723080c041d73b0abd6863d266c39d99049b572b52a28043f8d57f` |
| pythran_l2norm__l2_norm | `eb96a4374213bcd04c9aa257b2154660b637044dc6af387c699c76c451b4a836` | `76f091b9455d7a38d4ca2dba923bcbdb9ebd3bcc697ff98c7e0ac5672ef3ba55` |
| pythran_log_likelihood__log_likelihood | `2c7155ad3944b11da7dd67fa78f002576296d1dbede0281c004cd4b2edd20e6f` | `9a4f774acbb00d5e3816652038ff77c5b0fae88d6b992ca4f16cff1b4b8cd19f` |
| pythran_pairwise__pairwise | `b8cf70f174318013257f43bffa1d32fc2d99a7b50ec45a880be4df34c1c3993b` | `22def00612daa9ca7905c9b13c4c4f8618b86d7ed054fa65384798dc69ff0167` |
| pythran_projection_simplex__projection_simplex | `69d63a0da313a917f1b4baea2a9fd552b0d479076538dba42826e94a6c70e9db` | `a73a9280923b11c9d022afc3cdd79c15c0bd872c4f915d1514647d5fc41a7ceb` |
| pythran_smoothing__smoothing | `7f03a1fb363d0d05886abbbf50abaab68f9e81f58611d49e966de5f38ae0be58` | `ed13549961ee6ab70a0ff241913f66014470882bed03afdf543ebfa26c0029a2` |
| pythran_spectraldns__add_pressure_diffusion | `5bcc68b61ec47a8e3eb9654a0cb71e075d6039ba80fd806a1a50ec83b91b3ff0` | `fff37413b5199de0d1986a4cbff5557ac3be7436aa13b897d072953369ab20bf` |
| pythran_spectraldns__cross2 | `53c3253125ac9c0785149f06e9748e1021a13d129ccafac1f880d1b446072647` | `fdb9027d6a7a5d2451411ea3fe64fa4f9ccdc84e0e2dfe3526b98c2fd71e3054` |
| pythran_value_index__tri | `6fbad492f4f39a450881a50f058de321fbc522dee8d73ca92e7a5e6f50d5d0af` | `3acf81d6b5d74fd140a267e6d8f09ecee944f95687fd1c6347a01ad2722b79b5` |
