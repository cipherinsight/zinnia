# Literal NPBench/Pythran batch npb1

All 20 cases have Halo2 and Noir literal files. Eighteen cases pass both checkers; all 20 pass Noir. Bubble sort has a Halo2 diagnostic conflict, and scattering exceeds the Halo2 mock-run memory cap. These two are not cleanly validated in both systems; both files are retained with the reasons below.

Checks were run with `--no-prove` at the original fixture shapes, an 8 GiB memory cap, three Cargo jobs, `--profile dev-fast`, and `HW_CHECK_LOCK=/home/zhantong/Projects/zinnia-project/eval-redo/literal-work/lock-a`. Literal builds use `HW_VARIANT=literal` and `halo2_hw/target-literal`; optimized builds unset the variant and use `halo2_hw/target`. No Zinnia execution, IR/emitted-code inspection, keys, proofs, proof verification, timing study, or scaling runs.

Pass means Halo2 A/B MockProver constraints and NumPy outputs, or Noir compile/count plus A/B witness solving and NumPy outputs, together with rejection of the checker's representative out-of-domain input. It does not mean a generated proof or exhaustive input-domain testing. Tampered-public proof checks are intentionally skipped.

Evidence: `/home/zhantong/Projects/zinnia-project/eval-redo/literal-work/npb1/` (`bicg.json`, `poly.json`, `small.json`, `stencil.json`, `last.json`, `recheck.json`, `optimized.json`, logs, and source hash manifest). Later results replace earlier iteration failures.

## Counts and verdicts

Halo2 columns are advice + lookup = total; fixed and instance cells are excluded. Noir counts are unpadded UltraHonk gates with Nargo/noirc `1.0.0-beta.20` (commit `b4236c1957d0c26cb65d82adc9e5447b6ff1d629`), bb `5.0.0-nightly.20260324`, and `noir-recursive` settings. Verdicts are Halo2/Noir.

| Case | Literal Halo2 | Literal Noir | Optimized Halo2 | Optimized Noir | Literal verdict | Optimized verdict |
|---|---:|---:|---:|---:|---|---|
| npbench_deep_learning_mlp_mlp__mlp | 12410 + 2902 = 15312 | 4761 | 10758 + 2544 = 13302 | 4527 | pass/pass | pass/pass |
| npbench_polybench_bicg_bicg__kernel | 8836 + 2244 = 11080 | 4479 | 8836 + 2244 = 11080 | 4479 | pass/pass | pass/pass |
| npbench_polybench_durbin_durbin__kernel | 11167 + 2723 = 13890 | 5654 | 10753 + 2599 = 13352 | 5651 | pass/pass | pass/pass |
| npbench_polybench_gramschmidt_gramschmidt__kernel | 84391 + 20556 = 104947 | 17802 | 71286 + 17700 = 88986 | 17852 | pass/pass | pass/pass |
| npbench_polybench_ludcmp_ludcmp__kernel | 34231 + 7860 = 42091 | 9962 | 32835 + 7488 = 40323 | 9962 | pass/pass | pass/pass |
| npbench_polybench_syrk_syrk__kernel | 70306 + 18244 = 88550 | 14032 | 69361 + 18244 = 87605 | 14032 | pass/pass | pass/pass |
| npbench_scattering_self_energies_scattering_self_energies__scattering_self_energies | 20099988 + 4084704 = 24184692 | 1712099 | 8318940 + 2093952 = 10412892 | 1712048 | FAIL/pass | pass/pass |
| pythran_bubble_sort__bubble_sort | 6237 + 1463 = 7700 | 3958 | 4697 + 715 = 5412 | 3758 | FAIL/pass | pass/pass |
| pythran_crotg__CROTG | 3855 + 924 = 4779 | 3336 | 3059 + 753 = 3812 | 3296 | pass/pass | pass/pass |
| pythran_fbcorr__fbcorr | 1425654 + 358416 = 1784070 | 221600 | 1397622 + 358416 = 1756038 | 221600 | pass/pass | pass/pass |
| pythran_hasting__fweb | 1824 + 450 = 2274 | 2967 | 1279 + 351 = 1630 | 2967 | pass/pass | pass/pass |
| pythran_lap__laplacian | 5916 + 1480 = 7396 | 3506 | 4239 + 1105 = 5344 | 3505 | pass/pass | pass/pass |
| pythran_make_decision__md | 53953 + 12016 = 65969 | 9428 | 43280 + 10224 = 53504 | 9488 | pass/pass | pass/pass |
| pythran_multitype__check | 116 + 28 = 144 | 2831 | 104 + 28 = 132 | 2831 | pass/pass | pass/pass |
| pythran_multitype__check2 | 63 + 14 = 77 | 2825 | 59 + 14 = 73 | 2825 | pass/pass | pass/pass |
| pythran_multitype__times | 138 + 40 = 178 | 388 | 48 + 12 = 60 | 388 | pass/pass | pass/pass |
| pythran_periodic_bc__periodic_bc | 721 + 216 = 937 | 2947 | 720 + 216 = 936 | 2947 | pass/pass | pass/pass |
| pythran_pythagorean_triples__next_pythagorean_triples | 355 + 42 = 397 | 2847 | 222 + 42 = 264 | 2847 | pass/pass | pass/pass |
| pythran_specialconvolve__specialconvolve | 11840 + 2048 = 13888 | 4716 | 10440 + 2048 = 12488 | 4716 | pass/pass | pass/pass |
| pythran_vibr_energy__calculate_vibr_energy | 18656 + 4400 = 23056 | 5751 | 18656 + 4400 = 23056 | 5751 | pass/pass | pass/pass |

Automatic sizing is reported exactly as the checker prints it. The following k/lookup-width pairs expose any setting differences; counts at different lookup widths are not controlled fixed-width comparisons. No aggregate speedup claim is made.

| Case | Literal k/bits | Optimized k/bits |
|---|---:|---:|
| npbench_deep_learning_mlp_mlp__mlp | 9/8 | 9/8 |
| npbench_polybench_bicg_bicg__kernel | 9/8 | 9/8 |
| npbench_polybench_durbin_durbin__kernel | 9/8 | 9/8 |
| npbench_polybench_gramschmidt_gramschmidt__kernel | 9/8 | 9/8 |
| npbench_polybench_ludcmp_ludcmp__kernel | 9/8 | 9/8 |
| npbench_polybench_syrk_syrk__kernel | 9/8 | 9/8 |
| npbench_scattering_self_energies_scattering_self_energies__scattering_self_energies | 13/8 | 13/8 |
| pythran_bubble_sort__bubble_sort | 9/8 | 8/7 |
| pythran_crotg__CROTG | 9/8 | 9/8 |
| pythran_fbcorr__fbcorr | 12/8 | 12/8 |
| pythran_hasting__fweb | 8/7 | 7/6 |
| pythran_lap__laplacian | 9/8 | 9/8 |
| pythran_make_decision__md | 10/8 | 9/8 |
| pythran_multitype__check | 6/5 | 6/5 |
| pythran_multitype__check2 | 6/5 | 6/5 |
| pythran_multitype__times | 6/5 | 6/5 |
| pythran_periodic_bc__periodic_bc | 7/6 | 7/6 |
| pythran_pythagorean_triples__next_pythagorean_triples | 6/5 | 6/5 |
| pythran_specialconvolve__specialconvolve | 9/8 | 9/8 |
| pythran_vibr_energy__calculate_vibr_energy | 9/8 | 9/8 |

## Literal not larger than optimized

- **npbench_polybench_bicg_bicg__kernel**: Halo2 equal (11080 vs 11080); Noir equal (4479 vs 4479). The optimized form already is the two direct @ calls.
- **npbench_polybench_gramschmidt_gramschmidt__kernel**: Noir smaller (17802 vs 17852). Both retain the same dot/root/update schedule. Literal uses standard division instead of the optimized positive-divmod specialization; the compiled result is 50 gates smaller, so the optimized Noir gadget usage is not minimal here. No arithmetic shortcut was introduced.
- **npbench_polybench_ludcmp_ludcmp__kernel**: Noir equal (9962 vs 9962). Noir folds empty reductions and subtraction of zero, leaving the same LU/solve schedule.
- **npbench_polybench_syrk_syrk__kernel**: Noir equal (14032 vs 14032). Noir already implements the direct recurrence; explicit temporary arrays do not require extra gates.
- **pythran_fbcorr__fbcorr**: Noir equal (221600 vs 221600). The optimized version already rounds all products. Reordered independent output work and algebraically exact accumulation can compile to similar gates.
- **pythran_hasting__fweb**: Noir equal (2967 vs 2967). Noir performs its own CSE of repeated expressions. The authored file recomputes each occurrence.
- **pythran_make_decision__md**: Noir smaller (9428 vs 9488). The optimized argmin carries selected complex values; the direct form carries an index then gathers, a different cost tradeoff.
- **pythran_multitype__check**: Noir equal (2831 vs 2831). The compiler simplifies the explicit doubled-branch select to the same algebra as the optimized form.
- **pythran_multitype__check2**: Noir equal (2825 vs 2825). The compiler simplifies the explicit doubled-branch select to the same algebra as the optimized form.
- **pythran_multitype__times**: Noir equal (388 vs 388). Noir is already the direct bounded integer product. Halo2 uses standard input loaders as L6 requires.
- **pythran_periodic_bc__periodic_bc**: Noir equal (2947 vs 2947). This case only copies wires, so the optimized and direct work are essentially identical.
- **pythran_pythagorean_triples__next_pythagorean_triples**: Noir equal (2847 vs 2847). Noir compiles both forms of the same constant integer matrix product identically.
- **pythran_specialconvolve__specialconvolve**: Noir equal (4716 vs 4716). The optimized Noir already has these two stencil expressions; full array materialization is eliminated by SSA.
- **pythran_vibr_energy__calculate_vibr_energy**: Halo2 equal (23056 vs 23056); Noir equal (5751 vs 5751). The optimized arithmetic already matches the source; full temporary arrays only change scheduling.

## Per-case algorithm and arithmetic review

All files were reviewed against kernel.py and L1-L7: source loop/recurrence order, full intermediates, no authored CSE or arithmetic rewriting, select-based private branches, direct reduction/index algorithms, no forbidden baseline hints or proof reuse, and unchanged output plumbing. The shared gadget implementations themselves remain untouched.

| Case | Review notes |
|---|---|
| npbench_deep_learning_mlp_mlp__mlp | Full matmul then bias then source relu/softmax helpers; row max and sum scan; canonical exp. No bias/rescale fusion. |
| npbench_polybench_bicg_bicg__kernel | Reused earlier committed files; reviewed L1-L7 unchanged computationally (formatting only). Two full dots, tuple order. |
| npbench_polybench_durbin_durbin__kernel | Restored first beta multiplication and separate -r[0] expressions. Dot rescale once; reflected slice products completed before y update. |
| npbench_polybench_gramschmidt_gramschmidt__kernel | Source k/j recurrences, full Q and R, dot then sqrt/division; product array before A update. Constant lower R triangle omitted from publics. |
| npbench_polybench_ludcmp_ludcmp__kernel | Source LU/forward/backward order, one rescale per @ (including empty reductions), no merged subtract/divide statements. |
| npbench_polybench_syrk_syrk__kernel | Source i/k order; scalar alpha*A[i,k] once at source location, full broadcast products, then in-place slice addition. |
| npbench_scattering_self_energies_scattering_self_energies__scattering_self_energies | All eight loops and inner shape condition; gather, dHG, dHD, and update recomputed in place. No factor or index-proof hoisting. |
| pythran_bubble_sort__bubble_sort | Source bubble loops and fresh per-comparison bounds. Unconditional swap_test=True makes break shape-dependent/unreachable. |
| pythran_crotg__CROTG | Every abs(CA) occurrence recomputed; magnitude then square; conjugate before complex product; both branches computed and selected. |
| pythran_fbcorr__fbcorr | All seven loops in source nesting; each scalar product rounded then immediately added to the supplied output. |
| pythran_hasting__fweb | All three assignments in source order; repeated predation expressions computed again, no CSE. t is unread. |
| pythran_lap__laplacian | Nine slice statements in order, each full product before update; preserves var[1] quirk at lap[0]. Independent division per output. |
| pythran_make_decision__md | Full difference, magnitude, and square arrays; first-occurrence argmin keeps value/index, then standard index gather. |
| pythran_multitype__check | Full x+x array followed by selection on y!=0; tuple output unchanged. |
| pythran_multitype__check2 | Full x+x array followed by selection on the sole Boolean element; tuple output unchanged. |
| pythran_multitype__times | Exact integer product and source bounds. Standard Halo2 loaders replace forbidden local input load_witness. |
| pythran_periodic_bc__periodic_bc | Full source slice copied then assigned; overwritten unread plane is omitted from input-domain loading. |
| pythran_pythagorean_triples__next_pythagorean_triples | Constant integer matrix dot uses running sums in contraction order, then transpose/reshape wire movement. |
| pythran_specialconvolve__specialconvolve | Full intermediate arrays for each source elementwise add/multiply/subtract; exact integer arithmetic, original operand order. |
| pythran_vibr_energy__calculate_vibr_energy | Full negated, linear, squared, quadratic, difference, and exp arrays; negative harmonic is multiplied before rounding. |

## Interpretations and limitations

1. The current fairness arithmetic takes precedence over historical NPB_BASELINES section 5 shortcuts. CROTG and make_decision compute complex magnitude as rounded real-component squares, addition, then square root; the subsequent source square is retained. Scattering rounds all four real products of each complex multiplication before accumulating matrix terms, matching its current optimized arithmetic. This interprets L5's single-rescale dot row as the real dot contract, not permission to fuse the existing complex primitive schedule. These choices need author review.
2. MLP and vibr_energy use the same `canonical_exp` shared primitive as their optimized counterparts. LU and Hastings use `canonical_div_nonzero` to retain the existing derived-divisor domain. Laplacian also uses a fresh canonical nonzero division per output in Halo2; its divisor is already input-domain checked, and no magnitude/sign proof is retained or reused. No canonical compiler implementation outside the approved gadget sources was inspected.
3. CROTG computes both branches and selects their assigned outputs. A division by zero is totalized only for an inactive branch, with a guarded nonzero assertion. Other original inactive products remain as in the current optimized arithmetic contract; this work does not claim unrestricted binary64 behavior beyond shared Q32 intermediate domains.
4. Bubble sort's swap_test assignment is unconditional, including when no swap occurs. Each inner loop has at least one iteration, so its break is compile-time unreachable. No data-dependent early-exit optimization is introduced.
5. Halo2 `times` and scattering use standard checked input loaders instead of the optimized files' local input witnesses, which L6 forbids. Source preconditions/index bounds still apply. Input loading is statement plumbing, not a computational optimization.
6. Empty @ reductions in LU are still called and subtracted. Compiler constant folding of empty reductions, constants, wire copies, arrays, and repeated Noir expressions is allowed; manually removing them is not.
7. Lint reports zero errors. Its size warnings in Hastings refer to the source's explicit length-three output and in triples to the source's constant 9-by-3 matrices and coefficient 3; these are source constants, not shape-derived literals.

## Unfinished checks and tool problems

Scattering is not finishable as a Halo2 mock validation within the prescribed 8 GiB cap: A, B, and tampered-input mocks each exited -9. Stats completed at k=13, lookup_bits=8: 20,099,988 advice + 4,084,704 lookup = 24,184,692 cells. Noir completed all checks at 1,712,099 gates. The checker labels a killed tampered-input run as accepted because it did not return satisfied=false; that is a tool reporting defect, not evidence of acceptance. Both source files are retained per the brief.

Bubble sort cannot obtain a clean Halo2 checker verdict without changing either the checker or a literal rule: L5 mandates fresh `q.bound` calls at comparisons, but `hw_check.check` rejects the resulting `redundant_input_checks=11`. Its A/B constraints and outputs pass, and the invalid input is rejected; Noir passes fully. The files retain the required fresh bounds. No checker or gadget was modified.

- npbench_scattering_self_energies_scattering_self_energies__scattering_self_energies, literal halo2: `mock A: exit -9: ; mock B: exit -9: ; out-of-domain input neigh_idx:0 accepted`
- pythran_bubble_sort__bubble_sort, literal halo2: `11 inputs checked twice`

An initial Cargo command from the repository root missed the baseline-local build configuration and failed on a dependency feature gate; rerunning inside `halo2_hw` fixed it. An initial generated Hastings Noir file had a comment/signature newline error; it was corrected and rechecked. Python emits harmless relative-venv prefix warnings. These are recorded separately from final circuit verdicts.

## Source provenance

The SHA-256 manifest below binds the reviewed source and measurements. Each row lists kernel.py, literal Rust, literal Noir, optimized Rust, optimized Noir. Fixture and shared-gadget hashes are additionally in the external `source-hashes.json` manifest.

| Case | SHA-256 (kernel; literal Rust; literal Noir; optimized Rust; optimized Noir) |
|---|---|
| npbench_deep_learning_mlp_mlp__mlp | `e192f77c1df8641ec31a4d5973962c2cb1965742b033d041961f2f6c8de9e73a`; `c76a3ef29d2b71eac08f43b5900e36691015e834cf97bfa11ff32718889e989d`; `334af25986642ed0be18749d245987750dce9ebe2de46997252f3535038f40a2`; `c9bf4beb4c617d61c717f6ed5aad2c4a1e28f3aaaac20d18ab88ee7185466902`; `8414177df246136ddfb4be3108f855d01d1d71c69b33dd95596e04d1cfb0de4f` |
| npbench_polybench_bicg_bicg__kernel | `9c9b73ef2c7f0b889b1feadfd6f6adf175d81ef26fba3dc58cefa255d1af0750`; `ed7e653198cdf9c49a4b9336a91c1353ceee1740a7d1b6d466c9ab5138ec7b01`; `1f2f8dcedabe1f07115b63ad2f16ea0f7790623d91e34bb958c96424a5427eef`; `d3ddce5c7e475fae5b74a8361361165878424fb9fccba9344fd0bb3115242a35`; `3e77db61dbe90d2f4e22fd43d9a057a910f6391deb8468536b37ca00f2676514` |
| npbench_polybench_durbin_durbin__kernel | `001d7b597471dc96a45f803cd4768cdc38d9185fd1b58e823d5fd8f6bfc8319f`; `22cb7c5abbf3c27d593fb3fbd72d4650c25d9a7ae6f2dc38b6940ac124ef4104`; `e1f28219afa83416d468e9a6e7f46be656fd273b6e8beb031dbc973b06a48f12`; `91b12468f38bd504969f66c7067c4a1e5d08fd189f9d68f2524fc4dac68fe0a5`; `f585da33b2b177a9d8ea66fbab96413558b55b0d98c7459a8f1876fce0611a6f` |
| npbench_polybench_gramschmidt_gramschmidt__kernel | `c44a3b8bf2d6145f766117747f6c4947b4fce56595f402871a0118e44058836d`; `32679ce16ca454109199a7721a325129571178ec8a541d01517179cf15d0d905`; `5cf5f931f1b19f12155ab6b6df07be1e1ff9c563841402b342aa1634d9071f94`; `dd71948175336d17a412ad59a03e31a424c6d1903fbe52e740ee5f0601f8304b`; `a601ca6caf3747d33000816f3f882838177472e4780323b873bd406125341236` |
| npbench_polybench_ludcmp_ludcmp__kernel | `c30c2fbd6dc055c805f95f9243a141d575a24bea04d064b7d4c4a9d9787bb150`; `70e69d25cfccc1e53d4a04f36814506cf5489f039a1f8de01f9264f9433a3768`; `634a5a629871276bbf2d32f08fb1bcb297102fd31033c74e21fa025310be8f3f`; `d60c2893eb0418c86f9d75e4fb66910dac368907278d6bf0b7ffcb165808b4dd`; `b50d92b9e1b5d95e45480063ff897146595824a4fe6ab71e819dd7aa925791cb` |
| npbench_polybench_syrk_syrk__kernel | `b34e9bc73e18f659b0b5231a8e4924a8945d8916d30cf14aef8dd7e260513580`; `372656ec9056a219217d13ab8cba3b87fef34ff01afc5f09b415ee17dc1dc439`; `bc8ac49a096624439f5d637889656d44ee0854d5b1f10e76169977e95d2c9617`; `2d3d86dde602cdc770d886221c096fbf9064dea14b542d7350e6875ba310e8a2`; `9587e7b3c8a5230c323d5c262df2333245cd5e6fdfe38fc02520f10ae82044f7` |
| npbench_scattering_self_energies_scattering_self_energies__scattering_self_energies | `2a0b055128830fa48050a33f9795dd6fbe10d45104fa2d7ba65f448181d82433`; `9fc8eec368670d3762a09c489c0ebd383872f584f6e733790c65c45c6a81d2c4`; `4e5bdc96256557be65544a44747d31805ae10803824c0f27af140c75e9bc108e`; `768e7f5a728df7b972c5a4bf264982d28db7008bda36ac7e927c763b721e636b`; `cca9ea1d9f1e79fbc94595b791586acde68ad73f5b5a135f9bdd0915ccd0e31f` |
| pythran_bubble_sort__bubble_sort | `4c2eefe43aa89155b1268b93f844159c04d261bb8a6400f31e79b77b7f14d602`; `d304c19494a053c5b9692e283518ddc0502edc5a58eee2c6242d03f15d4ddf52`; `1c42118efe1c3a32e758bb7461935ed5b5af65b95b00086346220b076658b301`; `04793ef184c9d83d64989b5ea4424c261eafb416620535d571412efd76aaa012`; `c42901e9e357c5a64f91613a0072c1d2dfc3d22e1fcbae3d07daeee4fb804f01` |
| pythran_crotg__CROTG | `cbcea33742121f216359a96875af9286f805f116c9a7d5cf7c4ca0685ae2f45f`; `e87e6694bd618f0691ec4434f399a2cc2d9d384209596b4f99c904dbd422febd`; `83c46c0fb9353b1cb9201debaeafe6fbc3db421bcc4b0ff048ba21df1e329e44`; `bb61d19822a1245876242f2ab7badde9b397f32c82048bb9467cfaeea56d2ce0`; `195632d6ea5dce396e6a6da11f8618ff4e8722b05a3b5c02523bc44aa2d4ded5` |
| pythran_fbcorr__fbcorr | `c44057b3b83a1199eb53489a35dc6a130cf1dafbdbc560ca7e613f885605e1cd`; `5bbd282e82392259c8c00d22f8dc18fd6eb5517ea21b6dcc9a7089ae34cfa3ce`; `1d1f74d29ecad692c9f4e0dbbac4a0ff3d58f5bf40a8c5a72d945010167602cc`; `427f525db3c328ec772b8dcdcc5b38e9fe089aab616838e4f27e5a3e4d6aa0e3`; `93d199cdb330d7837b7b162d923da2c6a4aa705492f401ed2f54bce3b4d1b21c` |
| pythran_hasting__fweb | `443fc64f0ca5291b81e9c3763327f549b8a5f7b905d2d19bdfcad9bbda8f75fc`; `4557f8acd06626b3bc035b58be9c8e57dd78984f3695878ca033f56699311480`; `57f26e9ea711aa87670af24ed1a0ea2a9228847f73f25f2bf302d4c82951e7aa`; `8bdfa760cc7636c490e6a15dc47b15e258deb900e6a8006ae6b0808ac86490a1`; `58db6aeca8382d1b4bd9fbf14f2bcb4f70a1dc7bf1d6f3f674ee03041847502a` |
| pythran_lap__laplacian | `93f5a1f606decc96070fbaff571da0afeb4129fe5dcc323f064cd7c2801bdf4e`; `4036937f87d080f4e65c85c05d203e65e3b3604a6c7ccfc89b98191e7630fb30`; `79b9e700b421158a4970007f0abc5f367c1b8a4c71e2e749ca46658a987fcb00`; `fd4607abb5cd62d1342a599f4fd12441ed3d631b91017c077d5ec22a9cff9833`; `809ff6bd8b3e53740956c313f7aeba5f8e5abfdfe88e0d1805ed6daefa94bd29` |
| pythran_make_decision__md | `e5e08bd1b6719a6441d72266c154545370f5deece0b2727c182e029195a352cc`; `30a7e88b6abfce930a84ecde95883da4da520dfbf7244487c3771b5c0c3f2c55`; `0eb14acb38ea729cd45f6f09cb1e6d69df4c9f6253970f44ff9f9f16ff8995af`; `37896002d96abbcce37e7a7eb5fd3e6942440ea741384bcd39bf3412afdbcc27`; `e7468c98d0c1b4ebe3a3732dafee972bbc705c01b0900bd73591130d9d1ea0cd` |
| pythran_multitype__check | `5297e050850ff982cf1a5da58aeaf3c28f42bbefe58e4f082490eb91f9cb20a3`; `683628a030e11a439c5940c09cc44183d72f7deb4d2d68269f2d9c898c3bf01c`; `114599911123a069f3ff405e5e476cdf4f70f69bdd835752a91b88e014af6c9e`; `101e388f9c732d272bd4dbde375214533a484283729d36341d7fe5dbe4135336`; `bafb584cd2bdcb45f9b575910b8cf0f33fa5b78258ba8591f8443fc59cd97fe9` |
| pythran_multitype__check2 | `5297e050850ff982cf1a5da58aeaf3c28f42bbefe58e4f082490eb91f9cb20a3`; `112ee016c0eb0028674ce3e286aa608388483b962e5f12068f68e8c189df4208`; `306340bd9a45b5c90d8a01cf2bed83fc67feb295461c64c346b68c09ee5a05a1`; `72d97372e980c37bb65f002a73909d10e805a7ed90dc728b8b3a9a25ca531449`; `a02d48cf0ca79c01165625447147007a1870450f1422dc82b109d26504c577e4` |
| pythran_multitype__times | `5297e050850ff982cf1a5da58aeaf3c28f42bbefe58e4f082490eb91f9cb20a3`; `b686ac74f849408bff1130511138af8978db6f34feac04a1a056d562f3b81bd5`; `6834f70da03a6bebb5126fead678f5bea73439fe38fc344c79f34d2d59199321`; `b49bd700209dc515ba4748f29dbc89298821f1eb4d13db779570c2161c6c2e18`; `7c381d1a1a078b8232391e1e1e5cfb19bf31334f28514ed3bdb7e6b5f143b775` |
| pythran_periodic_bc__periodic_bc | `d96c7675ec36e5dc9fe7b7437fcd364efd4540c02cf29fe3eec9048b79b0f66a`; `8d0bafb9ea74b51fff606cc47a915af95f49d0aeafde9782ebe99ee308cba575`; `ebd83ca871d90ac445652996051bbcf60dd708ae66a1f7f91cd9ef46df72d0bd`; `044f95f076e9bc8c6d7a19fb8f16c0440cc5dc88c6a65f0837059465ba2a85d2`; `65a8ba2cd0c5571759c7b8c4095934b449de5feb6209151702b8220eaaaed927` |
| pythran_pythagorean_triples__next_pythagorean_triples | `8b02e649dce84b590be3826da1966e219b60a87b77768095184bf954ef584783`; `f0c44c273e5d66e8db3e9055919b27bba1b82172e4a0cc168c03a17006ccef59`; `54835702502b4766a2896bc98ea97d6375f6afc685ca61171b23d826103ba345`; `b7e2325a6a3f0cdbf5960c88761029dfaae462cc8acc202e9fcc18400b4cacdc`; `924973e85bdadc566235185f7659fb3f0cb46251c8c659a32ac9736ae76a5a20` |
| pythran_specialconvolve__specialconvolve | `99fe08f7cf3b265bc760c816be2269faf183231d74c0552341fae35eae613892`; `40b2125ba2e963002d4649cb5f4b725ef3e908ed20b1c52299c32608c83bcf73`; `fa2eec26e71a37ef379143ae0dd35b8d253be9212a992cc4fbe67cf3c04d43d1`; `9ec22b333de575f2de8eaeb5d8b64c5accc5213d6009378883287577fddfab9b`; `9e3014e046aa8c5a89d46e12478fdd9247b6889f6ac503a593780199cd262010` |
| pythran_vibr_energy__calculate_vibr_energy | `db54012657f3f673993dd200bd84c9dec2c6abaa9ffca9c88512b0f144a908a8`; `ab31106525de1aae403a0d0fd3c4036146f56c34151a8603134a5e616cd21586`; `6680fc5b105e64f460891a8720b15ab0aae8a4b13f692076d3ed4d866352dc37`; `2de04ef87d5c661095e122dd17bcccdd6bf1c7252e68e5006dd1f5dd7e244468`; `b67b7a6c1c2e3cc074269dca543f7cb3b55da56e10a79013d71b55a910e989a6` |
