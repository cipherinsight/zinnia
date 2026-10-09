# Literal NPBench/Pythran batch npb3

**Result: 18 of 20 cases pass both systems.** `npbench_go_fast_go_fast__go_fast` and `pythran_tax__tax` are not finishable under the unchanged input-recheck diagnostic, as detailed below; both pass Noir and their Halo2 failures report only redundant input checks. All 20 optimized checks pass. All 40 common-width counts succeeded. No memory-cap failures occurred.

Original fixture shapes; branch `literal-npb3`; 2026-10-09. This report covers only the 20 assigned entries. The earlier LU0 commit and LU1/CRC drafts were reused and reviewed against `kernel.py`; every other computation was authored from the source. Only literal source files and this report changed.

Validation uses `hw_check.check --no-prove`: Halo2 mock A/B, reference-output checks, stats and an out-of-domain input rejection; Noir compilation/gates, witness execution A/B, reference-output checks and an out-of-domain input rejection. No setup, key generation, proof, verification, timing benchmark, scaling run, Zinnia execution, IR or emitted-code inspection was performed. Fixture agreement is not an exhaustive domain proof.

Commands ran with the worktree venv, `HW_CHECK_LOCK=.../literal-work/lock-a`, temporary/work/log paths under `.../literal-work/npb3`, and an 8G memrun cap. Halo2 builds used `--profile dev-fast`, three jobs, restricted case lists, and separate `target-literal`/`target` directories. `HW_VARIANT=literal` selected literal files; it was unset for optimized checks. The pinned Noir toolchain and noir-recursive UltraHonk gates are those of `NOIR_BASELINES.md`.

## Checker results and automatic-size counts

Halo2 columns show advice + lookup = total cells; fixed and instance cells are excluded. These are the checker's automatic-size counts, not a controlled lookup-width comparison. Optimized entries were each checked once in this run. The next table gives the controlled Halo2 comparison.

| Case | Literal Halo2 | Literal Noir | Optimized Halo2 | Optimized Noir | Literal H/N | Optimized H/N | Notes |
|---|---:|---:|---:|---:|---|---|---|
| `npbench_crc16_crc16__crc16` | 141,152 + 136 = 141,288 | 43,657 | 134,627 + 248 = 134,875 | 45,256 | pass/pass | pass/pass | Reused draft; removed all local hints; direct signed bit recurrence and separate final masks/shifts. |
| `npbench_go_fast_go_fast__go_fast` | 41,761 + 9,744 = 51,505 | 10,004 | 41,326 + 9,616 = 50,942 | 10,004 | FAIL/pass | pass/pass | Unfinished: L6/input-check diagnostic conflict (16 rechecks). Source recurrence retained. |
| `npbench_polybench_covariance2_covariance2__kernel` | 48,536 + 6,256 = 54,792 | 11,819 | 21,307 + 4,246 = 25,553 | 9,748 | pass/pass | pass/pass | Full covariance matrix; means, centering, dot products, normalization separately. |
| `npbench_polybench_gemver_gemver__kernel` | 52,535 + 12,880 = 65,415 | 12,768 | 50,006 + 12,880 = 62,886 | 12,768 | pass/pass | pass/pass | Both outer products and every elementwise/matmul intermediate materialized. |
| `npbench_polybench_k3mm_k3mm__kernel` | 119,576 + 14,256 = 133,832 | 19,559 | 75,323 + 14,256 = 89,579 | 19,559 | pass/pass | pass/pass | Three left-associated matrix products; each dot scans its inner index. |
| `npbench_polybench_symm_symm__kernel` | 75,640 + 18,760 = 94,400 | 13,596 | 66,487 + 17,044 = 83,531 | 13,596 | pass/pass | pass/pass | temp2 retained; repeated alpha*B recomputed at each source occurrence. |
| `npbench_polybench_trmm_trmm__kernel` | 35,398 + 7,096 = 42,494 | 8,507 | 29,285 + 7,096 = 36,381 | 8,507 | pass/pass | pass/pass | In-place updates complete before final B*=alpha; empty dot is constant zero. |
| `npbench_weather_stencils_vadv_vadv__vadv` | 2,028,898 + 515,888 = 2,544,786 | 325,432 | 1,635,822 + 386,288 = 2,022,110 | 293,764 | pass/pass | pass/pass | All five source loops retained; complete horizontal planes per array operation; no coefficient sharing. |
| `pythran_create_grid__create_grid` | 433 + 128 = 561 | 3,908 | 432 + 128 = 560 | 3,908 | pass/pass | pass/pass | Zeros and both slice assignments retained; constant zero output column omitted from publics. |
| `pythran_deuxd_convolution__conv` | 308,848 + 77,288 = 386,136 | 9,125 | 40,187 + 9,236 = 49,423 | 9,125 | pass/pass | pass/pass | All four loops; no repeated-product or repeated-output CSE. |
| `pythran_goodexpomeasure__goodExpoMeasure` | 987,675 + 227,848 = 1,215,523 | 155,513 | 942,415 + 218,385 = 1,160,800 | 152,122 | pass/pass | pass/pass | Three casts, three full exp arrays, two products, nearest-even hundredths, scale and uint8 cast. |
| `pythran_ising__ising_step` | 4,764 + 1,036 = 5,800 | 3,578 | 4,555 + 1,004 = 5,559 | 3,575 | pass/pass | pass/pass | Source helper calls and sequential sweep; both branches computed and selected. |
| `pythran_ln__ln` | 21,600 + 5,616 = 27,216 | 4,971 | 11,232 + 2,928 = 14,160 | 4,869 | pass/pass | pass/pass | Every X-1 array and MSB-first power computed separately; source left-associated additions. |
| `pythran_lu__factorMatrix0` | 11,356 + 1,466 = 12,822 | 5,451 | 5,215 + 1,136 = 6,351 | 4,450 | pass/pass | pass/pass | Reused earlier commit; full abs before argmax, fresh comparisons/divisor bounds, full-row gather/scatter. |
| `pythran_lu__factorMatrix1` | 10,355 + 1,556 = 11,911 | 5,262 | 5,081 + 1,136 = 6,217 | 4,410 | pass/pass | pass/pass | Reused draft; repeat abs in branch, fresh comparisons/divisor bounds, source column swap loop. |
| `pythran_multi_export__a` | 46 + 14 = 60 | 2,822 | 46 + 14 = 60 | 2,822 | pass/pass | pass/pass | Identity: input-domain check and output only. |
| `pythran_perm__permutations` | 160 + 48 = 208 | 3,025 | 160 + 48 = 208 | 3,025 | pass/pass | pass/pass | Source cycles algorithm, with shape-only counters; no alternate permutation algorithm. |
| `pythran_rosen__rosen` | 2,713 + 668 = 3,381 | 3,232 | 2,680 + 668 = 3,348 | 3,232 | pass/pass | pass/pass | Complete elementwise intermediates and t0+t1 array before running sum. |
| `pythran_tax__tax` | 20,148 + 2,014 = 22,162 | 5,702 | 10,869 + 1,846 = 12,715 | 4,524 | FAIL/pass | pass/pass | Unfinished: L5/L6 input-check diagnostic conflict (70 rechecks); equality-test compaction and separate arrays. |
| `pythran_weights__weights` | 4,468 + 918 = 5,386 | 3,388 | 2,050 + 463 = 2,513 | 3,057 | pass/pass | pass/pass | Full equality arrays and row sums, threshold expression inside source loop, shared division. |

## Common lookup width

Additional `stats --input A --k 14 --lookup-bits 8` runs use the same 8-bit lookup width in both systems' Halo2 baselines. These are counts only, not new mock/proof checks or a proving-size recommendation. Noir uses the same toolchain/settings throughout. No aggregate fairness or proving-time ratio is claimed.

| Case | Literal Halo2 A + L = total | Optimized Halo2 A + L = total | Literal not larger in |
|---|---:|---:|---|
| `npbench_crc16_crc16__crc16` | 141,152 + 136 = 141,288 | 134,627 + 248 = 134,875 | Noir |
| `npbench_go_fast_go_fast__go_fast` | 41,761 + 9,744 = 51,505 | 41,326 + 9,616 = 50,942 | Noir |
| `npbench_polybench_covariance2_covariance2__kernel` | 48,536 + 6,256 = 54,792 | 21,307 + 4,246 = 25,553 | none |
| `npbench_polybench_gemver_gemver__kernel` | 52,535 + 12,880 = 65,415 | 50,006 + 12,880 = 62,886 | Noir |
| `npbench_polybench_k3mm_k3mm__kernel` | 119,576 + 14,256 = 133,832 | 75,323 + 14,256 = 89,579 | Noir |
| `npbench_polybench_symm_symm__kernel` | 75,640 + 18,760 = 94,400 | 66,487 + 17,044 = 83,531 | Noir |
| `npbench_polybench_trmm_trmm__kernel` | 35,398 + 7,096 = 42,494 | 29,285 + 7,096 = 36,381 | Noir |
| `npbench_weather_stencils_vadv_vadv__vadv` | 2,028,898 + 515,888 = 2,544,786 | 1,635,822 + 386,288 = 2,022,110 | none |
| `pythran_create_grid__create_grid` | 433 + 128 = 561 | 432 + 128 = 560 | Noir |
| `pythran_deuxd_convolution__conv` | 308,848 + 77,288 = 386,136 | 40,187 + 9,236 = 49,423 | Noir |
| `pythran_goodexpomeasure__goodExpoMeasure` | 987,675 + 227,848 = 1,215,523 | 942,415 + 218,385 = 1,160,800 | none |
| `pythran_ising__ising_step` | 4,764 + 1,036 = 5,800 | 4,555 + 1,004 = 5,559 | none |
| `pythran_ln__ln` | 21,600 + 5,616 = 27,216 | 11,232 + 2,928 = 14,160 | none |
| `pythran_lu__factorMatrix0` | 11,356 + 1,466 = 12,822 | 5,215 + 1,136 = 6,351 | none |
| `pythran_lu__factorMatrix1` | 10,355 + 1,556 = 11,911 | 5,081 + 1,136 = 6,217 | none |
| `pythran_multi_export__a` | 27 + 8 = 35 | 27 + 8 = 35 | Halo2 (8-bit), Halo2 (automatic), Noir |
| `pythran_perm__permutations` | 108 + 32 = 140 | 108 + 32 = 140 | Halo2 (8-bit), Halo2 (automatic), Noir |
| `pythran_rosen__rosen` | 2,713 + 668 = 3,381 | 2,680 + 668 = 3,348 | Noir |
| `pythran_tax__tax` | 20,148 + 2,014 = 22,162 | 10,869 + 1,846 = 12,715 | none |
| `pythran_weights__weights` | 4,468 + 918 = 5,386 | 1,637 + 328 = 1,965 | none |

## Interpretation and arithmetic audit

- L1/L2: each NumPy elementwise subexpression is materialized before the following operation. Shape-only views/copies are wire rearrangements, with no invented gates. Explicit scalar loops rescale each floating product; NumPy dot/matmul reductions accumulate raw products in index order and rescale once. Empty dots are compile-time zero. Integer multiplication and float-by-integer multiplication stay exact as permitted by the numeric contract.
- Shared `canonical_tanh`/`canonical_exp` match the current optimized baselines. Library internals may contain hints and internal range-proof reuse; none is authored in the literal files. Nonzero canonical division is used for vadv/tax to preserve their optimized divisor domain, and goodExpo's already-domain-checked sigma. Other divisions use the ordinary shared division/divmod gadgets.
- Covariance is interpreted as transpose, per-variable mean, full centering, full matrix of defining dot sums, then division by observations minus one. No symmetry reduction or quantized reciprocal. The `M` precondition retains its range plumbing; Halo2 uses the ordinary input loader because L6 disallows the optimized file's local witness load.
- LU0 materializes the entire abs slice before its stable argmax scan. Fancy row assignment gathers its RHS from the old matrix before writing. LU1 computes the second source abs even in the speculative branch. Both use full-axis standard indexed reads and equality/select writes at every row, followed by branch selection; no narrowed-axis gather or cached range proof.
- CRC represents integers as signed two's-complement bit arrays. Standard constrained bit decomposition replaces the draft's custom hints. Arithmetic right shift repeats the sign bit; the byte shift is logical. Complement/mask, shift/OR and final mask are separate source stages. Bit-wire movement and compile-time masks need no invented arithmetic gates.
- `permutations` follows the cycles/indices algorithm. Counters depend only on shape, so they remain compile-time/native counters. Noir bounds the outer loop by N! including the terminating iteration and gates the break/return effects; no private index is assumed in range. `create_grid` materializes zeros and slice assignments; constant third-column outputs remain outside the public instance.
- Convolution's clamp helper is evaluated on loop constants and retains the source's unusual bound `sw[0]` for both axes. No repeated product or clamped-output cache.
- ln uses a local hint-free MSB-first square-and-multiply helper built from shared `mul`: the current optimized arithmetic uses that schedule, whereas Halo2's generic `pow_const` is LSB-first and can round differently for powers 6/7. Each source power is independent, without cross-call sharing. This is an explicit L5 implementation choice requiring review.
- Ising retains the update helper and source iteration order; Noir spells stride-two constant loops with a parity guard. Both branches, including exp, are computed before selecting the assigned site, as L4 requires. `float32` casts in Ising/weights remain Q32 under NPB_FAIRNESS; no baseline-only binary32 rounding is added.
- goodExpo's missing shared round-to-hundredths wrapper is implemented without local hints: shared division/remainder gives floor(100*x+1/2), and an exact tie/parity test selects the even neighbour, followed by division by 100. Rounded Q32 values are materialized before scaling by 255 and truncation/wrap to uint8. The optimized per-backend cast domains are retained (Halo2 signed window, Noir nonnegative 64-bit scaled dividend).
- Tax's internal variable-length arrays use the required running-count/equality-test compaction, with a presence flag for each padded slot. Elementwise effects on padding are gated so padding contributes zero; internal lengths are not added to public outputs. Every source mask/comparison is independent and each subsequent array is evaluated in source order.
- No literal source contains a local witness hint, `unconstrained`, `unsafe`, forbidden optimized gadget or reused comparison window. For ln, Noir uses the shared constant-divisor gadget for every denominator, including powers of two; its rounding matches the optimized power-of-two specialization. Static lint has zero errors. Its warnings are source/format constants (CRC16 width, uint8 255/256, polynomial exponent 4, four tax segments), precondition-check widths, and casts of compile-time zero to u32.

## Not finishable under the unchanged checker

`go_fast` and `tax` have a direct conflict between L5/L6 and `hw_check.check`: ordinary `q.load_*` checks the input domain, L6 explicitly forbids `q.load_*_bounded`, and L5 requires fresh operand bounds at source comparisons (canonical_tanh also requires an I64). `q.bound` on such an input increments `redundant_input_checks`; the checker rejects any nonzero count unconditionally, including with `HW_VARIANT=literal`. go_fast reports 16, tax 70. Avoiding that result would require the forbidden bounded loader/proof reuse, altering shared tools, or hiding the diagnostic with an artificial wire copy. None was done. Complete source implementations are retained for review, but these are **not passing/finished baselines** until the contract/tool conflict is resolved. The report records their measured checker verdicts rather than treating the circuit size as a pass.

## Reuse and tool notes

LU0's earlier claimed pass was not inherited: it was rebuilt and checked after replacing fused abs/argmax, narrow comparison, narrowed scatter and the hoisted pivot bound. LU1 additionally removes CSE of repeated abs and restores the source's absolute row index and sequential swap. CRC removes all draft-local Rust/Noir hints and separates the three final source statements.

The venv emits harmless sys.prefix warnings when invoked through the prescribed relative path. Cargo must run inside halo2_hw so its local RUSTC_BOOTSTRAP setting applies; an initial invocation from the worktree root failed in poseidon-primitives and was corrected. The controlled stats CLI requires both --k and --lookup-bits; the initial width-only calls were rejected, then rerun with both. Shared-lock waiting is respected. No shared tool, fixture, optimized baseline or gadget was edited.

## Evidence

Logs and generated packages are under `/home/zhantong/Projects/zinnia-project/eval-redo/literal-work/npb3/`: `check-lu0.log`, `check2.log`, `check3.log`, `check4.log`, `check5.log`, `check-optimized.log`, `optimized.json`, `common-width.log`, `common-width.json`, `lint.log`, and build logs. These artifacts are not committed. Post-check changes to previously passing files only format Rust and remove unused imports/constants; source arithmetic is unchanged.

Optimized sources and shared libraries are unchanged from base commit `3155495c8f43ed39d1373d1a44f90c78bdb40cf1` (`literal`). Final literal source commit: `000e80591d23cd903dbb4529a948480b3c36cfbe`.

SHA-256 of kernel / literal Halo2 / literal Noir at that final source revision:

| Case | kernel.py | target literal | main literal |
|---|---|---|---|
| `npbench_crc16_crc16__crc16` | `3ebb6680f129474dc2588829e5af5775dd6779f7c95a0cbb349ec8f822709b7a` | `3921c257da72bcea582d517cd60b059e087cca6cc07e78a5bfa9c5859122ed1f` | `cc395a7161b3ed7fddc61b6d624d38a8136f01bbc99333e59562a22dadd78471` |
| `npbench_go_fast_go_fast__go_fast` | `2b79c8c68a5fa33fe723da8113d5fa40ddb2fe5b616c226244dc6d5dfe305bb0` | `9c06fc8428dd70376c50426585679804052d153d41465d6373d7996f78d171ea` | `36f0db6526eecf78baa2ac40d9ba4b95260dcc56e062b601562d7e71648f9ff1` |
| `npbench_polybench_covariance2_covariance2__kernel` | `caf29f2d151e2be5682bc8d8204bf822899b3c2e40a4a47d6371c457092bf459` | `809b15f905aab142617576cdff0918bd7d94050ae20cf5cbdc0e17f89c14e667` | `1cec5a227c187e8c147e2688b772d62939a8929ad484f70f494a72b03fafa4c6` |
| `npbench_polybench_gemver_gemver__kernel` | `cb623d7461a02759cb98780e34a1469f7d74d7d4716aa7c98fe50c08bd33c643` | `4100d633972fd83d4b895ff70e192cc999a67b44bbadcfe1aad845d298c58348` | `e49fa132155ec8ffe67ef5b4be470ac51a845eb7f5cd1db7b7929ae0c58daf6b` |
| `npbench_polybench_k3mm_k3mm__kernel` | `f37646cee2b51f22ee00c95f7175de2a3fb97b86df1d6e573f4a2119c0e7cc1a` | `7953918210defcf2e5a89fefe56dbc29a5964b8f3d66d6d80ae2a438416466ff` | `d4d0bc88a9450d2d2fd1766d4dcd2b48f0896aa24b306149dcc80aeede45b7ce` |
| `npbench_polybench_symm_symm__kernel` | `0ca8cf5ac38f31445960514b7560f63465fa612fb2c16c0408c2727f22ddf916` | `42d8a5e6f37f4fd04ae971f55aa121a1ae0b16ac83c2a491f10d734c6a96fe0f` | `8f9c592013a44f081bb05e6025d9349b7419edba39d307e0f243c82f90e5ca5a` |
| `npbench_polybench_trmm_trmm__kernel` | `f3838d91ff378dec8a5a6d7a885708fecc95bce903d8c060d95403a6a55fc5d0` | `b4fd5104d637d2a988aa16dc54077176720801d5d5064f14be5fd9a7d198478e` | `0289910d20ed4da94a2244b468a0a4414165bab1743545291d497acebbd3d97e` |
| `npbench_weather_stencils_vadv_vadv__vadv` | `3d27a2809da5a5015821194da5a2c2fb581f73b0ef8ab605a6a18cbdaa343ce9` | `6465df00f7dda1aa341e8d77c48bb426bfaea0ffc916abf658aef49c97ba76d5` | `6bc43eba5b16bee82e42aa3f167275b62a5c380ff45be17987999a36f5862990` |
| `pythran_create_grid__create_grid` | `aaef3e73e25deb15960cc39e0472035508f0a4a3f2a627477040674f32003f2c` | `7620f8dda7d84cc25587a69bf9353cb0fc2a8a55322d67c04b5ec7a3a9676d2a` | `20fabc4605c169096b229b13ff1b93b2cdb433d510d97c770a92a8645bfb7e99` |
| `pythran_deuxd_convolution__conv` | `1fec941af2989fa7e001f85b017ee23f74b735c48259f310be213918ccf602bf` | `050bd6c699e2997739d60c5565df9540594411210e933040bd9bdb9ec2c35ba8` | `c6ff0520c2ca48fee9f8bee51630a5c7f83e80b19d4c79332fccd6bd99757ac9` |
| `pythran_goodexpomeasure__goodExpoMeasure` | `37d75784b3981c0cdd1ed553febf4def0dd947cb00d0513b83b5b06bbe4910c5` | `15012bfb139e51f80966948e3c611093708500ffea959f39e09d85727988de78` | `5dd155dcbec7bc244cb938acf8a0436e203a93544718f97c2f1afbbe592c3cc9` |
| `pythran_ising__ising_step` | `cf5fe0f326779bcd2cbf1d84afb46d2100d5fa3ca134fea112abc0b5cb86d667` | `ccf5be5e6869e282a1200a56169a496f882e2ff1afe0b93884bbbef6542a22ed` | `a66ac620bba6e9fe4667bc8dcbf815a104b2a2e51093f60826b897c418cf65c0` |
| `pythran_ln__ln` | `f70d2b5e275e13a4d19c78b81cffd886950d762e235d79c1cbccc6c86a7a11da` | `31e47653c5a44170aff5480b9735f21356652a563d47de086a8d769e23adcad3` | `36ad664adad0e5c55d0cf9a631be5b9375ec42f0e4d9b6d85f02c7381069166d` |
| `pythran_lu__factorMatrix0` | `dcfe3d0fa74675d5dde30f76ed8791c6824094e31d1708574664285a39324334` | `cbbdc37bd675b56e46ff0af7d2ef2e0e25b33f050e49608ba4242bacc86a0dfb` | `85ea58df070510d50b3453f643b9b5fa725cc7cb0dcb2d61953291a8199974b0` |
| `pythran_lu__factorMatrix1` | `dcfe3d0fa74675d5dde30f76ed8791c6824094e31d1708574664285a39324334` | `720bda10bcffd59b760598a4402a6a5114e8bf0db1aac95cf6b0b0c4c648821d` | `2d1dc31c408f65a0099baa3f688f0e61bbd95159536d6844132f3b6e92e4fe84` |
| `pythran_multi_export__a` | `3faf8ebe0a74b54f3a544d7033f06bd4f72036fe1065719a94dc80b1a6419ab8` | `620710dfe7e3396594bd29b20db5ba91870cf57f04cce94143c7c2c38e8bceea` | `e415d2a7aa922c65fe89d6a5815f012e2fd52ac202cfa145b2a00b95d58a95dc` |
| `pythran_perm__permutations` | `633e8e48057c3a69cfaf20a8aa3c01cbde3c1434c075e6ff468babc46544e944` | `bfaedf0dde0d60d4e517b0604145c17293833b8d0afc64099c1efcf0f2f8ae2d` | `1a24bc57af46facd441fffd9259485210970bc76a84d3c0622ecccfc1d3a8134` |
| `pythran_rosen__rosen` | `8ab3f35f18d4fc01bc0fe2a43abcd13cfaf2ee46699ff04444cc30350d0be184` | `8607afe43b06d364cbe65c4f9aefe7df9fd44ee3305ace0ad8ae08b6b7124466` | `f7b4cd6d99f2c195aab2631b177e4294c488501c97df47bf437e0b80231b3c94` |
| `pythran_tax__tax` | `438a9cc3ce066b198c60b777551be83affcafcd3ce03fde3f55c5f4f924b679b` | `e782df606829eae0f1a2c16fa84b6bc9dcc1d39bae65081b32966e6b1768f909` | `8d79c70dfe91df28ebe1f7c9f1e1a7db038d29d1a5052f7f160b2df9ac81cc2b` |
| `pythran_weights__weights` | `5c2204cf036831dada0539f35ba8e4201ffdc212e5b277ad9164f095dfd95f7b` | `21a3071872fcf8a33696669b9ad3f301601be3d11d378a5e08bd3a98b44bda36` | `248c5966cd2e93838837fe14d9e38beb26ad09700d83da57c12e3b701aca13bd` |

## Explanation of non-larger results

- `npbench_crc16_crc16__crc16` (Noir): Noir is smaller (43,657 versus 45,256). The source recurrence is unchanged; the literal uses standard constrained `to_le_bits` rather than the optimized baseline's custom hinted bit/high-part decompositions and manual recomposition checks. The optimized file is not minimal for this toolchain. This is an allowed standard primitive, not a program-level algorithm shortcut.
- `npbench_go_fast_go_fast__go_fast` (Noir): Noir is equal (10,004). The optimized Noir already spells the source trace recurrence directly. Halo2 is larger and checker-blocked as explained above.
- `npbench_polybench_gemver_gemver__kernel` (Noir): Noir is equal (12,768). Separating full outer-product, elementwise and matrix-product arrays does not change the required arithmetic; Noir optimizes the temporary array structure. Halo2 retains the explicit scalar accumulation overhead.
- `npbench_polybench_k3mm_k3mm__kernel` (Noir): Noir is equal (19,559). The optimized Noir is already the same textbook left-to-right matrix-product algorithm, with one rescale per dot.
- `npbench_polybench_symm_symm__kernel` (Noir): Noir is equal (13,596). The literal repeats alpha*B at both source occurrences and restores temp2/row statement order; Noir can CSE the identical products and remove temporary storage. The Halo2 increase confirms the handwritten literal has not shared those products.
- `npbench_polybench_trmm_trmm__kernel` (Noir): Noir is equal (8,507). Moving the scaling back to the source's final separate array statement and materializing in-place updates exposes the same scalar arithmetic to Noir. No program-level simplification was added.
- `pythran_create_grid__create_grid` (Noir): Noir is equal (3,908). Both compute only data movement plus input-domain checks. Materializing the zero array and slice assignments costs no Noir arithmetic. Halo2 records one extra constant-zero cell.
- `pythran_deuxd_convolution__conv` (Noir): Noir is equal (9,125). The literal executes every source loop and repeated multiplication; Noir can eliminate the repeated constant-index expressions. Halo2 grows from 49,423 to 386,136 cells at the common width, showing that the explicit product/output caches were removed.
- `pythran_multi_export__a` (Halo2 (8-bit), Halo2 (automatic), Noir): Equal in both backends. The identity kernel has no computation beyond input-domain checking and returning the input, so the optimized baseline is already literal.
- `pythran_perm__permutations` (Halo2 (8-bit), Halo2 (automatic), Noir): Equal in both backends. The source cycles algorithm differs from the optimized host algorithm, but all indices depend only on shape; both circuits contain only the same checked inputs and output copies.
- `pythran_rosen__rosen` (Noir): Noir is equal (3,232). Full intermediate arrays and a running sum preserve the same scalar expressions; Noir eliminates array storage and normalizes the additions. Halo2 retains the separate t0+t1 additions and running accumulator.
