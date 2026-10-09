# Literal DS3 baselines

**Status:** 18 cases pass both non-proving checkers: 0403, 0409, 0414, 0416, 0422, 0425, 0426, 0428, 0429, 0433, 0434, 0435, 0438, 0439, 0440, 0441, 0442, 0443. Six complete implementations cannot receive a passing Halo2 checker verdict under the current fresh-bound rules: 0412, 0415, 0431, 0436, 0437, 0444; Noir passes for all six. Case 0420 is not finishable under the combined eager-evaluation and valid-input-domain requirements and has no final literal files.

All 25 cases were assessed; 24 retain complete literal implementations in both systems. Case 0420 was withdrawn after a concrete domain failure. Validation uses original shapes, `--no-prove`, `MEMRUN_MAX=8G`, `CARGO_BUILD_JOBS=3`, `dev-fast`, and `HW_CHECK_LOCK=/home/zhantong/Projects/zinnia-project/eval-redo/literal-work/lock-a`. Literal builds use `HW_VARIANT=literal` and `target-literal`; optimized builds unset the variant and use `target`. All workdirs/logs are under `/home/zhantong/Projects/zinnia-project/eval-redo/literal-work/ds3/`.

Halo2 sizes below are **advice + lookup = total**; fixed/instance cells are excluded. Lookup widths are automatically selected by the existing checker, not held common; these are automatic-size measurements, not a controlled common-width comparison or a proving-time ratio. Noir sizes are the checker's unpadded UltraHonk gate counts with the pinned Noir/BB toolchain (`noir-recursive`). No aggregate fairness claim is made.

`R(n)` means the Halo2 checker reports FAIL solely because `n inputs checked twice`: A/B mocks satisfy constraints, outputs match NumPy, and the out-of-domain input is rejected. This is a validation blocker, not a passing checker verdict. `P` means checker pass. See the blockers below.

| Case | Literal Halo2 | Literal Noir | Optimized Halo2 | Optimized Noir | H2/Noir verdict (literal; optimized) | Audit and notes |
|---|---:|---:|---:|---:|---|---|
| 0403 | 36 + 10 = 46 | 58 | 33 + 10 = 43 | 58 | P/P; P/P | Reused; defining C-stride sum in forward index order. Input dimension checks copied under L7; no generic extra bounds. **Not larger: Noir.** |
| 0409 | 2784 + 768 = 3552 | 3320 | 2784 + 768 = 3552 | 3320 | P/P; P/P | Reused; explicit array-copy aliases, then elementwise addition. Already direct. **Not larger: Halo2, Noir.** |
| 0412 | 3931 + 410 = 4341 | 3369 | 1530 + 240 = 1770 | 3079 | R(10)/P; P/P | Reused; removed forbidden bounded loader, added fresh comparison operands; retained equality-test prefix compaction. |
| 0414 | 508 + 144 = 652 | 2875 | 494 + 144 = 638 | 2875 | P/P; P/P | Reused unchanged after review; contiguous reshape, running sum, constant division. **Not larger: Noir.** |
| 0415 | 1178 + 312 = 1490 | 2917 | 736 + 216 = 952 | 2903 | R(8)/P; P/P | Two complete four-element pairwise maximum scans. |
| 0416 | 3048 + 744 = 3792 | 3218 | 2928 + 744 = 3672 | 3218 | P/P; P/P | Row-major two-element means; reshape is wiring. **Not larger: Noir.** |
| 0420 | 619 + 165 = 784 | 2875 | 529 + 142 = 671 | 2880 | R(2)/P; P/P | Withdrawn candidate counts (no final literal files). Eager branches and independent powers. NOT certified: valid clamped inputs can overflow the unused polynomial; see domain blocker. **Not larger: Noir.** |
| 0422 | 24750 + 1400 = 26150 | 6149 | 12250 + 1400 = 13650 | 6149 | P/P; P/P | Materialize 99-element hstack; valid-correlation defining sums in lag order, one rescale per output. **Not larger: Noir.** |
| 0425 | 46740 + 8080 = 54820 | 7151 | 3200 + 460 = 3660 | 4041 | P/P; P/P | Each scalar AND independently decomposes low bits; full masked array before >0 and integer cast. |
| 0426 | 71940 + 10960 = 82900 | 8071 | 3360 + 380 = 3740 | 4201 | P/P; P/P | Same construction as 0425 with m=15. |
| 0428 | 4844 + 1070 = 5914 | 3270 | 1804 + 443 = 2247 | 3094 | P/P; P/P | Both tuple expressions independently compute mean and std; std computes its own mean and rounds every squared deviation. |
| 0429 | 4844 + 1070 = 5914 | 3270 | 1804 + 443 = 2247 | 3094 | P/P; P/P | Same as 0428 with spread multiplier 2. |
| 0431 | 3128 + 766 = 3894 | 3145 | 1624 + 395 = 2019 | 3053 | R(16)/P; P/P | Independent mean/std calls, full > and < arrays, then logical_and and invert. |
| 0433 | 1253 + 216 = 1469 | 2989 | 1004 + 216 = 1220 | 2967 | P/P; P/P | Test both signed spellings per axis position; complete row write, then column write with equality/select updates. |
| 0434 | 1805 + 200 = 2005 | 3077 | 1304 + 240 = 1544 | 3009 | P/P; P/P | As 0433 with each two-index array processed in index order, including duplicate indices. |
| 0435 | 361 + 108 = 469 | 2872 | 360 + 108 = 468 | 2872 | P/P; P/P | Source-order constant writes; unread overwritten inputs omitted; constant output layout retained. **Not larger: Noir.** |
| 0436 | 6270 + 1400 = 7670 | 3766 | 3680 + 880 = 4560 | 3574 | R(50)/P; P/P | Full row-max array by pairwise scans, then broadcast equality. |
| 0437 | 6270 + 1400 = 7670 | 3766 | 3680 + 880 = 4560 | 3574 | R(50)/P; P/P | Full row-min array by pairwise scans, then broadcast equality. |
| 0438 | 4034 + 930 = 4964 | 3264 | 742 + 192 = 934 | 2930 | P/P; P/P | Full sample covariance matrix and both normalizations, full clipping, then [0][1]; arithmetic caveat below. |
| 0439 | 93400 + 25600 = 119000 | 21673 | 53800 + 14800 = 68600 | 15373 | P/P; P/P | All 2000 broadcast products, including both symmetric halves. |
| 0440 | 6301 + 1650 = 7951 | 3742 | 6100 + 1650 = 7750 | 3742 | P/P; P/P | Preserve enumerate order; all five diagonal square roots per iteration, then slice += including additions to zero. **Not larger: Noir.** |
| 0441 | 311 + 70 = 381 | 2853 | 266 + 70 = 336 | 2848 | P/P; P/P | Four equalities followed by an any/OR scan. |
| 0442 | 11852 + 230 = 12082 | 5219 | 3663 + 230 = 3893 | 3687 | P/P; P/P | Membership by equality/OR, invert full mask, prefix counts and all-pairs equality compaction. |
| 0443 | 11772 + 230 = 12002 | 5219 | 3583 + 230 = 3813 | 3690 | P/P; P/P | Membership by equality/OR, prefix counts and all-pairs equality compaction. |
| 0444 | 18002 + 2184 = 20186 | 5479 | 5886 + 724 = 6610 | 4030 | R(160)/P; P/P | Four independent comparison arrays and separate logical operations, then all-pairs equality compaction. |

## Validation blockers and semantic limits

**Checker conflict (0412, 0415, 0420, 0431, 0436, 0437, 0444).** L5 requires fresh `q.bound` operands at each comparison and L6 forbids `load_*_bounded`/I64 proof reuse. The unmodified `hw_check/check.py:220` unconditionally rejects any nonzero `redundant_input_checks`, including these mandatory literal comparison checks. Retained source files use the required fresh bounds; no fake copy/add-zero workaround or forbidden proof reuse was introduced. Successful combined validation cannot be finished under the current rules/tool. These implementations are complete, but their Halo2 checker verdict is not pass.

**0420 domain blocker.** The literal computes both where branches and independently computes x**2 and x**3 as required by L3/L4. For x=10000, x_min=0, x_max=1, NumPy returns 1, while the unused cube exceeds Q32. The optimized baseline explicitly substitutes zero before computing the polynomial outside the selected interval. Copying that transformation would violate the eager literal form; silently narrowing the input domain would violate fairness. This case is not certified/finished, regardless of A/B fixture agreement. Both Halo2 mock B and Noir execute B reject this input (`edge0420.json`, `edge0420.log`). The candidate pair was removed from the worktree rather than leaving a known failing baseline. It remains in commit `b127bbf4` and the external `rejected0420/` evidence directory.

**Arithmetic comparison limits.** 0428/0429/0431 follow L2/L5: squared deviations round separately before summation. Their optimized counterparts fuse the squared-deviation sum into a dot/rescale and share repeated statistics. 0438 uses Q32 sample covariance, two square-root normalizations and clipping; its optimized counterpart rewrites Pearson correlation using scaled integer centered sums and one root/division. Thus these optimized-vs-literal numbers are not evidence of exact arithmetic/domain equivalence. Intermediate Q32 overflows still reject witnesses under the shared gadget contract; all-input numerical equivalence was not proved.

## Rule interpretations

- 0403: ravel_multi_index uses the defining C-stride inner product in index order. The library inner-product gadget loads raw private inputs (L7 plumbing, not hints for computed values); copied dimension checks imply int64 membership as the named CONVENTIONS 1.6 exception. No direct `load_witness` appears.
- Array copies, slices, transpose, reshape and newaxis are wire views/copies; shape arithmetic and literal masks are compile-time constants. They do not create gratuitous constraints. Constant writes in 0435 and unread off-diagonal elements in 0440 retain the original input/output plumbing.
- Means divide the Q32 running sum by a constant integer using the shared floor-division gadget; this is exactly division by the corresponding Q32 constant without an extra scaling stage. Multiplication by an integer literal uses exact integer-times-Q32 arithmetic.
- 0425/0426: the unlisted scalar integer bitwise-AND operation uses two's-complement bit decomposition independently for each broadcast element. Halo2 decomposes the low M bits using the permitted division gadget; Noir decomposes the signed-offset 64-bit representation. The masked integer array, positivity comparison and boolean-to-integer cast remain separate. No program-level bit-decomposition reuse.
- 0428/0429/0431: textbook population std is its own mean, all deviations, all individually rounded squares, index-order sum, division by N, sqrt. Repeated source mean/std calls are not shared in source.
- 0433/0434: a private write equality-tests both valid signed spellings of each axis position, asserts exactly one hit for bounds, and selects the full slice. This uses no ordered comparison, hinted position or algebraically factored equality. Row and column statements remain separate.
- 0438: the unlisted corrcoef call expands to the full 2x2 sample covariance matrix (/3), square roots of both diagonal entries, row division then column division, and clip to [-1,1]. All entries are built even though only [0][1] is returned. Covariance dot sums each rescale once.
- 0441/0442/0443: membership is a scan of exact equality tests with OR. Every mask compaction uses prefix counts and equality testing for each input/output pair, including padded output positions.
- Compiler CSE/dead-result elimination is allowed by the rulebook. Identical Noir counts do not imply source-level sharing; no compiler IR or Zinnia output was inspected.

## Not-larger flags

- 0403: Noir. Noir sees the same constant linear expression and dimension assertions; statement ordering adds no gates after compilation.
- 0409: Halo2, Noir. The kernel is already a direct elementwise addition. Explicit array copies are wiring, so both sizes are identical.
- 0414: Noir. The same constant-divisor mean remains after Noir folds zero accumulator initialization and reshape wiring.
- 0416: Noir. The same pairwise constant-divisor means remain after Noir simplifies zero accumulator initialization and reshape wiring.
- 0420: Noir. Withdrawn candidate only: omitting the optimized branch-domain protection saves a few Noir gates but rejects valid inputs. This is a failed fairness check, not a usable smaller literal baseline.
- 0422: Noir. The hstack/slice is compile-time wiring and the same dot sums reach Noir. Halo2 still explicitly performs each raw product and accumulation.
- 0435: Noir. Constant writes and output omissions are wiring; both implement the same nine surviving input-domain checks. Noir removes overwritten cells.
- 0440: Noir. Noir folds the literal additions to zero and compiles the same 50 roots; iteration/output order does not require extra gates.

## Verification and evidence

Every retained final source was reviewed against kernel.py and L1-L7. Fixtures A/B and the checker's one out-of-domain tamper were run in both systems; exceptions are explicitly identified above. No proofs, key generation, proof verification, scaling, timing experiments or Zinnia runs/IR/size inspection. No shared libraries, fixtures, optimized sources, case sets or tools were edited.

Raw evidence: `reused.json`, `group2.json`, `group3.json`, `bits.json` (supersedes the initial Noir type errors in 0425/0426), `group4.json`, `writes.json` (final 0433/0434), `review.json` (final 0403/0409), `optimized.json`, and `edge0420.json`, all in the work directory above. Final lint of retained files: zero errors; two advisory cast warnings for compile-time bit masks in 0425/0426. The initial Noir shift-width error was corrected to u64/u64. Python emitted harmless virtualenv-prefix warnings. An initial build invoked from the worktree root missed the crate's nightly configuration; all measured builds were rerun successfully from halo2_hw. No case hit the 8 GiB cap.

| Case | Literal k / lookup bits | Optimized k / lookup bits |
|---|---|---|
| 0403 | 6 / 5 | 6 / 5 |
| 0409 | 9 / 8 | 9 / 8 |
| 0412 | 8 / 7 | 7 / 6 |
| 0414 | 6 / 5 | 6 / 5 |
| 0415 | 7 / 6 | 7 / 6 |
| 0416 | 8 / 7 | 8 / 7 |
| 0420 | 7 / 6 | 6 / 5 |
| 0422 | 9 / 8 | 9 / 8 |
| 0425 | 9 / 8 | 8 / 7 |
| 0426 | 10 / 8 | 9 / 8 |
| 0428 | 9 / 8 | 7 / 6 |
| 0429 | 9 / 8 | 7 / 6 |
| 0431 | 9 / 8 | 7 / 6 |
| 0433 | 7 / 6 | 7 / 6 |
| 0434 | 8 / 7 | 7 / 6 |
| 0435 | 7 / 6 | 7 / 6 |
| 0436 | 9 / 8 | 9 / 8 |
| 0437 | 9 / 8 | 9 / 8 |
| 0438 | 8 / 7 | 7 / 6 |
| 0439 | 11 / 8 | 11 / 8 |
| 0440 | 9 / 8 | 9 / 8 |
| 0441 | 6 / 5 | 6 / 5 |
| 0442 | 8 / 7 | 8 / 7 |
| 0443 | 8 / 7 | 8 / 7 |
| 0444 | 9 / 8 | 9 / 8 |

## Source fingerprints

SHA-256 of each kernel and final literal sources (full hashes).

| Case | kernel.py | target.literal.rs | main.literal.nr |
|---|---|---|---|
| 0403 | 2e75752a4d50768a1083ad952daa2c4de1d9ef6edee6f79c6215b7ada717140a | 912e12d017d81004c6a6a1ebadce05dcbc7ff58b2577d630fb73e0536523ba7a | f24dacc7f27e1693f214c5a9ae963aa260e8a327ea59af74c9bd0ef0d7c7c128 |
| 0409 | 39b750b005a47c4efd2bf5151dbe97906fa5f699ad090cb4657b04297747f290 | a446caf3ac2c6172723275b1559e41aaef87026fb83ccb0d25b1796701b779de | c94f857e6bd152a56e660aca03a937f9577d96cc8a74d6d1c6f96ae065a9b65f |
| 0412 | 4d3116254d53f743c2380fe849e52546a0ade3caf990a20257babca00d351919 | 8091eb3685b78ce881c78be0991993aff8c82c04a6ee478975428ad1d34db6e2 | 5176a919cc5ee670d57cf17291b72bb46a45298d526d53d33a76b00014128bf3 |
| 0414 | d675567119b16f68f05284b0f89d9130ca1c6025e987ac148889ca9d48f53e5a | 5a3d0345518dfc9633b2623a7c2d828fbf0984b67b268118e30268447dddc791 | c80f5a8e796646090226eab8801c687b695cc06df8dc70e813cba23f3e25e6ca |
| 0415 | a7452331560511210d3709cab22c3444b5ce33b1b134e695b2f33f1922ce64f4 | 2d2699b3d9c58d1cac61aedaf02adba573b115d599adc25d5abed1406a8b5b0f | c867cce4f66eba77fe4de83af54c977a7bb9a89fe07142d5d875a93ee0c2a59f |
| 0416 | daea2da89d3487810280e34f0085f05602f95a4a943b113d2d2e5d0384531de0 | ca7f3f832f8b65f4341cd07f6805f6e47fec164dbe2012469dbccb38da21eee0 | aeb3a5d30f47284c76e8fa6a1f8f91891f78a8638b02b4408ad1740c180c232d |
| 0420 | bf00cc91d37e058c50a65333591ae7ad7d3d09d1becc2220e935d9d03f373929 | absent | absent |
| 0422 | 0140f690b5c7c9da659b3080a67076a800a80c837fc435584b6498b37ed9d416 | 939ead47878d125a628c923126d5444c10add2463ca69263a64cae988093ec18 | ec3446d643a3598a9281339a016ab9b1fc3aed4cfcf7d81b39712c94d1d8a872 |
| 0425 | fdd4e15e7dba8a74b8e1570547501663b40114d0e3d25e42e7b1ba8031a8d786 | d482b00c551c9a4c9d8cf33f3262987df46f3f883752b1e3886bd22300b52a75 | ccb30e5085fe61c1d5270b47a757a1f227e19ebb6e7a950b430b67ad96cfb55c |
| 0426 | d9e21da27a2b11131f36825c3ca475d4d60bff8c978033b0d1d7a778c22cfdab | 6c5f0cb9643a5e23787fda1f70859c630edd21635e690f755ed2cd426a6f7d33 | f94a08930ddbee0ddca5e2524259314cb2e6b02a63fefec77c07e23a687c358a |
| 0428 | da87cfbcbc43c963578ccf4e1efe5437c1c0cea0f73901370c0f4cbccdefba88 | aa2cace8ba0ec6bc0203bb5a964703f9a68d40b0157cd3d992434beada9e9af4 | 51d9b1b386fe69cc82bb64e2bd53daf5a8fb4a8562838af5f5f5cde17ba6796e |
| 0429 | a4f2c5e5b1e6c1c88da66eb0d251411e7496a8c690311727ab84edd6cc632b59 | 612519fdd055b17f97c3eed5d404f2116caaacff1c9b5a220421b08792c26d4f | 7df5a39fe43bde4bdc81f754d5ccb376abef1f26d7328bec874c97ab8dfafb58 |
| 0431 | a58e0b9a1169a950e12616d67bc4fcfd8eb41474fbc2fe4d8af8836f9a68f87e | b4968fa71374db0f1260ca0e5bd140bc23cff2b1295f7494c0746fd7cfe5431c | f9afd9114a6ecbba137e5f14c47e7cd7f63454288f357a43d3b1e1df2463c9b5 |
| 0433 | 3ab240abe4d52a18efcb5d80ce79bfb768e272b61097ba05e6aea5c3fff5a0ed | f495fba6af33eb194659a45f739b475c124671050c101044ece7ae2213741f2f | 39d2a82c1c23124fd931c294b95569160599518d07945c0ae09b61ea5b695945 |
| 0434 | 7a4bd2b0a32b53117a7f4ce7d384573e80c2cc0151483254192b77b0e5eb3fab | da880018309009b34f3993aadf52a6a31c98ba566c44074ee02518ba227ca8a2 | f29e4f6ccbeed1074e0ec9ea10a90c91e2727ec0fc944182310af6aefb2079de |
| 0435 | 1b6fa645ff55b591dc99a68bb5aa83533d334376a3ae9e1456afd91fa25e6e7a | e8317bc9a93cf0346989ff98a930fd588c786f90f3f7a00411a301b7c87626a8 | f94940b3a84a0e232410d7fdba41d96462a6571113b44fdf2e123228dcfbf66d |
| 0436 | 9cbd36af4ac29da63ef613dd3e85861c17806cad4eddc4a1f416b29f819e753a | 2d90dd017b08d84146c84cd038f845904f6a3873660b4f38d8bfa04d1c9d42a7 | cace0b86fbcbab983ab7cdaebd409a941620d786647eb556bf06ee8c989ebd42 |
| 0437 | 3c7b558899bf8510fb7ad16f8bb1768274497af508866848ba0efd67f37b1103 | 31e6eaeaeaca6574ac7799a08139afa2ffd3e778bcb52a511c4636194c498ac7 | 7583ab9142c2ccabb9a1d799e383353e632f235e7861e5a1957ef470c0cd6029 |
| 0438 | 04f84690c7d72be6fe1eb903526acd1f7b3f5ec6d094cdd0e127d178c63a1056 | 1431bf73eb414a00447f40cbeb7ac9b196753d76fd5d2f6d0653448a4646cebc | 966892cd7d32bd9b38c53b7679bc999c102ffb22e9f6c34fd4e666b7b432399d |
| 0439 | ac0e96ffc23f52cfd8d914dd59e2db4c70d9cc9a1ec4215fcec54c8419714fa9 | d2474a3560638425231929bef1d9256131a8cc1e71ba253e953731c1a09900a0 | cc9af3d8302aabeee92c1112f2069f6cfca8308729edd987a7dbe2a357ab56e8 |
| 0440 | 973a7ce2680880c2063da96d35a3af83758f402adc9e4a7cced6ee88d066da2f | 69b13b19252e8b6a1706c219e41d4a98093e250ee4ba677ad600bbc2b566c0aa | f07e8cc748f9a7aa7aa737649a7a0d73529bdef1bbadd81534fdea933115d59d |
| 0441 | 9a87fc99754fb54f06c8e4668abecaca1ce6322792b5ba721e84e999e639ca4c | 5fb762bd74994a585c8181f66a5ebbb51058e2a97a4c0a6cb5db5e68add45ebf | f3f3d620c2628bd9b817c693ec557bac51b4c5a93f17e12ae9dfeeb5d280276c |
| 0442 | 203d0ff788ac2b0c5c37f12fe94f140d72eeef18b5460fb8c9467abc79ad0001 | ab7a67e05ce4d51e95a6b94f63d2f5796048c58e0e71e4a7bbefd94e96123f16 | 67bb430f0a3c420cdf729886e3eb74a11bd3d02264e7dfd8e8e28b9a757de661 |
| 0443 | fd93ca95a23424d73622c9880e800ee3c9e660113a911bc838a087018deaa6fa | 977fc3ba37b0c22a06993d89a5d76690222f0c46cd807a4015179984f695cb88 | 73c6a45de58717a4b726fa4efa72d0f1c71ffb0d756abc6cdfe47d9e68337496 |
| 0444 | e3df526c71b0c7c3b32f7c07da6fa5ed820b2c87cabee8213e5c60734954eae9 | 74700f46271f878bd07b9a3e7519b167960a36e53444be8937fd3eceb1cdecc3 | 72c92ed6496140ea07a78c064f66b3339994a76ae548f4cceb1db4715a350bb5 |
