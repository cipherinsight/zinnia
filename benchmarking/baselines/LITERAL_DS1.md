# Literal ds1 baselines

Branch `literal-ds1`; original fixture shapes; 2026-10-09.

**All 25 literal cases completed functional/domain checks. All 25 optimized checks pass in both systems.**
Fourteen have clean checker passes in both systems; eleven have only the Halo2 policy conflict described below.
All 25 cases have both files. No keys or proofs were generated. Every checker invocation uses `--no-prove`.

## Measurements and verdicts

Halo2 cells are `advice + lookup = total`, excluding fixed/instance cells, at the checker's automatic sizing. Noir counts are unpadded UltraHonk gates with the pinned toolchain. These are not proving-time measurements. `POLICY` means only the checker's redundant-input-check policy failed; constraints, NumPy output comparisons, and input-domain rejection passed. It is not reported as a clean checker pass.

<!-- RESULTS -->
Case labels below abbreviate `ds1000_caseNNNN`. `L<=O` flags every system where the literal count is not larger. Optimized verdicts are all pass/pass.

| Case | Literal Halo2 A + L = total | Literal Noir | Optimized Halo2 A + L = total | Optimized Noir | Literal H / N verdict | L<=O | Notes |
|---|---:|---:|---:|---:|---|---|---|
| 0298 | 4889 + 360 = 5249 | 3370 | 1624 + 180 = 1804 | 3196 | POLICY / pass | - | Reused draft corrected: stable bubble sort, full unique values, inverse ranks, matrix scatter, output packing. |
| 0300 | 22123 + 5023 = 27146 | 6895 | 3331 + 692 = 4023 | 3368 | POLICY / pass | - | Stable bubble sort and linear percentile with separate Q32 weight and product. |
| 0301 | 800 + 240 = 1040 | 2941 | 800 + 240 = 1040 | 2941 | pass / pass | Halo2, Noir | Full reshape as wire aliases. |
| 0309 | 6373 + 1559 = 7932 | 3645 | 4305 + 807 = 5112 | 3495 | POLICY / pass | - | First argmax, index-order scan. |
| 0310 | 6373 + 1559 = 7932 | 3645 | 4305 + 807 = 5112 | 3495 | POLICY / pass | - | First argmin, index-order scan. |
| 0311 | 6391 + 1563 = 7954 | 3666 | 4323 + 811 = 5134 | 3516 | POLICY / pass | - | First argmax then Fortran-order unravel. |
| 0312 | 6391 + 1563 = 7954 | 3670 | 4323 + 811 = 5134 | 3520 | POLICY / pass | - | First argmax then C-order unravel. |
| 0314 | 18290 + 3917 = 22207 | 5404 | 11308 + 1657 = 12965 | 4906 | POLICY / pass | - | Separate argmax, unravel, minimum, tuple scatter, argmax, unravel. |
| 0317 | 27000 + 8000 = 35000 | 9066 | 27000 + 8000 = 35000 | 9066 | pass / pass | Halo2, Noir | Full array copy as wire aliases. |
| 0318 | 10207 + 432 = 10639 | 3519 | 2898 + 432 = 3330 | 3466 | pass / pass | - | Permutation precondition; inverse scatter then column gather. |
| 0319 | 92393 + 3136 = 95529 | 7867 | 21128 + 3136 = 24264 | 7786 | pass / pass | - | Permutation precondition; inverse scatter then plane gather. |
| 0320 | 3961 + 969 = 4930 | 3351 | 3159 + 623 = 3782 | 3259 | POLICY / pass | - | First argmin then C-order unravel. |
| 0321 | 3961 + 969 = 4930 | 3351 | 3159 + 623 = 3782 | 3259 | POLICY / pass | - | First argmax then C-order unravel. |
| 0322 | 1430 + 156 = 1586 | 2935 | 521 + 108 = 629 | 2901 | POLICY / pass | - | Pairwise minimum, full equality mask, prefix count, equality-test compaction. |
| 0323 | 1317 + 340 = 1657 | 2960 | 1317 + 340 = 1657 | 2960 | pass / pass | Halo2, Noir | Integer deg2rad then shared sin. |
| 0324 | 1321 + 340 = 1661 | 2960 | 1321 + 340 = 1661 | 2960 | pass / pass | Halo2, Noir | Integer deg2rad then shared cos. |
| 0325 | 1921 + 459 = 2380 | 3104 | 1921 + 459 = 2380 | 3104 | pass / pass | Halo2, Noir | Separate degree sine, radian sine, fresh comparison, boolean-to-int. |
| 0326 | 1232 + 336 = 1568 | 2986 | 1228 + 336 = 1564 | 2974 | pass / pass | - | Local arcsin arithmetic wrapper; separate degrees conversion. Review interpretation below. |
| 0327 | 467 + 140 = 607 | 2879 | 460 + 140 = 600 | 2879 | pass / pass | Noir | Full constant pad; omit constant public outputs. |
| 0328 | 138 + 42 = 180 | 2835 | 138 + 42 = 180 | 2835 | pass / pass | Halo2, Noir | Constant shape branch; full alias copy then zero tail. |
| 0329 | 7632 + 2112 = 9744 | 4131 | 7632 + 2112 = 9744 | 4131 | pass / pass | Halo2, Noir | Elementwise shared power operation; same square-and-multiply order in Noir. |
| 0334 | 8000 + 2100 = 10100 | 4241 | 6250 + 1700 = 7950 | 4241 | pass / pass | Noir | Full stack, index-order sum, shared float division by 3. |
| 0335 | 14850 + 3700 = 18550 | 4491 | 9700 + 2800 = 12500 | 4504 | POLICY / pass | Noir | Full stack and pairwise max with fresh bounds. |
| 0336 | 320 + 96 = 416 | 2866 | 320 + 96 = 416 | 2866 | pass / pass | Halo2, Noir | Full flipped view, diagonal view. |
| 0338 | 640 + 192 = 832 | 2916 | 640 + 192 = 832 | 2916 | pass / pass | Halo2, Noir | Separate diagonal, flipped view, second diagonal, vstack. |

Automatic Halo2 settings (these differ for some literal/optimized pairs):

| Case | Literal k / lookup bits | Optimized k / lookup bits |
|---|---:|---:|
| 0298 | 8 / 7 | 7 / 6 |
| 0300 | 9 / 8 | 8 / 7 |
| 0301 | 7 / 6 | 7 / 6 |
| 0309 | 9 / 8 | 9 / 8 |
| 0310 | 9 / 8 | 9 / 8 |
| 0311 | 9 / 8 | 9 / 8 |
| 0312 | 9 / 8 | 9 / 8 |
| 0314 | 9 / 8 | 9 / 8 |
| 0317 | 10 / 8 | 10 / 8 |
| 0318 | 9 / 8 | 9 / 8 |
| 0319 | 9 / 8 | 9 / 8 |
| 0320 | 9 / 8 | 8 / 7 |
| 0321 | 9 / 8 | 8 / 7 |
| 0322 | 7 / 6 | 7 / 6 |
| 0323 | 7 / 6 | 7 / 6 |
| 0324 | 7 / 6 | 7 / 6 |
| 0325 | 9 / 8 | 9 / 8 |
| 0326 | 7 / 6 | 7 / 6 |
| 0327 | 6 / 5 | 6 / 5 |
| 0328 | 6 / 5 | 6 / 5 |
| 0329 | 9 / 8 | 9 / 8 |
| 0334 | 9 / 8 | 9 / 8 |
| 0335 | 9 / 8 | 9 / 8 |
| 0336 | 7 / 6 | 7 / 6 |
| 0338 | 7 / 6 | 7 / 6 |
<!-- END RESULTS -->

## Rule interpretations and source audit

1. **Fresh comparison bounds versus input-domain bookkeeping (L5/L6/L7).** Input loaders perform the required once-per-element domain check. Every comparison then takes fresh `q.bound` operands, exactly as L5 requires; no bounded loaders or retained I64/window proofs cross operations. `hw_check.check` still treats these required comparison bounds as redundant input checks and fails eleven Halo2 verdicts: 0298, 0300, 0309, 0310, 0311, 0312, 0314, 0320, 0321, 0322, 0335. The library/checker conflict cannot be fixed within this batch's allowed files. No tricks hide the rechecks, and no shared tool was modified. These cases are not finishable as *clean checker passes* under the simultaneous rules; their complete literal circuits are retained with the exact failure recorded.
2. **Textbook library algorithms (L5).** 0298 uses stable bubble sort carrying original indices, adjacent equality tests, a running distinct count, full equality-test compaction of the unique values, then inverse ranks. The unique-values array is computed even though only its length reaches the output; Noir may eliminate it. 0300 sorts all 20 elements with 190 compare-and-swaps. All arg scans keep the first tie. 0314 completes each source call before the next, including the first unravel and the full private write. 0322 forms the entire equality mask and compacts coordinates by a prefix count and an equality test for every input/output pair. Pairwise reductions scan index order. Local argmax functions implement the library calls, not hoisted work.
3. **Percentile arithmetic (0300).** The textbook default linear percentile uses exact integer `19*p = 100*lo + remainder`, `hi = min(lo+1,19)`, Q32 `weight = remainder/100`, then `s_lo + mul(weight, s_hi-s_lo)`. This deliberately preserves a separate Q32 weight and rounded product rather than the optimized file's single `floor(remainder*diff/100)` expression. Both languages use the same boundaries and check `0 <= p <= 100`. It is a documented L5 interpretation, not a claim of bit-identical internal arithmetic to the optimized interpolation.
4. **Wire-only transformations and domain plumbing (L2/L7).** Reshape, copy, flip, diagonal and stacking are represented explicitly as complete wire/host-view mappings, without value arithmetic outside constraints. All input-dependent arithmetic is in the circuit. In 0328 the constant false shape branch copies all five aliases and overwrites the last two with constants; only the three retained private values are loaded/domain-checked, matching L7 and the optimized statement. In 0336/0338, NumPy flip/diagonal are views: build the complete view mapping but check only its selected inputs, again matching L7. Padding constants are materialized, then omitted from public instances as required. Compiler removal of unused wire aliases is allowed.
5. **Dynamic shape plumbing (0298).** Use fixed-capacity internal arrays; the actual unique count is public. Materialize the padded matrix by the private column scatter, then pack it at the dynamic row stride for the existing `dynamic2` output format. Padding/packing is instance plumbing, not a replacement for the source scatter.
6. **Permutation preconditions and empty storage (0318/0319).** Enforce exactly one occurrence of each legal permutation value, as in the optimized statement. `empty_like` starts as zero storage solely because the precondition guarantees every slot will be overwritten. The inverse permutation is fully built by per-position selects, then used by standard `q.index` / Noir `index_pos` reads. The precondition's equality tests are not reused for the source scatter or gather. A Noir plane read selects its whole row; Halo2 selects each component through the same standard scalar indexing API.
7. **Scalar arithmetic calls (0323–0325, 0329, 0334).** Integer deg2rad is exact integer times Q32 pi/180, consistent with the existing numeric contract; conversion of integer radians to float precedes sin. Boolean-to-int reuses the constrained 0/1 wire. A power call uses shared `pow_const`; Noir's local power-six wrapper uses the same three shared products (`x2`, `x4`, `x2*x4`) inside that one library operation. This is permitted library arithmetic, not CSE across source calls. Mean is an index-order sum followed by shared float division by constant 3, with no program-level cancellation of Q32 scale factors.
8. **Missing arcsin gadget (0326).** Neither shared library has a direct arcsin entry. The local wrapper uses the half-angle definition `2*atan(v/(1+sqrt(1-v*v)))`, the existing case's degree-9 atan approximation coefficients and exact raw-root arithmetic as its numeric contract, and only shared sqrt/div/product/polynomial gadgets. This is an explicit interpretation of the brief's permission to write a missing local helper and the requirement to preserve approximation arithmetic. Unlike the optimized expression, it materializes radians before a separate multiplication by Q32 `180/pi`; it does not fold the factor 2 into `360/pi`. Review this local arithmetic-contract interpretation. Inputs outside [-1,1] are rejected.

Only 0298 was reused from the earlier `WIP (unverified draft)` commit. Its forbidden bounded-loader/range-proof reuse was removed, its omitted unique-values array restored, matrix construction separated from packing, and its explicit `std::as_witness` call removed. All other literal files were authored from the kernels. Source-order and L1–L6 checks were performed case by case; the final scan found no forbidden API calls or hints, and all Noir output-layout declarations exactly match their optimized counterparts.

## Cases no larger than optimized

<!-- COMPARISONS -->
| Cases | System(s) | Explanation |
|---|---|---|
| 0301, 0317, 0328, 0336, 0338 | Both, equal | Reshape/copy/slice/diagonal/view wiring is already direct. Explicit intermediate wire mappings do not add arithmetic gates; domain and output plumbing is identical. |
| 0323, 0324, 0325 | Both, equal | The optimized scalar trigonometric kernels already use the direct shared arithmetic calls. Preserving source order changes no operation count. |
| 0329 | Both, equal | One power operation per element in both variants; L5 requires the shared power gadget, whose repeated-squaring arithmetic is permitted inside that operation. |
| 0327 | Noir equal (2,879); Halo2 larger | Noir eliminates the explicit constant padding array and output extraction. Halo2 retains seven explicit constant assignments (607 total cells versus 600). |
| 0334 | Noir equal (4,241); Halo2 larger | Constant propagation/simplification compiles the shared float division by 3 to the same Noir circuit as the optimized raw constant division. Halo2 retains the direct float-divisor gadget construction (10,100 versus 7,950 cells). |
| 0335 | Noir **smaller** (4,491 versus 4,504); Halo2 larger | Two standard pairwise maxima per column are the required L5 scan. The optimized Noir hint-based `max_of` checks a witnessed maximum via differences/product. It is 13 gates more expensive for this three-value reduction with this compiler, so the optimized alternative is not the cheapest here. The literal source still performs the full scan and no forbidden optimization. |

Every other case is strictly larger in both systems. These flags were generated from the completed JSON measurements; no size was inferred from a previous run. No aggregate compiler-overhead claim is made, particularly where automatic lookup widths differ or a local arithmetic interpretation needs review.
<!-- END COMPARISONS -->

## Reproduction and evidence

Working directory: `benchmarking/baselines`; Python: `../../.venv/bin/python` (NumPy 2.2.2). Builds run from `halo2_hw` so its `.cargo/config.toml` supplies `RUSTC_BOOTSTRAP=1`. `CARGO_BUILD_JOBS=3`, profile `dev-fast`, `MEMRUN_MAX=8G`; literal `CARGO_TARGET_DIR=target-literal` and `HW_VARIANT=literal`; optimized `CARGO_TARGET_DIR=target` and `HW_VARIANT` unset. Builds restrict `HALO2_HW_CASES` to the active group. Checks use `HW_CHECK_LOCK=/home/zhantong/Projects/zinnia-project/eval-redo/literal-work/lock-a` and all work roots/TMPDIR under `/home/zhantong/Projects/zinnia-project/eval-redo/literal-work/ds1/`.

The literal checks are `python -m hw_check.check <cases> --no-prove --halo2-bin halo2_hw/target-literal/dev-fast/halo2_hw --workroot <scratch>/check --out <scratch>/<result>.json`. Optimized checks use the analogous `target/dev-fast/halo2_hw` and `<scratch>/check-optimized`, once per case. Consolidated literal evidence is also saved as `literal-results.json`. JSON evidence files: `check0298.json`, `check-arg.json`, `check-reductions.json`, `check-views.json`, `check0300.json`, `check-perm.json`, `check-arithmetic.json`, `check-optimized.json`. Logs have corresponding names. `lint.log` records **0 errors** for `HW_VARIANT=literal python -m hw_check.lint` on all 25 cases. Headers/output layouts and forbidden APIs were separately scanned; `git diff --check` passed.

No job in this batch hit the 8G memory cap. The sole unresolved completion issue is the eleven Halo2 POLICY verdicts, not missing files or failed arithmetic checks.

Noir: Nargo 1.0.0-beta.20, noirc commit `b4236c1957d0c26cb65d82adc9e5447b6ff1d629`; bb 5.0.0-nightly.20260324, the driver's pinned UltraHonk settings. Halo2 settings (`k`, `lookup_bits`) are recorded with each measurement below/JSON. Auto-sized counts are not a controlled-common-lookup-width comparison.

Verified scope: builds, Halo2 MockProver A/B, NumPy output comparisons on both inputs, Noir compilation/gate counting/witness solving A/B, and the checker's selected out-of-domain input rejection. No proofs, key generation, public-proof tamper tests, timing/scaling runs, Zinnia runs, IR inspection, or Zinnia size inspection. No claim that two fixture inputs establish correctness for all inputs; the source and soundness audit is separate from fixture checks.

Tool issues: (a) the fresh-bound policy conflict above; (b) the first build was accidentally launched from the repository root and hit stable Rust E0554 in `poseidon-primitives`, resolved by rebuilding from the required `halo2_hw` directory; (c) the specified relative Python path emits harmless `sys.prefix/sys.exec_prefix` warnings, and the checker's reference input probing emits the expected invalid-arcsin warning for 0326; (d) final runs waited for the shared lock held by other heavy jobs. Shared tools, optimized sources, fixtures and case sets were not changed.

## Source SHA-256

These identify the exact kernel and literal sources used for this report.

| Case | kernel.py | target.literal.rs | main.literal.nr |
|---|---|---|---|
| 0298 | `5e6a36129dcc02e4a0d9743b2a968c99b130536ca03375252612c440f616f43b` | `5d303f81a2164239d2006a00717c6ede666f88785f3312953b80e2329993faed` | `006cdee5187f94108c1f5b721da5236a00064ace5383cd664cf06014a0cdc302` |
| 0300 | `d950713f0c8ba3d50cdea9c1c440abade6696522004f269867e8c5bdb20aa61e` | `5ca97905a0e73a3f997a5248100735c37a5ce8787680380e5b1c123e23447c0c` | `108392dd1a64bb29fe82bfbf4c86cb756bc1302d94be480e5e518561a7936b0e` |
| 0301 | `7ab6e39cf7dabd9c6138f9da2072e01a539e477669834aaf4866c67c68fde34b` | `44b41db3bc9d8ad8ecb1236774f2682b1cdb0ad1f9bc74ddf516cd59a3886776` | `5c7405a8177f4141a0551892c27024ccf15a5061249527aaa552699c4d603d78` |
| 0309 | `4317cee01bb4dd2b1824c19d140031c0ef82dd47b9baaa8a0da17bf6c596d615` | `5f6c1e0f230f766a214f1a0a74d59b944197452b320f1b24edc6d38a6e887071` | `f702027b2c23af757e5ce10421ce4e9e72c92fcaea9ef23513cfe64e39b61a39` |
| 0310 | `6cffedce8b97b3eb80c91487d5b28d18c56b4f5aaf69f4e2164d4a3484f12f11` | `c90de751d05bb788b81e1c0b9cbbc7255159c76513a30fa5dabc2f09621fdd92` | `049243380259bae9ce2876cbc7084d69fcf681f2b3895e82fdc504be76bffd8a` |
| 0311 | `9d5e19e7f898f74f822b609d99113142b09906bd6a2a5049adb6c143cdfee0ca` | `22b0d8f08f2aa0a2d93b08a8544203b94ebf8ba9c42fa2a1c6ae94f9014ae0d0` | `2915fb955f59bfa9a4f9b945e56d26a20ea18af722e769691c23419f328f6fb2` |
| 0312 | `1eafa0c12ff17bb7dd486c914cf65417ecefbd198b2f88e2adf3924389a8dfc5` | `54574317071f53fa891bca986214e25da7b21612c47423cf0f4a00666b4ebad8` | `ee9a5cf73781ea80fb7fbc4593a68975e058386a4033639882bfcda245dce054` |
| 0314 | `d69f2773df1b1e8da227369eae247635cdedc03c28637198cbfcc3b4e73ea2c7` | `584bb7c12f9c4d019677c344b620989ab6a10a8c4c981fa816a22e0bc88254bf` | `8d7df52c6236325f612db4e59e6e8023251d706b96e4e2b2eb6cc9086c75563b` |
| 0317 | `f6e9b3b36f535834c11000bcd929cf89bfc79c9c95829b9d54aa56e1717c9223` | `9516882c13c1d5dba51a1bedeed277b576d582e1b173da29e8e9e649c87cdf7e` | `8ca721136f607524a48b5b81da988838ec8ac52225d27927c77b6777b0309c7c` |
| 0318 | `f7f4b55b3cde15ba491603328098371c971d42a448d80392c52aed9683908f81` | `599d76c5988ed166f9eb8351d0600a313e58cce1fd656f4d68710afda11171a5` | `1c00ff1b7cf3f0294ef16efd2074a3285a5f9119d9213fc0be84b1a8923cf97c` |
| 0319 | `b2d4c466e6616b44741bc44105321310e700c0d7cc8e50aa8f08095ddcf3edab` | `d59cac717f7c2375d514a18e54a140c049ca3cbb80df1cf19fa587ea05485d60` | `e5ba431bbf927bd30444e01d75b18634ec9ef1a37c7772f2dcac9447777bbde0` |
| 0320 | `321a766782f2465174b05d286d84e44d6cda27f79482d2d2fd7e1b40b739ffed` | `3bbc48e39af78e3fbcfd309cb2371494a7d0c07f04becc70ce7ea878affe089d` | `65056bd6ffc471501ddc1cd3c0b640159ab7a4b4d32db5f18d6dc3e1fe89fdc6` |
| 0321 | `7c2ce43b4c7d7dbade8e21c80a5e2da8ee58fb98f9cb0ff56534d87f3ebc213a` | `0b30dbf66c58b867a3ff389f47df5ba118a03c5a55f4d55f3cef27825bdd68f1` | `3b38b3450c8760ab293320662373cbbf59b81b82df2b9d6ff91404cf4a9e1faa` |
| 0322 | `da61ae098a11c2cabbdb6cb5661a238ce018ad9352d0a69cd82dbf9d608b942a` | `baa70bf237c349464ef3019a7d68c90d511b314e3a2b52750be83a9f54782741` | `1f1d9bd1bc6aa9687a616a37ab9c753ab943acf7dee964b5e5aae619837f201c` |
| 0323 | `03ac48c87dec9d8170704bf2f77935d195be9fdbb05be22d17ec2c5a6e9b65b3` | `07b6549868a70ae195092ad30bb747facff7aa7ebf17927f4102e57f309dbbe6` | `24e328751e3ad57e7ccd2f404af6115bfc99c2cf3a85bdc3a35309d5d45170c6` |
| 0324 | `a0424c49f059917230b2d9af4bea348ab2fff44acb81bfbd95c99548bdf0b624` | `0acdbae5c5f9b108bb923cabc2b20dd07a219cff3f454b2c069460eea887a0b5` | `fbbb6972bc5a9e2c3c87adec6f0284889aae851fc56a0e2ab4cdbacd2483b706` |
| 0325 | `f7ca3368117d97c0024a866abcca47ae4322aa9621c854e2e5f5426e1cf3b0ff` | `dcd49284963fc89a52e2dbaefb27d0f5b9840b39d8fddad29b3393c451ca6439` | `c7bea4b862a0df57a009fb3306f5d027e6808f20c87e58228f87cf37d2d01f0b` |
| 0326 | `a16482fa497533709d2066f46a316aeafcfa1f13443e1c2afd9bfd58e3911d17` | `0e873905e7f0da11f5ba2d2cb13de02c79b3e6acbcd65487b08ecf12702eb165` | `df1794ef5f9add02b539c3dfcffa9f044c22912322e6a161ec2b5081cf92cab9` |
| 0327 | `df092bb76131f459d67f531cbbff5c7232a46daf1263e8f2387259ae4c5c4be4` | `a01cd83b460432953a62eba42629afeb012bda67f67f9d2fbcc19ec1a56f0277` | `826dc76aace40e386936078dec6ebc8ef0141422a83628195944246d3b7b447d` |
| 0328 | `e32664ec39ea1fab00c367056bc8da495999ae516cbef0f8264c87e19e80ffbf` | `895b559302ae9174f83e1e934017ad0a76d5aada3ba3fdc18f8854e2e98d3de5` | `39f4e68ad524895694e65a326a93a1fdc9b980d88a0621c0b78d82c85f13d178` |
| 0329 | `5521337a6233c2aaa5855ee0c4273eba6ff1e56a5bb4e8cf64bcd01c3b12062f` | `9c5ae8d260346aaef24eb68afa743887020cce2e0a1fae2d25eb22a3069c965a` | `bd33bdfb1fb52c5c22d51509e2106436ed2bb01abadcdadca58aaa5b90935d59` |
| 0334 | `9b4afa0d4bba61d0424578ca46ab18e7055520088502eeae6da40cef76e5f8af` | `8a8136160e1423bc2914f80dd00d8cda36a5b5075ed398b173670c48280e9541` | `90529f1677ec9313a39d26e1078c371a3b82b5b0e61087b4aa22b3047e49faa0` |
| 0335 | `345a59619d5dcb0f35422f651f3bdfe0bcfa1165c6f77787142fc8fe94787b3e` | `e9367e29e18a8056949903b8a4f619549b88a08711962b7e9069b0f856a80076` | `2322d24a83f91b68f31e3501abbe6c41a1e374d74e80234e816b82ba5c044041` |
| 0336 | `480386f65b0eab59ff8f4535783b73855539bf67e696c347908d191b7ca08f9f` | `c2e72b6652d7c2acf150d55951890a25607aae7fdff8cc05cd397b9dd270f189` | `efff45258542af3dde7742226f72073bf63aae92800478e7bf7b2fc1e18bc231` |
| 0338 | `c1396c22c82b140863571d2960e8a74977d537eb78d5801f7d9b06ec6b6ecea8` | `094221251918fb9bab442740eeec35ce30b7f26d9466afd76973d570f972166f` | `45752c738a487a81b3ecc0146f2292b88af1415f4cfd8b155f19582fe9e67ab2` |
