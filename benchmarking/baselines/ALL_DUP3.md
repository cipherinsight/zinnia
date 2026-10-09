# Batch dup3: complete

All eight cases, all 32 circuits, pass. No unfinished cases.
Each case was committed separately on `all-dup3`. Sources were audited against
its own `kernel.py` and L1-L7; no Zinnia compilation, IR, emitted code, or sizes
were used.

## Files and no-proof checks

Paths below are relative to this report. OH/ON are optimized Halo2/Noir;
LH/LN are literal Halo2/Noir. Every linked file passed its corresponding check.

| Case | OH | ON | LH | LN |
|---|---|---|---|---|
| 0417 | [target.rs](../datasets/ds1000/ds1000_case0417/target.rs): pass | [main.nr](../datasets/ds1000/ds1000_case0417/main.nr): pass | [target.literal.rs](../datasets/ds1000/ds1000_case0417/target.literal.rs): pass | [main.literal.nr](../datasets/ds1000/ds1000_case0417/main.literal.nr): pass |
| 0418 | [target.rs](../datasets/ds1000/ds1000_case0418/target.rs): pass | [main.nr](../datasets/ds1000/ds1000_case0418/main.nr): pass | [target.literal.rs](../datasets/ds1000/ds1000_case0418/target.literal.rs): pass | [main.literal.nr](../datasets/ds1000/ds1000_case0418/main.literal.nr): pass |
| 0419 | [target.rs](../datasets/ds1000/ds1000_case0419/target.rs): pass | [main.nr](../datasets/ds1000/ds1000_case0419/main.nr): pass | [target.literal.rs](../datasets/ds1000/ds1000_case0419/target.literal.rs): pass | [main.literal.nr](../datasets/ds1000/ds1000_case0419/main.literal.nr): pass |
| 0430 | [target.rs](../datasets/ds1000/ds1000_case0430/target.rs): pass | [main.nr](../datasets/ds1000/ds1000_case0430/main.nr): pass | [target.literal.rs](../datasets/ds1000/ds1000_case0430/target.literal.rs): pass | [main.literal.nr](../datasets/ds1000/ds1000_case0430/main.literal.nr): pass |
| 0448 | [target.rs](../datasets/ds1000/ds1000_case0448/target.rs): pass | [main.nr](../datasets/ds1000/ds1000_case0448/main.nr): pass | [target.literal.rs](../datasets/ds1000/ds1000_case0448/target.literal.rs): pass | [main.literal.nr](../datasets/ds1000/ds1000_case0448/main.literal.nr): pass |
| 0449 | [target.rs](../datasets/ds1000/ds1000_case0449/target.rs): pass | [main.nr](../datasets/ds1000/ds1000_case0449/main.nr): pass | [target.literal.rs](../datasets/ds1000/ds1000_case0449/target.literal.rs): pass | [main.literal.nr](../datasets/ds1000/ds1000_case0449/main.literal.nr): pass |
| 0463 | [target.rs](../datasets/ds1000/ds1000_case0463/target.rs): pass | [main.nr](../datasets/ds1000/ds1000_case0463/main.nr): pass | [target.literal.rs](../datasets/ds1000/ds1000_case0463/target.literal.rs): pass | [main.literal.nr](../datasets/ds1000/ds1000_case0463/main.literal.nr): pass |
| 0502 | [target.rs](../datasets/ds1000/ds1000_case0502/target.rs): pass | [main.nr](../datasets/ds1000/ds1000_case0502/main.nr): pass | [target.literal.rs](../datasets/ds1000/ds1000_case0502/target.literal.rs): pass | [main.literal.nr](../datasets/ds1000/ds1000_case0502/main.literal.nr): pass |

`hw_check.check --no-prove` verified Halo2 MockProver on A/B, Noir compilation,
gate counting and witness execution on A/B, decoded outputs against NumPy,
and rejection of the checker's selected out-of-domain input for each backend
and variant. Optimized Halo2 reports zero redundant input checks. Both variants
passed `hw_check.lint` on all eight cases (zero errors).

No proof generation, key generation, proof verification, tampered-public
verification, benchmark timing, or scaling runs were performed. Fixture checks
and source audit do not establish exhaustive full-domain equivalence.

## Counts from the no-proof checks (automatic Halo2 sizing)

Halo2 numbers are advice + lookup = total; fixed and instance cells are excluded.
Noir numbers are unpadded UltraHonk gates with the pinned toolchain/settings.
`k/lb` gives automatic k and lookup bit width, optimized then literal.

| Case | Optimized Halo2 | Literal Halo2 | Optimized Noir | Literal Noir | k/lb opt; literal |
|---|---:|---:|---:|---:|---|
| 0417 | 494 + 144 = 638 | 508 + 144 = 652 | 2875 | 2875 | 6/5; 6/5 |
| 0418 | 2928 + 744 = 3672 | 3048 + 744 = 3792 | 3218 | 3218 | 8/7; 8/7 |
| 0419 | 2928 + 744 = 3672 | 3048 + 744 = 3792 | 3218 | 3218 | 8/7; 8/7 |
| 0430 | 1804 + 443 = 2247 | 2956 + 462 = 3418 | 3086 | 3086 | 7/6; 8/7 |
| 0448 | 960 + 288 = 1248 | 960 + 288 = 1248 | 2966 | 2966 | 7/6; 7/6 |
| 0449 | 960 + 288 = 1248 | 960 + 288 = 1248 | 2966 | 2966 | 7/6; 7/6 |
| 0463 | 3589 + 200 = 3789 | 11126 + 200 = 11326 | 3650 | 5096 | 8/7; 8/7 |
| 0502 | 22308 + 4056 = 26364 | 22308 + 4056 = 26364 | 6661 | 7759 | 9/8; 9/8 |

For a controlled lookup-width comparison, additional count-only `stats` calls
used original shapes, `--k 10 --lookup-bits 8` for both variants of every case.
These are counts only; the MockProver runs above used automatic sizing.

| Case | Optimized Halo2, 8-bit lookup | Literal Halo2, 8-bit lookup |
|---|---:|---:|
| 0417 | 304 + 84 = 388 | 318 + 84 = 402 |
| 0418 | 2208 + 600 = 2808 | 2328 + 600 = 2928 |
| 0419 | 2208 + 600 = 2808 | 2328 + 600 = 2928 |
| 0430 | 1330 + 297 = 1627 | 2524 + 374 = 2898 |
| 0448 | 648 + 192 = 840 | 648 + 192 = 840 |
| 0449 | 648 + 192 = 840 | 648 + 192 = 840 |
| 0463 | 3389 + 160 = 3549 | 10926 + 160 = 11086 |
| 0502 | 22308 + 4056 = 26364 | 22308 + 4056 = 26364 |

## Reuse and rule interpretations

- **0417:** Adapted all four 0414 twins by wiring reversed input before binning. Literal means scan each bin in source order; the optimized Halo2 sum uses the library accumulator. Reversal, slice and reshape are wires.
- **0418:** Adapted 0416: reverse all six columns of each input row before pairwise means; all 24 means remain in reversed bin order.
- **0419:** Adapted 0416: reverse input columns, compute all pairwise means, then reverse the three output bins per row.
- **0430:** Adapted 0428 (identical kernel). Literal independently computes both mean/std occurrences, with full deviations and raw squares; one rescale per squared-deviation reduction preserves twin arithmetic (literal section 3). Optimized Noir now uses the shared constant-divisor gadget for 30 instead of divmod_pos. Noir compiler CSE leaves equal counts.
- **0448:** Reused 0361 copy-only loading/output plumbing for 24 cells. Changed to two 3x4 parameters and interleaved x/y output of shape 3x4x2. Literal explicitly wires both transposes, vstack and final transpose. Both kinds have equal counts in both backends.
- **0449:** Same adaptation of 0361 as 0448, with its own standalone files and its own fixture. Both kinds have equal counts in both backends.
- **0463:** Adapted 0462: Halo2 input shape is 20x1; Noir flat ABI holds the same 20 cells. Final reshape(-1,1) is wiring. Public dynamic count denotes rows of unit width one, followed by up to 20 values padded with zeros. Literal retains complete masks and equality-per-pair compaction.
- **0502:** Adapted 0501 (identical kernel): NumPy negative-index wrap and bounds through standard gadgets. Literal Halo2 q.index is the same implementation as optimized (equal counts); literal Noir uses index_pos/native indexing, while optimized uses pick for the length-two axis.

Shared Q32 division/rescale/sqrt gadgets and acceptance guards are identical
between each case's variants. In 0430, literal section 3 requires a single
rescale of the full raw squared-deviation sum, matching the twin; it does not
permit merging the two std calls. The rescaled squared-deviation sum and
quotients retain the twin's signed-window guards; this is the inherited Q32
acceptance domain, not an assertion that all finite binary64 std computations
are accepted. Multiplication by integer 3 is exact in both variants. Constant
30 uses `floor_div_const` in both Noir variants, improving the optimized twin's
choice without changing its rounding or domain.

Fixed transposes, reshapes, slices and concatenations are wiring, including
literal intermediate arrays. 0463 has one public row count and unit-width
padding, rather than an extra public constant column dimension. Noir's flat
input ABI is row-major and matches its declared 20x1 kernel input. 0502 uses
standard library indexing with negative wrap and bounds: L5 allows `q.index`
for literal Halo2 even though the library internally chooses one-hot indexing.
All source-defined input domain checks are retained; no fixture-specific guards.

**Literal not larger:** 0417, 0418, 0419 and 0430 have equal Noir counts.
The mean kernels are already direct; Noir simplifies zero-initialized sums.
0430's separately written identical calls are merged by Noir's compiler (allowed
by LITERAL_BASELINES.md). 0448 and 0449 have equal counts in both backends because
all work is wire copies plus the same domain checks. 0502 has equal Halo2 counts
because both variants call the same standard indexing gadget once per output.
No literal count is smaller, under either automatic or common-width sizing.

## Environment, evidence and fixture issue

Built in `benchmarking/baselines/halo2_hw` with `--profile dev-fast`,
`CARGO_BUILD_JOBS=3`, `MEMRUN_MAX=8G` and `eval-redo/tools/memrun.sh`.
Optimized: `HW_VARIANT` unset, `CARGO_TARGET_DIR=target`.
Literal: `HW_VARIANT=literal`, `CARGO_TARGET_DIR=target-literal`.
Checks ran from `benchmarking/baselines` using this worktree's `.venv`;
`HW_CHECK_LOCK=/home/zhantong/Projects/zinnia-project/eval-redo/all-work/lock-b`.
Work directories and `TMPDIR` are under the evidence directory below.

All duplicate case directories lacked `fixture.json`, so the default first
check reported a missing fixture. Checks were then run with `--fixture-root`
pointing at staged copies of the committed acceptance fixtures in
`benchmarking/datasets/ds1000_numpy/generated/caseNNNN/fixture.json`.
For 0417/0418/0419, only `bin_size` moved from parameter/input lists to constants
(4/2/2), matching the performance kernels; all array values are unchanged.
Other fixture schemas and values are unchanged apart from case names/notes.
No repository fixtures or shared tools were modified. Integration must provide
these fixtures in the performance folders or keep using the override.

The relative `.venv` executable produced harmless Python prefix warnings;
subsequent checks used its absolute path. An initial batch build invoked from
the repository root missed the crate-local Cargo configuration and failed on
a nightly feature; rebuilding from the required crate directory passed.
There are no unresolved circuit or checker failures.

Evidence directory:
`/home/zhantong/Projects/zinnia-project/eval-redo/all-work/dup3/`.
Raw checks: `0417-opt.json`, `0417-literal.json`, `rest-opt.json`,
`rest-literal.json`. Common-width counts: `common-width-stats.json`.
Staged fixtures: `fixtures/ds1000_caseNNNN/fixture.json`.
Build logs: `build-0417-opt.log`, `build-0417-lit.log`, `build-opt.log`,
`build-literal.log`. Check commands used `--no-prove`, the corresponding
`--halo2-bin halo2_hw/target[-literal]/dev-fast/halo2_hw`, the staged fixture
root, and work roots `opt/` or `literal/` in that directory.

## Source identity

`source-hashes.json` in the evidence directory records SHA-256 for each kernel,
each of the four circuit files, original acceptance fixture, and staged fixture.
Below is a per-case SHA-256 of the concatenated ordered `filename + NUL + bytes`
for `kernel.py`, `target.rs`, `main.nr`, `target.literal.rs`, `main.literal.nr`.

| Case | Source bundle SHA-256 |
|---|---|
| 0417 | `93e26cfb41a3b2a9a54cbaf195c04f3444d6584de1b70abc2da6cbb5a89e6237` |
| 0418 | `1861e382fb1dc1e2284b3e9a8921c7eca4f6e8bc0060ce708ca663e4dc4e1f25` |
| 0419 | `075f11f1f122ac16f485c7477dc412aec3c314a8c3e6dfa14cbff53739448b0f` |
| 0430 | `acf9b52a3100ac3814a5193c9e67d73d53cfef66b89af25ef9aef5309da1aaf8` |
| 0448 | `17e43943188fae6388069c74bbd3fd64cb429f29990e16936c321c4453062ac8` |
| 0449 | `d7c89718c80772a9c231c05716eec77dbc9c686e33dd378e1de3fba7ecf16f42` |
| 0463 | `b56649257ec98d9778daa3b56c2489578b7280ebc4b622b5f465aa6ba6f88263` |
| 0502 | `0b4c5f4682f374975fde73db53422fb98717571507e98dcd43d95f4c811071db` |
