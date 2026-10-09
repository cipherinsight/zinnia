# dup1 handwritten baselines

All nine assigned cases are complete. Each of the 36 circuits passed its own
`hw_check.check --no-prove` run using the case-specific A/B fixtures described
below. Both optimized and literal static lint report **0 errors**.

## Files and validation

Paths are relative to this report. O = optimized, L = literal.
Each linked file has the verdict shown.

| Case | Optimized Halo2 | Optimized Noir | Literal Halo2 | Literal Noir |
|---|---|---|---|---|
| 0302 | [pass: target.rs](../datasets/ds1000/ds1000_case0302/target.rs) | [pass: main.nr](../datasets/ds1000/ds1000_case0302/main.nr) | [pass: target.literal.rs](../datasets/ds1000/ds1000_case0302/target.literal.rs) | [pass: main.literal.nr](../datasets/ds1000/ds1000_case0302/main.literal.nr) |
| 0303 | [pass: target.rs](../datasets/ds1000/ds1000_case0303/target.rs) | [pass: main.nr](../datasets/ds1000/ds1000_case0303/main.nr) | [pass: target.literal.rs](../datasets/ds1000/ds1000_case0303/target.literal.rs) | [pass: main.literal.nr](../datasets/ds1000/ds1000_case0303/main.literal.nr) |
| 0304 | [pass: target.rs](../datasets/ds1000/ds1000_case0304/target.rs) | [pass: main.nr](../datasets/ds1000/ds1000_case0304/main.nr) | [pass: target.literal.rs](../datasets/ds1000/ds1000_case0304/target.literal.rs) | [pass: main.literal.nr](../datasets/ds1000/ds1000_case0304/main.literal.nr) |
| 0313 | [pass: target.rs](../datasets/ds1000/ds1000_case0313/target.rs) | [pass: main.nr](../datasets/ds1000/ds1000_case0313/main.nr) | [pass: target.literal.rs](../datasets/ds1000/ds1000_case0313/target.literal.rs) | [pass: main.literal.nr](../datasets/ds1000/ds1000_case0313/main.literal.nr) |
| 0330 | [pass: target.rs](../datasets/ds1000/ds1000_case0330/target.rs) | [pass: main.nr](../datasets/ds1000/ds1000_case0330/main.nr) | [pass: target.literal.rs](../datasets/ds1000/ds1000_case0330/target.literal.rs) | [pass: main.literal.nr](../datasets/ds1000/ds1000_case0330/main.literal.nr) |
| 0337 | [pass: target.rs](../datasets/ds1000/ds1000_case0337/target.rs) | [pass: main.nr](../datasets/ds1000/ds1000_case0337/main.nr) | [pass: target.literal.rs](../datasets/ds1000/ds1000_case0337/target.literal.rs) | [pass: main.literal.nr](../datasets/ds1000/ds1000_case0337/main.literal.nr) |
| 0339 | [pass: target.rs](../datasets/ds1000/ds1000_case0339/target.rs) | [pass: main.nr](../datasets/ds1000/ds1000_case0339/main.nr) | [pass: target.literal.rs](../datasets/ds1000/ds1000_case0339/target.literal.rs) | [pass: main.literal.nr](../datasets/ds1000/ds1000_case0339/main.literal.nr) |
| 0341 | [pass: target.rs](../datasets/ds1000/ds1000_case0341/target.rs) | [pass: main.nr](../datasets/ds1000/ds1000_case0341/main.nr) | [pass: target.literal.rs](../datasets/ds1000/ds1000_case0341/target.literal.rs) | [pass: main.literal.nr](../datasets/ds1000/ds1000_case0341/main.literal.nr) |
| 0342 | [pass: target.rs](../datasets/ds1000/ds1000_case0342/target.rs) | [pass: main.nr](../datasets/ds1000/ds1000_case0342/main.nr) | [pass: target.literal.rs](../datasets/ds1000/ds1000_case0342/target.literal.rs) | [pass: main.literal.nr](../datasets/ds1000/ds1000_case0342/main.literal.nr) |

## Sizes

Halo2 entries show advice + lookup = total cells. Noir entries are unpadded
UltraHonk gates, from the same no-prove checks. Automatic Halo2 sizing was
used; these are not timings or controlled common-lookup-width comparisons.

| Case | Optimized Halo2 | Literal Halo2 | Optimized Noir | Literal Noir | Halo2 k (O/L) |
|---|---:|---:|---:|---:|---|
| 0302 | 800 + 240 = 1040 | 800 + 240 = 1040 | 2941 | 2941 | 7/7 |
| 0303 | 800 + 240 = 1040 | 800 + 240 = 1040 | 2941 | 2941 | 7/7 |
| 0304 | 800 + 240 = 1040 | 800 + 240 = 1040 | 2941 | 2941 | 7/7 |
| 0313 | 4289 + 807 = 5096 | 6373 + 1559 = 7932 | 3495 | 3645 | 9/9 |
| 0330 | 7632 + 2112 = 9744 | 7632 + 2112 = 9744 | 4131 | 4131 | 9/9 |
| 0337 | 320 + 96 = 416 | 320 + 96 = 416 | 2866 | 2866 | 7/7 |
| 0339 | 640 + 192 = 832 | 640 + 192 = 832 | 2916 | 2916 | 7/7 |
| 0341 | 1200 + 360 = 1560 | 1200 + 360 = 1560 | 3004 | 3004 | 7/7 |
| 0342 | 1200 + 360 = 1560 | 1200 + 360 = 1560 | 3004 | 3004 | 7/7 |

## Twin reuse and source audit

All four twin files supplied the initial plumbing and backend API usage.
Each literal computation was reviewed against its own kernel under L1-L7.
No twin files, shared gadgets, tools, or repository fixtures were modified.

| Case | Twin | Adaptation and interpretation |
|---|---|---|
| 0302 | 0301 | Reshape to 5 rows and 4 columns; unchanged row-major output order. |
| 0303 | 0301 | Accept 23 inputs; select the first 20, then reshape to 4 by 5. Do not check the discarded last three inputs. |
| 0304 | 0301 | Accept 23 inputs; slice the last 20, reverse, then reshape to 4 by 5. Output A[22] through A[3]. |
| 0313 | 0309 | Same flat first-maximum argmax. Removed stale unravel comments. Optimized Halo2 omits the unused final maximum-value update; literal retains every scan step. |
| 0330 | 0329 | Same exponent 6, same 8 by 6 input and output. Literal assigns the powered array to result, matching this kernel. |
| 0337 | 0336 | Change shape to 8 by 10, row stride to 10, and anti-diagonal column to 9-i. Literal fliplr materializes all 80 aliases before diag. |
| 0339 | 0338 | Change shape to 8 by 10; crop to the leading 8 by 8 before extracting diagonal and anti-diagonal. Keep stride 10, anti column 7-i, diagonal row before anti-diagonal row. |
| 0341 | 0340 | Same 5 by 6 integer flatten. Literal explicitly appends/copies all 30 values in row-major order. |
| 0342 | 0340 | Same 5 by 6 integer flatten; only whitespace differs in the source kernel. Literal explicitly appends/copies all 30 values. |

Views, reshapes, slices, flips, diagonals, stacking, and copies are fixed wire
aliases, not arithmetic. Literal code forms all source intermediate views;
only the selected/read input cells receive domain checks (CONVENTIONS 1.6,
L7, as in the twins). Host-side selection in Halo2 uses only constant indices
and copies the input values into checked private cells; it is not a computed
hint. Shapes and `min(a.shape)` are compile-time constants.

0313 uses a strict greater-than scan, so ties keep the first row-major index.
Optimized Halo2 carries I64 bounds; literal bounds both operands at each
comparison. Noir literal calls `gt` afresh, with no source-level proof reuse.

0330 uses `pow_const` and the equivalent Noir shared `mul` sequence:
x2 = mul(x,x), x4 = mul(x2,x2), result = mul(x2,x4). Every product has the same
floor rescale and signed-window check in all four files. Repeated squaring
inside a power operation follows L5 and section 3 of LITERAL_BASELINES;
it does not replace the power with six differently rounded products.

## Equal-size cases

- 0302: equal in Halo2 and Noir. The source only moves wires; literal aliases/copy loops add no constraints.
- 0303: equal in Halo2 and Noir. The source only moves wires; literal aliases/copy loops add no constraints.
- 0304: equal in Halo2 and Noir. The source only moves wires; literal aliases/copy loops add no constraints.
- 0330: equal in Halo2 and Noir. The same three shared Q32 products implement the single source power operation.
- 0337: equal in Halo2 and Noir. The source only moves wires; literal aliases/copy loops add no constraints.
- 0339: equal in Halo2 and Noir. The source only moves wires; literal aliases/copy loops add no constraints.
- 0341: equal in Halo2 and Noir. The source only moves wires; literal aliases/copy loops add no constraints.
- 0342: equal in Halo2 and Noir. The source only moves wires; literal aliases/copy loops add no constraints.

No literal circuit is smaller than its optimized counterpart.

## Validation evidence and limits

Built with `CARGO_BUILD_JOBS=3`, `--profile dev-fast`, under `memrun.sh` with
`MEMRUN_MAX=8G`. Optimized uses `HW_VARIANT` unset and `CARGO_TARGET_DIR=target`;
literal uses `HW_VARIANT=literal` and `CARGO_TARGET_DIR=target-literal`.
Checks use the assigned `HW_CHECK_LOCK=.../all-work/lock-c` and TMPDIR/workdirs
under `/home/zhantong/Projects/zinnia-project/eval-redo/all-work/dup1/`.

The checked-in duplicate folders lack `fixture.json`; the first attempted
0302 check reported this and did not validate any circuit. All successful
checks use `--fixture-root` pointing to the external `dup1/fixtures` directory.
`dup1/generate_fixtures.py` reproduces the fixtures with the repository's
`perf_fixtures.build_fixture`, using each case's own record, signature,
constants, kernel, and seeds `1000 * case + 1` and `1000 * case + 2`. Existing draws cover 0302,
0303, 0304, 0337, and 0339. For 0313/0330/0341/0342, which have no draw registered,
the script registers the exact twin's draw in memory before generating with
the new case's seeds. No fixture generator source is modified. Generation
checks shapes, domains, differing A/B outputs, and NumPy execution.

Raw successful checks in that external directory:
`0302-opt.json`, `0302-lit.json`, `remaining-opt.json`, `remaining-lit.json`.
To reproduce, generate those external fixtures, build each variant with
`HALO2_HW_CASES=0302,0303,0304,0313,0330,0337,0339,0341,0342`, then run from
`benchmarking/baselines`:

```sh
../../.venv/bin/python -m hw_check.check 0302 0303 0304 0313 0330 0337 0339 0341 0342 \
  --no-prove --fixture-root /home/zhantong/Projects/zinnia-project/eval-redo/all-work/dup1/fixtures \
  --halo2-bin halo2_hw/target/dev-fast/halo2_hw \
  --workroot /home/zhantong/Projects/zinnia-project/eval-redo/all-work/dup1/optimized
```

For literal, set `HW_VARIANT=literal`, use the `target-literal` binary and
`dup1/literal` workroot. Apply the environment/lock settings above.

Validation covers Halo2 MockProver A/B, Noir compilation/counting and A/B
witness solving, output comparison with NumPy, and the checker's out-of-domain
input rejection for each file. Optimized Halo2 reports no redundant input
checks. These checks and source review do not constitute exhaustive testing
of the input domain. No proofs, key generation, public-tamper proof tests,
benchmark timing, scaling, or Zinnia compilation/IR/output/size inspection.

Nothing remains unfinished. Tool limitation: repository fixtures for these
former duplicates are missing; validation uses the documented external
fixtures. The shared lock queued checks behind other batches, without bypass.
