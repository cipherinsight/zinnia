# Batch dup3

In progress. Each completed case has four standalone files under
`benchmarking/datasets/ds1000/ds1000_caseNNNN/`: optimized `target.rs` and
`main.nr`, literal `target.literal.rs` and `main.literal.nr`.

Checks use `hw_check.check --no-prove`, dev-fast builds under memrun's 8 GiB
cap, three Cargo jobs, and `HW_CHECK_LOCK=eval-redo/all-work/lock-b`.
Evidence and staged fixtures: `eval-redo/all-work/dup3/`.
The duplicate folders lack fixtures; checks use the corresponding committed
acceptance fixtures from `datasets/ds1000_numpy/generated/caseNNNN/fixture.json`
via `--fixture-root`. For 0417/0418/0419 only, `bin_size` is moved from the
parameter/input lists to constants (4/2/2), matching these performance kernels.
Fixture values are unchanged. No repository fixtures or tools are modified.

| Case | Four-file check status | Opt Halo2 advice + lookup = total | Literal Halo2 advice + lookup = total | Opt Noir gates | Literal Noir gates |
|---|---|---:|---:|---:|---:|
| 0417 | all pass | 494 + 144 = 638 | 508 + 144 = 652 | 2875 | 2875 |
| 0418 | all pass | 2928 + 744 = 3672 | 3048 + 744 = 3792 | 3218 | 3218 |
| 0419 | all pass | 2928 + 744 = 3672 | 3048 + 744 = 3792 | 3218 | 3218 |
| 0430 | all pass | 1804 + 443 = 2247 | 2956 + 462 = 3418 | 3086 | 3086 |
| 0448 | all pass | 960 + 288 = 1248 | 960 + 288 = 1248 | 2966 | 2966 |
| 0449 | all pass | 960 + 288 = 1248 | 960 + 288 = 1248 | 2966 | 2966 |
| 0463 | all pass | 3589 + 200 = 3789 | 11126 + 200 = 11326 | 3650 | 5096 |

0417: adapted all four 0414 twins by wiring the reversed input before binning.
Literal means scan each bin in source order. Noir counts are equal because the
compiler removes the literal accumulator's zero initialization; the arithmetic
is already direct. No added arithmetic rounding points or domain guards.

Verified for completed rows: Halo2 MockProver A/B, Noir compilation/count and
witness solving A/B, decoded outputs against NumPy, out-of-domain input rejection.
No proofs, keys, tampered-public verification, timing or scaling runs.

0418: Adapted 0416: reverse all six columns of each input row before pairwise means; all 24 means remain in reversed bin order.

0419: Adapted 0416: reverse input columns, compute all pairwise means, then reverse the three output bins per row.

0430: Adapted 0428 (identical kernel). Literal independently computes both mean/std occurrences, with full deviations and raw squares; one rescale per squared-deviation reduction preserves twin arithmetic (literal section 3). Optimized Noir now uses the shared constant-divisor gadget for 30 instead of divmod_pos. Noir compiler CSE leaves equal counts.

0448: Reused 0361 copy-only loading/output plumbing for 24 cells. Changed to two 3x4 parameters and interleaved x/y output of shape 3x4x2. Literal explicitly wires both transposes, vstack and final transpose. Both kinds have equal counts in both backends.

0449: Same adaptation of 0361 as 0448, with its own standalone files and its own fixture. Both kinds have equal counts in both backends.

0463: Adapted 0462: Halo2 input shape is 20x1; Noir flat ABI holds the same 20 cells. Final reshape(-1,1) is wiring. Public dynamic count denotes rows of unit width one, followed by up to 20 values padded with zeros. Literal retains complete masks and equality-per-pair compaction.
