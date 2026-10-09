# Literal DS3 baselines

Work in progress. All checks use `--no-prove`, the 8 GiB cap and assigned lock.

| Case | Literal Halo2 advice + lookup | Literal Noir gates | Optimized Halo2 | Optimized Noir | Verdict / notes |
|---|---:|---:|---:|---:|---|
| 0403 | 306 + 80 = 386 | 58 | not run | not run | Both pass. Reused; replaced direct witness loading with standard integer loader (L6); C-order defining sum retained. Dimension checks remain semantic index checks. |

No proofs, key generation, timing/scaling runs or Zinnia inspection performed.
