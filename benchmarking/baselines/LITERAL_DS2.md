# DS2 literal baselines

In progress. All checks use `--no-prove`; no keys or proofs are generated.

| Case | Literal halo2 advice + lookup = total | Literal Noir gates | Optimized halo2 | Optimized Noir | Verification / notes |
|---|---:|---:|---:|---:|---|
| 0340 | 1200 + 360 = 1560 | 3004 | not run | not run | Both pass A/B and input-domain checks. Reused unchanged after line-by-line audit; flattening is a row-major wire copy. |
