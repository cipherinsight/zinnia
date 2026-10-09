# Literal npb4

In progress. All checks use `--no-prove`, the 8G memory cap and lock-b.

| Case | Literal Halo2 advice + lookup = total | Literal Noir gates | Optimized Halo2 / Noir | Verdict and review |
|---|---:|---:|---|---|
| npbench_polybench_atax_atax__kernel | 8485 + 2140 = 10625 | 4397 | not run | Both pass A/B and domain rejection. Reused unchanged after L1-L7 review: two left-associated matrix products, full intermediate, one rescale per dot. |

Build environment: stable Rust needs `RUSTC_BOOTSTRAP=1` for the dependency's nightly feature; initial build without it failed (E0554), then succeeded. No dependency or tool files changed.
