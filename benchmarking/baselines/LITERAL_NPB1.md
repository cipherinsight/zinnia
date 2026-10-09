# Literal NPBench/Pythran batch npb1

In progress. Checks use `HW_VARIANT=literal`, `--no-prove`, an 8 GiB cap,
and HW_CHECK_LOCK `eval-redo/literal-work/lock-a`. Evidence is under
`eval-redo/literal-work/npb1/`. No proofs or Zinnia runs.

| Case | Literal Halo2 advice + lookup = total | Literal Noir gates | Optimized Halo2 | Optimized Noir | Verdict / review |
|---|---:|---:|---:|---:|---|
| npbench_polybench_bicg_bicg__kernel | 8836 + 2244 = 11080 | 4479 | not run | not run | Both pass. Reused existing files unchanged after line-by-line L1-L7 review: tuple order, full outputs, each @ uses one dot rescale. |
