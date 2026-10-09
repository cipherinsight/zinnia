# dup1 handwritten baselines

Each listed case has standalone `target.rs`, `main.nr`, `target.literal.rs`,
and `main.literal.nr` in `benchmarking/datasets/ds1000/ds1000_caseNNNN/`.
Sizes below are Halo2 advice + lookup cells and Noir unpadded gates.

| Case | Twin | Optimized Halo2 | Literal Halo2 | Optimized Noir | Literal Noir | Four checks | Adaptation |
|---|---|---:|---:|---:|---:|---|---|
| 0302 | 0301 | 800 + 240 = 1040 | 800 + 240 = 1040 | 2941 | 2941 | pass | Reshape to 5 rows and 4 columns; same row-major outputs. |

Validation uses `hw_check.check --no-prove`: Halo2 MockProver A/B, Noir
compile/count and witness execution A/B, NumPy output comparison, and rejection
of the checker's out-of-domain input. No proofs, key generation, timing study,
scaling, or Zinnia compilation/output inspection.

These duplicate folders have no checked-in fixtures. Fixtures generated through
`perf_fixtures.build_fixture` live outside the repository at
`/home/zhantong/Projects/zinnia-project/eval-redo/all-work/dup1/fixtures` and are
passed with `--fixture-root`. Check JSON is in the same batch directory
(`0302-opt.json`, `0302-lit.json`). Repository fixtures and tools are unchanged.

0302 has equal literal and optimized sizes in both backends: reshape is wire
aliasing with exactly the same input-domain checks and public cells. Literal
code materializes the full reshape; there is no arithmetic to optimize.
