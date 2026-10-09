# dup2 handwritten baselines

Each case directory is `benchmarking/datasets/ds1000/ds1000_caseNNNN/`.
The four files for every completed case are `target.rs` (optimized Halo2),
`main.nr` (optimized Noir), `target.literal.rs` (literal Halo2), and
`main.literal.nr` (literal Noir). All four stand alone.

Validation uses the worktree venv, dev-fast Cargo builds (3 jobs, 8 GiB cap),
`HW_CHECK_LOCK=eval-redo/all-work/lock-a`, and `hw_check.check --no-prove`.
Pass means Halo2 MockProver A/B or Noir witness execution A/B matches NumPy,
and the checker's out-of-domain input is rejected. No keys, proofs, proof
verification, timing study, scaling, Zinnia execution, IR, emitted code or
Zinnia size inspection. Counts below are automatic sizing, not a controlled
common-lookup-width experiment. L1-L7 reviewed against each kernel.

Evidence and staged packages: `/home/zhantong/Projects/zinnia-project/eval-redo/all-work/dup2/`.
The repository has no fixtures for these duplicates. Fixtures are generated
outside the repository in `fixtures/`, using `perf_fixtures.build_fixture`
and the case's own kernel and deterministic seeds. Missing DRAWS registrations
are supplied in memory from the listed twin (0365/0363, 0356/0355, etc.);
no shared tools or repository fixtures are modified. Checks use `--fixture-root`.

| Case | Optimized Halo2 advice + lookup = total | Literal Halo2 advice + lookup = total | Optimized Noir gates | Literal Noir gates | Four checks |
|---|---:|---:|---:|---:|---|
| 0343 | 1200 + 360 = 1560 | 1200 + 360 = 1560 | 3004 | 3004 | pass/pass/pass/pass |

## Case notes

- **0343 (twin 0340):** All four files reuse input/output plumbing. Transpose
  changes the output to column-first traversal of X (j then i). Literal append
  loop directly follows X.T.flat. Equal sizes are expected: both variants only
  permute checked input wires; no arithmetic can be removed. Literal lint passes.
