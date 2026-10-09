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
| 0356 | 3807 + 837 = 4644 | 6913 + 1107 = 8020 | 3376 | 3731 | pass/pass/pass/pass |
| 0357 | 3807 + 837 = 4644 | 6913 + 1107 = 8020 | 3376 | 3731 | pass/pass/pass/pass |
| 0365 | 5599 + 824 = 6423 | 15117 + 3376 = 18493 | 3891 | 4804 | pass/pass/pass/pass |
| 0370 | 1335 + 345 = 1680 | 2768 + 675 = 3443 | 3005 | 3024 | pass/pass/pass/pass |

## Case notes

- **0343 (twin 0340):** All four files reuse input/output plumbing. Transpose
  changes the output to column-first traversal of X (j then i). Literal append
  loop directly follows X.T.flat. Equal sizes are expected: both variants only
  permute checked input wires; no arithmetic can be removed. Literal lint passes.

- **0356 (twin 0355):** Same kernel, shapes, constants and outputs. Reused all four circuits after source review, changing the case header. Optimized code selects payloads using three comparisons per column; literal code bubble-sorts all three keys with indices, materializes static indices, then gathers. No Q32 arithmetic or changed rounding. Stable ties; literal sizes should exceed optimized.

- **0357 (twin 0355):** Reused four circuits and reversed axis 0 after ascending argsort. Optimized payload outputs swap first/last; literal code explicitly creates the reversed index array before static indices and gathers. This reverses ties as well (it is not a descending stable sort).

- **0365 (twin 0363):** Same insertion kernel. Reused thermometer-hint optimized construction and direct prefix/element/suffix literal construction. Optimized Halo2 now reuses each adjacent-output product, as Noir already did. Added a Q32 window check after integer-to-float conversion in all four files: the twin omitted this output-overflow guard. Both variants accept insertion endpoints and wrap negative positions in [-100,100]. The literal local sign test and per-output comparisons retain fresh bounds. Lint cast warnings are inside the constrained-and-checked optimized Noir hint only. Additional non-proving checks in both variants/backends accepted pos=-100/100 with inserted values -2^31/2^31-1, and rejected pos=101 and inserted value 2^31.

- **0370 (twin 0368):** Same isclose/all kernel. Optimized tolerance hoisted per column, self-row omitted; literal full 3x3 broadcast computes tolerance per element then a running all reduction. Reused Q(1e-5)=42950 and floor-rescaled products. Optimized Noir uses abs_int64 for differences (equivalent domain to the twin abs followed by window), matching the other files. Retained the documented shared-gadget domain: abs operands and differences must have magnitude below 2^63, including rejection of Q32 i64::MIN. This is the inherited arithmetic contract, not a claim of unrestricted binary64 NumPy equivalence.
