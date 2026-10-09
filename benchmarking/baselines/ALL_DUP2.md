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
are supplied in memory for 0356 from 0355, 0365 from 0363, and 0370 from 0368;
no shared tools or repository fixtures are modified. Checks use `--fixture-root`.

| Case | Optimized Halo2 advice + lookup = total | Literal Halo2 advice + lookup = total | Optimized Noir gates | Literal Noir gates | Four checks |
|---|---:|---:|---:|---:|---|
| 0343 | 1200 + 360 = 1560 | 1200 + 360 = 1560 | 3004 | 3004 | pass/pass/pass/pass |
| 0356 | 3807 + 837 = 4644 | 6913 + 1107 = 8020 | 3376 | 3731 | pass/pass/pass/pass |
| 0357 | 3807 + 837 = 4644 | 6913 + 1107 = 8020 | 3376 | 3731 | pass/pass/pass/pass |
| 0365 | 5194 + 824 = 6018 | 15117 + 3376 = 18493 | 3891 | 4804 | pass/pass/pass/pass |
| 0370 | 1335 + 345 = 1680 | 2768 + 675 = 3443 | 3005 | 3024 | pass/pass/pass/pass |
| 0382 | 66135 + 12570 = 78705 | 616600 + 124550 = 741150 | 5366 | 125038 | pass/pass/pass/pass |
| 0387 | 540000 + 160000 = 700000 | 540000 + 160000 = 700000 | 127816 | 127816 | pass/pass/pass/pass |
| 0392 | 1200 + 360 = 1560 | 1200 + 360 = 1560 | 3004 | 3004 | pass/pass/pass/pass |
| 0393 | 1620 + 480 = 2100 | 1620 + 480 = 2100 | 3191 | 3191 | pass/pass/pass/pass |

## Case notes

- **0343 (twin 0340):** All four files reuse input/output plumbing. Transpose
  changes the output to column-first traversal of X (j then i). Literal append
  loop directly follows X.T.flat. Equal sizes are expected: both variants only
  permute checked input wires; no arithmetic can be removed. Literal lint passes.

- **0356 (twin 0355):** Same kernel, shapes, constants and outputs. Reused all four circuits after source review, changing the case header. Optimized code selects payloads using three comparisons per column; literal code bubble-sorts all three keys with indices, materializes static indices, then gathers. No Q32 arithmetic or changed rounding. Stable ties; literal sizes exceed optimized in both backends.

- **0357 (twin 0355):** Reused four circuits and reversed axis 0 after ascending argsort. Optimized payload outputs swap first/last; literal code explicitly creates the reversed index array before static indices and gathers. This reverses ties as well (it is not a descending stable sort).

- **0365 (twin 0363):** Same insertion kernel. Reused thermometer-hint optimized construction and direct prefix/element/suffix literal construction. Optimized Halo2 keeps the twin fused multiply-add/subtract gates; Noir keeps its shared adjacent-output products. Added a Q32 window check after integer-to-float conversion in all four files: the twin omitted this output-overflow guard. Both variants accept insertion endpoints and wrap negative positions in [-100,100]. The literal local sign test and per-output comparisons retain fresh bounds. Lint cast warnings are inside the constrained-and-checked optimized Noir hint only. Additional non-proving checks in both variants/backends accepted pos=-100/100 with inserted values -2^31/2^31-1, and rejected pos=101 and inserted value 2^31.

- **0370 (twin 0368):** Same isclose/all kernel. Optimized tolerance hoisted per column, self-row omitted; literal full 3x3 broadcast computes tolerance per element then a running all reduction. Reused Q(1e-5)=42950 and floor-rescaled products. Optimized Noir uses abs_int64 for differences (equivalent domain to the twin abs followed by window), matching the other files. Retained the documented shared-gadget domain: abs operands and differences must have magnitude below 2^63, including rejection of Q32 i64::MIN. This is the inherited arithmetic contract, not a claim of unrestricted binary64 NumPy equivalence.

- **0382 (twin 0381):** Removed output reversal and the redundant full-length slice in all four circuits. Optimized Halo2 retains the packed-key merge-exchange network; optimized Noir retains a hinted permutation pinned by bounded indices and strictly increasing (value,index) keys. Literal versions bubble-sort all 100 elements with indices (4950 comparisons), then directly return them. Stable ties follow the twin and L5; NumPy default argsort does not promise stable ties at this length, so the fixture check (distinct random values) does not certify its unspecified tie ordering. No rounding operations.

- **0387 (twin 0385):** Removed transpose(1,0,2,3); patches are in row-block-first order i*100+j. Literal code materializes reshape and swapaxes wire arrays, then the final reshape. Optimized code directly traverses blocks in that order. Equal sizes are expected because reshapes/permutations need no arithmetic beyond the same 20000 input checks.

- **0392 (twin 0340):** Reused the 30-cell copy plumbing but changed X integer[5,6] to a float[20,10], selected rows 4:7 (30 floats), and changed both output declarations to float[30]. Only read rows are loaded/range-checked. Literal source constants and nested slice traversal remain explicit. Equal sizes are expected: both forms expose the same 30 checked wires.

- **0393 (twin 0391):** Reused column slice 4:7, with literal min(high,a.shape[1]) represented as a compile-time min(7,10). Optimized code folds this constant. Only the 60 selected cells are read and checked. Equal sizes are expected because the additional statement is compile-time arithmetic.

## Final validation and limitations

All nine cases are complete: 36 passing variant/backend checks. Both variant
lint runs report zero errors. No case is unfinished and no literal circuit is
smaller than its optimized counterpart. Both sizes are equal for 0343, 0387, 0392, 0393
(the four copy/permutation/slice kernels); each arithmetic/sort case is larger
in literal form in both backends. No memory-cap failures occurred.

Raw per-case results are `NNNN-opt.json` and `NNNN-lit.json` under the evidence
root; each contains outputs, domain-check outcomes and sizes. The insertion
edge results are `0365-edge-results-{opt,literal}.json`. Source SHA-256 hashes
are recorded in `source-sha256.json`. Noir uses Nargo 1.0.0-beta.20 and bb
5.0.0-nightly.20260324 with the driver default noir-recursive settings.

Missing duplicate fixtures/draw registrations were the only blocking tool/data
issue and were handled through external fixture staging. The venv prints benign
sys.prefix/sys.exec_prefix warnings when invoked through ../../.venv/bin/python.
No shared tools, gadgets, case sets, other cases or committed fixtures changed.
Fixture agreement and constraint review do not claim a full-domain binary64
NumPy equivalence proof; the stable-sort and Q32 abs-domain interpretations
above remain explicit. No proving or compiler-size comparison was performed.

Automatic Halo2 settings (k / lookup bits):

| Case | Optimized | Literal |
|---|---:|---:|
| 0343 | 7 / 6 | 7 / 6 |
| 0356 | 8 / 7 | 9 / 8 |
| 0357 | 8 / 7 | 9 / 8 |
| 0365 | 9 / 8 | 9 / 8 |
| 0370 | 7 / 6 | 8 / 7 |
| 0382 | 9 / 8 | 11 / 8 |
| 0387 | 15 / 8 | 15 / 8 |
| 0392 | 7 / 6 | 7 / 6 |
| 0393 | 9 / 8 | 9 / 8 |
