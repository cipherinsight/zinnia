# Independent GPT-5.6 implementation pilot: round 1

**All 60 independently submitted programs compiled and passed correctness checks on their first compiler evaluation.** Each of the six tasks received ten submissions. No compiler fixes, algorithm rewrites, replacements, or compatibility-driven author revisions were needed in this pilot.

## Sample and procedure

Ten agents were launched with fresh contexts (`fork_turns="none"`), without the earlier discussion or curated implementations: four GPT-5.6 Sol, three GPT-5.6 Terra, and three GPT-5.6 Luna. All used medium reasoning effort. Each solved all six tasks. The same [brief](brief.md) allowed ordinary Python and NumPy without prescribing an algorithm. Authors could self-check with NumPy but were instructed not to inspect the compiler, curated inventory, or other submissions, and not to run Zinnia. The agents shared a filesystem; separation was maintained by instructions and separate output directories.

The [protocol](protocol.json) was written before sampling. The complete [frozen manifest](frozen.json) was created at `2026-09-20T17:27:11.379181+00:00`, after all authors finished and before any compiler feedback. It records every program's model assignment and source hash. All 60 frozen files are retained under [submissions](submissions). The compiler source and loaded extension fingerprints stayed unchanged throughout this pilot.

The evaluator adds only required input annotations and specializes the supplied public `R = 3` configuration for other round counts. It does not rewrite the algorithms. It compiles each original function separately from an instrumented version that checks output shape and values. NumPy execution of each submitted function is compared with the existing independent scalar oracles before the compiler's results are checked.

## Results

| Task | Submitted programs passing | Implementation/shape checks passing | Distinct structural forms |
|---|---:|---:|---:|
| Calibrate readings | 10/10 | 30/30 | 2 |
| Subtract channel background | 10/10 | 30/30 | 4 |
| Thresholded energy | 10/10 | 30/30 | 3 |
| Pairwise squared distances | 10/10 | 30/30 | 3 |
| Sum image blocks | 10/10 | 30/30 | 2 |
| Repeated spatial smoothing | 10/10 | 40/40 | 2 |

- **190/190** original and instrumented compilation cases pass.
- **3,610/3,610** NumPy reference checks pass and the same **3,610/3,610** mock-backend correct-output checks pass: seven predefined fixtures plus twelve independently seeded random fixtures per shape profile.
- **190/190** incorrect-output controls are rejected.
- **60/60** real Halo2 proofs are generated and verified, one per submitted program on profile 0, nonzero fixture 6. Their separate negative controls use the mock backend.
- **3/3** evaluator regression tests pass, covering preservation of early returns, local helper returns, variable-name collisions, and specialization of only interface/configuration details.
- No submission is skipped, marked expected-failure, or excluded because it duplicates another submission.

The ordinary three profiles per task and four smoothing profiles yield 190 cases. The compiler checks are empirical checks on these shapes and inputs; real proofs cover one profile/input per program, not the entire mock test matrix.

Evidence: [first-pass report](results/first-pass.json), [real-proof report](results/proofs.json), [source-overlap report](results/diversity.json).

## What independence and coverage mean here

These are ten separately generated submissions per task, not ten different algorithms. There are 21 distinct function bodies after ignoring formatting, comments, and docstrings, and 16 structural forms after also ignoring local variable names. Structural forms still distinguish different NumPy spellings; they are not counts of distinct algorithms. For example, all ten distance implementations use broadcasting, and nine calibration implementations use `np.clip`.

Authors converged on vectorized NumPy solutions. This supports compatibility for the styles that these agents naturally chose, but adds limited algorithm diversity. Shared model-family priors also mean the submissions are not statistically independent human observations. There is no justified inference that every unforeseen natural program will compile, and no human usability or Rust/RISC Zero comparison was measured.

**The gate passes for this controlled sample.** The earlier curated suite remains useful for alternative styles, including ordinary loops, filtered generators, and conditional appends that these agents did not choose. Human piloting is still needed to establish task difficulty, training, and session duration.

## Reproduction and fingerprints

Run from `zinnia-src` using the existing extension build:

```sh
.venv/bin/python -m studies.user_study.pilots.validate --output /tmp/pilot-validation.json
.venv/bin/python -m studies.user_study.pilots.validate --backend halo2 --profile 0 --fixture 6 --output /tmp/pilot-proofs.json
.venv/bin/python -m pytest -q testing/lang/test_user_study_pilot_harness.py
```

- Git HEAD: `0c8fd2795920a95200f6c17b2310223c263bdb60`; dirty checkout, including existing development changes.
- Working compiler source SHA-256: `133b45d2b554c35900c02d06537ddcda87929ded197156c807c4a9daa0dbba8c`.
- Loaded extension SHA-256: `c754b6cce548f56123cbe24dc97b0792366f97214fb2c1ce73d8494f3b01173a`.
- Frozen manifest SHA-256: `1911376b315e9ec68f4bf04c51afc358fbc5a80641b536a846dde51163d8f5b2`.
- Pilot evaluator SHA-256: `0f30a6829c3cc1a2f2a475fe290e75b50e4ebf264c5e51b3d5eb73f08a3469c1`.
- Python `3.14.7`, NumPy `2.2.2`.

These fingerprints match the initial protocol, frozen manifest, first-pass evaluation, real-proof evaluation, and current checkout where applicable. All submission hashes were checked again when generating this report. Model generation itself is not deterministically reproducible because model sampling seeds were not controlled; the frozen code and compiler evaluation are reproducible artifacts.
