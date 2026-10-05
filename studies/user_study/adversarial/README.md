# Six-task Astra compatibility audit

Six `gpt-6-astra` agents each reviewed one task for plausible Python/NumPy implementations that Zinnia would reject or compile incorrectly. They produced **148 candidate programs**. Compiler repairs retain all six task specifications and the candidate algorithms. Each task's `REPORT.md`, `inventory.json`, raw programs, and baseline results preserve the review evidence. These are six reviewers producing multiple strategies, not 148 independent participants.

The final gate compiles every original program, then compiles an assertion-instrumented copy. Native NumPy and the compiled circuit must agree with a separate scalar oracle. Each shape profile has 19 fixtures, including boundary cases and 12 seeded random cases, followed by an incorrect-output negative control. Smoothing uses **R=3 throughout**, as in the online study. Every program also produces and verifies a real Halo2 proof for profile 0, fixture 6.

| Task | Programs | Passing baseline profile checks | Passing repaired profile checks | Compiled input cases | Halo2 proofs |
|---|---:|---:|---:|---:|---:|
| Calibration | 24 | 36/72 | 72/72 | 1,368 | 24 |
| Background subtraction | 24 | 53/72 | 72/72 | 1,368 | 24 |
| Threshold energy | 25 | 36/75 | 75/75 | 1,425 | 25 |
| Pairwise distances | 25 | 57/75 | 75/75 | 1,425 | 25 |
| 2×2 pooling | 25 | 48/75 | 75/75 | 1,425 | 25 |
| Three-round smoothing | 25 | 69/100 | 100/100 | 1,900 | 25 |
| **Total** | **148** | **299/469** | **469/469** | **8,911** | **148** |

There are also **469 successful wrong-output rejection checks**. Passing means compilation and checked semantics; some baseline failures compiled successfully but produced incorrect results.

## Repairs

- Source entry selection, ordinary local/global helpers, lexical captures, lambda/map/filter/fromiter, supported finite NumPy callbacks, and NumPy imports/aliases. Existing decorators, explicit chips, and preconditions remain effective.
- Array semantics through enumerate, starred/zip calls, reduction inputs, ndarray starts for `sum`, split views, and numeric casts.
- Reduction axis/keepdims/initial/where/out handling; ufunc output writeback through aliases and strided basic slices.
- Bounded filtered iteration, conditional `continue`/`break` writes, preservation of old lazy-array values when writes are skipped, and lazy selection of statically known conditional-expression branches.
- Bounded dot/inner/compress/take; ndindex, scans, reduceat, sliding windows, roll, edge padding, square, abs, select, copyto/putmask, squeeze, and median operations needed by the reviewed candidates.
- Correct floor division for floating buffers in mock, preprocessing, Halo2, and Halo2-base. Supplemental regressions exercise NumPy's default floating buffer allocation for smoothing, both divisor signs, and remainder semantics.

The shared study worker now accepts a source module with a `kernel` entry and helper functions/imports. Instrumentation checks only the entry's returns and retains helper bodies. The tutorial reflects this interface.

## Study and regression checks

All **148 programs pass the actual isolated study worker**, with **1,357 visible/hidden input cases**. The worker compiles the original module, executes NumPy, and checks compiled/native agreement; the trusted host compares results with the scalar oracle. The website uses mock constraint execution for feedback. The 148 cryptographic proofs above are a separate offline validation gate.

The previous **55 curated strategies / 172 profile checks** and **60 independently sampled GPT-5.6 pilot programs / 190 profile checks** also pass. The earlier pilot source hashes remain unchanged.

Language regression validation includes 799 passing tests from the broader run, followed by 200 focused checks covering the final repairs and added tests (these sets overlap). The binary-search regression passed when allowed to finish, in 296 seconds. Native Rust test targets type-check with both optional backends enabled. This is not a claim that every native Rust test was executed.

All 11 site tests pass. Two Playwright sessions complete onboarding, practice, all six tasks, and final grading in their assigned languages: Zinnia and Rust/RISC Zero. The browser report records 12 completed tasks, 12 hidden grades, and zero browser errors.

Machine-readable results are in `results/`. `final-manifest.json` records every candidate's hash, compiler-source/extension fingerprints, and result-file hashes. `summarize.py` verifies those identities and refuses to freeze a failing gate. Historical baseline formats differ; in particular the background baseline records names and outcomes but has no per-program source hashes, so the final manifest does not retroactively establish cryptographic identity for that baseline.

## Reproduce

Use the repository's Python virtual environment and built extension. The final validation build enabled Halo2-base and Noir as well as the default Halo2 backend. This workstation used its available stable Rust compiler with `RUSTC_BOOTSTRAP=1` for a dependency's nightly feature; the repository's build scripts specify `nightly-2025-12-01` for these optional backends.

```sh
RUSTC_BOOTSTRAP=1 PYTHON=.venv/bin/python scripts/develop.sh --offline --features halo2-base-backend,noir-backend
.venv/bin/python -m studies.user_study.adversarial.check --task calibrate --output studies/user_study/adversarial/results/calibrate-after.json
.venv/bin/python -m studies.user_study.adversarial.check --task calibrate --backend halo2 --profile 0 --fixture 6 --timeout 90 --output studies/user_study/adversarial/results/calibrate-halo2.json
# Repeat the two task checks for background, threshold_energy, distances, pool, and smooth.
.venv/bin/python -m studies.user_study.validate --output studies/user_study/adversarial/results/curated-regression.json
.venv/bin/python -m studies.user_study.pilots.validate --output studies/user_study/adversarial/results/pilot-regression.json
.venv/bin/python -m studies.user_study.adversarial.site_check --output studies/user_study/adversarial/results/site-worker.json
.venv/bin/python -m studies.user_study.adversarial.summarize
```

The site check requires a Linux context that allows bubblewrap to create namespaces. It uses a temporary runtime and does not create participant records. See [the site README](../site/README.md) for browser tests and deployment setup.

## Interpretation

The defensible conclusion is **“all 148 plausible implementations identified by the six reviewers compile and pass the stated gates.”** A finite audit cannot establish that every possible sensible implementation is supported. Shapes, values, APIs, floating-point precision, and resource limits outside the checked cases remain outside this evidence.

Zinnia still has an explicitly bounded language/API surface. Examples include restrictions on escaping/mutating closures and callback signatures, unsupported third-party packages, and writable negative-stride views. Task-local reports and focused probes document API-specific restrictions. This audit does not claim general NumPy compatibility or prove algorithmic correctness for all inputs. Additional natural implementations from human pilots should enter the same no-exclusion gate, with a compiler repair or an explicitly justified task change before enrollment. These engineering results provide no evidence that either language is more usable.
