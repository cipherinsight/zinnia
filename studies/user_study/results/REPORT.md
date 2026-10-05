# Final compatibility validation

This is compiler/task validation for the proposed Zinnia–RISC Zero user study. It contains no participant results and does not establish a usability advantage for either language.

## Results

- **55/55 solution variants** pass across **172/172 implementation/shape combinations**.
- **1,204 correct-output cases** agree with independent scalar oracles. Each reference implementation is also run in NumPy; each original, uninstrumented implementation is compiled before its instrumented checks run.
- **172/172 incorrect-output controls** are rejected by the mock backend.
- **55/55 real Halo2 proofs** are generated and verified on a nonzero randomized private input, one proof per variant at profile 0. Their additional incorrect-output controls use the mock backend.
- Full operator, syntax, and language regression run: **1238 passed, 2 skipped in 387.15s (0:06:27)**.
- The study inventory has **no skipped cases, expected failures, or exclusions**.

| Task | Variants | Implementation/shape checks |
|---|---:|---:|
| Calibrate and clip readings | 12 | 36 |
| Subtract channel background | 8 | 24 |
| Energy above a per-channel noise floor | 10 | 30 |
| Pairwise squared distances | 10 | 30 |
| Sum non-overlapping image blocks | 8 | 24 |
| Repeated spatial smoothing | 7 | 28 |

The two retained regression-suite skip markers concern chained-index assignment and negative-base fractional powers. The chained-index test was also called directly and passed; chained reads/writes are now covered explicitly by 13 task/profile checks. Fractional complex-valued powers are outside these integer task contracts. No skipped regression is counted as a pass.

## Evidence and limits

The [mock report](validation.json) and [real-proof report](proofs.json) contain every case and source hash. The [regression log](regressions.txt) records the final suite result. [Task specifications](../tasks.md) leave algorithm choice to participants; [the inventory](../tasks.py) is for preparation, not participant starter code.

Fixtures include different per-channel thresholds, equality with thresholds, empty/full selections, numeric extremes, singleton dimensions, rectangular tensors, and grids with no interior cells. A compiled case is reused across seven changing private inputs. Bounds derive from public shapes and the fixed smoothing round count.

The real-proof check uses the original Halo2 backend (`halo2` alias), not Halo2-base, and covers profile 0 only. All shapes and seven fixtures receive the broader mock correctness checks. Neither mode proves acceptance of every future program, complete NumPy compatibility, or correctness on every possible input. Independent pilots must add unseen valid solutions to the inventory; any failure requires a compiler fix or task revision before freezing the study. The RISC Zero templates and participant study have not been run by this validation.

## Build and reproduction

Recorded at `2026-09-20T17:09:23.495536+00:00`. Git HEAD: `0c8fd2795920a95200f6c17b2310223c263bdb60`; the checkout contains local changes, so HEAD alone is not the tested version.

- Python: `3.14.7`; NumPy: `2.2.2`.
- Rust: `1.96.1 (31fca3adb 2026-06-26)`; maturin: `1.14.1`.
- Working compiler source SHA-256: `133b45d2b554c35900c02d06537ddcda87929ded197156c807c4a9daa0dbba8c`.
- Loaded native extension SHA-256: `c754b6cce548f56123cbe24dc97b0792366f97214fb2c1ce73d8494f3b01173a`.
- Inventory SHA-256: `6dd9bf8f9ace2ae126d5efc7cd6d943d3a0eb704b65380f70a788f7636f730f0`.
- Validator SHA-256: `8ad71b64ff073f49186f90000791d3edf82b4380dda8f80519cdf143ee919653`.

The source fingerprint includes existing unrelated development changes in this checkout. Source and extension fingerprints match between the mock and proof reports. They identify the build used here; they are not a replacement for freezing a clean study release.

The exact local build enabled all optional backends so the full regression suite could execute. Two optional dependency crates require nightly Rust features; this local run enabled those features only for those crates on the installed compiler. Repository build-matrix scripts instead use a pinned nightly toolchain. Dependencies were downloaded before the final offline build.

```sh
RUSTC_BOOTSTRAP=poseidon_primitives,halo2_base PYTHON=.venv/bin/python scripts/develop.sh --offline --features halo2-base-backend,noir-backend
.venv/bin/python -m studies.user_study.validate --output studies/user_study/results/validation.json
.venv/bin/python -m studies.user_study.validate --backend halo2 --profile 0 --fixture 6 --output studies/user_study/results/proofs.json
.venv/bin/python -m pytest -q testing/operator testing/syntax testing/lang
```

Initial broader runs exposed missing optional-backend build flags, then an overflow in signed-integer field conversion. The final result above follows enabling those backends and repairing the conversion, including a regression that failed on the prior native extension with the minimum 64-bit integer. The historical [before report](before.json) uses a smaller 41-variant inventory and must not be compared as a participant outcome or a controlled performance result.
