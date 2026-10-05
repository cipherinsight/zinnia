# Benchmarks

- `datasets/`: the programs.
  - `ds1000/ds1000_case<NNNN>/`: the cross-system performance set (Zinnia `kernel.py`,
    `fixture.json` with inputs A and B, and the handwritten halo2-base, Noir, Cairo, SP1
    and RISC Zero baselines).
  - `ds1000_numpy/`, `npbench/`, `pythran/`: the acceptance account (see
    `datasets/FILTERING.md`).
  - `case_sets/*.json`: which cases each study uses, with reasons and tolerances.
- `baselines/`: the baseline programs and their rules (`CONVENTIONS.md`,
  `NOIR_BASELINES.md`); each system directory provides the `bench` entry point below.
- `noir_fpc/`: the Q32 fixed-point Noir library shared by the handwritten Noir baselines
  and Zinnia's Noir backend (embedded by `src/prove/noir/mod.rs`, so it stays here).
- `harness/`: the single benchmark harness (this file describes it).
- `results/<YYYY-MM-DD>-<label>/`: one directory per epoch, written by the harness.
- `archive/`: results and runners kept for traceability only (see its README).

## Usage

```sh
.venv/bin/python -m benchmarking.harness check-env          # versions and paths; non-zero on a mismatch
.venv/bin/python -m benchmarking.harness build              # release extension + build record
.venv/bin/python -m benchmarking.harness run --label main   # one epoch, default systems and case set
.venv/bin/python -m benchmarking.harness analyze benchmarking/results/<epoch>
.venv/bin/python -m benchmarking.harness accept --epoch benchmarking/results/<epoch>
```

`run` options: `--systems zinnia_halo2,halo2_hw,zinnia_noir,noir_hw,sp1,sp1_plonk,risc0,cairo`
(default: all but `sp1_plonk`), `--case-set FILE` (default `datasets/case_sets/ds1000_perf.json`),
`--cases 0309,0336`, `--fixtures-dir DIR` (trial fixtures outside the dataset), `--repeats N`
(default 5), `--timeout S` (per step, default 1800), `--memory-max 16G|none`, `--threads N`,
`--seed N`, `--halo2-k K --halo2-lookup-bits L` (fixed halo2 parameters on both halo2 sides,
for a sensitivity run), `--results-dir DIR`. Tests: `.venv/bin/python -m pytest -q
benchmarking/harness/tests`.

`build` runs `scripts/develop.sh --release --features halo2-base-backend,noir-backend` with
`RUSTC_BOOTSTRAP=1` and writes `target/zinnia-harness-build.json` (`.so` SHA-256, profile,
features, rustc, source state). `run` refuses to time a Zinnia system unless the installed
`.so` is the recorded one, equals Cargo's release artifact, is not the debug artifact and has no
debug sections. It also refuses when a toolchain a requested system needs has the wrong version,
or when `RISC0_DEV_MODE` is set.

## The protocol

For each case, in a seeded random order, and each requested system:

1. **Setup** with input A (compile, keys), in its own child process.
2. **Warm-up**: one prove with input B, not used in the statistics.
3. **Timed proves**: `--repeats` proves with input B, each in its own child process. Systems
   are interleaved: sample i of every system runs before sample i+1 of any, in an order rotated
   per sample.
4. **Verify** every timed proof, each in its own child process.
5. **Output check**: every proof's public outputs, decoded, compared element by element in the
   kernel's output order with the kernel body run as plain NumPy on input B: integers and
   booleans exactly, floats within the case-set tolerance (default 1e-6 absolute + 1e-6
   relative, recorded as such). Counts must agree; permutations fail.
6. **Tamper check** (once per case): a copy of the first proof with one public value
   incremented must be rejected.

Every step runs under the same memory cap (a transient systemd scope with `MemoryMax`, no swap,
as `eval-redo/tools/memrun.sh`) and the same timeout. Children see no GPU
(`CUDA_VISIBLE_DEVICES=""`, `SP1_PROVER=cpu`, `RISC0_DEV_MODE` unset). KDE's file indexer is
suspended during an epoch (`balooctl6`/`balooctl`) and resumed after. A raw record's status is
`ok` only if every step succeeded, every output matched and the tampered proof was rejected (or
the proof had no public value). Otherwise its status is the first failure class:

`no_program`, `reference_failed`, `unsupported`, `setup_failed`, `witness_dependent`,
`nondeterministic_build`, `prove_failed`, `verify_failed`, `wrong_output`, `output_unchecked`,
`tamper_accepted`, `tamper_inconclusive`, `timeout`, `out_of_memory`, `crashed`, `harness_error`
(`harness/protocol.py` describes each). Nothing is dropped.

### Clocks per system

| system | prove core | end to end | verify |
|---|---|---|---|
| `zinnia_halo2` | `create_proof` + transcript finalize, timed in the backend | the whole `prove()` call, **including** IPA parameter generation and keygen (the API couples them) | `verify_proof_artifact`, **including** re-keygen from the shipped IR |
| `zinnia_noir`, `noir_hw` | `bb prove` process wall clock (includes bb start-up; the epoch manifest records the start-up floor on a one-line circuit) | `nargo execute` + `bb prove` | `bb verify` process |
| contract programs | `timings_s.prove` | `timings_s.witness` + `timings_s.prove` | `timings_s.verify` |

Every raw record carries its system's window descriptions.

### halo2 parameters (same policy on both halo2 sides)

Zinnia's halo2-base backend (`src/prove/halo2_base/mod.rs`, `estimate_params`): for every
`k` in 4..=22, `lookup_bits = min(8, k-1)`; cell counts from a dry keygen-stage synthesis at that
`lookup_bits`; skip `k` whose usable rows (`2^k - 20`) cannot hold the instance column or the
lookup table; cost = committed polynomials (from a halo2 `ConstraintSystem` configured for the
column layout) × `2^k`; pick the minimum (first on ties). IPA over BN254, Blake2b transcript,
PSE `halo2_proofs` v0.3.0 through halo2-base `community-edition` with `halo2-pse`. `halo2_hw`
applies the same rule to its own circuit and reports `k`, `lookup_bits`; `--halo2-k/--halo2-lookup-bits`
fixes both sides.

### Noir

Both Noir sides go through `harness/adapters/noir_toolchain.py`: nargo 1.0.0-beta.20 and bb
5.0.0-nightly.20260324 at their pinned paths, the package with `noir_fpc` copied in, and
`--scheme ultra_honk --verifier_target noir-recursive` (UltraHonk, zero-knowledge) for
`write_vk`, `prove`, `verify` and `gates`. Zinnia's `main.nr` comes from `emit_structural`
and must be byte-identical for inputs A and B.

## The per-system program contract

Each baseline system directory (`baselines/halo2_hw`, `baselines/sp1`, `baselines/risc0`,
`baselines/cairo`) provides an executable `bench`:

```
bench --case-dir DIR --fixture PATH --input A|B --mode setup|prove|verify
      --work-dir DIR --out PATH [--proof PATH] [--tamper] [system flags]
```

- `setup` (input A) builds and writes keys to `--work-dir`; `prove` (input B) writes one proof
  to exactly `--proof` with those keys (the proof's size is reported as the proof size); `verify` checks `--proof` (with `--tamper`, after changing one
  public value of it; the harness passes a copy).
- It writes one JSON object to `--out` and exits non-zero on any failure (still writing the JSON):
  `{"ok": bool, "error": str, "error_class": str, "publics": [...], "proof_path": str,
  "verified": bool, "timings_s": {"setup", "witness", "prove", "verify"}, "details": {...}}`.
  `publics` is the kernel's return value flattened row-major (tuple parts in order), decoded to
  numbers; a data-dependent length is decoded too (only the true elements).
- System flags: `sp1_plonk` passes `--variant plonk`; `halo2_hw` accepts `--k K --lookup-bits L`
  in setup and reports the chosen `k`/`lookup_bits` in `details`.

The handwritten Noir baselines are driven by the harness itself. Their `Prover.toml` writer and
public-output decoder are taken from `baselines/noir_hw/prover_toml.py`
(`write_prover_toml(case_dir, fixture_path, input_name) -> str`,
`decode_publics(case_dir, fields) -> list`) when that module exists; otherwise the harness
writes `Prover.toml` from the fixture and decodes outputs by the reference's types.

## Results

`results/<epoch>/`: `manifest.json` (repository commit, diff hash and untracked files; build
record and current `.so` hash; every toolchain version and path; baseline entry points and
`Cargo.lock` hashes; `noir_fpc` hashes; machine; tracked environment variables; configuration;
case order; indexer log; bb calibration), `case_set.json`, `raw/<system>/<case>.json` (every
sample with both clocks, proof size and hash, decoded publics of the first samples, output
check, verification, tamper result, CPU state, file hashes), `raw/_reference/<case>.json`,
`logs/`, `work/` (keys and proofs, not committed), and `summary/` from `analyze`:
`per_case.csv` (median and IQR per clock), `summary.json`/`summary.md` with per-system coverage,
the **all-supported subset** (cases every system completed) and on it the geometric mean of
per-case ratios (Zinnia / baseline) with a bootstrap 95 % interval, the same per pair of
systems, exclusions, and the acceptance table when `acceptance/results.json` exists.

## Acceptance

`accept` reads `datasets/case_sets/acceptance.json` and, for every program, in a capped child
process: applies the scope rule (imports other than numpy and math are out of scope), compiles,
mock-runs input B and compares with the NumPy reference. Outcomes: `accepted_correct`,
`accepted_wrong_output`, `accepted_unchecked`, `rejected_out_of_scope`,
`rejected_unsupported_construct`, `rejected_no_provable_bound`, `rejected_solver_budget` (rejected
at the default SMT budget, accepted with a larger one), `rejected_compiler_error`,
`rejected_resource_limit`, `rejected_unclassified` (`harness/acceptance/classify.py`).
