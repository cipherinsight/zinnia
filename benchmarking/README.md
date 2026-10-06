# Benchmarks

- `datasets/`: the programs.
  - `ds1000/ds1000_case<NNNN>/`: the cross-system performance set (Zinnia `kernel.py`,
    `fixture.json` with inputs A and B, and the handwritten halo2-base, Noir, Cairo, SP1
    and RISC Zero baselines).
  - `ds1000_numpy/`, `npbench/`, `pythran/`: the acceptance account (see
    `datasets/FILTERING.md`).
  - `case_sets/*.json`: which cases each study uses, with reasons and tolerances.
- `baselines/`: the baseline programs and their rules (`CONVENTIONS.md`,
  `NOIR_BASELINES.md`, `zkvm_common/README.md`): `halo2_hw/`, `noir_hw/` + `hw_check/`,
  `sp1/`, `risc0/`, `cairo/`.
- `noir_fpc/`: the Q32 fixed-point Noir library shared by the handwritten Noir baselines
  and Zinnia's Noir backend (embedded by `src/prove/noir/mod.rs`, so it stays here).
- `harness/`: the single benchmark harness (this file describes it).
- `results/<YYYY-MM-DD>-<label>/`: one directory per epoch, written by the harness.
- `archive/`: results and runners kept for traceability only (see its README).

## Usage

```sh
.venv/bin/python -m benchmarking.harness check-env          # versions and paths; non-zero on a mismatch
.venv/bin/python -m benchmarking.harness build              # release extension + build record
.venv/bin/python -m benchmarking.harness build --skip-zinnia --baselines halo2_hw,sp1,risc0,cairo
.venv/bin/python -m benchmarking.harness run --label main   # one epoch, default systems and case set
.venv/bin/python -m benchmarking.harness analyze benchmarking/results/<epoch>
.venv/bin/python -m benchmarking.harness accept --epoch benchmarking/results/<epoch>
```

`run` options: `--systems` from `zinnia_halo2, halo2_hw, zinnia_noir, noir_hw, sp1, sp1_groth16,
risc0, risc0_groth16, cairo` (default: all but the two Groth16 variants), `--case-set FILE` (default `datasets/case_sets/ds1000_perf.json`),
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
as `eval-redo/tools/memrun.sh`, plus `OOMPolicy=continue` so an OOM-killed tool is reported as
`out_of_memory` by its step) and the same timeout. On a machine without a systemd user manager
(a Docker container), `ZINNIA_HARNESS_MEMORY_CAP=watchdog` enforces the cap instead by summing
the RSS of the step's processes every 0.1 s and killing them all above the cap
(`harness/isolation.py`); `run` refuses to start when the chosen cap cannot be applied, and the
manifest records the mechanism (`memory_cap`). Work files count against the cap when the
results directory is on tmpfs; keep epochs on disk. Children see no GPU
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
| `zinnia_halo2` | `prove_s`: `create_proof` + transcript finalize, timed in the backend | `preprocess_s` + `witness_s` + `prove_s` | `verify_s`: transcript + `verify_proof` with the verifying key read from setup's key directory |
| `zinnia_noir`, `noir_hw` | `bb prove` process wall clock (includes bb start-up; the epoch manifest records the start-up floor on a one-line circuit) | `nargo execute` + `bb prove` (`nargo execute` re-elaborates the package source before running the ACIR: see Noir below) | `bb verify` process |
| `halo2_hw` | `prove_s` (create_proof) | `witness_s` + `prove_s` | `verify_s` |
| `sp1`, `sp1_groth16` | `prove` (the SDK executes inside the prove call) | `prove` | `verify` |
| `risc0` (`_groth16`) | `prove` (+ `compress_succinct` + `wrap_groth16`) | `execute` + prove core | `verify` |
| `cairo` | `scarb prove` process | `scarb execute` + `scarb prove` processes | `scarb verify` process |

Every raw record carries its system's window descriptions.

### The halo2 pair, step by step

Both halo2 sides run the same steps in separate processes and write the same files:
setup writes the IPA parameters (`params.bin`), the proving key (`pk.bin`), the verifying key
(`vk.bin`, both `SerdeFormat::RawBytes`) and the pinned layout; prove reads the parameters
and the proving key; verify reads the parameters and the verifying key. Zinnia's keys are
generated without a witness (the synthesis a verifier runs); halo2_hw's with input A. Every
prove uses input B. The per-step timings in the raw records (`timings_s`):

| step | `zinnia_halo2` | `halo2_hw` | in a clock |
|---|---|---|---|
| read the inputs | Python input builder + `parse_inputs_s` | `load_input_s` | no |
| read parameters / key | `load_params_s`, `load_pk_s` (prove); `load_params_s`, `load_vk_s` (verify) | same names | no |
| resolve external calls | `preprocess_s`: one evaluation pass over the IR | (none) | end to end |
| witness | `witness_s`: IR walk into the builder pinned to keygen's layout | `witness_s`: the circuit function into the pinned builder | end to end |
| start the proving thread pool | `thread_pool_s` | `thread_pool_s` | no |
| prove | `prove_s`: `create_proof` + transcript finalize | `prove_s`: same calls | prove core, end to end |
| verify | `verify_s`: transcript + `verify_proof` | `verify_s`: same calls | verify |

Outside every clock on both sides: process start-up, Zinnia's recompilation in each step
(the IR digest must equal setup's; halo2_hw's circuit is compiled into its binary), Zinnia's
check that the keys belong to the compiled circuit (an IR hash computed once per process),
writing the proof, and the whole-call diagnostics (`prove_call_s`, `verify_call_s`,
`process_wall`).

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

The two sides run the same commands on the same files: setup compiles the package and writes
the verifying key (`bb write_vk`); every prove runs `nargo execute` with `Prover_B.toml` and
`bb prove -k <setup's vk>`; every verify runs `bb verify -k <setup's vk>`. Writing
`Prover_B.toml` (Zinnia's input builder, or `hw_check.prover_toml`) is outside the windows on
both sides. `nargo execute` is the witness step on both sides, but nargo has no mode that runs
the compiled ACIR alone: each call re-elaborates the package's source and libraries (it skips
code generation when the result matches the compiled artifact), then runs the ACVM. Its floor on
a one-line circuit is in the manifest (`calibration.bb.nargo_execute_s`, about 60 ms), and it
grows with the source size: on 0381 Zinnia's emitted `main.nr` is 669 KB and `nargo check`
alone takes about 170 ms of its about 285 ms `nargo execute`, against 60 ms of 66 ms for the
2.5 KB handwritten one.

## The per-system program contract

Each baseline program takes the case folder, the fixture, the input name and a mode, writes one
JSON object (`ok`, `publics` = the kernel's return value flattened row-major and decoded,
`proof_path`, `timings_s` per step, `verified` in verify mode, optionally `output_shape`) and
exits non-zero on any failure. The harness calls (`harness/adapters/contract.py`):

```
halo2_hw/target/release/halo2_hw <mode> --case DIR --fixture F --input A|B --workdir W --out J
                                 [--k K --lookup-bits L] [--tamper-index 0]
sp1/target/release/sp1-baseline  --case-dir DIR --mode <mode> --work-dir W --fixture F --input A|B
                                 --out J --proof-mode core|groth16 [--tamper]
risc0/target/release/risc0-baseline  ... --proof-mode composite|groth16 [--tamper]
python cairo/cairo_baseline.py   ... --proof-mode stwo [--tamper]
```

Setup gets its own work directory; every prove runs in a fresh copy of it (`cp --reflink=auto`)
inside the sample's proof directory, because the programs overwrite their proof in place; that
copy is what the sample's verification and the tamper check read. The clocks are sums of the
program's step timings as in the table above. The handwritten Noir baselines are driven by the
harness itself (same commands as Zinnia's), with `hw_check.prover_toml` writing `Prover.toml`
and `hw_check.noir_driver` decoding the outputs from each `main.nr`'s `// @outputs:` line.

## Results

`results/<epoch>/`: `manifest.json` (repository commit, diff hash and untracked files; build
record and current `.so` hash; every toolchain version and path; baseline entry points and
`Cargo.lock` hashes; `noir_fpc` hashes; machine; tracked environment variables; configuration;
case order; indexer log; nargo/bb start-up calibration), `case_set.json`, `raw/<system>/<case>.json` (every
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
