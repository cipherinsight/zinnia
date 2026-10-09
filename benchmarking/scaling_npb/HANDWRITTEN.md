# Handwritten NPBench/Pythran scaling

The current study budget is **64 GiB per count job**, with a six-hour step
timeout (`--memory 64G --timeout 21600`). Run these jobs on the evaluation
server with sufficient physical RAM; the development machine has only 46 GiB.
At most two such jobs run concurrently. The 8 GiB commands and local results
below describe the earlier preparation/smoke configuration, not the final
resource-limit verdict. Noir final counts use key-free `stats` mode.

After merging the original-size authoring branches, audit their sources against
already generated scaling fixtures. The baseline root supplies both the sources
and `benchmarking/baselines/hw_check/sizes.py`, so the size-formula language has
one implementation, including `factorial`.

```sh
python -m benchmarking.scaling_npb handwritten \
  --data /path/to/npb-data --baseline-root /path/to/npb-hw \
  --out /path/to/handwritten-source-audit.json
```

Add `--write` to materialize checked `target.<function>.rs` and
`main.<function>.nr` beside their scaled fixtures. Add `--caps 32,64` to limit
the audit. This never launches builds, proving, remote jobs, or timing.

The original and scaled fixture shapes determine every annotated literal. The
independent checker verifies both values, rejects unsigned-type overflow, and
compares **every byte outside annotated value spans**, including formula text,
comments, whitespace, and literals inside the function body. The report records
all formula values and SHA-256 hashes of original source, candidate source, and
the source with size values masked. A conflicting existing output is an error;
identical outputs can be reused.

Each point and target remains in the report:

- `rewritten`: source checks pass; written only with `--write`.
- `needs_review`: source checks pass, but the size linter found unexplained
  size-like literals. The warnings are listed and no file is written.
- `not_scalable`: formula, source, file, or representability checks failed;
  the reason is retained and the command returns a nonzero status.
- `missing`: the original implementation has not been supplied.
- `unchanged` / `fixed_size`: carried from the fixture generator, including its
  reason and earlier equivalent cap. They are not independently materialized.

No compile or output-validation success is implied. Compile each scaled Noir
source and generate `Prover.<function>.toml` from that ABI and scaled fixture;
do not copy the original-size Prover TOML. The handwritten driver handles this.
Halo2 build discovery must run against the materialized folders before executing
them. These checks are required before reporting circuit sizes.

## Build and count (no proving benchmark)

The integrated handwritten checkout needs the build override and embedded source
provenance changes. Build one cap at a time, with a separate target directory:

```sh
MEMRUN_MAX=8G RUSTC_BOOTSTRAP=1 \
HALO2_HW_NPB_SOURCE_ROOT=/path/to/npb-data/cap32 \
CARGO_TARGET_DIR=/path/to/targets/cap32 \
/path/to/memrun.sh cargo build --release --offline --jobs 2 \
  --manifest-path /path/to/npb-hw/benchmarking/baselines/halo2_hw/Cargo.toml

python -m benchmarking.scaling_npb handwritten-counts \
  --audit /path/to/handwritten-source-audit.json --cap 32 \
  --halo2-bin /path/to/targets/cap32/release/halo2_hw \
  --memory-runner /path/to/memrun.sh --memory 8G --timeout 300 \
  --work /path/to/scaled-count-work --out /path/to/scaled-counts-cap32.jsonl
```

`HALO2_HW_CASES` can restrict compilation to original IDs (`program__function`);
`handwritten-counts --cases` selects scaled IDs (`program__function_cap32`). The
external source root replaces original NPBench/Pythran discovery. Runtime case
resolution still uses the program folder and function, so no CAP suffix is
inserted into compiled module names. The counter checks the binary's embedded
source text against each audited SHA-256, rejecting wrong-cap or stale builds.

Counting runs serially under the supplied memory limiter and shares the
original-size checker's `/tmp/hw_check_heavy.lock`. Halo2 runs `stats`; Noir
runs `stats`, which compiles and obtains gates without keys or proofs. Results append
to JSONL. Re-running resumes successful records only when fixture, source,
compiled binary (Halo2), gadgets/driver/tools (Noir), and memory settings match.
Timeouts and failures remain records with log paths and are retried on resume.
Tool duration fields are diagnostic metadata, not timing-benchmark measurements.
Noir stats does not validate outputs; execute the witness before reporting an
unchecked circuit as correct.

The CAP32 atax smoke compiled from the scaled source and passed Halo2 MockProver
and Noir witness execution for both A/B, with outputs matching NumPy (maximum
Noir error 4.3e-9). Count-only results: Halo2 30,698 advice + 7,608 lookup cells
(k9); Noir 8,572 gates. No proof or timing benchmark was run.

The initial dry audit on 2026-10-09 (partial authoring integration) covered all
474 point-target rows: 102 rewritten, 22 initially needing literal review, 144 missing,
18 fixed size and 188 unchanged; zero source/formula failures. Review warnings
were in deriche, d2q9, lapl3d and periodic_dist, on both targets. Most are stencil
coefficients or offsets; d2q9 also has fixed nine-direction arrays. Their
algorithmic roles and any fixed dimensions must be reviewed before certification.
All 22 warnings were subsequently reviewed in `handwritten_reviews.json`,
yielding 124 rewritten and no pending review. Reviews bind the exact source
SHA-256, carry a reason, and guard d2q9's fixed third dimensions at nine; any
source edit invalidates the review. The counts are an intermediate snapshot; regenerate after completing integration.

Validation: `python -m pytest -q benchmarking/scaling_npb/tests` (22 tests),
including tampered body literals, operations, comments, whitespace, formula
text, original values, overflow, factorial, per-function fixture filenames,
dry runs, write conflicts, missing sources, and warning exclusions.

## Full integrated source audit

After all 79 original-size baselines were merged (612b1f36), every materialized
scaled source passed: CAP32 98/98, CAP64 92/92, CAP128 78/78 point-targets.
The full manifest also retains 188 unchanged and 18 fixed-size point-targets.
There are no missing sources, source-diff errors, or pending literal reviews.
Additional explicit reviews cover covariance's fixed range widths, sumarray3d's
three coordinates (guarded at three), CRC16's fixed output width, stencil halo
and arithmetic constants, and goodExpoMeasure's byte conversion and test data.
All source files were materialized under the existing cap folders.

To compare completed count records with existing Zinnia circuit-size records:

```sh
python -m benchmarking.scaling_npb handwritten-summary \
  --audit /path/to/handwritten-source-audit.json \
  --counts /path/to/cap32.jsonl /path/to/cap64.jsonl /path/to/cap128.jsonl \
  --zinnia /path/to/measure_final_cap32.json /path/to/heavy_halo2_cap32.json \
    /path/to/changed_final_cap32.json \
  --out /path/to/handwritten-comparison.json
```

Supply the corresponding Zinnia files for all caps of interest, with corrected
`changed_final` files last. The JSON and Markdown report retains uncounted points,
timeouts and failures, and labels ratios provisional until scaled-output checks
are completed. OOM is diagnosed only from a matching systemd scope's journal
record, saved beside the count log; an unexplained signal is kept as such.

Regression validation now includes 23 Python tests plus the standalone Rust
build-discovery test (`rustc --edition 2021 --test build.rs`).
