# What the 83K lines of Rust count, and a defensible figure for the paper

Survey of 2026-10-12, run on the local machine (line counts do not depend on hardware).
Sources: commit `2569c9ed` (`git archive 2569c9ed src Cargo.toml Cargo.lock build.rs
benchmarking/noir_fpc`); `src/`, `zinnia/` and `Cargo.toml` are identical at `HEAD` (`1f7f5bfb`).
Paper sentence: `implementation.tex` line 7, "a compiler core in Rust (about 83K lines of code)".

## Method

- A *code line* is a line that is neither blank nor only comment, the definition of tokei's and
  cloc's "code". Doc comments (`///`, `//!`) are comments under this definition (item 4 below).
- The starting point is the README's D4: all Rust in `src/` without test code (`strip_tests.py`)
  and without `src/phase0`, `src/phase1`. With tokei 15.0.0 installed locally
  (`cargo install tokei`) the same tree gives **82,944**, the README's number exactly.
- The finer counts below need line ranges of individual items, so they use a small counter
  (`survey/count.py`: blank out comments with `strip_tests.py`'s lexer, count non-empty lines).
  On D4 it gives 82,903 and on all of `src/` 149,518, identical to the README's cloc numbers
  (tokei counts 41 more lines on D4, 0.05%). Differences below are computed with this counter;
  totals are quoted against both.
- **Unreachable code** (item 2): rustc's `dead_code` lint does not report `pub` items of a
  library, and almost every item in the crate is `pub`. `survey/privatize.py` copies the stripped
  tree and rewrites every bare `pub` to `pub(crate)`, so the only roots left are what PyO3
  exports to Python (the `#[no_mangle]` module init, the `#[pyclass]`/`#[pymethods]` impls and
  the registered `#[pyfunction]`s). `cargo check --features halo2-base-backend,noir-backend`
  (the paper's build) then reports every item no exported function reaches.
  `survey/deadlines.py` turns each report into the item's line range (from its first attribute
  or doc line to its closing brace) and adds the `impl` blocks of types that are never
  constructed. The crate has no blanket `#![allow(dead_code)]` (one item-level
  `allow(dead_code)`, in `helpers/static_array/shape.rs`).
- Removal experiments (`survey/mk.sh`): the module or entry point is deleted from the
  privatized copy and the crate must still type-check; what becomes unreachable is then counted.

## Where the 82,944 lines go

| Category | Code lines | Examples | Status |
|---|---:|---|---|
| **A. Retired halo2 (IPA) backend** | **3,711** | `prove/halo2/` (`synthesizer.rs` 2,172, `config.rs` 542, `mod.rs` 456, `vk_cache.rs` 240, `minimal.rs` 197) | Not one of the paper's two backends. `prove::create_prover_backend` refuses `"halo2"`/`"halo2-ipa"` (`LEGACY_HALO2_RETIRED`); its own header says it is kept for its unit tests. Deleting the module leaves the crate compiling, and no other code becomes unreachable. Compiled only because feature `halo2-base-backend` implies `halo2-backend`. |
| **B. Research-prototype glue** | 90 | `lib.rs`: `analyze_shapes`, `build_candidates`, `mod phase0/phase1` | Entry points of `src/phase0`, `src/phase1` (already excluded from D4). The core never references phase0/phase1. |
| **C. Test and evaluation entry points** | 280 | `lib.rs` `emit_structural_circuit` (witness-independence test), `execute_noir_circuit` (Noir differential test), `noir_param_identifier` (harness), `get_and_reset_prove_counters`, `get_elided_publics` (documented as superseded); `prove/halo2_base/python.rs` `halo2_base_cell_stats`/`cell_profile`/`configured_polys`; `halo2_base::structural_fingerprint*`, `noir::driver` execute path | 127 lines of entry points plus 153 lines only they reach. Callable from Python, but used by `testing/` and `benchmarking/`, not by compiling or proving a program. |
| **D. Unreachable code** | 1,846 (+55 unused `use` lines) | `ops/` 753 (the op-object layer ported from Python `op_def`: `AddOp`, `EqualOp`, `SelectOp`, ... "never constructed", with their impls); `types/` 349 (`ZinniaType::from_annotation`, `AnnotationArg`); `lib.rs`/`builder.rs`/`ir.rs`/`scope.rs` helpers 206; `optim/` 313 (`predicates/cache.rs`, unused resolver constructors, `prove_in_scope`); `prove/` 90 (`kernel.rs` unused coefficient tables, `NoExternalCallback`) | rustc reports 228 items (`survey/dead-items.txt`) after A–C are removed. Items used only by test code count here too. This is a lower bound: rustc does not report unused methods of trait impls, and code reachable only through a configuration value or a string the pipeline never passes counts as live (examples under "Not determined"). |
| **E. Measurement, ablation and debugging instrumentation** | about 910 | `optim/telemetry.rs` 422 (SMT telemetry, `ZINNIA_TELEMETRY_DIR`), `lower/local.rs` 131 (`ZINNIA_ABLATE_JOINT`), `lower/forcing.rs` 83 (`ZINNIA_FORCE_CONSTRUCTION`), `lower/report.rs` 73 (`ZINNIA_LAYER_REPORT`), about 205 lines of call sites (telemetry handles, `eprintln!` traces, `ZINNIA_DEBUG_*`, `ZINNIA_Z3_TRACE`, prove counters, `dump_layer_input`) | Live code, off unless an environment variable is set; the compiled circuit does not depend on it. The four files are exact; the call-site count is a grep of lines naming these facilities and undercounts multi-line blocks by a few dozen lines. The ablation switches drive the paper's RQ2 ablation, so this is evaluation tooling rather than compiler. |
| **F. Mock backend and IR interpreter** | 1,004 | `prove/mock/` 626, `prove/interpreter.rs` 378 | Kept in every definition below: calling a `@zk_circuit` function runs the program on it (`zinnia/api/zk_circuit.py`), and the paper names the interpreter. Listed for a stricter count. |
| **G. Python bindings (FFI layer)** | about 750 | `lib.rs` after B and C (535), `prove/halo2_base/python.rs` (175 live) | Kept: this is how the Python front end calls the core. |
| Rest: front end, symbolic execution, IR, optimizations, lowering, Halo2-lib and Noir backends | about 74,300 | see the per-directory table | |

Per directory (code lines, local counter), from D4 to definition 3 below:

| Directory | D4 | after A–C | unreachable (D) | instrumentation files (E) | definition 3 |
|---|---:|---:|---:|---:|---:|
| `prove` | 16,569 | 12,852 | 234 | 0 | 12,618 |
| `ops` | 13,696 | 13,696 | 753 | 0 | 12,943 |
| `helpers` | 11,739 | 11,739 | 53 | 0 | 11,686 |
| `ir_gen` | 10,608 | 10,608 | 15 | 0 | 10,593 |
| `optim` | 9,963 | 9,963 | 322 | 422 | 9,219 |
| `lower` | 9,814 | 9,814 | 38 | 287 | 9,489 |
| top-level files | 6,441 | 6,224 | 206 | 0 | 6,018 |
| `builder` | 2,718 | 2,718 | 29 | 0 | 2,689 |
| `types` | 977 | 977 | 349 | 0 | 628 |
| `ast` | 378 | 378 | 0 | 0 | 378 |
| **Total** | **82,903** | **78,969** | **1,999** | **709** | **76,261** |

(The unreachable column includes the 153 lines of C that only C reaches; the 55 unused `use`
lines and the about 205 instrumentation call-site lines are not in the per-directory columns.)
After the exclusions `prove` holds the Halo2-lib backend (`prove/halo2_base`, 5,910), the Noir
backend (`prove/noir`, 3,750) and about 3,000 lines shared by the backends (kernel, witness
preprocessing, interpreter, mock, backend trait).

## The questions asked

1. **Code outside the compiler the paper describes.** The large item is the retired halo2 IPA
   backend (A, 3,711). The rest is small: prototype glue (B, 90), test and evaluation entry
   points (C, 280), measurement and ablation instrumentation (E, about 910). Benchmark harness
   code is not in `src/` (it is in `benchmarking/` and `eval/`). There is no CLI. The two cargo
   features `halo2-base-backend` and `noir-backend` are both enabled in the paper's build, and
   no other `cfg(feature)` code exists apart from what feature `halo2-backend` gates, which is
   exactly A. Optimization passes off by default: `DynamicNDArrayMetaAssertInjection`
   (66 lines, enabled by a config flag that defaults to off) and `ExternalCallRemover`
   (25 lines, named in `apply_pass` but never put in the pipeline).
2. **Unreachable code.** 1,846 lines by rustc after A–C are removed (5,465 before removing A,
   since rustc also flags most of the retired backend), plus 55 unused import lines. It is
   concentrated in the op-object layer of `ops/` and in type-annotation parsing in `types/`.
3. **Generated or table-like code.** None of note. No file is generated (no `@generated`/"do not
   edit" markers, no build-script output in `src/`). Macros are counted as written, not as
   expanded: 14 `macro_rules!` keep, for example, each `ops/np_like/np_*.rs` file to one line.
   Lines that hold only numeric literals (Remez coefficients and similar) are 95, mostly in
   `prove/kernel.rs`; lines that hold only a quoted name (operator-name lists in
   `optim/predicates/contracts.rs`) about 55. Text embedded in multi-line string literals (Noir snippets) is 225 lines.
   The Noir fixed-point library is pulled in with `include_str!` and is not Rust.
4. **Doc comments.** Confirmed excluded. tokei reports doc comments as embedded Markdown
   counted under "comments": in D4, 15,117 doc-comment lines and 6,253 other comment lines,
   none in the 82,944.
5. **The Python package `zinnia/`.** 5,813 code lines (tokei) in 53 files. Compiler logic is
   `zinnia/compile/` (3,542: the AST transformer to the JSON the Rust core reads, `@requires`
   extraction, source-level lowering of local functions, callbacks, imports and iterator loops)
   and the input parser `zinnia/exec/input_parser.py` (341); the rest is the user API,
   configuration, type annotations and the runtime of external functions. The paper already
   calls this "a Python front end", separate from the Rust count; if a size is wanted:
   "a Python front end of about 6K lines".

## Candidate definitions

| # | Definition | Code lines |
|---|---|---:|
| 1 | All non-test Rust in `src/` except the two research prototypes (the current 83K) | 82,944 (tokei) |
| 2 | 1 without the retired halo2 IPA backend, prototype glue, test and evaluation entry points, and code no exported function reaches (A + B + C + D) | 76,915 |
| 3 | 2 without measurement, ablation and debugging instrumentation (E): the compiler as the paper describes it, from the AST through symbolic execution, IR, optimizations and lowering to the Halo2-lib and Noir backends, with the mock backend and interpreter | about 76,000 (76,261 by directory minus about 205 call-site lines and the 55 unused imports) |
| 3' | 3 without the mock backend and IR interpreter (F) | about 75,000 |

## Recommendation

Use definition 3 and write "a compiler core in Rust (about 76K lines of code, excluding tests)".
It counts what the sentence describes: the stages it lists ("symbolic execution, IR
optimization, lowering, and backend-specific synthesis") and the two backends the next paragraph
names, plus the interpreter the paper mentions. Every exclusion is either not part of that
compiler (a backend the code itself refuses, prototypes, test and measurement hooks) or code
nothing calls. Definition 2 (about 77K) differs by 1%, so "about 76K" also holds under the
looser reading. 83K overcounts by about 7K lines, mostly the retired halo2 backend (3.7K);
the rest is unreachable code (1.9K), instrumentation (0.9K) and test entry points (0.3K).

## Not determined

- **Legacy and native array representations side by side.** NumPy arrays have a native
  representation (`Value::StaticArray`, `helpers/static_array/`, 6,803 lines) and an older one,
  nested `Value::List` (used by much of `ops/static_ndarray_ops/` and `helpers/array_ops/`).
  Operators without a native path convert at their entry (`to_value_list`, 22 files, about 90
  call sites marked "legacy"), and some operators with a native path keep the list path as a
  fallback. Both are live, so neither rustc nor file boundaries separate the duplicated part;
  measuring it needs line coverage (`-C instrument-coverage`; `llvm-cov` is installed) over the
  test suite, which I did not run. A coverage run over only the benchmark programs would not
  answer it either, because unexecuted lines there include every NumPy feature the 100
  programs do not use.
- **Reachable but never taken code.** rustc treats a branch behind a configuration value or a
  pass name as live (for example the `ExternalCallRemover` arm and `enable_memory_consistency`).
  I found only the small cases listed under question 1 by reading `compile_circuit_impl`, not by
  a systematic search.
- **Unused trait-impl methods** are not reported by rustc, so D is a lower bound.

## Reproducing

Scripts in `survey/` (paths in `mk.sh` are relative to a scratch directory holding `work/`, the
`git archive` above, and `crate0/`, a copy of `Cargo.toml`, `Cargo.lock`, `build.rs`,
`benchmarking/noir_fpc` with `src/` = `work/nontest/src`):

```sh
python strip_tests.py work/src work/nontest/src              # D4 tree
tokei -t Rust work/nontest/src -e phase0 -e phase1             # 82,944
python survey/count.py work/nontest/src phase0 phase1          # 82,903 (= cloc)
# env: LIBRARY_PATH=<z3 wheel lib>, PYO3_PYTHON=.venv/bin/python,
#      PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 (local Python is 3.14), RUSTC_BOOTSTRAP=1
./mk.sh e3 "emit_structural_circuit,execute_noir_circuit,noir_param_identifier,get_and_reset_prove_counters,get_elided_publics" \
  "sed -i '/halo2_base_cell_stats, m\|halo2_base_cell_profile, m\|halo2_base_configured_polys, m/d' src/prove/halo2_base/python.rs"
```

`survey/dead-items.txt` lists the 241 `dead_code` reports of that build (the 228 of D plus those only C reaches), and
`survey/dead-items-by-file.txt` their code lines per file.
