# SP1, RISC Zero and Cairo baselines

Handwritten baselines for the DS-1000 performance set on two RISC-V zkVMs (SP1,
RISC Zero) and on Cairo. Every program computes what the case's `kernel.py`
computes, in Q32 fixed-point integers like Zinnia (author decision 2026-10-06),
reads every input privately and makes every output public.

## Layout

| Path | What |
|---|---|
| `datasets/ds1000/<case>/zkvm.rs` | the case's Rust program, shared by SP1 and RISC Zero |
| `datasets/ds1000/<case>/main.cairo` | the case's Cairo program |
| `baselines/zkvm_common/q32/` | Q32 arithmetic (`no_std`, no floating point) |
| `baselines/zkvm_common/ds1000/` | crate `ds1000-zkvm`: includes every `zkvm.rs` as a module; input reader trait, public-value decoding, fixture loading, the shared host CLI/report, and the native runner `ds1000-native` |
| `baselines/zkvm_common/check_outputs.py` | output check against NumPy |
| `baselines/zkvm_common/sync_cases.py` | regenerates the per-case glue (case list, guest binaries, Cairo targets) |
| `baselines/sp1/` | SP1 6.8.1: `guest/` (one binary per case), `host/` (`sp1-baseline`) |
| `baselines/risc0/` | RISC Zero 3.0.3: `methods/guest/` (one binary per case), `host/` (`risc0-baseline`) |
| `baselines/cairo/` | Scarb 2.20.1 package `ds1000_cairo` (one executable per case), `src/q32.cairo`, driver `cairo_baseline.py` |

One build per system covers every case: one guest package with a binary per
case (one target directory), one host binary that loads the case's guest at
run time, one Scarb package with an executable target per case.

## Building

```sh
cd benchmarking/baselines
python3 zkvm_common/sync_cases.py                       # after adding/removing a case
(cd zkvm_common/ds1000 && CARGO_TARGET_DIR=../../target cargo build --release --features std)   # native runner
(cd sp1 && PATH=$HOME/.sp1/bin:$PATH cargo build --release)        # builds every SP1 guest too
(cd risc0 && cargo build --release)                                # builds every RISC Zero guest too
(cd cairo && ~/.local/share/scarb/scarb-v2.20.1-x86_64-unknown-linux-gnu/bin/scarb build)
```

Build heavy targets one at a time under `eval-redo/tools/memrun.sh`.

## Program contract

All three systems take the same arguments and print one JSON object:

```sh
sp1/target/release/sp1-baseline     --case-dir D [--fixture F] [--input A|B] --mode setup|execute|prove|verify [--proof-mode core|compressed|groth16] [--work-dir W] [--out R] [--tamper]
risc0/target/release/risc0-baseline --case-dir D ... [--proof-mode composite|succinct|groth16]
python3 cairo/cairo_baseline.py     --case-dir D ... [--proof-mode stwo]
target/release/ds1000-native        --case-dir D ... --mode execute
```

The JSON has `ok`, `publics` (decoded output values in kernel output order:
floats from Q32, integers, booleans), `proof_path`, `proof_size_bytes`,
`timings_s` (one entry per step, e.g. `client_init`, `setup`, `execute`,
`prove`, `compress_succinct`, `wrap_groth16`, `verify`), cycle counts
(`cycles` for SP1; `user_cycles`, `total_cycles`, `segments` for RISC Zero;
`resources.n_steps` for Cairo) and `peak_rss_mib`. `--tamper` (verify mode)
flips one bit of the last public output byte; the run succeeds only if
verification rejects the proof. Keys and proofs live in `--work-dir`
(default `<system>/work/<case>/`, git-ignored).

## Writing a case

A case has the same parameters, shapes, constants and algorithm as
`kernel.py`, with NumPy semantics (the output check compares with NumPy).

### Numbers

* Floats are `Q32` (`q32` crate; `Q32` struct in `cairo/src/q32.cairo`): the
  host encodes `round(v * 2^32)`, Zinnia's quantizer.
* `+`, `-`, comparisons and `==` are exact. `a * b` rescales with floor,
  `a / b` is Euclidean; `x * n` and `x / n` for an integer `n` (Cairo:
  `mul_int`, `div_int`) are exact multiplication and `x / from_int(n)`.
* Library contractions (`np.dot`, `tensordot`, `correlate`, `matmul`,
  `einsum`) accumulate raw products and rescale once: Rust `dot`, Cairo
  `raw_product` + `from_raw_product`.
* `np.sqrt` is `.sqrt()` (`floor(sqrt(x * 2^32))`); `np.sin`, `np.cos`,
  `np.arcsin` are `.sin()`, `.cos()`, `.asin()` (Zinnia's own definitions; see
  the `q32` crate docs); `np.deg2rad`/`np.degrees` multiply by `DEG_TO_RAD` /
  `RAD_TO_DEG`.
* Integers are `i64`, booleans `bool`. Integer `//` and `%` follow Python
  (`floor_div`, `floor_mod`); plain `/` and `%` only where both operands are
  known non-negative. `int(x)` / `astype(int)` is `trunc_to_int`;
  `astype(float)` / int-to-float promotion is `Q32::from_int`.
* No `f64`/`f32` anywhere in a guest.

### Inputs and outputs

* Read every parameter in signature order: arrays flattened row-major
  (`input.read_array::<Q32, N>()`, `read_matrix::<T, R, C>()`,
  `read_vec::<T>(len)`), scalars with `input.read::<i64>()` etc. Floats are
  `Q32`, ints `i64`, bools `bool`. Cairo: `main` takes `Array<Q32>`,
  `Array<i64>`, `Array<bool>`, `Q32`, `i64`, `bool` in the same order and
  asserts the array lengths.
* `pub type Output` is the kernel's return value: a scalar, a `Vec` (an
  array flattened row-major; its length may depend on the data), or a tuple
  for a tuple return. The output dtype follows NumPy (a mean of integers is a
  float, an index is an `i64`, a mask is `bool`). Cairo returns the same
  structure and declares it on a line `// output: <layout>` with `q32`, `i64`,
  `bool`, `[T]` (array) and `(T, U, ...)` (tuple).
* Shape parameters that the kernel fixes are constants in the program.

### Style

Write what a competent developer would write for this zkVM: direct indexing
and branches (cheap in a zkVM), the standard library (`sort_unstable`,
iterators), one pass where one pass suffices, no dead work, no transliteration
of Zinnia's circuit lowering (no one-hot selects or select chains where an
index works). Cairo arrays are append-only: build outputs by appending in
order, compute gathers directly, and use `Felt252Dict` only where a real
scatter is needed. Panic where NumPy would raise (out-of-range index, empty
reduction).

## Output check

```sh
.venv/bin/python benchmarking/baselines/zkvm_common/check_outputs.py \
    --systems native,sp1,risc0,cairo [--cases a,b] [--fixtures-dir DIR]
```

For each case it runs the kernel body as plain NumPy on fixture input `B` and
compares the public values position by position: integers and booleans
exactly, floats within the case-set tolerance (default `atol = rtol = 1e-6`).
