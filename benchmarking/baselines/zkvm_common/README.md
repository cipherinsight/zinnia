# SP1, RISC Zero and Cairo baselines

Handwritten baselines on two RISC-V zkVMs (SP1, RISC Zero) and on Cairo, for
the DS-1000 performance set (all three) and the NPBench/Pythran acceptance
programs (SP1, RISC Zero, and the Cairo entries with a source file). Every program computes what its `kernel.py`
computes, in Q32 fixed-point integers like Zinnia (author decision 2026-10-06),
reads every input privately and makes every output public.

## Layout

| Path | What |
|---|---|
| `datasets/ds1000/<case>/zkvm.rs` | the case's Rust program, shared by SP1 and RISC Zero (fixed shapes) |
| `datasets/{npbench,pythran}/<program>/zkvm.rs` | the program's Rust entry functions, shape-generic (one case per entry function, `<program>/<function>`) |
| `datasets/ds1000/<case>/main.cairo` | the fixed-shape Cairo program |
| `datasets/{npbench,pythran}/<program>/cairo/<function>.cairo` | one shape-generic Cairo entry, target `<program>__<function>` |
| `baselines/zkvm_common/q32/` | Q32 arithmetic (`no_std`, no floating point) |
| `baselines/zkvm_common/kernels/` | crate `zkvm-kernels`: includes every `zkvm.rs` as a module; inputs (`src/io.rs`: `Input`, `Args`, `Element`), outputs (`src/publics.rs`), NumPy behaviours shared by several cases (`src/np.rs`), fixture loading, the shared host CLI/report, and the native runner `kernels-native` |
| `baselines/zkvm_common/check_outputs.py` | output check against NumPy |
| `baselines/zkvm_common/sync_cases.py` | regenerates the per-case glue (case list, guest binaries, Cairo targets) |
| `baselines/sp1/` | SP1 6.8.1: `guest/` (one binary per case), `host/` (`sp1-baseline`) |
| `baselines/risc0/` | RISC Zero 3.0.3: `methods/guest/` (one binary per case), `host/` (`risc0-baseline`) |
| `baselines/cairo/` | Scarb 2.20.1 package `ds1000_cairo` (one executable per case), `src/q32.cairo`, shared NumPy behaviours `src/np.cairo`, driver `cairo_baseline.py` |

One build per system covers every case: one guest package with a binary per
case (one target directory), one host binary that loads the case's guest at
run time, one Scarb package with an executable target per case.

## Building

```sh
cd benchmarking/baselines
python3 zkvm_common/sync_cases.py                       # after adding/removing a case
(cd zkvm_common/kernels && CARGO_TARGET_DIR=../../target cargo build --release --features std)   # native runner
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
target/release/kernels-native        --case-dir D ... --mode execute
```

For an NPBench/Pythran program with several entry functions, add
`--function F`. Its fixture defaults to `D/fixture.F.json` if that exists,
else `D/fixture.json`. A tuple parameter travels as its items, one parameter
buffer each, in order.

The JSON has `ok`, `publics` (decoded output values in kernel output order:
floats from Q32, integers, booleans), `proof_path`, `proof_size_bytes`,
`timings_s` (one entry per step, e.g. `client_init`, `setup`, `execute`,
`prove`, `compress_succinct`, `wrap_groth16`, `verify`), cycle counts
(`cycles` for SP1; `user_cycles`, `total_cycles`, `segments` for RISC Zero;
`resources.n_steps` for Cairo) and `peak_rss_mib`. `--tamper` (verify mode)
flips one bit of the last public output byte; the run succeeds only if
verification rejects the proof. Keys and proofs live in `--work-dir`
(default `<system>/work/<case>/`, git-ignored).

## Proof modes and resources

Measured on this machine (Ryzen 7 9800X3D, CPU only) on the calibration
program `empty_program` and case 0300; `peak_rss_mib` in every JSON report
gives the figure per run. Use one heavy job at a time.

| System | Mode | What it is | Peak memory | Notes |
|---|---|---|---|---|
| SP1 6.8.1 | `setup` | `CpuProver::new()` + `setup(elf)` | 7.1 GB | prover start-up alone is 16-21 s (`client_init`), reported apart |
| SP1 | `core` | one STARK per execution shard | 9.6-9.7 GB for small programs; more than 16 GB for 0385 (1.7M cycles; OOM-killed at a 16 GB cap) | ~16 s for 2k-7k cycles (mostly fixed cost); proof 2.8 MB |
| SP1 | `compressed` | shards recursively compressed to one STARK | 17.1 GB | 60 s for 0300; proof 1.3 MB; needs a cap above 16 GB |
| SP1 | `groth16` | compressed proof wrapped in Groth16 over BN254 (gnark linked in-process, `native-gnark`) | > 17.8 GB before the gnark step (measured); not yet run to completion | first use downloads 6.2 GB of circuit artifacts to `~/.sp1/circuits/groth16/v6.1.0`; run alone with a cap of at least 32 GB |
| SP1 | `verify` | `LightProver` | < 0.1 GB | 40-70 ms |
| RISC Zero 3.0.3 | `composite` | one STARK per segment | 0.3-0.6 GB for one 2^15-2^16-cycle segment; 9.4 GB for 0385 (two 2^20 segments) | 2.2-2.6 s small, 137 s for 0385; receipt ~210 KB small, 880 KB for 0385 |
| RISC Zero | `succinct` | segments compressed to one STARK by recursion | 1.4 GB small, 9.4 GB for 0385 | composite + 6-10 s (small), + 22 s (0385) |
| RISC Zero | `groth16` | succinct receipt wrapped in Groth16 (Docker image `risczero/risc0-groth16-prover:v2025-04-03.1`) | not measured | the Docker step runs outside this process and outside `memrun.sh`'s cgroup, so `peak_rss_mib` does not include it; not yet run |
| RISC Zero | `verify` | `Receipt::verify` | < 0.1 GB | 10-20 ms |
| Cairo (Scarb 2.20.1) | `stwo` | `scarb execute --target bootloader` + `scarb prove --proof-format binary` (Stwo; the same proof again as JSON, untimed, for `scarb verify`) | 13.7 GB (3.6k steps) to 14.7 GB (0385, 1.01M steps) | 10-19 s from the empty program to 1M steps (fixed cost dominates); binary proof 0.95 MB for the empty program (JSON 11-16 MB) |
| Cairo | `verify` | `scarb verify` | < 0.1 GB | 30-55 ms wall clock including process start-up |

Why Groth16 for SP1's SNARK mode (rather than PLONK): gnark's Groth16 prover
does less work than its PLONK prover on the same circuit, so it is expected to
be the faster wrap (not measured here), and its proof is the smaller one
(SP1 documents about 260 bytes against about 870), so it is the stronger
baseline. It is also the only SNARK wrapper RISC Zero offers, so both zkVMs end
in the same kind of proof. Its cost is a larger one-time download (6.2 GB
against 2.3 GB for PLONK). The submitted paper measured SP1's PLONK wrap; the
mode is one enum value away (`SP1ProofMode::Plonk`) if that series is wanted
too.

## Writing a case

A case has the same parameters, shapes, constants and algorithm as
`kernel.py`, with NumPy semantics (the output check compares with NumPy). The
rules below hold for both kinds of program; "Shape-generic programs" adds what
is specific to NPBench/Pythran.

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
  `RAD_TO_DEG`. `np.exp`, `np.log`, `np.tanh`, `np.arctan` are `.exp()`,
  `.ln()`, `.tanh()`, `.atan()` (textbook fixed-point algorithms, about 1e-9;
  Zinnia's halo2-base backend has none of them); `x ** n` for an integer `n`
  is `.powi(n)`, `x ** y` for a float `y` is `.powf(y)`.
* Complex numbers are `C32 { re, im }` (both Q32): `+ - * /`, `.conj()`,
  `.abs()`, `.norm_sqr()`, `.exp()`, `C32::I`; products rescale once per part.
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
* When the output's shape depends on the data and is not implied by its
  length (a 2-D crop whose rows and columns both vary), commit the shape with
  the values: Rust `Shaped { shape, values }`, Cairo a tuple
  `(Array<i64>, Array<T>)` declared as `([dim], [T])`. The hosts report the
  shape as `output_shape`; `publics` keeps only the values.
* Shape parameters that the kernel fixes are constants in the program.
* A parameter the kernel never reads (e.g. 0402's `a`) is declared on a line
  `// unused parameters: a` in `zkvm.rs` and `main.cairo`; the hosts and the
  Cairo driver leave it out of the private input, as Zinnia removes it from
  the circuit. Run `sync_cases.py` after adding such a line.

### Shape-generic programs (NPBench, Pythran)

* One top-level `pub mod <function> {` per in-scope entry function of the
  program (the function's name as in `kernel.py`; add
  `#[allow(non_snake_case)]` when needed). It defines `pub type Output` and
  `pub fn main(args: &mut Args<impl Input>) -> Output`. Helpers shared by the
  entry functions are private items of the file.
* Read the parameters in signature order: `args.scalar::<T>()`,
  `args.array1::<T>()` ... `args.array4::<T>()` (`Array1`...`Array4` of
  ndarray 0.16), with `T` = `Q32` (float), `i64` (int), `bool`, `C32`
  (complex). Arrays carry their shapes, so the program works for any shape
  the fixture gives; take extents from the arrays (`a.dim()`, `a.nrows()`),
  never from `kernel.py`'s annotation. The guest commits the input shapes,
  so the proof binds them.
* `Output` is the returned value: `Q32`, `i64`, `bool`, `C32`, an ndarray
  (committed with its shape), a `Vec` (1-D output whose length depends on the
  data), a tuple, or `Option<T>` where the function can return `None`.
* An entry function that returns `None` has as outputs the array arguments
  its body writes (subscript, slice or augmented assignment, `out=`), in
  parameter order: `Output` is that array or a tuple of them, and the module
  carries a line `// outputs: mutated A, B` naming them (the output check
  reads it).
* Preconditions (`@requires`) are asserted.
* Use ndarray for storage, indexing (`a[(i, j)]`) and slicing (`s![..]`);
  write the arithmetic as loops or `Zip`/`mapv` as reads naturally. Matrix
  products: `np::matmul`, `np::matvec`, `np::vecmat`, `np::inner` (fused dot
  products). Follow NumPy's evaluation order (`alpha * A @ B` is
  `(alpha * A) @ B`).

### Shape-generic Cairo entries

Write **one `cairo/<function>.cairo` file**, then run `sync_cases.py`. The
function must already be an entry in the folder's `zkvm.rs`. Sync generates
its symlink, `cases.cairo` module and Scarb executable target
`<program>__<function>`. Fixed DS-1000 `main.cairo` files keep their original
signature, encoding and output declarations. No Rust file or case set needs
to change to add a Cairo entry.

Each shaped Cairo executable takes `Array<Array<i64>>` and returns
`Array<i64>`. Use `crate::io::{ArgsTrait, Word, tensor, vector}`:

```cairo
// output: tensor<1,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::Q32;

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let values: Array<Q32> = args.array1();
    // Read every other parameter, then compute the result here.
    let mut out = args.output();
    tensor(ref out, array![values.len()].span(), values.span());
    out
}
```

* Parameters travel in signature order, tuple items expanded recursively.
  `args.scalar::<T>()` reads one scalar, `args.array1::<T>()` an array;
  `array2` returns `(rows, cols, values)`, `array3`/`array4` similarly return
  extents followed by values. `args.array(rank)` returns `(dims, values)`.
  Supported elements are `i64`, `bool`, `Q32`, and `C32`.
* The driver writes Rust's parameter words: array elements first, dimensions
  last, with Cairo Serde lengths around the buffers. It infers dimensions
  from rectangular fixture values and checks the fixture's shape metadata.
  The guest validates dimensions, element counts, scalar lengths and booleans.
  Derive bounds from those dimensions, never from annotation or fixture sizes.
* `args.output()` rejects unread parameters and starts the output with
  `[number_of_input_dimensions, input_dimensions...]`, exactly the Rust guest's
  shape-vector commitment. Append outputs **after** this prefix. The returned
  Cairo array adds one Serde length around the full word buffer; after removing
  that envelope, `public_words` is bit-identical to Rust. Negative words travel
  as signed felts in Cairo, and signed little-endian i64 words in Rust.
* `Word::write(value, ref out)` appends a scalar; `tensor(ref out, dims, values)`
  appends dimensions and elements without a redundant data length; `vector`
  appends a Rust `Vec` (length, elements). A rank-one ndarray also encodes
  length then elements, but declares `tensor<1,T>`.
* Declare the **result only** on `// output:`: `q32`, `i64`, `bool`, `c32`,
  `tensor<1,q32>` through rank 4, `[T]` for a Rust `Vec`, `(T,U,...)` for a
  tuple, `()` for unit, and `option<T>` for Rust `Option` (append 0 for None
  or 1 followed by T for Some; do not use Cairo Option's enum tags).
  The driver strips input shapes before decoding result values. `input_shapes`
  and `public_words` are separate report fields; tensor dimensions are also
  reported in `output_shape`.
* Copy each Rust entry's `// unused parameters: a, b` and
  `// outputs: mutated A, B` declarations into its Cairo file. Sync checks
  unused declarations against Rust. The output checker checks mutated
  declarations and compares NumPy's written arguments in that order.
* Cairo arrays are append-only. Use ordinary arrays/spans for reads and
  sequential outputs; use `Felt252Dict<felt252>` for genuine mutations (see
  bubble sort). `Felt252Dict<i64>` is not supported by Cairo's dictionary
  value trait; store the signed integer as a felt and convert on reads.
* Fused contractions are `np::inner(a,b)`, `matmul(a,b,rows,inner,cols)`,
  `matvec(a,b,rows,cols)` and `vecmat(a,b,rows,cols)`, on flattened Q32 spans.
  They accumulate `i128` raw products and rescale once. Preserve Rust's
  evaluation order and broadcasting: GEMM scales A before matmul and
  broadcasts the product to C as `ndarray::zip_mut_with` does.
* `Q32Trait` includes `exp`, `ln`, `tanh`, `atan`, `atan2`, `powi`, `powf`,
  and `floor`, in addition to the existing operations. `powi` multiplies
  sequentially, not by exponentiation by squaring. `C32Trait` provides
  `new`, `from_real`, `conj`, `norm_sqr`, `abs`, `exp`, `scale(Q32)` and
  `div_real(Q32)`; `C32` supports `+ - * /` and unary `-`. Constants in
  `q32` include `C_ZERO`, `C_ONE`, `I`, `HALF`, `LN_2`, `LOG2_E`.

Build only the entries being ported (Scarb 2.20.1 supports `--target-names`):

```sh
# From the repository root. Put the Scarb distribution's bin on PATH too:
# scarb discovers scarb-execute and scarb-cairo-test there.
export PATH="$HOME/.local/share/scarb/scarb-v2.20.1-x86_64-unknown-linux-gnu/bin:$PATH"
.venv/bin/python benchmarking/baselines/zkvm_common/sync_cases.py
MEMRUN_MAX=8G /home/zhantong/Projects/zinnia-project/eval-redo/tools/memrun.sh \
  scarb --manifest-path benchmarking/baselines/cairo/Scarb.toml build \
  --target-names npbench_crc16_crc16__crc16,npbench_polybench_gemm_gemm__kernel
# SCARB_TARGET_NAMES accepts the same comma-separated list. Driver --mode setup
# also builds only its selected case.
```

For validation on this machine, point `SCARB` at a wrapper that exports
`MEMRUN_MAX=8G` and calls `memrun.sh scarb "$@"`; this caps each driver execution.
The phase-1 wrapper is `eval-redo/cairo-work/p1/scarb-mem`. Execute only:

```sh
export SCARB=/home/zhantong/Projects/zinnia-project/eval-redo/cairo-work/p1/scarb-mem
.venv/bin/python benchmarking/baselines/zkvm_common/check_outputs.py \
  --systems native,cairo --cases npbench_crc16_crc16/crc16 --input A
.venv/bin/python benchmarking/baselines/zkvm_common/check_outputs.py \
  --systems native,cairo --cases npbench_crc16_crc16/crc16 --input B
.venv/bin/python benchmarking/baselines/cairo/cairo_baseline.py \
  --case-dir benchmarking/datasets/npbench/npbench_crc16_crc16 \
  --function crc16 --input B --mode execute
# Stronger check: all committed i64 words, including input and result shapes,
# against the unchanged native Rust program, on both A and B:
.venv/bin/python benchmarking/baselines/cairo/tests/check_public_words.py \
  --cases npbench_crc16_crc16/crc16
```

`check_outputs.py` accepts both `program/function` and `program__function`;
`check_public_words.py` uses `program/function` and accepts `--fixtures-dir`
for scratch fixtures. Build `kernels-native` first as above; the word checker
links a read-only oracle to its existing Rust library. The harness locates
`cairo/<function>.cairo` using the case set's `shape_generic` and `function`
fields and passes `--function` automatically. No harness run is needed for
local validation, and **never prove locally** for the Cairo porting work.

Library checks (execute tests, no proofs):

```sh
.venv/bin/python benchmarking/baselines/cairo/tests/generate_q32_vectors.py
(cd benchmarking/baselines/cairo && "$SCARB" test)
```

The generator compiles the unchanged `zkvm_common/q32/src/lib.rs` with rustc
and writes exact integer expectations to `src/q32/rust_vectors.cairo`.
Tests cover argument-reduction boundaries, underflow, powers of two, tiny
positive logarithm inputs, quadrants, repeated powers, complex rounding and
invalid domains. Phase 2 should keep case-specific helpers in its own file;
report any further shared-library helper needed rather than editing the
phase-1-owned `q32.cairo` or `np.cairo` concurrently.

### How the Rust guests move data

Every parameter travels as one buffer of 64-bit little-endian words (a Q32
by its encoding, an integer as itself, a boolean as 0/1, a complex number as
two Q32 words; a shape-generic program's arrays end with their dimensions),
read with the zkVM's raw-slice call (SP1 `io::read_vec`, RISC Zero
`env::read_slice` of a length word and then the words). The output is
encoded into one word buffer (`Publics::encode`: a scalar is its words, a
vector its length then its items, an ndarray its dimensions then its
elements, a tuple its items in order; a shape-generic guest commits the
input shapes first) and committed
with one raw call (SP1 `io::commit_slice`, RISC Zero `env::commit_slice`).
This is the bulk-data path both zkVMs document; their serde codecs
(`io::read`/`env::read`, `commit`) cost hundreds of cycles per element
(RISC Zero, 1,000-element copy 0317: 493k user cycles with serde, 52k with
raw words; SP1 0385, 20,000 elements: 13.7M cycles with serde and software
SHA-256, 1.7M now). The SP1 guest also uses SP1's patched `sha2`, so the
public values are hashed on SP1's SHA-256 precompile.

### Style

NumPy behaviours that several cases need (`np_index`, `argmax`, `mean`,
`isclose`, `argsort`, `np.insert`'s position, the bounding-box crop, ...) live
once in `ds1000/src/np.rs` (Rust: `np::argmax(&a)` through the prelude) and
`cairo/src/np.cairo` (`use crate::np::argmax;`); use them instead of a local
copy, and move a helper there once a second case needs it.

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

For each case it runs the kernel as plain NumPy on fixture input `B` (or
`--input A`) and compares the public values position by position: integers
and booleans exactly, floats within the case-set tolerance (default
`atol = rtol = 1e-6`), complex values as (real, imaginary) pairs. For a
shape-generic entry function that returns `None`, NumPy's outputs are the
arguments named on its `// outputs: mutated` line, after the call. Cases are
named as in the hosts: `ds1000_case0300`, `pythran_lu/factorMatrix0`.
