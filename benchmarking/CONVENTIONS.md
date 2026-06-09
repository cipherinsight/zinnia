# DS-1000 baseline conventions

This file documents the apples-to-apples rules every per-case baseline
(`target.rs`, `sp1.prog.rs`/`sp1.driver.rs`, `risc0.prog.rs`/`risc0.driver.rs`,
`main.cairo`, `main.nr` + `Prover.toml`) must satisfy. The primary
benchmarking goal is efficiency comparison across frameworks, so each
baseline must do the same logical work in the same shape.

## 1. Algorithm and control flow

All 5 baselines for a case are **high-level reimplementations of the
kernel.py**: same algorithm, same control flow, same loop bounds, same
intermediate variables, same operation order. The handwritten halo2
baseline mirrors what a Rust developer would write to express the
kernel in halo2-graph idioms — **not** a one-to-one transliteration of
Zinnia's IR (Zinnia's IR is an artifact of lowering, not the spec).

Idiomatic gadget usage is encouraged:

- **`fpc.qcos` / `qsin` / `qpow` / `qsqrt` / `qabs` / `qexp` / `qlog`
  / `inner_product`** — use these for the corresponding NumPy ops
  (`np.cos`, `np.sin`, `np.linalg.norm`, etc.). They match Zinnia's
  semantic of the operation; the fact that Zinnia lowers them to
  polynomial chains is an implementation detail of the lowering, not
  a requirement on the handwritten baseline.
- `.powi` / `.pow` / `**` for compile-time exponents in SP1/Risc0 are
  still **disallowed** — expand to an explicit multiplication chain so
  the cost is comparable to halo2's `qmul` chain.
- `f64::cos` / `.sin` / `.sqrt` / `.exp` / `.ln` ARE allowed in
  SP1/Risc0 (the user explicitly authorized native transcendentals
  there: "do not use taylor series; they have builtin operators").
- Cairo lacks native transcendentals — use a 4-term polynomial
  approximation (see `ds1000_case0323/main.cairo` for the cos Taylor
  template).

## 2. Halo2 lowering style

Treat halo2-graph as the substrate (FixedPointChip + GateChip +
RangeChip). Write the kernel like a Rust developer using that library
— Rust `for` loops, idiomatic gadget calls, helper closures where they
clarify the algorithm. The structure should mirror `kernel.py`, not
Zinnia's emitted IR.

Common gadget mappings:

| NumPy op                  | halo2-graph call                                                       |
|---------------------------|------------------------------------------------------------------------|
| `np.add` / `+` (float)    | `fpc.qadd(ctx, a, b)`                                                  |
| `np.sub` / `-` (float)    | `fpc.qsub(ctx, a, b)`                                                  |
| `np.mul` / `*` (float)    | `fpc.qmul(ctx, a, b)`                                                  |
| `np.div` / `/` (float)    | `fpc.qdiv(ctx, a, b)`                                                  |
| integer arithmetic        | `gate.add` / `gate.sub` / `gate.mul`                                   |
| `np.cos` / `np.sin`       | `fpc.qcos(ctx, x)` / `fpc.qsin(ctx, x)`                                |
| `np.exp` / `np.log`       | `fpc.qexp(ctx, x)` / `fpc.qlog(ctx, x)`                                |
| `np.sqrt`                 | `fpc.qsqrt(ctx, x)`                                                    |
| `np.power(a, b)`          | `fpc.qpow(ctx, a, b)`                                                  |
| `np.abs`                  | `fpc.qabs(ctx, x)`                                                     |
| inner product / dot       | `fpc.inner_product(ctx, a_vec, b_vec)` (also `np.tensordot`, `np.dot`) |
| `<` (int/float)           | `range_chip.is_less_than(ctx, lhs, rhs, 128)`                          |
| `==` (int)                | `gate.is_equal(ctx, lhs, rhs)`                                         |
| `==` (float)              | `qsub` + two `range_chip.is_less_than(±ε)` + `gate.and`                |
| `select(cond, t, f)`      | `gate.select(ctx, t, f, cond)`                                         |
| `not` / `or` / `and`      | `1 - x` / `a + b - a·b` / `gate.mul`                                   |
| assert                    | `gate.assert_is_const(ctx, &cond, &F::ONE)`                            |
| public output             | `make_public.push(value)`                                              |

## 3. Number representation per framework

| Framework | Native rep                | Notes                                                                                      |
|-----------|---------------------------|--------------------------------------------------------------------------------------------|
| Halo2     | FixedPointChip<F, 32>     | **2^32 fixed-point** to match Zinnia's `precision_bits = 32` default (`src/prove/halo2/artifact_format.rs:166`, `src/prove/types.rs:22`). Use `quantization(f64)` for every input. |
| SP1       | native `f64`              | zkVMs natively support `f64` — this is the realistic comparison point.                     |
| Risc0     | native `f64`              | Same as SP1.                                                                               |
| Cairo     | u128/i128 integer quantisation | **×2^32 = 4_294_967_296** to match Zinnia's `precision_bits = 32` default (see `src/prove/halo2/artifact_format.rs:166`, `src/prove/types.rs:22`). Use `i128` when the kernel uses signed values. |
| Noir      | Field (with u64/u128 view)| Same quantisation as Cairo (×2^32). Order comparisons require an integer cast.             |

The Cairo scale is fixed at 2^32 to match Zinnia's quantizer
(`prove/field.rs::quantize`, which computes `(value * 2^precision_bits) as i128`),
so the integer literals encode the same numeric value the halo2 backend
sees. Inside Cairo, this means:

- A `f64` literal `v` becomes `(v * 2^32) as i128` — pre-compute as
  `int_part * Q + frac_part`, e.g. `0.11 ≈ (11 * Q) / 100` since
  Rust truncates.
- Multiplication of two Q-scaled values needs an explicit `/ Q`
  rescale to stay in Q-scale (mirrors `FixedPointChip::qmul`).
- Square root of a Q-scaled value `q_x` returns `q_y = sqrt(q_x * Q)`
  (mirrors the legacy fixed-point sqrt witness).
- Integer u128/i128 values fit Q-scale floats safely: a single Q-scaled
  product peaks at 2^64 (still well under 2^128).

## 4. Assertion shape

Every baseline reads inputs **and** a claimed result, recomputes, and
asserts equality. Equality is realised per framework:

- Halo2 (float result): EqualFIR pattern (`qsub` + two `is_less_than` ± ε
  + `and` + `assert_is_const`). ε defaults to **0.001**; cases that
  accumulate Taylor or qmul rounding (e.g. 0323) loosen it.
- Halo2 (integer result): `gate.is_equal` + `assert_is_const`.
- Cairo / Noir (integer values): exact `==`. If a tolerance is needed
  (e.g. shared Taylor with quantisation rescaling), express it as two
  range checks (`x + tol ≥ y` and `x ≤ y + tol`) since there's no `abs`.
- SP1 / Risc0 (f64): `(computed - claimed).abs() < ε` with ε matching
  the halo2 baseline's ε.

## 5. Sample inputs

Sample inputs are deterministic and chosen so the assertion is
**meaningful but not trivially passing**:

- argmax / argmin: place the extremum at a non-trivial cell (not the
  first or last element).
- mean: choose inputs so the per-index sum is exactly divisible by the
  count, so the multiplication-form check (`sum == n * mean`) is exact.
- gather (e.g. `diag`): use distinct, deterministic values so the
  indexing is observably correct.
- transcendentals: pick a degree / value with a recognisable result
  (cos(60°)=0.5, sin(90°)=1, etc.).

Inputs are baked into Cairo as literals, shipped to Noir via
`Prover.toml`, and written from the driver Rust files for SP1 / Risc0.
The halo2 baseline expects them via the `CircuitInput` struct (JSON).

## 6. Files per case

| File                    | Framework | Notes                                                          |
|-------------------------|-----------|----------------------------------------------------------------|
| `kernel.py`             | Zinnia    | Pre-existing — the canonical kernel.                            |
| `record.json`           | DS-1000   | Pre-existing — original test cases and metadata.                |
| `target.rs`             | Halo2     | Self-contained example for `halo2-graph`'s `cargo run --example`.|
| `sp1.prog.rs`           | SP1       | Guest program.                                                  |
| `sp1.driver.rs`         | SP1       | Host driver (proves + verifies).                                |
| `risc0.prog.rs`         | Risc0     | Guest program.                                                  |
| `risc0.driver.rs`       | Risc0     | Host driver.                                                    |
| `main.cairo`            | Cairo     | Hardcoded inputs as literals.                                   |
| `main.nr`               | Noir      | Function taking inputs + claimed result.                        |
| `Prover.toml`           | Noir      | Noir prover input.                                              |
| `SKIPPED.md` (optional) | n/a       | Per-case notes on simplifications or skipped frameworks.        |

## 7. Zinnia-incompatible cases

When the Zinnia compiler rejects a kernel (dynamic shapes, `np.gcd`,
`np.pad` with runtime width, `np.bincount` with runtime values, etc.),
the case has **no** `target.rs` / halo2 baseline — the compiler can't
emit it, and a handwritten halo2 baseline would no longer be
apples-to-apples. The remaining zkVM baselines (`sp1.*`, `risc0.*`,
`main.cairo`) are still written and executed; these frameworks are
general-purpose and can express the kernel directly.

The 28 cases in this bucket (as of 2026-05-26): 0295–0299, 0327–0328,
0331–0332, 0354, 0362, 0366, 0405–0408, 0411, 0420, 0425–0427, 0440,
0471, 0489, 0495–0499. Compile-failure reasons are recorded in
`/tmp/zinnia_compile_status.txt`.
