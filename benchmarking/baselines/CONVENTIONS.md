# Handwritten circuit baselines: the rules

This file is the rulebook for the handwritten circuit baselines of the DS-1000
performance set: the halo2-base circuits (`datasets/ds1000/<case>/target.rs`,
built by `baselines/halo2_hw`) and the Noir circuits (`datasets/ds1000/<case>/main.nr`,
run by `baselines/hw_check/noir_driver.py`). The handwritten baselines of the
NPBench/Pythran performance set follow the same rules; `NPB_BASELINES.md` gives their
file layout and what is specific to that set. The zkVM and Cairo baselines have their
own documentation in `baselines/{sp1,risc0,cairo}`; they share the statement rules of
section 1.

Terms:

- **kernel**: `datasets/ds1000/<case>/kernel.py`, the program Zinnia compiles.
- **NumPy reference**: the kernel body run as plain NumPy (`hw_check.reference`).
- **Q32**: fixed point with 32 fractional bits; a float `v` is the integer
  `round(v * 2^32)` (Zinnia's quantizer, `src/prove/field.rs::quantize`).
- **window**: the signed 64-bit range `[-2^63, 2^63)`. Zinnia's integers and Q32
  values live in it, and Zinnia's halo2-base comparisons are 64-bit comparisons on it.
- **hint**: a value the prover computes outside the circuit (a quotient, a square
  root, a sorted order) and loads as a witness.
- **pinned**: a hint is pinned when the constraints leave exactly one value it can
  take, so a dishonest prover gains nothing by choosing another.

## 1. The statement

1. **Same function as the kernel.** A baseline computes what `kernel.py` computes
   under NumPy semantics, for every input of the declared shapes and dtypes whose
   NumPy result is finite and fits Q32. It is written from `kernel.py`, never from a
   Zinnia IR dump or Zinnia's emitted code.
2. **Where NumPy raises, there is no proof.** An out-of-range index (`IndexError`),
   a division by zero, the square root of a negative number, an overflow of the
   window: the circuit has no satisfying witness. The baseline does not reject any
   other input (no assertion keyed to a fixture value, no narrower bin sizes or
   index ranges than NumPy accepts).
3. **Inputs private, outputs public.** Every kernel parameter is a private input.
   The return value is public, flattened row-major, tuple elements in order.
   Outputs that are compile-time constants of the circuit (a shape fixed by the
   signature, a constant array) are not public cells, exactly as Zinnia leaves
   them out of its instance; the baseline still declares them so the output check
   sees the whole value. An output of data-dependent length (`x[x >= 0]`,
   `np.argwhere`, `nonzero`) is exposed as one public length followed by the
   values padded with zeros to the maximum length.
4. **Shape parameters are constants.** In the performance set, DS-1000 shape
   parameters (`ncol`, `bin_size`, `patch_size`, `size`, `power`, ...) are
   compile-time constants of the kernel (author's decision, 2026-10-06); a baseline
   uses the kernel's constant, and a kernel that still takes one at runtime keeps it
   runtime.
5. **Input-independent structure.** The circuit's gates and layout never depend on
   input values. The checker generates keys with input A and proves with input B.
6. **Input domain.** Every input element the circuit reads is checked to lie in its
   dtype's domain, so a proof means "some valid NumPy input gives these outputs"
   (author's decision; Zinnia does the same since `frontend-2`): an integer is an
   int64, a float's Q32 code lies in the same signed 64-bit window (that is,
   -2^31 <= v < 2^31), a boolean is 0 or 1. One check per element, and none for
   elements the kernel never reads. halo2: the `q.load_*` loaders do it (the
   `*_bounded` variants return the window proof for comparisons to reuse; a mock
   run reports `redundant_input_checks`, which must be 0); Noir:
   `noir_hw::check_inputs` / `check_input` / `checked` (the window check, which
   Noir shares with any later `window` of the same value). A check that the kernel's
   own semantics already impose and that implies the domain replaces it (0402/0403:
   NumPy's `0 <= index < dim`).

## 2. Numbers

- Floats are Q32 as above; integers are themselves; booleans are `{0, 1}`.
  Negative values are `p - |x|` in the field.
- Rounding is Zinnia's:
  - a product is `floor(a * b / 2^32)`;
  - a sum of products that NumPy computes as one reduction (`np.dot`, `@`,
    `tensordot`, `correlate`, `einsum`, `(x*y).sum()`) may be rescaled once,
    `floor(sum(a_i * b_i) / 2^32)`; this is closer to the real result, and it is
    what Zinnia's fused product-sum does;
  - a float quotient is `floor(a * 2^32 / b)` with remainder in `[0, |b|)`;
  - `int(x)` and `astype(int)` truncate toward zero; `//` and `%` follow Python
    (floor, remainder with the sign of the divisor);
  - `sin`/`cos` reduce the argument by pi, reflect it into `[0, pi/2]` and
    evaluate the degree-14 minimax polynomial Zinnia evaluates
    (`q32::SIN_COEFS`); `cos(x) = sin(x + pi/2)`. With Q32 coefficients the
    polynomial is accurate to 2.4e-7 on `[0, pi/2]` but only 1.4e-4 near pi, so
    the reflection is what lets the result meet the case set's 1e-6 tolerance.
- Comparisons are on the window (64-bit), the width Zinnia uses.
- `==` in the kernel is exact equality, also on floats. `np.isclose` uses NumPy's
  formula `|a - b| <= atol + rtol * |b|` with the kernel's (or NumPy's default)
  `rtol` and `atol`.

## 3. Idiomatic and cheapest sound

A baseline is what a competent developer writes with the library's standard
gadgets. Concretely:

- Use the cheapest construction that is sound (section 4) and computes the
  kernel's function. If Zinnia uses a cheaper sound construction for an operation,
  the baseline may use it too; if the baseline knows a cheaper one, it uses it.
- Do the kernel's work and no more: no dead computation (work whose result never
  reaches an output), no re-check of an output against a private claimed value, no
  constant outputs exposed as public cells, loop-invariant tests hoisted out of
  loops, each value range-checked once (halo2: keep the `I64` returned by
  `q32.bound`; Noir: keep the `u64` returned by `noir_hw::window`).
- Do the kernel's work and no less (section 1.2): negative-index wrap, bounds,
  the full output (all matches, not the first), the NumPy tie-breaking rule
  (`argmax` returns the first maximum, `argsort` is the stable sort NumPy's
  default `quicksort` gives for the fixture's distinct values; see the case notes
  where ties matter).
- Gathers by a private index use the library's indexing: halo2-base
  `select_from_idx` behind `q.index`, Noir native array indexing after
  `noir_hw::index_pos` (a NumPy index) or `noir_hw::array_index` (a position,
  e.g. one read from a hint). For short axes a one-hot over the valid index values
  `-len..len` is cheaper and needs no range check (`q.index` switches to it up to
  8 elements, measured by the `index_cost` test; in Noir use `noir_hw::pick` for
  rows of up to 8). Never scan the whole input per output when one index selects
  it.
- Constant divisors use the constant-divisor gadget (one remainder check against
  the constant), not the variable-divisor one.
- Integers that are squared or multiplied stay integers (no Q32 rescale).

## 4. Soundness

- Every hint is pinned, and the code says what pins it next to the hint: a
  decomposition `x = q * d + r` checks `r` against `d` and puts `q` in the window,
  so the field identity is an integer identity and `(q, r)` is unique; a square
  root is bracketed `y^2 <= x * 2^32 < (y + 1)^2`; a parity bit is pinned by
  `q = 2h + bit` with `h` range-checked.
- Private inputs are whatever field elements the prover loads. A construction
  may rely on an input lying in its declared range (an int64, a Q32 value) only
  where the circuit checks it (section 1.6 checks every read input).
  Field identities that hold for every field element need no range (equality,
  a product of differences being zero); identities that hold only for small
  integers do (a sum of squares being zero means all-zero only below 2^64:
  in BN254's scalar field -1 is a square).
- Constants are circuit constants: halo2-base `Constant(..)` or
  `ctx.load_constant`, Noir literals. Never `load_witness` for a constant.
- Noir range checks are `x.assert_max_bit_size::<N>()` on the original `Field`. A
  cast `x as u64` truncates and proves nothing, so a cast only follows such a
  check. Every `unsafe { hint(..) }` is followed by the checks that pin it, with a
  `// Safety:` comment naming them.
- A Noir array index that is not built from loop constants is proven in range
  by the circuit itself, never left to Barretenberg's memory check: the
  compiler can drop a range check of a value used as a memory index, and the
  pinned bb (5.0.0-nightly.20260324) admits a read at index p - 1. Use
  `noir_hw::array_index` (checks of `2 k` and `last - k`, which the compiler
  keeps; the construction Zinnia's Noir library uses) or `index_pos`, which
  calls it. An index that is a select of loop constants, or arithmetic on
  positions already proven in range, is in range by construction and needs
  none (say so in a comment).
- The shared gadgets carry malicious-hint tests: `cargo test --lib` in
  `baselines/halo2_hw` (MockProver rejects neighbouring and non-integer quotients,
  flipped signs and parities) and `nargo test` in `baselines/noir_hw`
  (`should_fail` tests).

## 5. Gadgets

halo2-base (`q32::Q32`, built on `GateChip`/`RangeChip` of halo2-base 0.4.1):

| NumPy | gadget |
|---|---|
| float input / constant | `q.load_floats(ctx, &v)` / `q.float_const(ctx, v)` or `Constant(fe(quantize(v)))` |
| int input / constant | `q.load_ints` / `q.int_const`, `Constant(fe(v))` |
| `+`, `-` (float or int) | `q.add`, `q.sub` (exact) |
| float `*` | `q.mul` (rescaled, returns `I64`) |
| float dot / tensordot / correlate | `q.dot` (one rescale) |
| int `*`, float `*` int | `q.gate.mul`, `q.mul_int` (exact) |
| float `/` | `q.div` (any sign), `q.div_pos` (positive divisor), `q.div_const`; `q.div_by_abs` when the circuit already holds the divisor's magnitude from `q.abs_int64` (a pivot chosen by `argmax(abs(..))`) |
| int `//`, `%` | `q.floor_div_const`, `q.floor_div_pos` |
| `astype(float)` | `q.int_to_float` |
| `int(x)`, `astype(int)` | `q.float_to_int` |
| `np.floor` | `q.floor` |
| `x ** n` (constant n) | `q.pow_const` |
| `np.sqrt` | `q.sqrt` (exact floor square root); `q.isqrt` for a raw integer |
| `np.sin`, `np.cos` | `q.sin`, `q.cos` |
| `<`, `<=`, `>`, `>=` | `q.bound` once per value, then `q.lt`/`le`/`gt`/`ge`, `q.lt_const` |
| `max`, `min`, `abs` | `q.max`, `q.min`, `q.abs`; `q.max_of` / `q.min_of` (hint pinned by range checks and a product) for the extremum of a few values |
| `==`, `!=` | `q.eq`, `q.gate.not` |
| `where`, `if` on data | `q.gate.select`, `q.select_i64` |
| `a[i]` (private `i`) | `q.index` (NumPy wrap and bounds; one-hot or range-checked form by length), `q.index_i64` if `i` already has a window proof |
| `x[mask]`, `nonzero`, `argwhere` | `q.select_kept` (values) or `q.compact` (other payloads): a running one-hot of the next free slot, then `out.dynamic` |
| `np.sort` (values only) | `q.sort` (Batcher's merge exchange on the window offsets; `merge_exchange(n)` gives the network) |
| `k // d`, `k % d` for a small known-non-negative `k` (an index built from loop constants) | `q.divmod_small` (halo2-base `div_mod`) |
| `&`, `|`, `~` on bools | `q.gate.and`, `q.gate.or`, `q.gate.not` |

Noir (`noir_hw`, on top of the Noir standard library):

| NumPy | Noir |
|---|---|
| `+`, `-`, int `*` | `Field` arithmetic |
| float `*` | `noir_hw::mul` |
| float dot | `noir_hw::dot`, or accumulate products and `noir_hw::rescale` once |
| float `/` | `noir_hw::div`; `div_by_abs` when `abs_int64` already gave the divisor's magnitude |
| int `//`, `%` | `noir_hw::divmod_pos` |
| `astype(float)`, `int(x)`, `np.floor` | `from_int`, `to_int`, `floor` |
| `np.sqrt`, `np.sin`, `np.cos` | `sqrt` (`isqrt` for a raw integer), `sin`, `cos` |
| `//` by a constant `2^k` | `floor_div_pow2` |
| `max`/`min` of a few values | `max_of` |
| `x[mask]`, `nonzero`, `argwhere` | `select_kept`, `compact`, `compact_rows` |
| comparisons | `let w = window(x)` once per value, then `u64` comparisons; `lt`, `gt`, ... for one-off tests |
| `max`, `min`, `abs` | `max`, `min`, `abs` (or `window` and select); `max_of` for the maximum of a few values |
| `a[i]` (private `i`) | `a[index_pos::<8>(i, len)]` (native indexing; `::<28>` above 128 elements); `pick(row, i)` for rows of up to 8 |
| `a[k]` (private position `k`, e.g. from a hint) | `a[array_index::<8>(k, len - 1)]` (`::<28>` above 128 elements) |
| `u32` index arithmetic (`k // 6` on an index built from loop constants) | native `u32` operators |

Noir specifics (toolchain, ZK target, driver) are in `NOIR_BASELINES.md`.

## 6. Files

`target.rs` (one per case) is a module of the `halo2_hw` crate:

```rust
//! DS-1000 case NNNN, halo2-base baseline.
//! Kernel: <one line>.
//! <only the non-obvious construction choices>

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[8, 6])?; // name and shape from the kernel signature
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    ...
    out.floats(&result); // or out.int / out.bools / out.const_ints / out.dynamic
    Ok(())
}
```

`main.nr` (one per case) is the `src/main.nr` of a Nargo package the driver stages
with the dependency `noir_hw` (the baselines do not use Zinnia's `noir_fpc`). Its
parameters carry the kernel's parameter names (arrays flattened row-major or
nested, `Field` or native integer types); its header names the case and kernel
and carries the output layout the driver decodes:

```noir
// DS-1000 case NNNN, Noir baseline.
// Kernel: <one line>.
// @outputs: float[6]
```

`@outputs` segments (separated by `;`): `float[N]`, `int[N]`, `bool[N]`;
`const int[N] = [...]` / `const float[N] = [...]` for circuit constants (not
public); `dynamic <kind>[IxU]` for a public length of up to `I` items of `U`
cells each, then `I * U` padded cells; `dynamic2 <kind>[RxC]` for a 2-D output
of data-dependent shape: public `(rows, cols)`, then `R * C` cells holding the
`rows * cols` values compact and row-major, zero-padded (halo2: `out.dynamic2`);
`repeat[k < N](SEG | SEG ...)` for a group of segments repeated `N` times, whose
counts may use `k` (a constant column or triangle between public cells; a `const`
segment may give one value for all its cells: `const float[k] = 0`).

`Prover.toml` (one per case) is generated from `fixture.json` input B by
`python -m hw_check.prover_toml <case dir>`; do not edit it by hand.

## 7. Checking and running

From `benchmarking/baselines`:

```bash
# build (release = the measured profile; dev-fast for iteration)
(cd halo2_hw && cargo build --release)              # all cases
(cd halo2_hw && HALO2_HW_CASES=0309,0310 cargo build --profile dev-fast)
python -m hw_check.lock_match                        # same dependency revisions as Zinnia
python -m hw_check.check 0309 0310                   # or --all
```

`hw_check.check` runs, per case and system: halo2 `mock` with A and B, `setup`
with A, `prove` with B, `verify`, and `verify` with public value 0 tampered;
Noir `setup`, `prove` with A and B, `verify`, tampered `verify`. It also moves
one input element the kernel reads out of its domain (halo2 `mock
--tamper-input name:index`, Noir `prove --tamper-input name:index`) and requires
both systems to reject it. Every proved
output must equal the NumPy reference position by position (integers and booleans
exactly, floats within `1e-6 + 1e-6 * |x|` unless the case set gives a tolerance),
and the tampered public must be rejected. A baseline that fails is wrong.

The program contract (`halo2_hw <mode> --case <dir> --input A|B --workdir <dir>`,
and `python -m hw_check.noir_driver <mode> ...`) prints one JSON object with `ok`,
`publics` (decoded, kernel order), `proof_path` and per-step `timings_s`
(halo2: `params_s`, `synthesize_s`, `keygen_vk_s`, `keygen_pk_s`, `witness_s`,
`prove_s`, `verify_s`, loads and writes separately; Noir: `compile_s`,
`write_vk_s`, `execute_s`, `prove_s`, `verify_s`). halo2's `prove_s` covers
`create_proof` and `transcript.finalize()`, the window of Zinnia's
`prove_seconds`. For Noir, the benchmark harness does not use the driver's
timings: it runs the tools itself, and its witness step is `noir-execute` on the
compiled artifact, not the driver's `nargo execute` (`benchmarking/README.md`,
section Noir). `halo2_hw` sizes the circuit by Zinnia's rule
(`params.rs`); `--k` and `--lookup-bits` override it.

## 8. Cases without a baseline

Listed in `NOIR_BASELINES.md` (Noir) and in the bench-circuits report
(`eval-redo/fix/bench-circuits.md`) with the reason for each.
