# Handwritten baselines for the NPBench/Pythran set

This guide is for the authors of the handwritten halo2-base and Noir circuits of
the NPBench/Pythran performance set. That set,
`datasets/case_sets/npbench_pythran_perf.json`, has 79 entries with status
`perf`. Each entry is one entry function of a program folder under
`datasets/npbench/` or `datasets/pythran/`. The goal is a halo2-base and a Noir
baseline for every entry, written under the same rules as the DS-1000
baselines, so that Zinnia can be compared with expert circuits on this set too:
circuit size first (halo2-base cells, Noir gates), proving time later.

Read `CONVENTIONS.md` (the rules) and `NOIR_BASELINES.md` (Noir specifics)
first. This file adds only what differs for this set. Where a rule of
`CONVENTIONS.md` does not fit an NPBench/Pythran program, the interpretation
used here is stated in section 5; follow it, and add any new one there.

## 1. Terms and layout

- **entry**: one row of the case set, named by its case id
  `<program>__<function>` (`pythran_lu__factorMatrix0`). Several entries can
  share a program folder (`pythran_lu` holds `factorMatrix0` and
  `factorMatrix1`).
- **program folder**: `datasets/{npbench,pythran}/<program>/`, holding
  `kernel.py` (all entry functions and their helpers), the fixtures and the
  zkVM port `zkvm.rs`.
- **fixture**: the entry's inputs A and B. The case set names the file:
  `fixture.<function>.json` when the program has several entry functions,
  `fixture.json` otherwise.
- **shape-generic**: the kernels read their sizes from the arrays
  (`M.shape[0]`), and the fixture fixes those sizes. The performance set runs
  them at the fixture's shapes; the scaling study runs them at larger ones.

An entry's baseline files sit in its program folder and carry the entry
function's name:

| file | what |
|---|---|
| `target.<function>.rs` | the halo2-base circuit, a module of the `halo2_hw` crate |
| `main.<function>.nr` | the Noir circuit (the `src/main.nr` the driver stages) |
| `Prover.<function>.toml` | generated from the fixture's input B (`hw_check.prover_toml`); never edit by hand |

The name is per function even when the program has a single entry function
(`pythran_insertion_sort/target.insertion_sort.rs`), so one rule covers every
folder. The DS-1000 layout (`<case>/target.rs`, `main.nr`, `Prover.toml`) is
unchanged.

The three pilot entries are the examples to start from:

| entry | kind | files |
|---|---|---|
| `npbench_polybench_atax_atax__kernel` | reductions (`A @ x @ A`) | `datasets/npbench/npbench_polybench_atax_atax/{target,main}.kernel.*` |
| `pythran_lu__factorMatrix0` | data-dependent pivot row, in-place output | `datasets/pythran/pythran_lu/{target,main}.factorMatrix0.*` |
| `pythran_insertion_sort__insertion_sort` | sorting, in-place output | `datasets/pythran/pythran_insertion_sort/{target,main}.insertion_sort.*` |

## 2. Building, checking and measuring one entry

Run everything from `benchmarking/baselines` with the repository's virtual
environment (`../../.venv/bin/python`, NumPy 2.2 as the harness; the system
Python's NumPy may lack functions some kernels call). Run cargo, nargo and
proving under `eval-redo/tools/memrun.sh` with `MEMRUN_MAX` at most `16G`, and
keep to six parallel jobs (`CARGO_BUILD_JOBS=6`).

```bash
PY=../../.venv/bin/python
MEMRUN=/home/zhantong/Projects/zinnia-project/eval-redo/tools/memrun.sh
E=pythran_lu__factorMatrix0

# Build only this entry (dev-fast for iteration; the harness and the
# measurements use --release). A bare number still names a DS-1000 case.
(cd halo2_hw && HALO2_HW_CASES=$E CARGO_BUILD_JOBS=6 MEMRUN_MAX=16G $MEMRUN cargo build --profile dev-fast)

# The full check of both baselines (section 7 of CONVENTIONS.md): A/B mock
# runs, keys from A, proof from B, outputs against NumPy with the case set's
# tolerance, tampered public rejected, one read input moved out of its
# domain and rejected.
$PY -m hw_check.check $E --halo2-bin halo2_hw/target/dev-fast/halo2_hw
$PY -m hw_check.check --npb --all          # every NPBench/Pythran entry with a baseline (release binary)

$PY -m hw_check.lint $E                    # static rules, including the size constants (section 3)
$PY -m hw_check.sizes $E --file main.nr    # the size constants and their formulas
$PY -m hw_check.prover_toml $E             # writes Prover.<function>.toml (input B)

# Circuit size. halo2: cells at the size Zinnia's rule chooses (also printed
# by hw_check.check); Noir: gates (the driver's setup prints `gates`).
halo2_hw/target/release/halo2_hw stats --case ../datasets/pythran/pythran_lu --function factorMatrix0 --input A
$PY -m hw_check.noir_driver setup --case ../datasets/pythran/pythran_lu --function factorMatrix0 --workdir /tmp/w
```

`halo2_hw` and the Noir driver take an entry as its program folder plus
`--function` (the fixture defaults to the case set's file), or `halo2_hw` as
its case id with `--fixture`. `hw_check.domain_edges` accepts entry ids too;
add NPBench/Pythran domain-edge inputs to `domain_edges.json` under the entry
id.

The benchmark harness runs these entries as systems `halo2_hw` and `noir_hw`
under the DS-1000 protocol (build the release binary and Noir's executor
first: `python -m benchmarking.harness build --skip-zinnia --baselines
halo2_hw --noir-execute`):

```bash
.venv/bin/python -m benchmarking.harness check-env --systems halo2_hw,noir_hw
.venv/bin/python -m benchmarking.harness run --label npb-hw --systems halo2_hw,noir_hw \
    --case-set benchmarking/datasets/case_sets/npbench_pythran_perf.json [--cases ID,..]
```

An entry without a baseline file is recorded as `no_program` and listed in the
summary under "Cases without a program", apart from the failures; the run goes
on.

Do not run Zinnia on these programs, read its IR or emitted code, or look at
its circuit sizes for them while writing a baseline (`harness.costs_halo2`
prints Zinnia's numbers: do not use it either). Write each baseline from
`kernel.py`, the fixture and the case set alone.

## 3. Size constants

Every number in a baseline that is derived from the input shapes is a named
constant declared at the top of the file, before the first function, with the
formula that derives it from the fixture's dimensions. The rest of the file
uses only the names. Rewriting the constants alone then instantiates the
baseline at other shapes, which is what the scaling study will do (its
checker for DS-1000, `benchmarking/scaling/rewrite.py` on branch
`scaling-study`, admits changes to size constants and formula-derived
constants only).

```rust
// Sizes (hw_check.sizes): from fixture.factorMatrix0.json.
const N: usize = 5; // = M_0
const COLS: usize = 5; // = M_1
```

```noir
// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 10; // = list2_0
global KEY_SHIFT: u32 = 6; // = 14 * cdiv(64 + clog2(list2_0), 14) - 64
global KEY_BITS: u32 = 64 + KEY_SHIFT;
global INDEX_BITS: u32 = 8; // = 8 + 20 * min(1, (list2_0 - 1) // 128)
```

The rule in detail:

- A dimension is named `<parameter>_<axis>`, axis 0 first (`M_0`, `list2_0`),
  as in the scaling checker. A formula may use dimensions, integer constants,
  `+ - * // % ** <<` and `clog2` (the bits of an index into n positions),
  `bits` (the bit length of n), `cdiv`, `max`, `min`.
- A constant whose value is a literal carries `// = <formula>` on its line.
  A constant computed from other constants by the language
  (`global KEY_BITS: u32 = 64 + KEY_SHIFT;`) needs none.
- Sizes include more than dimensions: loop bounds, bit widths of keys and
  indices, the `W` of `noir_hw::array_index::<W>` / `index_pos::<W>` (8 up to
  128 elements, 28 above), table lengths. A number fixed by the number format
  (32 fractional bits, the 64-bit window) or by the kernel's text (the `5` of
  `1.0 / 5`, the `1024` of an `@requires` bound) is not a size.
- The Noir `// @outputs:` line names its counts by these constants
  (`// @outputs: float[N * COLS]`; the driver evaluates integer expressions
  over the file's integer globals).
- Loops run over the constants. No hand-unrolled code with literal indices,
  no lists whose length is a size: such a baseline cannot be rescaled by
  rewriting numbers.
- A construction may depend on the size (a one-hot below some length, a
  range check above it), but only through a constant with a formula, as
  `INDEX_BITS` above. A choice that is only right at the fixture's size is
  not allowed.

`hw_check.lint` checks the formulas against the fixture (an error when a value
differs, or when a size constant follows a function) and warns about every
number outside the constants that looks like a size (a dimension, a product
of consecutive dimensions, one less or more, their `clog2`) and every literal
index width: each warning is either made a size constant or is not a size.
`python -m hw_check.sizes <entry> --file target.rs --fixture <other fixture>`
prints the file instantiated at another fixture's shapes; this is how a
baseline is checked for scalability.

## 4. What is specific to NPBench/Pythran

**Shape-generic kernels at the fixture's sizes.** A circuit is written for the
fixture's shapes, given as the size constants; `inputs.floats("M", &[N, COLS])`
rejects any other shape. Shape arithmetic in the kernel (`n - 1`, `range(line,
m)`, slices) becomes arithmetic on the constants, resolved at compile time.
Keep the kernel's quirks: `deuxd_convolution` clamps both indices to
`sw[0]`, `factorMatrix1` swaps only the columns from `line` on (unlike
`factorMatrix0`), and a baseline computes exactly that.

**Outputs of a function that returns `None`** (`mutated_outputs` in the case
set). The outputs are the arguments the case set lists, in parameter order,
after the call: every element of each, flattened row-major, including
elements the function never writes (they are copies of the input, and public
like the rest). This is what Zinnia, the zkVM ports (`// outputs: mutated`)
and the harness check (`harness.outputs.call_outputs`). An argument that is
overwritten before it is read (`hdiff`'s `out_field`) is never read, so it is
not loaded and gets no domain check. A function that returns a value outputs
that value only, also when it writes arguments on the way (`ln` returns `Y`,
`d2q9`'s `one_time_step` returns `f2` after `periodic_bc` wrote `f1`,
`add_pressure_diffusion` returns `P_hat` after writing `dU`); writes that do
not reach the return value are dead work and are left out.

**Several entry functions in one program.** Each entry is its own circuit:
its own `target.<f>.rs` (a separate module of `halo2_hw`) and
`main.<f>.nr` (a separate Nargo package). Code two entries share either is
repeated in both files (as `factorMatrix0`/`factorMatrix1` would) or, when it
is a general construction, becomes a gadget (below). The fixture differs per
entry (`fixture.<f>.json`).

**Tolerances.** `hw_check.check` takes the entry's `float_tolerance` from the
case set (currently `1e-6` absolute and relative for every entry). Integers
and booleans are compared exactly.

**Complex numbers** (`clip`, `CROTG`, `make_decision`, `scattering_self_energies`,
`spectraldns`). A complex value is two floats, real part first, in the
fixture (`[re, im]`), in the inputs (`inputs.complexes(name, &shape)` returns
`[re0, im0, re1, im1, ...]`; `inputs.complex(name)` a scalar; in Noir declare
`[Field; 2 * N]` or `[[Field; 2]; N]`) and in the outputs (`out.floats` on the
interleaved cells; `float[2 * N]`).

**Scalars that bound work** (`fibo(n)`, `times(n, m)`, `covariance2`'s `M`)
stay private runtime inputs (CONVENTIONS 1.4: a kernel that takes a size at
runtime keeps it runtime). A loop over such a scalar runs to the bound of the
kernel's `@requires` precondition, each step conditional on the counter.

**Gadgets.** Use the existing ones (`halo2_hw/src/q32.rs`,
`noir_hw/src/lib.nr`; tables in CONVENTIONS section 5). The pilots added
`q.sort` with `merge_exchange` (sorting network on window offsets),
`q.div_by_abs` and `noir_hw::div_by_abs`. Batch 2 added, on both sides
unless noted: `clip_const` (`np.clip` by constant bounds, region bits pinned
by one 64-bit range check), `div_fraction` (`a / b` for `0 <= a < b`, a
32-bit quotient check), `arctan2` and `arctan2_nonneg` (Q32 in and out,
within 1e-8; `q.arctan2(ctx, y: I64, x: I64) -> Cell`,
`q.arctan2_nonneg(ctx, y: Cell, x: Cell) -> Cell` for inputs proven in
`[0, 2^63)`, `noir_hw::arctan2(y, x)`, `noir_hw::arctan2_nonneg(y, x)`),
`q.index_hits` (the pinned one-hot of a NumPy index, reused by many gathers)
and `noir_hw::floor_div_const::<D>` (floor division by a constant up to 256
with 8-bit remainder checks). A new gadget is a new function with
malicious-hint tests (`cargo test --lib`, `nargo test`); never change an
existing function's circuit, because the DS-1000 and pilot baselines are
measured with them. Before committing a gadget change, compare the DS-1000
cells/gates of a few cases before and after (`hw_check.check 0300 0309 0381
0438`), and run the gadget tests.

**Sorting** (`bubble_sort`, `insertion_sort`, `projection_simplex`,
`value_index`). halo2: `q.sort` on the loaded window offsets, 31
compare-exchanges for 10 values. Noir: the order computed unconstrained and
verified by strictly increasing keys `window(v) * 2^KEY_SHIFT + i` read
through it (as 0381), which was cheaper than the same merge-exchange network
in Noir on the pilot (3,063 against 3,251 gates for 10 values) and grows
linearly. When ties matter (`argsort`, `value_index`), the index in the key
gives NumPy's stable order (CONVENTIONS section 3).

## 5. Interpretations of the rules for this set

1. **In-place outputs** (`mutated_outputs`): as in section 4. The written
   arguments' final contents are public outputs, unwritten elements included.
2. **`@requires` preconditions** are part of the statement and asserted in
   the circuit, as the zkVM ports assert them and as the DS-1000 set's
   permutation precondition is enforced (`datasets/FILTERING.md`). A
   parameter that only a precondition reads (`covariance2`'s `M`) is loaded,
   domain-checked and checked against the precondition; a parameter nothing
   reads (`covariance2`'s `float_n`, `hasting`'s `t`) is not loaded.
3. **Sums of squares and complex products as one reduction.** CONVENTIONS
   section 2 lets a sum of products that NumPy computes as one reduction be
   rescaled once. Here that covers `np.sum((x - y) ** 2)`,
   `np.sum(np.abs(x) ** 2, 1)` and loops that accumulate `tmp * tmp` into one
   variable (`pairwise`, `matrix_multiply`'s `r += m0[i][k] * m1[k][j]`), and
   the real and imaginary parts of a complex product
   (`re = a c - b d`, `im = a d + b c`, each one rescale). Elementwise
   products that NumPy materializes before adding are rounded one by one, in
   NumPy's evaluation order: in `syrk`'s
   `C[i, :i + 1] += alpha * A[i, k] * A[:i + 1, k]`, `alpha * A[i, k]` is
   rounded, then each element's product; `alpha * A @ B` is
   `(alpha * A) @ B`.
4. **`float32`** (`conv2d_bias`'s `np.empty(.., dtype=np.float32)`,
   `weights`, `ising`'s `np.float32(beta)`): computed in Q32 like any float.
   float32 rounding changes a value by at most `6e-8` relative, inside the
   `1e-6` tolerance. Where it would change a discrete outcome (a comparison,
   a `uint8` cast) the author must check that the fixture's and the domain's
   values keep the outcome, and say so in the file header.
5. **Transcendental functions without a rounding rule** (`exp`, `log`,
   `tanh`, `arctan2`, `2.0 ** x`, `np.round(x, 2)`). CONVENTIONS fixes the
   rounding of `sin` and `cos` only. For the others the baseline computes a
   value within the case's tolerance of NumPy's for every input of the
   domain where the result fits Q32, with the cheapest sound construction
   (range reduction plus a minimax polynomial, as for `sin`), designed
   independently of Zinnia's (do not read Zinnia's approximation). The gadget
   states its error bound and its tests check it. One shared gadget per
   function, written once (section 6 says by whom).
6. **Data-movement-only entries** (`create_grid`, `multi_export.a`,
   `periodic_bc`, `permutations`) are baselines like the others: every read
   input is loaded with its domain check and the copies are output. Outputs
   that are constants of the circuit (`create_grid`'s third column of zeros)
   are not public cells (CONVENTIONS 1.3; halo2 `out.const_floats` between
   the public segments, Noir `const float[..]` segments).
7. **Integer overflow.** As for DS-1000, the statement covers inputs whose
   NumPy result fits the window (CONVENTIONS 1.1); exact field arithmetic is
   used without overflow checks except where a gadget needs a range anyway.
   Python integers do not overflow (`fibo` returns a Python `int`): a result
   outside int64 has no proof.
8. **Inputs that reach no output.** An input element whose value reaches no
   output is not loaded and gets no domain check, also where NumPy touches it
   on the way: a slice that drops it after an elementwise operation
   (`harris`'s `I[0, 0]`), a copy that is overwritten before it is read
   (the border of `d2q9`'s `f1`, which `periodic_bc` overwrites with copies of
   the interior). This is the rule for overwritten arguments (section 4)
   applied element by element.
9. **Magnitude of a complex value** (`clip`, `CROTG`): `abs(z)` is the exact
   floor square root of `re^2 + im^2`, taken on the raw sum (scale 2^64), so
   one rounding as NumPy rounds `hypot` once. A comparison of `abs(z) ** 2`
   values (`make_decision`'s `argmin`) compares the exact raw sums, whose
   order is the real one; NumPy's float64 order differs only between values
   equal to within its rounding. The raw sums are range-checked as Q32
   intermediates: a squared distance whose Q32 value leaves the window has no
   proof.

## 6. The remaining 76 entries in 4 batches

The batches are balanced by expected effort: each entry was rated 1 (direct
use of existing gadgets), 2 (several gadgets, sequential dependencies or a
large output), 3 or 4 (a new gadget, complex numbers, data-dependent control
flow or indexing), and each batch totals 34 to 36 points over 19 entries.
Entries that need the same new gadget are grouped with the batch that writes
it: batch 1 writes `exp`, `log` and `tanh` (halo2 and Noir), batch 2 writes
`arctan2` and does most of the complex-number entries, batch 3 has the
sorting and pivoting entries that reuse the pilots. Three entries in other
batches need batch 1's `exp`: `deriche` (batch 3), `goodExpoMeasure` and
`ising_step` (batch 4); do them last, against the gadget's committed
interface (`q.exp(ctx, x: I64) -> I64`, `noir_hw::exp(x: Field) -> Field`,
Q32 in and out; batch 1 publishes the interface first).

### Batch 1 (exp, log, tanh; 36 points)

| entry | rating | notes |
|---|---|---|
| `npbench_deep_learning_mlp_mlp__mlp` | 4 | relu, matmuls, softmax (max, exp, division) |
| `npbench_deep_learning_softmax_softmax__softmax` | 4 | 1,024 exps, row maxima, divisions |
| `npbench_go_fast_go_fast__go_fast` | 3 | `tanh` of the diagonal |
| `pythran_log_likelihood__log_likelihood` | 3 | exp, log, sqrt constant; `sigma != 0` precondition |
| `pythran_vibr_energy__calculate_vibr_energy` | 2 | exp, once the gadget exists |
| `npbench_polybench_covariance2_covariance2__kernel` | 2 | `np.cov` (mean, `ddof=1`); `M` precondition, `float_n` unused |
| `npbench_polybench_gramschmidt_gramschmidt__kernel` | 2 | sqrt, divisions; tuple output |
| `npbench_polybench_symm_symm__kernel` | 2 | NumPy evaluation order; in place `C` |
| `npbench_polybench_trmm_trmm__kernel` | 2 | in place `B` |
| `pythran_deuxd_convolution__conv` | 2 | clamped indices are loop constants |
| `pythran_sumarray3d__summation` | 2 | `1 / sqrt`, nested function; tuple output |
| `npbench_polybench_bicg_bicg__kernel` | 1 | two matrix-vector products |
| `npbench_polybench_k3mm_k3mm__kernel` | 1 | three matrix products |
| `pythran_create_grid__create_grid` | 1 | data movement; constant zero column |
| `pythran_l2norm__l2_norm` | 1 | sum of squares, sqrt |
| `pythran_multi_export__a` | 1 | data movement |
| `pythran_pairwise__pairwise` | 1 | sum of squares, sqrt |
| `pythran_reverse_cumsum__reverse_cumsum` | 1 | |
| `pythran_wdist__slow_wdist` | 1 | division, sqrt |

### Batch 2 (arctan2, complex numbers; 35 points)

| entry | rating | notes |
|---|---|---|
| `npbench_pythran_arc_distance_arc_distance__arc_distance` | 4 | sin, cos, sqrt, `arctan2` |
| `npbench_scattering_self_energies_scattering_self_energies__scattering_self_energies` | 4 | complex, gathers by a private index (`neigh_idx`), in place `Sigma` |
| `pythran_clip__clip` | 3 | complex magnitude, division by it, branch |
| `pythran_crotg__CROTG` | 3 | complex scalars, sqrt, division, branch on `abs(CA) == 0` |
| `pythran_arc_distance__arc_distance` | 2 | as the NPBench one, 5 points |
| `pythran_make_decision__md` | 2 | complex, `argmin`, gather from `symbols` |
| `pythran_spectraldns__add_pressure_diffusion` | 2 | complex times float; returns `P_hat` (the `dU` write is dead) |
| `npbench_polybench_trisolv_trisolv__kernel` | 2 | forward substitution, divisions; in place `x` |
| `pythran_d2q9_nxnyns__one_time_step` | 2 | many constant products; returns `f2` |
| `pythran_lapl3d__laplacien` | 2 | abs, global maximum, divisions |
| `pythran_spectraldns__cross2` | 1 | complex times float |
| `npbench_compute_compute__compute` | 1 | integer `clip`, arithmetic |
| `npbench_polybench_gesummv_gesummv__kernel` | 1 | |
| `pythran_allpairs__sqr_dists` | 1 | sums of squares |
| `pythran_allpairs__sqr_dists_loops` | 1 | the same function written as loops |
| `pythran_harris__harris` | 1 | |
| `pythran_ln__ln` | 1 | powers, constant divisors; returns `Y` |
| `pythran_rosen__rosen` | 1 | |
| `pythran_smoothing__smoothing` | 1 | sequential recurrence |

### Batch 3 (sorting, pivoting, control flow; 34 points)

| entry | rating | notes |
|---|---|---|
| `npbench_polybench_deriche_deriche__kernel` | 4 | exp of a private scalar, `2.0 ** -alpha`, recurrences; needs batch 1's `exp` |
| `pythran_projection_simplex__projection_simplex` | 3 | sort, cumsum, last index meeting a condition, division by it |
| `pythran_periodic_dist__dist` | 3 | runtime boolean flags, masked updates, sqrt, `d == 0` |
| `npbench_polybench_cholesky_cholesky__kernel` | 2 | sqrt, divisions; in place `A` |
| `npbench_polybench_gemver_gemver__kernel` | 2 | outer products; in place `A`, `w`, `x` |
| `npbench_polybench_lu_lu__kernel` | 2 | LU without pivoting; in place `A` |
| `npbench_polybench_ludcmp_ludcmp__kernel` | 2 | LU plus two triangular solves |
| `pythran_cronbach__cronbach` | 2 | variances (`ddof=1`), division |
| `pythran_fibo_seq__fibo` | 2 | runtime trip count up to 1024 |
| `pythran_value_index__tri` | 2 | stable argsort of a column, row gather |
| `pythran_weights__weights` | 2 | equality counts, threshold, `1 / total`; float32 output |
| `pythran_lu__factorMatrix1` | 1 | `factorMatrix0`'s pilot with the column-range swap |
| `pythran_bubble_sort__bubble_sort` | 1 | integer sort (`q.sort` / sorted-key check) |
| `npbench_polybench_k2mm_k2mm__kernel` | 1 | in place `D` |
| `pythran_hasting__fweb` | 1 | divisions; `t` unused |
| `pythran_matmul__matrix_multiply` | 1 | |
| `pythran_perm__permutations` | 1 | data movement, 24 permutations |
| `pythran_pythagorean_triples__next_pythagorean_triples` | 1 | constant integer matrix |
| `pythran_specialconvolve__specialconvolve` | 1 | integer stencil |

### Batch 4 (stencils, large kernels; 36 points)

| entry | rating | notes |
|---|---|---|
| `pythran_goodexpomeasure__goodExpoMeasure` | 4 | exp, `np.round(x, 2)`, `uint8` cast; needs batch 1's `exp` |
| `pythran_ising__ising_step` | 4 | exp compared with 0.5, data-dependent sequential updates; needs batch 1's `exp` |
| `npbench_crc16_crc16__crc16` | 3 | bitwise operations on 16-bit values |
| `npbench_weather_stencils_vadv_vadv__vadv` | 3 | tridiagonal solve with divisions; in place `utens_stage` |
| `npbench_deep_learning_conv2d_bias_conv2d__conv2d_bias` | 2 | 1,800 outputs, float32 output |
| `npbench_polybench_durbin_durbin__kernel` | 2 | sequential divisions |
| `npbench_polybench_syr2k_syr2k__kernel` | 2 | per-product rounding order; in place `C` |
| `npbench_polybench_syrk_syrk__kernel` | 2 | as `syr2k` |
| `npbench_weather_stencils_hdiff_hdiff__hdiff` | 2 | stencil with `np.where` on products; `out_field` is overwritten, never read |
| `pythran_fbcorr__fbcorr` | 2 | accumulates into the input `output` |
| `pythran_tax__tax` | 2 | masked sums by constant thresholds, division |
| `npbench_polybench_gemm_gemm__kernel` | 1 | in place `C` |
| `npbench_polybench_mvt_mvt__kernel` | 1 | in place `x1`, `x2` |
| `pythran_euclidean_distance_square__euclidean_distance_square` | 1 | |
| `pythran_lap__laplacian` | 1 | constant products, division by `dh2` (precondition) |
| `pythran_multitype__check` | 1 | branch on an integer; tuple output |
| `pythran_multitype__check2` | 1 | branch on a boolean |
| `pythran_multitype__times` | 1 | product with two range preconditions |
| `pythran_periodic_bc__periodic_bc` | 1 | data movement |

## 7. Pilot results

Checked with `hw_check.check` (both systems pass every step), sizes at the
fixture's shapes. halo2: advice and lookup cells at the `k` and lookup width
Zinnia's sizing rule chooses (`halo2_hw stats`); Noir: Barretenberg gates
(`bb gates`, UltraHonk).

| entry | halo2 k | advice | lookup | Noir gates |
|---|---|---|---|---|
| `npbench_polybench_atax_atax__kernel` | 9 | 8,485 | 2,140 | 4,397 |
| `pythran_insertion_sort__insertion_sort` | 8 | 2,208 | 441 | 3,063 |
| `pythran_lu__factorMatrix0` | 9 | 5,215 | 1,136 | 4,450 |
