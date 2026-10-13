# Task B, follow-up: are the capacities tight?

The paper (§8.1, "Cost of bounds") measures, for the 36 programs with data-dependent
shapes, how full their arrays are on the two test inputs A and B: 54% of their capacity on
average. A reviewer asked how large the overhead can be when the bound is much larger
than the runtime shape. This study asks the converse question: for each array, is there a
valid input on which the array actually reaches its capacity? If so, any circuit that
accepts every valid input must reserve that room, whoever writes it.

## Definitions

- **Array set and capacity.** Exactly the inventory of `../final/arrays.json` (the data
  behind the paper's 54%): 36 programs, 141 array values (source expressions with an
  input-dependent shape, one row per evaluation and per tuple/list element). The capacity
  is the number of element slots in the compiler's exported layout, as there. The
  capacities were re-read from the circuits compiled by current main (ce1b259e, built
  locally): all 141 are unchanged (`mock/*.json`, `capacity_mismatch` empty).
- **Size.** The number of elements of the array on an input, measured by running the
  program as NumPy with every array expression observed, with the same instrumentation as
  the original measurement (`../final/probe.py`). On fixtures A and B this reproduces all
  282 recorded lengths.
- **Valid input.** An input of the declared signature (the shapes of the compiled
  circuit's inputs) whose values lie in the compiler's value domain (Integer: int64; Float:
  Q32 fixed point, |v| < 2^31; Complex: a pair of such floats; Boolean) on which the
  NumPy program runs without raising, including floating-point errors (division by zero
  and the like are treated as invalid). None of the 36 programs has a `@requires`
  precondition, so there is nothing more to satisfy. Every reported worst-case input was
  also run through the compiled circuit with the mock backend (constraint satisfaction only,
  no keys or proofs): all 53 checked inputs satisfy the circuit, and the length the
  circuit outputs equals the NumPy size for every array (`mock/`).
- **Fill.** Size divided by capacity. Three program-level summaries, each averaged over
  the 36 programs like the paper's 54%:
  - *per-array worst case*: every array at its own worst-case input; the program's value
    is the mean of its arrays' fractions (the measure asked for);
  - *summed worst case*: the same inputs, but total size over total capacity, which is how
    the paper sums its fill per program;
  - *one input*: a single input for the whole program that maximises total size over
    total capacity. This is the paper's measure exactly, evaluated at the worst input
    instead of at A and B. It is lower when no single input fills every array at once.

## Results

| Measure (mean over 36 programs) | Test inputs A, B | Worst-case inputs |
|---|---:|---:|
| Fill, summed per program (the paper's 54%) | 53.6% | 99.3% (each array at its own worst input) |
| Fill, mean of per-array fractions | 53.9% | 99.3% |
| Fill, one input per program | 53.6% (A and B) | 95.4% (median 100%) |
| Arrays that reach their capacity | 20 of 141 (on A or B) | 135 of 141 |
| Programs whose every array reaches capacity | | 35 of 36 |
| Programs with one input filling every array at once | | 33 of 36 |
| Median program capacity / elements | 2.0 | 1.0 (largest 1.46, periodic_dist) |

Weighting arrays equally instead of programs, the mean worst-case fraction is 97.7%.

With the narrower domain L > 0 for `periodic_dist` (see below), the per-array and summed
worst-case fills are 98.5%, one input 95.3%, 129 of 141 arrays reach capacity
(array-weighted 95.5%), and the largest program capacity/elements ratio is 2.13.

**Only one program has arrays below capacity: `pythran_periodic_dist`.** In the other 35
programs every array reaches its capacity on some valid input, and in 33 of them one input
fills every array at the same time. The exceptions are explained next, each with an upper
bound proved by hand and confirmed by z3 (`smt.json`).

### periodic_dist: 6 arrays reach 120 of 256 (46.9%), and the bound is loose

For each axis, `dx[i, j] = x[j] - x[i]` is a 16 x 16 matrix and the program builds four
arrays: `dx[dx > L/2]` and `dx[dx > L/2] - L` (the "first mask"), then, after updating
`dx`, `dx[dx < -L/2]` and `dx[dx < -L/2] + L` (the "second mask"). The capacity of each
is 256, the whole matrix.

- *First mask.* With L < 0 and all points equal, every entry 0 exceeds L/2, so the first
  mask holds all 256 entries. L is declared as a plain Integer with no precondition, so
  this input is valid. If only L > 0 is admitted (a box length), the first mask holds at
  most 120: the diagonal is 0, not above L/2 > 0, and of the pair `dx[i, j] = d`,
  `dx[j, i] = -d` at most one exceeds L/2 >= 0, so at most 16 * 15 / 2 = 120.
- *Second mask, any L.* At most 120 of 256 on every valid input. The first mask replaces
  every entry d > L/2 by d - L, which is greater than L/2 - L = -L/2, so an updated entry
  never qualifies for the second mask. A diagonal entry is 0; if it is not updated, then
  0 <= L/2, so L >= 0 and 0 is not below -L/2. Two entries of a pair, d and -d, that are
  both not updated satisfy d <= L/2 and -d <= L/2; if both were below -L/2, then
  d < -L/2 and d > L/2, impossible for L >= 0, and for L < 0 the bound d <= L/2 < 0 together
  with -d <= L/2 < 0 is already impossible. So at most one entry per off-diagonal pair
  qualifies, 120 in total. The argument holds in Q32 and in float64, because both compute
  `x[i] - x[j]` as the exact negation of `x[j] - x[i]`. Points 2 apart with L = 2 reach 120.
- *One input.* For L < 0 every entry falls in exactly one mask, so the four arrays of an
  axis hold 2 * 256 of 4 * 256 slots (50%); for L > 0 every off-diagonal entry can fall in
  one mask, 2 * 240 (46.9%).

z3 confirms each maximum as an optimum, with a proof that one more is unsatisfiable, for
16 points and for 4, 6 and 8 (first mask 256 for any L and 120 for L > 0, second mask 120,
both masks together 256 and 240). One query is not fully proved by z3: for 16 points and
any L, the optimum of both masks together is 256, but the check of 257 returned "unknown"
after the 15-minute limit. That bound needs no solver: an updated entry never qualifies for
the second mask (shown above), so each of the 256 entries is in at most one mask. The compiler's capacity of 256 for the second mask is an
over-approximation: it does not know that `dx` is antisymmetric. A circuit that knew it
could reserve 120.

### Programs where no single input fills every array

The arrays of these programs each reach capacity separately, but they partition the same
elements, so the bound is not loose for any single array:

- `case0490`, per row, `np.where(temp < t1)`, `np.where(temp >= t2)` and the mask
  `t1 <= temp < t2` (counted twice: the selection and its `+ 5`). Every element
  contributes to at most two of the four arrays (the mask excludes the other two; an
  element outside it is in at most both `np.where` results), so a row fills at most 100 of
  200 slots, 50%. Reached with t2 < t1 (both `np.where` results full) or with every element
  in [t1, t2). z3: 100, and 101 unsatisfiable.
- `pythran_tax`: the four income brackets are disjoint, and the brackets have 3, 3, 6 and
  5 arrays. An income contributes to at most 6 arrays, so at most 60 of 170 slots (35.3%),
  reached when every income is in bracket 3. z3: 60, and 61 unsatisfiable.
- `periodic_dist`, above.

A circuit that must accept every valid input still needs every one of these capacities,
since each array reaches it on some input. Sharing storage between arrays that are never
full together (for example the four tax brackets, which hold at most 10 elements in
total) would save space, but this is a different optimisation from a tighter bound.

### Which methods found the worst cases

- (a) **Construction** by reading the program (`constructions.py`, with the reason for each
  input): every maximum in the table was reached by a construction. Examples: `ncol = 1`
  for the reshape-after-truncation programs, `bin_size = 1` for the binning programs,
  `low = 0, high = 10/20` for the slices, `N = 100` for `argsort(a)[::-1][:N]`, an
  all-nonzero (or all-zero) image for the crop programs, `size = [8, 8]` so that every
  window covers the whole 4 x 4 array, `z = 1000` so that every condition of
  projection_simplex holds.
- (b) **Search** that knows only each parameter's dtype and shape (`run_search.py`): a sweep
  of generic patterns (domain extremes, constants, ramps, alternations, the fixture values),
  20,000 random combinations per program and hill climbing on each array's size and on the
  total. It reached the same maximum as the construction for all 141 arrays and the same
  one-input total for all 36 programs, independently, and never exceeded a construction or
  a capacity.
- (c) **z3** (`smt.py`) on models of the size expressions for the programs where the
  maximum is below capacity or not simultaneous (periodic_dist, 0490, tax): optimal values
  equal to the constructions, with unsatisfiability proofs one above. Every z3 model
  for 0490, tax and 16-point periodic_dist was turned into a program input and measured
  with NumPy; the sizes equal the z3 optima.

## Caveats

- The domain is the declared one: types, shapes and the compiler's value ranges, with no
  preconditions in these programs. Inputs such as L < 0 in periodic_dist, `N = 100` in
  case0383 or `low = 0, high = 20` in case0392 are valid by the signature, though a user
  of the DS-1000 or Pythran originals might never pass them. A program annotated with
  stricter `@requires` preconditions would be measured against those. The L > 0 variant
  is reported separately above.
- Sizes are the number of elements of NumPy values, as in the original measurement. As
  there, they describe the compiler's logical array values, not the distinct storage a
  circuit allocates (`../REPORT.md`, "Definitions and scope").
- Values were checked against the mock backend; no key or proof was generated and nothing
  was timed. Everything ran locally.

## Reproduce

From this folder, with a zinnia-src checkout whose extension is built
(`PYTHON=.venv/bin/python scripts/develop.sh`; here a release build of main ce1b259e in a
separate worktree, so that the shared editable install was left alone):

```sh
ZINNIA_CHECKOUT=/path/to/zinnia-src-checkout ./reproduce.sh
```

`run_search.py` (methods a and b, `search/`), `smt.py` (method c, `smt.json`),
`validate_mock.py` (`mock/`), `report.py` (`tightness.csv`, `summary.json`, the generated
tables below). The search is seeded and deterministic.

## Tables

<!-- generated:begin -->
### Per program

| Program | Arrays | Capacity | Fill A/B (paper) | Worst, per-array mean | Worst, summed | Worst, one input | Arrays at capacity |
|---|---:|---:|---:|---:|---:|---:|---:|
| case0298 | 2 | 30 | 50.0% | 100.0% | 100.0% | 100.0% | 2/2 |
| case0301 | 1 | 20 | 100.0% | 100.0% | 100.0% | 100.0% | 1/1 |
| case0302 | 1 | 20 | 100.0% | 100.0% | 100.0% | 100.0% | 1/1 |
| case0303 | 2 | 46 | 87.0% | 100.0% | 100.0% | 100.0% | 2/2 |
| case0304 | 3 | 69 | 87.0% | 100.0% | 100.0% | 100.0% | 3/3 |
| case0322 | 1 | 8 | 62.5% | 100.0% | 100.0% | 100.0% | 1/1 |
| case0353 | 1 | 6 | 50.0% | 100.0% | 100.0% | 100.0% | 1/1 |
| case0383 | 1 | 100 | 7.0% | 100.0% | 100.0% | 100.0% | 1/1 |
| case0389 | 3 | 216 | 100.0% | 100.0% | 100.0% | 100.0% | 3/3 |
| case0391 | 1 | 200 | 30.0% | 100.0% | 100.0% | 100.0% | 1/1 |
| case0392 | 1 | 200 | 15.0% | 100.0% | 100.0% | 100.0% | 1/1 |
| case0393 | 1 | 200 | 30.0% | 100.0% | 100.0% | 100.0% | 1/1 |
| case0412 | 1 | 10 | 35.0% | 100.0% | 100.0% | 100.0% | 1/1 |
| case0413 | 1 | 10 | 35.0% | 100.0% | 100.0% | 100.0% | 1/1 |
| case0414 | 3 | 24 | 75.0% | 100.0% | 100.0% | 100.0% | 3/3 |
| case0415 | 3 | 24 | 75.0% | 100.0% | 100.0% | 100.0% | 3/3 |
| case0416 | 3 | 144 | 83.3% | 100.0% | 100.0% | 100.0% | 3/3 |
| case0417 | 3 | 24 | 75.0% | 100.0% | 100.0% | 100.0% | 3/3 |
| case0418 | 3 | 144 | 83.3% | 100.0% | 100.0% | 100.0% | 3/3 |
| case0419 | 4 | 192 | 75.0% | 100.0% | 100.0% | 100.0% | 4/4 |
| case0442 | 1 | 20 | 72.5% | 100.0% | 100.0% | 100.0% | 1/1 |
| case0443 | 1 | 20 | 30.0% | 100.0% | 100.0% | 100.0% | 1/1 |
| case0444 | 1 | 20 | 72.5% | 100.0% | 100.0% | 100.0% | 1/1 |
| case0462 | 1 | 20 | 45.0% | 100.0% | 100.0% | 100.0% | 1/1 |
| case0463 | 2 | 40 | 50.0% | 100.0% | 100.0% | 100.0% | 2/2 |
| case0467 | 16 | 256 | 39.1% | 100.0% | 100.0% | 100.0% | 16/16 |
| case0468 | 16 | 256 | 39.1% | 100.0% | 100.0% | 100.0% | 16/16 |
| case0481 | 2 | 18 | 11.1% | 100.0% | 100.0% | 100.0% | 2/2 |
| case0490 | 20 | 1000 | 27.6% | 100.0% | 100.0% | 50.0% | 20/20 |
| case0507 | 3 | 41 | 70.7% | 100.0% | 100.0% | 100.0% | 3/3 |
| case0508 | 2 | 300 | 6.8% | 100.0% | 100.0% | 100.0% | 2/2 |
| case0509 | 3 | 120 | 59.6% | 100.0% | 100.0% | 100.0% | 3/3 |
| case0510 | 3 | 120 | 82.5% | 100.0% | 100.0% | 100.0% | 3/3 |
| pythran_periodic_dist | 12 | 3072 | 0.0% | 73.4% | 73.4% | 50.0% | 6/12 |
| pythran_projection_simplex | 2 | 20 | 50.0% | 100.0% | 100.0% | 100.0% | 2/2 |
| pythran_tax | 17 | 170 | 17.6% | 100.0% | 100.0% | 35.3% | 17/17 |
| pythran_periodic_dist (L>0) | 12 | 3072 | 0.0% | 46.9% | 46.9% | 46.9% | 0/12 |

### Per array

Max = largest size found over all methods; "construction" names the method-(a) input that reaches it (constructions.py); "search" is the largest size the program-agnostic search (sweep, random, hill climbing) found on its own.

| Program | Array (source expression, occurrence) | Capacity | A | B | Max | Max/cap | Construction | Search |
|---|---|---:|---:|---:|---:|---:|---|---:|
| case0298 | `np.unique(a, return_inverse=True)` #0[0] | 5 | 2 | 3 | 5 | 100.0% | distinct | 5 |
| case0298 | `np.zeros((a.size, vals.size))` #0[] | 25 | 10 | 15 | 25 | 100.0% | distinct | 25 |
| case0301 | `np.reshape(A, (-1, ncol))` #0[] | 20 | 20 | 20 | 20 | 100.0% | ncol=4 | 20 |
| case0302 | `np.reshape(A, (nrow, -1))` #0[] | 20 | 20 | 20 | 20 | 100.0% | nrow=4 | 20 |
| case0303 | `A[:col]` #0[] | 23 | 20 | 20 | 23 | 100.0% | ncol=1 | 23 |
| case0303 | `np.reshape(B, (-1, ncol))` #0[] | 23 | 20 | 20 | 23 | 100.0% | ncol=1 | 23 |
| case0304 | `A[len(A) - col:]` #0[] | 23 | 20 | 20 | 23 | 100.0% | ncol=1 | 23 |
| case0304 | `A[len(A) - col:][::-1]` #0[] | 23 | 20 | 20 | 23 | 100.0% | ncol=1 | 23 |
| case0304 | `np.reshape(B, (-1, ncol))` #0[] | 23 | 20 | 20 | 23 | 100.0% | ncol=1 | 23 |
| case0322 | `np.argwhere(a == np.min(a))` #0[] | 8 | 4 | 6 | 8 | 100.0% | all-equal | 8 |
| case0353 | `A[~np.in1d(np.ravel_multi_index(A.T, dims), np.ravel_multi_index(B.T, dims))]` #0[] | 6 | 3 | 3 | 6 | 100.0% | A-rows-not-in-B | 6 |
| case0383 | `np.argsort(a)[::-1][:N]` #0[] | 100 | 7 | 7 | 100 | 100.0% | N=100 | 100 |
| case0389 | `a.reshape(h // nrows, -1, nrows, ncols)` #0[] | 72 | 72 | 72 | 72 | 100.0% | h=18,w=4 | 72 |
| case0389 | `a.reshape(h // nrows, -1, nrows, ncols).swapaxes(1, 2)` #0[] | 72 | 72 | 72 | 72 | 100.0% | h=18,w=4 | 72 |
| case0389 | `a.reshape(h // nrows, -1, nrows, ncols).swapaxes(1, 2).reshape(h, w)` #0[] | 72 | 72 | 72 | 72 | 100.0% | h=18,w=4 | 72 |
| case0391 | `a[:, low:high]` #0[] | 200 | 60 | 60 | 200 | 100.0% | low=0,high=10 | 200 |
| case0392 | `a[low:high, :]` #0[] | 200 | 30 | 30 | 200 | 100.0% | low=0,high=20 | 200 |
| case0393 | `a[:, low:high]` #0[] | 200 | 60 | 60 | 200 | 100.0% | low=0,high=10 | 200 |
| case0412 | `x[x >= 0]` #0[] | 10 | 5 | 2 | 10 | 100.0% | all-nonnegative | 10 |
| case0413 | `x[x.imag != 0]` #0[] | 10 | 4 | 3 | 10 | 100.0% | all-imag-nonzero | 10 |
| case0414 | `data[:data.size // bin_size * bin_size]` #0[] | 8 | 8 | 8 | 8 | 100.0% | bin_size=1 | 8 |
| case0414 | `data[:data.size // bin_size * bin_size].reshape(-1, bin_size)` #0[] | 8 | 8 | 8 | 8 | 100.0% | bin_size=1 | 8 |
| case0414 | `data[:data.size // bin_size * bin_size].reshape(-1, bin_size).mean(axis=1)` #0[] | 8 | 2 | 2 | 8 | 100.0% | bin_size=1 | 8 |
| case0415 | `data[:data.size // bin_size * bin_size]` #0[] | 8 | 8 | 8 | 8 | 100.0% | bin_size=1 | 8 |
| case0415 | `data[:data.size // bin_size * bin_size].reshape(-1, bin_size)` #0[] | 8 | 8 | 8 | 8 | 100.0% | bin_size=1 | 8 |
| case0415 | `data[:data.size // bin_size * bin_size].reshape(-1, bin_size).max(axis=1)` #0[] | 8 | 2 | 2 | 8 | 100.0% | bin_size=1 | 8 |
| case0416 | `data[:, :data.shape[1] // bin_size * bin_size]` #0[] | 48 | 48 | 48 | 48 | 100.0% | bin_size=1 | 48 |
| case0416 | `data[:, :data.shape[1] // bin_size * bin_size].reshape(data.shape[0], -1, bin_size)` #0[] | 48 | 48 | 48 | 48 | 100.0% | bin_size=1 | 48 |
| case0416 | `data[:, :data.shape[1] // bin_size * bin_size].reshape(data.shape[0], -1, bin_size).mean(axis=-1)` #0[] | 48 | 24 | 24 | 48 | 100.0% | bin_size=1 | 48 |
| case0417 | `new_data[:data.size // bin_size * bin_size]` #0[] | 8 | 8 | 8 | 8 | 100.0% | bin_size=1 | 8 |
| case0417 | `new_data[:data.size // bin_size * bin_size].reshape(-1, bin_size)` #0[] | 8 | 8 | 8 | 8 | 100.0% | bin_size=1 | 8 |
| case0417 | `new_data[:data.size // bin_size * bin_size].reshape(-1, bin_size).mean(axis=1)` #0[] | 8 | 2 | 2 | 8 | 100.0% | bin_size=1 | 8 |
| case0418 | `new_data[:, :data.shape[1] // bin_size * bin_size]` #0[] | 48 | 48 | 48 | 48 | 100.0% | bin_size=1 | 48 |
| case0418 | `new_data[:, :data.shape[1] // bin_size * bin_size].reshape(data.shape[0], -1, bin_size)` #0[] | 48 | 48 | 48 | 48 | 100.0% | bin_size=1 | 48 |
| case0418 | `new_data[:, :data.shape[1] // bin_size * bin_size].reshape(data.shape[0], -1, bin_size).mean(axis=-1)` #0[] | 48 | 24 | 24 | 48 | 100.0% | bin_size=1 | 48 |
| case0419 | `new_data[:, :data.shape[1] // bin_size * bin_size]` #0[] | 48 | 48 | 48 | 48 | 100.0% | bin_size=1 | 48 |
| case0419 | `new_data[:, :data.shape[1] // bin_size * bin_size].reshape(data.shape[0], -1, bin_size)` #0[] | 48 | 48 | 48 | 48 | 100.0% | bin_size=1 | 48 |
| case0419 | `new_data[:, :data.shape[1] // bin_size * bin_size].reshape(data.shape[0], -1, bin_size).mean(axis=-1)` #0[] | 48 | 24 | 24 | 48 | 100.0% | bin_size=1 | 48 |
| case0419 | `new_data[:, :data.shape[1] // bin_size * bin_size].reshape(data.shape[0], -1, bin_size).mean(axis=-1)[:, ::-1]` #0[] | 48 | 24 | 24 | 48 | 100.0% | bin_size=1 | 48 |
| case0442 | `A[~np.in1d(A, B)]` #0[] | 20 | 12 | 17 | 20 | 100.0% | A-disjoint-from-B | 20 |
| case0443 | `A[np.in1d(A, B)]` #0[] | 20 | 5 | 7 | 20 | 100.0% | A-inside-B | 20 |
| case0444 | `A[np.logical_and(A > B[0], A < B[1]) \| np.logical_and(A > B[1], A < B[2])]` #0[] | 20 | 12 | 17 | 20 | 100.0% | A-in-(B0,B1) | 20 |
| case0462 | `a[selection]` #0[] | 20 | 10 | 8 | 20 | 100.0% | alternating-nonzero | 20 |
| case0463 | `a[selection]` #0[] | 20 | 9 | 11 | 20 | 100.0% | alternating-nonzero | 20 |
| case0463 | `a[selection].reshape(-1, 1)` #0[] | 20 | 9 | 11 | 20 | 100.0% | alternating-nonzero | 20 |
| case0467 | `window(a, size)` #0[0] | 16 | 4 | 4 | 16 | 100.0% | size=[8,8] | 16 |
| case0467 | `window(a, size)` #0[1] | 16 | 6 | 6 | 16 | 100.0% | size=[8,8] | 16 |
| case0467 | `window(a, size)` #0[2] | 16 | 6 | 6 | 16 | 100.0% | size=[8,8] | 16 |
| case0467 | `window(a, size)` #0[3] | 16 | 4 | 4 | 16 | 100.0% | size=[8,8] | 16 |
| case0467 | `window(a, size)` #0[4] | 16 | 6 | 6 | 16 | 100.0% | size=[8,8] | 16 |
| case0467 | `window(a, size)` #0[5] | 16 | 9 | 9 | 16 | 100.0% | size=[8,8] | 16 |
| case0467 | `window(a, size)` #0[6] | 16 | 9 | 9 | 16 | 100.0% | size=[8,8] | 16 |
| case0467 | `window(a, size)` #0[7] | 16 | 6 | 6 | 16 | 100.0% | size=[8,8] | 16 |
| case0467 | `window(a, size)` #0[8] | 16 | 6 | 6 | 16 | 100.0% | size=[8,8] | 16 |
| case0467 | `window(a, size)` #0[9] | 16 | 9 | 9 | 16 | 100.0% | size=[8,8] | 16 |
| case0467 | `window(a, size)` #0[10] | 16 | 9 | 9 | 16 | 100.0% | size=[8,8] | 16 |
| case0467 | `window(a, size)` #0[11] | 16 | 6 | 6 | 16 | 100.0% | size=[8,8] | 16 |
| case0467 | `window(a, size)` #0[12] | 16 | 4 | 4 | 16 | 100.0% | size=[8,8] | 16 |
| case0467 | `window(a, size)` #0[13] | 16 | 6 | 6 | 16 | 100.0% | size=[8,8] | 16 |
| case0467 | `window(a, size)` #0[14] | 16 | 6 | 6 | 16 | 100.0% | size=[8,8] | 16 |
| case0467 | `window(a, size)` #0[15] | 16 | 4 | 4 | 16 | 100.0% | size=[8,8] | 16 |
| case0468 | `window(a, size)` #0[0] | 16 | 4 | 4 | 16 | 100.0% | size=[8,8] | 16 |
| case0468 | `window(a, size)` #0[1] | 16 | 6 | 6 | 16 | 100.0% | size=[8,8] | 16 |
| case0468 | `window(a, size)` #0[2] | 16 | 6 | 6 | 16 | 100.0% | size=[8,8] | 16 |
| case0468 | `window(a, size)` #0[3] | 16 | 4 | 4 | 16 | 100.0% | size=[8,8] | 16 |
| case0468 | `window(a, size)` #0[4] | 16 | 6 | 6 | 16 | 100.0% | size=[8,8] | 16 |
| case0468 | `window(a, size)` #0[5] | 16 | 9 | 9 | 16 | 100.0% | size=[8,8] | 16 |
| case0468 | `window(a, size)` #0[6] | 16 | 9 | 9 | 16 | 100.0% | size=[8,8] | 16 |
| case0468 | `window(a, size)` #0[7] | 16 | 6 | 6 | 16 | 100.0% | size=[8,8] | 16 |
| case0468 | `window(a, size)` #0[8] | 16 | 6 | 6 | 16 | 100.0% | size=[8,8] | 16 |
| case0468 | `window(a, size)` #0[9] | 16 | 9 | 9 | 16 | 100.0% | size=[8,8] | 16 |
| case0468 | `window(a, size)` #0[10] | 16 | 9 | 9 | 16 | 100.0% | size=[8,8] | 16 |
| case0468 | `window(a, size)` #0[11] | 16 | 6 | 6 | 16 | 100.0% | size=[8,8] | 16 |
| case0468 | `window(a, size)` #0[12] | 16 | 4 | 4 | 16 | 100.0% | size=[8,8] | 16 |
| case0468 | `window(a, size)` #0[13] | 16 | 6 | 6 | 16 | 100.0% | size=[8,8] | 16 |
| case0468 | `window(a, size)` #0[14] | 16 | 6 | 6 | 16 | 100.0% | size=[8,8] | 16 |
| case0468 | `window(a, size)` #0[15] | 16 | 4 | 4 | 16 | 100.0% | size=[8,8] | 16 |
| case0481 | `idx_list.nonzero()` #0[0] | 9 | 1 | 1 | 9 | 100.0% | all-match | 9 |
| case0481 | `idx_list.nonzero()[0]` #0[] | 9 | 1 | 1 | 9 | 100.0% | all-match | 9 |
| case0490 | `np.where(temp < t1)` #0[0] | 50 | 21 | 28 | 50 | 100.0% | all-below-t1 | 50 |
| case0490 | `np.where(temp >= t2)` #0[0] | 50 | 19 | 16 | 50 | 100.0% | all-at-least-t2 | 50 |
| case0490 | `a[np.logical_and(temp >= t1, temp < t2)]` #0[] | 50 | 10 | 6 | 50 | 100.0% | all-in-[t1,t2) | 50 |
| case0490 | `a[np.logical_and(temp >= t1, temp < t2)] + 5` #0[] | 50 | 10 | 6 | 50 | 100.0% | all-in-[t1,t2) | 50 |
| case0490 | `np.where(temp < t1)` #1[0] | 50 | 34 | 23 | 50 | 100.0% | all-below-t1 | 50 |
| case0490 | `np.where(temp >= t2)` #1[0] | 50 | 12 | 24 | 50 | 100.0% | all-at-least-t2 | 50 |
| case0490 | `a[np.logical_and(temp >= t1, temp < t2)]` #1[] | 50 | 4 | 3 | 50 | 100.0% | all-in-[t1,t2) | 50 |
| case0490 | `a[np.logical_and(temp >= t1, temp < t2)] + 5` #1[] | 50 | 4 | 3 | 50 | 100.0% | all-in-[t1,t2) | 50 |
| case0490 | `np.where(temp < t1)` #2[0] | 50 | 18 | 23 | 50 | 100.0% | all-below-t1 | 50 |
| case0490 | `np.where(temp >= t2)` #2[0] | 50 | 27 | 24 | 50 | 100.0% | all-at-least-t2 | 50 |
| case0490 | `a[np.logical_and(temp >= t1, temp < t2)]` #2[] | 50 | 5 | 3 | 50 | 100.0% | all-in-[t1,t2) | 50 |
| case0490 | `a[np.logical_and(temp >= t1, temp < t2)] + 5` #2[] | 50 | 5 | 3 | 50 | 100.0% | all-in-[t1,t2) | 50 |
| case0490 | `np.where(temp < t1)` #3[0] | 50 | 26 | 26 | 50 | 100.0% | all-below-t1 | 50 |
| case0490 | `np.where(temp >= t2)` #3[0] | 50 | 16 | 22 | 50 | 100.0% | all-at-least-t2 | 50 |
| case0490 | `a[np.logical_and(temp >= t1, temp < t2)]` #3[] | 50 | 8 | 2 | 50 | 100.0% | all-in-[t1,t2) | 50 |
| case0490 | `a[np.logical_and(temp >= t1, temp < t2)] + 5` #3[] | 50 | 8 | 2 | 50 | 100.0% | all-in-[t1,t2) | 50 |
| case0490 | `np.where(temp < t1)` #4[0] | 50 | 25 | 19 | 50 | 100.0% | all-below-t1 | 50 |
| case0490 | `np.where(temp >= t2)` #4[0] | 50 | 20 | 25 | 50 | 100.0% | all-at-least-t2 | 50 |
| case0490 | `a[np.logical_and(temp >= t1, temp < t2)]` #4[] | 50 | 5 | 6 | 50 | 100.0% | all-in-[t1,t2) | 50 |
| case0490 | `a[np.logical_and(temp >= t1, temp < t2)] + 5` #4[] | 50 | 5 | 6 | 50 | 100.0% | all-in-[t1,t2) | 50 |
| case0507 | `np.flatnonzero((~mask).sum(axis=1))` #0[] | 5 | 4 | 4 | 5 | 100.0% | all-nonzero | 5 |
| case0507 | `np.flatnonzero((~mask).sum(axis=0))` #0[] | 6 | 5 | 5 | 6 | 100.0% | all-nonzero | 6 |
| case0507 | `result (branch-merged crop)` | 30 | 20 | 20 | 30 | 100.0% | all-nonzero | 30 |
| case0508 | `np.argwhere(A)` #0[] | 200 | 12 | 8 | 200 | 100.0% | all-nonzero | 200 |
| case0508 | `A[ystart:ystop, xstart:xstop]` #0[] | 100 | 15 | 6 | 100 | 100.0% | all-nonzero | 100 |
| case0509 | `np.flatnonzero(mask.sum(axis=1))` #0[] | 10 | 5 | 7 | 10 | 100.0% | all-zero | 10 |
| case0509 | `np.flatnonzero(mask.sum(axis=0))` #0[] | 10 | 5 | 6 | 10 | 100.0% | all-zero | 10 |
| case0509 | `result (branch-merged crop)` | 100 | 48 | 72 | 100 | 100.0% | all-zero | 100 |
| case0510 | `np.flatnonzero((~mask).sum(axis=1))` #0[] | 10 | 9 | 9 | 10 | 100.0% | all-nonzero | 10 |
| case0510 | `np.flatnonzero((~mask).sum(axis=0))` #0[] | 10 | 9 | 9 | 10 | 100.0% | all-nonzero | 10 |
| case0510 | `result (branch-merged crop)` | 100 | 81 | 81 | 100 | 100.0% | all-nonzero | 100 |
| pythran_periodic_dist | `dx[dx > L / 2]` #0[] | 256 | 0 | 0 | 256 (120 with L>0) | 100.0% | L=-2,equal | 256 |
| pythran_periodic_dist | `dx[dx > L / 2] - L` #0[] | 256 | 0 | 0 | 256 (120 with L>0) | 100.0% | L=-2,equal | 256 |
| pythran_periodic_dist | `dx[dx < -L / 2]` #0[] | 256 | 0 | 0 | 120 | 46.9% | L=2,spread | 120 |
| pythran_periodic_dist | `dx[dx < -L / 2] + L` #0[] | 256 | 0 | 0 | 120 | 46.9% | L=2,spread | 120 |
| pythran_periodic_dist | `dy[dy > L / 2]` #0[] | 256 | 0 | 0 | 256 (120 with L>0) | 100.0% | L=-2,equal | 256 |
| pythran_periodic_dist | `dy[dy > L / 2] - L` #0[] | 256 | 0 | 0 | 256 (120 with L>0) | 100.0% | L=-2,equal | 256 |
| pythran_periodic_dist | `dy[dy < -L / 2]` #0[] | 256 | 0 | 0 | 120 | 46.9% | L=2,spread | 120 |
| pythran_periodic_dist | `dy[dy < -L / 2] + L` #0[] | 256 | 0 | 0 | 120 | 46.9% | L=2,spread | 120 |
| pythran_periodic_dist | `dz[dz > L / 2]` #0[] | 256 | 0 | 0 | 256 (120 with L>0) | 100.0% | L=-2,equal | 256 |
| pythran_periodic_dist | `dz[dz > L / 2] - L` #0[] | 256 | 0 | 0 | 256 (120 with L>0) | 100.0% | L=-2,equal | 256 |
| pythran_periodic_dist | `dz[dz < -L / 2]` #0[] | 256 | 0 | 0 | 120 | 46.9% | L=2,spread | 120 |
| pythran_periodic_dist | `dz[dz < -L / 2] + L` #0[] | 256 | 0 | 0 | 120 | 46.9% | L=2,spread | 120 |
| pythran_projection_simplex | `ind[cond]` #0[] | 10 | 5 | 5 | 10 | 100.0% | large-z | 10 |
| pythran_projection_simplex | `cssv[cond]` #0[] | 10 | 5 | 5 | 10 | 100.0% | large-z | 10 |
| pythran_tax | `d[d > 256303]` #0[] | 10 | 0 | 0 | 10 | 100.0% | all>256303 | 10 |
| pythran_tax | `d[d > 256303] * 0.45` #0[] | 10 | 0 | 0 | 10 | 100.0% | all>256303 | 10 |
| pythran_tax | `d[d > 256303] * 0.45 - 16164.53` #0[] | 10 | 0 | 0 | 10 | 100.0% | all>256303 | 10 |
| pythran_tax | `d[(d > 54057) & (d <= 256303)]` #0[] | 10 | 10 | 10 | 10 | 100.0% | all-in-(54057,256303] | 10 |
| pythran_tax | `d[(d > 54057) & (d <= 256303)] * 0.42` #0[] | 10 | 10 | 10 | 10 | 100.0% | all-in-(54057,256303] | 10 |
| pythran_tax | `d[(d > 54057) & (d <= 256303)] * 0.42 - 8475.44` #0[] | 10 | 10 | 10 | 10 | 100.0% | all-in-(54057,256303] | 10 |
| pythran_tax | `d[(d > 13769) & (d <= 54057)]` #0[] | 10 | 0 | 0 | 10 | 100.0% | all-in-(13769,54057] | 10 |
| pythran_tax | `d[(d > 13769) & (d <= 54057)] - 13769` #0[] | 10 | 0 | 0 | 10 | 100.0% | all-in-(13769,54057] | 10 |
| pythran_tax | `d[(d > 8820) & (d <= 13769)]` #0[] | 10 | 0 | 0 | 10 | 100.0% | all-in-(8820,13769] | 10 |
| pythran_tax | `d[(d > 8820) & (d <= 13769)] - 8820` #0[] | 10 | 0 | 0 | 10 | 100.0% | all-in-(8820,13769] | 10 |
| pythran_tax | `seg3 * 2.2376e-06` #0[] | 10 | 0 | 0 | 10 | 100.0% | all-in-(13769,54057] | 10 |
| pythran_tax | `seg3 * 2.2376e-06 + 0.2397` #0[] | 10 | 0 | 0 | 10 | 100.0% | all-in-(13769,54057] | 10 |
| pythran_tax | `seg4 * 1.00727e-05` #0[] | 10 | 0 | 0 | 10 | 100.0% | all-in-(8820,13769] | 10 |
| pythran_tax | `seg4 * 1.00727e-05 + 0.14` #0[] | 10 | 0 | 0 | 10 | 100.0% | all-in-(8820,13769] | 10 |
| pythran_tax | `seg3 * prog_seg3` #0[] | 10 | 0 | 0 | 10 | 100.0% | all-in-(13769,54057] | 10 |
| pythran_tax | `seg3 * prog_seg3 + 939.57` #0[] | 10 | 0 | 0 | 10 | 100.0% | all-in-(13769,54057] | 10 |
| pythran_tax | `seg4 * prog_seg4` #0[] | 10 | 0 | 0 | 10 | 100.0% | all-in-(8820,13769] | 10 |
<!-- generated:end -->
