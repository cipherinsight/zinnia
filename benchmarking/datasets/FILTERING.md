# Which programs are in scope

This file states, once, the rules that decide which benchmark programs the
evaluation uses. They are applied the same way to every dataset: the DS-1000
NumPy slice (220 programs), NPBench (54) and Pythran (115). The code that applies
them is `tools/scope_rules.py` (shared), with `tools/ds1000_scope.py` for
DS-1000 and `tools/acceptance_case_set.py` for NPBench and Pythran. The results
are in `case_sets/acceptance.json` (one entry per program and function) and, for
DS-1000, `ds1000_numpy/manifest.json` and `ds1000_numpy/SCOPE.md`.

## Principle

A rule looks at the program text and at the inputs the program is meant to
receive. No rule looks at whether Zinnia compiles the program. A program that is
in scope and that Zinnia rejects counts as a rejection in the acceptance study;
it is never filtered out for that reason.

## The rules

A program is out of scope when at least one rule fires.

| Rule | Fires when the program ... |
| --- | --- |
| `imports` | imports or uses a module other than `numpy` (including its submodules, such as `numpy.linalg`) and `math`. This is the author's rule (2026-10-06). `from __future__ import ...` is a compiler directive and does not count. |
| `exception_control_flow` | uses `try` / `except` / `finally`. |
| `external_io` | reads or writes files, the network, stdin or stdout (`print` included), or uses NumPy file I/O. |
| `nondeterminism` | draws random numbers or reads the time, the process or the environment. |
| `non_numeric_data` | works on strings, bytes, Python objects, ragged sequences (lists of arrays of different shapes), datetimes or record arrays, in its operations or in its inputs. |
| `non_finite_values` | uses NaN or infinity (`np.nan`, `np.inf`, `isnan`, `nan*` reductions, `float("inf")`, ...) or receives NaN or infinity as an input. Zinnia and every baseline compute reals in Q32 fixed point, which has no such values; this is a stated limit of the claim, not a compiler gap. |
| `opaque_runtime` | uses `eval`, `exec`, `compile`, `getattr`, `setattr`, `globals`, `locals` or import machinery. |
| `plotting` | draws plots or uses interactive display. |
| `no_callable_kernel` | has no top-level function to call as the kernel (NPBench and Pythran only). |

Control flow, loop counts, recursion, program size and the NumPy functions a
program calls are not filters.

### Rules removed on 2026-10-06

* `excluded_numpy_family` (DS-1000): excluded programs that call `np.linalg`,
  `np.lib` or `np.polyfit`. It contradicted the policy above (these are NumPy
  operators Zinnia may lack, which makes them rejections, not exclusions) and
  removed 9 deterministic numeric programs: 0348, 0384, 0386, 0456, 0457, 0458,
  0477, 0482, 0483 (audit DB-09). They are back in scope.
* `unbounded_runtime_behavior` (DS-1000): excluded any `while` loop. NPBench and
  Pythran never had it ("control flow is not a filter"), and it fired on no
  DS-1000 program, so removing it makes the datasets uniform without changing
  any status.
* The DS-1000 checks for pandas attribute names (`.values`, `.iloc`, ...) and
  `.numpy()` are replaced by the precise checks below (library objects among
  the test inputs, names bound by library imports).

## DS-1000

Source: `xlang-ai/DS-1000`, `data/ds1000.jsonl.gz` (cached as
`ds1000_numpy/ds1000.jsonl.gz`, byte-identical to upstream `main` on
2026-10-06), the 220 records with `library == "Numpy"`.

A DS-1000 program is the reference solution as DS-1000 runs it: inside the
record's `exec_context`, fed by its test inputs. The rules are applied to

* the solution text, with the `def f(...)` wrapper DS-1000 adds for function
  problems, and the `exec_context` lines after the solution;
* the `exec_context` statements that compute an input the solution reads
  (problem 0477 reads singular values computed with `np.linalg.svd`);
* the `exec_context` imports whose names the solution uses (`scipy`, `pd`,
  `MinMaxScaler`, ...): rule `imports`;
* every DS-1000 test input, produced by running DS-1000's own
  `define_test_input` with NumPy only: a pandas, torch or TensorFlow object is
  rule `imports`; a string, object, ragged or datetime value is
  `non_numeric_data`; NaN or infinity is `non_finite_values`. (Before
  2026-10-06 only the solution text was checked, so 0344, 0459, 0460, 0469,
  0470, 0473 and 0492 were kept and their string, complex-infinity, ragged or
  NaN inputs were typed as ordinary numbers; audit DB-08, DB-09.)

Two programs (0398, 0399) receive a `pandas.Series` of floats and only take its
length and index it by position, which a NumPy array supports; they are in scope
and their kernels take the array. This is the one place the `imports` rule is
read as "the program uses the library", not "the test harness built the input
with it"; see the report's open questions.

Result (2026-10-06): 163 in scope, 57 out of scope. `ds1000_numpy/SCOPE.md`
lists every exclusion with its evidence.

### Kernels

`tools/generate_ds1000_kernels.py` turns each in-scope program into a kernel,
mechanically:

* Parameters: the names the solution reads that the `exec_context` binds, in
  binding order, plus DS-1000 test inputs nothing reads (kept so the signature
  matches the DS-1000 program).
* Types: each parameter's shape is its shape in the record's last test case; its
  dtype is the narrowest of Boolean < Integer < Float < Complex that holds its
  value in every test case. An input with no Zinnia type is never guessed: it
  makes the program out of scope by the rules above.
* Body: the reference text. Function problems keep their own `return`; helper
  functions move to module level as `@zk_chip`; a solution that is one function
  called with its own parameter names is inlined.
* Return value: the value DS-1000 checks (`result = <expr>` after the solution
  in `exec_context`, otherwise `result`).
* No preconditions (`@requires`) are added, with one exception: 0318 and 0319
  (`c[permutation] = np.arange(n)`, an inverse permutation) state their input
  as a permutation of `0..n`, `@requires(lambda a, permutation:
  is_permutation(permutation))` (author decision 2026-10-06). NumPy leaves
  `np.empty` entries uninitialized for any other input, so the program's
  result is defined only on permutations; the precondition is enforced in
  circuit, and the handwritten baselines enforce the same statement (every
  position hit exactly once by `[p_k == j]`), so neither side handles
  duplicate or negative entries. The fixtures draw permutations.

The acceptance kernels (every parameter a runtime input) are in
`ds1000_numpy/generated/`. The performance set (`ds1000/`, see
`case_sets/ds1000_perf.json`) uses the same generator with DS-1000 shape
parameters bound as compile-time constants (author decision 2026-10-06). The
generator refuses to write into the performance set.

## NPBench and Pythran

Source pools: NPBench (`spcl/npbench`, the 54 `*_numpy.py` kernels) and the
Pythran test cases (`serge-sans-paille/pythran`, `pythran/tests/cases/`; 131
files, 126 kernel families after keeping the smallest file per family, 115
after the early exclusions below). Each program folder holds one `kernel.py`;
`upstream_sources.json` names its upstream file and the upstream commits the
kernels were checked against on 2026-10-06.

The rules are applied to the code a kernel retains from upstream (the kernel
without its Zinnia import, decorators and annotations) and to the upstream
input types of each entry function: an entry function whose every exported
signature has a string, dict, set or module parameter, or whose upstream
example is a ragged sequence, fires `non_numeric_data`.

Entry functions: NPBench's `func_name` for the benchmark; for Pythran, every
exported function (`#pythran export`) that no other exported function calls.
An exported function that another one calls is a `@zk_chip` helper of its
caller and is compiled as part of it (listed in `case_sets/acceptance.json` with
`entry_function: false`).

What is verbatim: the name, parameters, defaults and body of every retained
function (equal ASTs once docstrings are dropped), the helper functions and
module constants the kernel needs, and the imports it needs. Runners, data
generators, unused alternate kernels and scaffolding were removed.

What is added: the `@zk_circuit` / `@zk_chip` decorators, Zinnia type
annotations on the parameters of entry functions, and, on 46 entry functions,
`@requires` preconditions. (The 2026-05-19 version of this file said "no
function signatures were modified"; that was wrong.)

### Signatures (corrected 2026-10-06, audit DB-10)

Until 2026-10-06 the annotations were machine drafts (`contracts/*.json`,
`"review_status": "draft"`): 43 of the 169 programs had an annotation whose rank
or dtype contradicted upstream (for example `polybench/mvt` took vectors as
scalars, `crc16` took a byte array as a float, `lapl2d` took 2-D arrays as
1-D). `tools/acceptance_case_set.py` now derives every entry-function
annotation from upstream:

* dtype and rank: NPBench's `initialize()` output for the argument (or its
  preset parameter); Pythran's exported signature (Pythran lists become
  arrays; a list of k-tuples becomes an array whose last axis has length k).
* shape: the small inputs of the next section. The shape at the upstream
  example input is kept in `case_sets/acceptance.json` as
  `upstream_signature` (NPBench: preset S, its smallest; Pythran: the first
  `#runas` call of the function that an exported signature accepts, else the
  first such `#bench` call). When a function has several exported signatures,
  the one that accepts the example is used; without an example, the first
  all-numeric one.
* helper functions carry no annotations (`@zk_chip` infers them at the call).

Other kernel changes made on 2026-10-06:

* Zinnia decorators and annotations that the drafts had put on functions
  nested inside a body (`monte_carlo`, `sumarray3d`) are removed, so those
  bodies are verbatim again.
* Exported functions the extraction had left out are copied verbatim from
  upstream: `emin.run`, `shallow_water.run`, `multitype.check`,
  `make_circle_two_points._make_circle_one_point`. The drafts had made a
  helper the entry instead (for example `emin.Eandg`).
* `@requires` preconditions are kept on entry functions and listed per entry
  in `case_sets/acceptance.json` with their kind (trip-count bound or non-zero
  divisor) and whether the example input (at the small size) satisfies them;
  every kept precondition of an in-scope entry does. They are machine drafts
  that nobody reviewed; preconditions on functions that are now helpers are
  dropped (listed per program), and so is `spectraldns.add_pressure_diffusion`'s `0 <= K <= 1024`,
  drafted when the array `K` was taken for an integer (listed under
  `preconditions_dropped`). Their bounds (1024 for trip counts) are not
  lowered to the small sizes.

### Sizes (2026-10-06)

The acceptance study asks which programs Zinnia accepts, not how it scales, and
the upstream inputs are far too large for any ZK system (NPBench's preset S, its
smallest, multiplies a 1000x1200 by a 1200x1100 matrix; `deep_learning/mlp`
takes 64 million input elements). Every NPBench and Pythran entry is therefore
typed and run at a small size, stated in `tools/acceptance_sizes.py`:

* every array dimension of an input is at most 16;
* scalars that are sizes or trip counts are at most 16; other scalars keep their
  upstream value (crc16's polynomial 33800, `is_square(12**5)`,
  `gf2mulmod(2**10, 2**6-1, 2**5-1)`, `times(200, 3)`);
* relations among dimensions are kept: what is equal stays equal, n and n+1
  stay n and n+1, a dimension computed from a parameter is still computed from
  it;
* where a loop runs once per permutation of n items (`fannkuch`, `n_queens`,
  `permutations`), n is 4: 24 iterations instead of 120 to 362880.

NPBench: preset "T" is preset S with every integer parameter p replaced by
max(min(p, 2), ceil(p / d)), for the smallest whole d such that NPBench's
`initialize()` succeeds and every integer parameter, integer input and array
dimension is at most 16 (gemm: NI, NJ, NK = 1000, 1100, 1200 -> 14, 15, 16 with
d = 75). Inputs come from `initialize()` at T, so shapes stay consistent with
each other (spmv's M+1 row pointers and nnz entries, hdiff's I+4 halo,
stockham's N = R^K). Two exceptions:

* `deep_learning/lenet`: N = 2 and H = W = 16. Two 5x5 convolutions and two 2x2
  poolings need H, W >= 16 for a non-empty feature map. The fully connected
  widths 120, 84 and 10 are constants of lenet's `initialize()`, not
  parameters, so `fc1w`, `fc1b`, `fc2w`, `fc2b`, `fc3w` keep dimensions 120 and
  84.
* `nbody`: the trip count Nt = ceil(tEnd / dt) comes from the float tEnd, which
  the integer rule leaves alone; tEnd = 0.8 (S: 2.0) gives Nt = 16.

The divisor and preset T of each benchmark are in the entry's `sizing` record.

Pythran: the upstream example line (`#runas`, else `#bench`) with its sizes
scaled down by hand, keeping how the line builds its arguments; each scaled
line is listed in `PYTHRAN_SMALL_EXAMPLES` next to what it changes (for
example `diffusion`: grid 2**7 -> 2**4 with the impulse still at the centre,
trip count 500 -> 16; `factorize_naive`: 12222 -> 12, since n bounds the loop
and its precondition n <= 1024 rejects 12222). A line whose arrays and size
scalars are already at most 16 is used as it is. Six exported functions that
no upstream example calls get an example chosen here
(`PYTHRAN_NO_UPSTREAM_EXAMPLE`: `sqr_dists_loops`, `fbcorr`, `fweb`,
`Decoding_logBP`, `_make_circle_one_point`, `solve`); before 2026-10-06 they
had every axis of length 8.

Sizes written in a function body are not inputs and stay as upstream wrote
them, since bodies are verbatim: emin's 1,000,000 steps, the 600 x d and
200 x d point sets of `allpairs_distances` (500 x d and 200 x d in
`allpairs_distances_loops`), guerre's 50-entry work lists.

### Inputs A and B (fixtures)

Every in-scope entry function has two inputs, A and B, in a `fixture.json` next
to its kernel (`fixture.<function>.json` when the program has several entry
functions), with the schema of the performance set's fixtures
(`tools/acceptance_fixtures.py`):

* NPBench: A is `initialize()` at preset T (NumPy's global generator seeded
  with 42 first, since mlp draws its input from it).
* Pythran: A is the arguments of the example call above; `np.empty` in an
  example becomes `np.zeros` (spectraldns's `cross2` output buffer would
  otherwise be uninitialized memory).
* DS-1000: the performance set's draws A and B for the problem when it has a
  performance case (135 functions), with the shape parameters the
  performance kernel binds as constants at their DS-1000 value; otherwise
  (28) A is DS-1000's own last test input.
* B, when not given: a seeded perturbation of A that keeps sizes, trip counts
  and integer scalars: Gaussian noise of 10 % of the largest magnitude on
  floats with integer arrays redrawn in their range, else multiplicative noise
  of 5 %, else the same with integer arrays kept, else one random factor in
  [1.05, 1.25] on every float (keeps the Cholesky inputs positive definite).
  A kernel with only integer scalars gets each minus one (plus one for 0 and
  1). The first B on which the kernel runs is kept.

Check: for A and B the kernel body runs as plain NumPy (division by zero,
overflow and invalid operations raise) and every output, and every argument
after the call, is finite. Zinnia is not run. The notes would flag values Q32
cannot hold (inputs or outputs of magnitude 2^31 or more, nonzero inputs below
2^-32); no fixture has any. Running the generator twice gives byte-identical
fixtures.

Inputs changed for Q32 (`PYTHRAN_Q32_INPUTS` and `NPBENCH_Q32_INPUTS` in
`tools/acceptance_fixtures.py`). Where the example input puts a program where
no Q32 implementation can follow NumPy, the input is changed, and the reason
is in the fixture notes and in the entry's `sizing` (`q32_change`, with the
example it replaces):

* `log_likelihood`: the data are drawn (seeded) within 2 sigma of the mean;
  `np.arange(16.)` reaches 13.5 sigma, where the density is about 1e-40, 0 in
  Q32, and its log is undefined.
* `lu.factorMatrix0` and `lu.factorMatrix1`: A and B are seeded strictly
  diagonally dominant 5x5 matrices, one per function and input. Upstream's
  `np.arange(1., 26.).reshape(5, 5)` is singular (an exact zero pivot in Q32),
  and upstream calls `factorMatrix1` on `factorMatrix0`'s output.
* `hyantes_core.run`: the towns are moved by 0.007 rad, so none sits exactly on
  a grid point (acos at exactly 1 turns Q32's ~1e-9 trigonometry error into
  0.28 km); the two that did are now 51 and 63 km away, still within range.
* `calculate_u.timeloop`: t_stop 2.0 -> 1.45, so 5 time steps instead of 10;
  the explicit scheme grows about 30x per step and 10 steps reach 2.3e13,
  beyond 2^31. (1.45 keeps the loop test `t <= t_stop` 0.05 away from equality,
  whatever the rounding of 0.1.)
* NPBench `vadv`: `wcon` is NPBench's draw times 0.1, which makes the
  tridiagonal system strictly diagonally dominant; NPBench's own draw makes the
  Thomas sweep divide by values down to 3.7e-5.

Output tolerance: every entry of `case_sets/acceptance.json` has
`float_tolerance`. A float output passes when |actual - expected| <= atol +
rtol * |expected| against NumPy on the same input; integer and boolean outputs
are compared exactly. The default is atol = rtol = 1e-6. Two entries have a
larger atol, because their NumPy program itself is not reproducible in Q32,
each with the measured effect (recomputed by the tool, which stops if it
exceeds atol):

* `nbody`: atol 2e-3. The 16-step integration is chaotic: rounding only the
  inputs to Q32 moves NumPy's own output by 9.9e-4.
* `resnet`: atol 2e-6. `conv2d` writes into a float32 buffer, which Q32 cannot
  imitate; the float32 rounding alone moves the output by 1.7e-6.

Result (2026-10-06, entry functions): NPBench 54 in scope, 0 out; Pythran 110
in scope, 12 out (`imports`: `descent` (itertools), `extrema` (functools),
`ramsurf` (cmath); `nondeterminism`: `rand_mat_stat` (`from numpy.random import
randn`, which the rule missed until 2026-10-06 because it only looked for
`np.random.*` attributes); `non_numeric_data`: `build_system` (a module
parameter), `comp_unrolling` (its only example is a ragged list),
`euler_challenge13` (`str()`), `rc4`, `roman_decode`, `scrabble` (2 functions),
`train_equalizer` (string parameters)). Parameters an export does not list keep their default and
are annotated from it (`check_mask.mask`, `ising_step.beta`, `weights.threshold`);
`nd_local_maxima.local_maxima`'s `mode=wrap` (a function) stays unannotated.

Early exclusions (unchanged, from the 2026-05-19 pass, under the rules above):

| Source | Rule |
| --- | --- |
| `pythran:arc_distance_list.py` | `nondeterminism` |
| `pythran:brownian.py` | `nondeterminism` |
| `pythran:count_perm_sig.py` | `nondeterminism` |
| `pythran:empirical.py` | `external_io` |
| `pythran:frequent_itemsets.py` | `external_io` |
| `pythran:grayscott.py` | `nondeterminism` |
| `pythran:julia_pure_python.py` | `imports` (`time`) |
| `pythran:kmeans.py` | `nondeterminism` |
| `pythran:loopy_jacob.py` | `exception_control_flow` |
| `pythran:monte_carlo_pricer.py` | `nondeterminism` |
| `pythran:pi_buffon.py` | `nondeterminism` |
