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
* No preconditions (`@requires`) are added.

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
annotations on the parameters of entry functions, and, on 47 entry functions,
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
* shape: the upstream example input. NPBench: preset S, its smallest. Pythran:
  the first `#runas` call of the function that an exported signature accepts,
  else the first such `#bench` call; with neither (8 functions), every axis
  has length 8. When a function has several exported signatures, the one that
  accepts that example is used; without an example, the first all-numeric one.
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
  divisor) and whether the upstream example satisfies them. They are machine
  drafts that nobody reviewed; preconditions on functions that are now
  helpers are dropped (also listed).

NPBench's preset S is large for a circuit compiler (up to 64 million input
elements, `deep_learning/mlp`); `--npbench-preset` selects another preset.

Result (2026-10-06, entry functions): NPBench 54 in scope, 0 out; Pythran 111
in scope, 11 out (`imports`: `descent` (itertools), `extrema` (functools),
`ramsurf` (cmath); `non_numeric_data`: `build_system` (a module parameter),
`comp_unrolling` (its only example is a ragged list), `euler_challenge13`
(`str()`), `rc4`, `roman_decode`, `scrabble` (2 functions), `train_equalizer`
(string parameters)). Parameters an export does not list keep their default and
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
