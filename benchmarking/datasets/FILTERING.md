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
after the early exclusions below). Each program folder holds one `kernel.py`.

What was kept verbatim: every retained function's name, parameter names,
defaults and body; the helper functions and module constants the kernel needs;
the imports it needs. Runners, data generators, unused alternate kernels and
scaffolding were removed.

What was added (the 2026-05-19 version of this file wrongly said "no function
signatures were modified"): the `@zk_circuit` / `@zk_chip` decorators, Zinnia
type annotations on every parameter of the entry function, and, in 67 kernels,
`@requires` preconditions. The annotations and preconditions were machine drafts
(`contracts/*.json`, `"review_status": "draft"`). They are replaced by
annotations derived mechanically from upstream type information; see
`case_sets/acceptance.json` for the derivation per program.

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
