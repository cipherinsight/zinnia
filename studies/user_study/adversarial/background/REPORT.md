> Coordinated final validation: **72/72 profile checks and 24/24 Halo2 proofs pass**, and all 24 task programs pass the actual study worker. The task review below preserves the baseline and repair history. See [the final audit report](../README.md) and [result manifest](../final-manifest.json) for the verified build and complete evidence.

# BACKGROUND adversarial review

Designed independently by the task's Astra reviewer. Retained 24 standalone submissions (one `kernel`, `np` provided), covering ordinary NumPy and Python implementations without restricting candidates to supported compiler forms. `inventory.json` indexes every candidate; `baseline-results.json` preserves baseline evidence. `design.py` records generation text and `probe.py` is the executable baseline probe; neither is a submission.

## Contract and validation

Input is integer x[T,C], T,C positive, values [-512,4096]. Output x minus each column minimum, preserving shape. No task redesign is warranted: this is a natural bounded array computation, and local helpers should remain allowed.

Only public shape annotations were injected via pilots.validate.specialize. Original programs compiled before assertion instrumentation via pilots.validate.instrument (which preserves helper returns). Python reference execution used validate.python_function, existing tasks.py profiles/signatures/oracle and all 19 fixtures per profile (7 fixtures + 12 seeded random cases). Profiles are (3,2), (1,1), (16,4). Every successful profile checked all 19 outputs using the mock backend and rejected an independently perturbed expected output. This is finite empirical coverage, not a proof that arbitrary Python/NumPy is supported.

All 24 candidates passed Python oracle checks on all three profiles: 1,368 reference executions. Baseline compiler/mock validation passed 53/72 candidate-profile checks. There are 19 failures across seven candidates. Failures in `execute` are silent compiler misbehavior: the original and instrumented code compiled but a correct expected answer was rejected. Singleton shapes mask zip-star and subtract-out bugs.

## Failures and recommended repairs

- `ufunc_reduce` and `ufunc_accumulate`: `np.minimum.reduce(x, axis=0)` and `np.minimum.accumulate(x, axis=0)[-1]` are legitimate reduction idioms. Transformer currently rewrites only add/multiply reduce (zinnia/compile/transformer/base.py visit_Call). Extend recognized ufunc methods preserving their axis=0 defaults, initial/where/dtype/out semantics where applicable. A minimum accumulate needs prefix reduction along the specified axis.
- `nested_helper` and `helper_closure`: support local pure helper lifting and closure conversion for captured arrays/static dimensions. Nonclosure helper using builtin min is also rejected by current source compilation path; the diagnostic incorrectly presumes capture. Public single-kernel interface makes advice to add a module-level chip unusable.
- `zip_columns`: src/ir_gen/named_attr/mod.rs visit_named_attr expands starred expressions only when already List/Tuple; StaticArray x is pushed as ONE argument, then later materialized. Thus zip(*x) is compiled as zip(x). Expand the first-axis iterable before argument collection, including StaticArray. Apply consistent handling to visit_expr_attr and linalg call path; audit list/tuple starred lowering. builtin zip itself transposes its provided arguments correctly.
- `subtract_out`: np.subtract(..., out=out) must mutate the destination array storage, with aliases seeing the update, and return that destination. Ignoring out silently returns the zero-filled empty_like allocation.
- `min_out`: np.min(..., out=low) must write the reduction to low. The static reduction fast path in named_attr/mod.rs returns early with only axis/keepdims handled, bypassing out. Thread all supported reduction options through fast paths, and apply writes before returning. Also audit initial/where because domain-safe initial/where candidates passing does not show these keywords are respected generally.

## Exact baseline results

| Candidate | Profile | Stage | Checked outputs before failure | Error |
|---|---:|---|---:|---|
| ufunc_reduce | 0 | compile_original | 0 | ZinniaException: IR generation failed: Missing required argument: x1 |
| ufunc_reduce | 1 | compile_original | 0 | ZinniaException: IR generation failed: Missing required argument: x1 |
| ufunc_reduce | 2 | compile_original | 0 | ZinniaException: IR generation failed: Missing required argument: x1 |
| ufunc_accumulate | 0 | compile_original | 0 | ZinniaException: IR generation failed: Missing required argument: x1 |
| ufunc_accumulate | 1 | compile_original | 0 | ZinniaException: IR generation failed: Missing required argument: x1 |
| ufunc_accumulate | 2 | compile_original | 0 | ZinniaException: IR generation failed: Missing required argument: x1 |
| zip_columns | 0 | execute | 2 | AssertionError: correct answer rejected |
| zip_columns | 2 | execute | 2 | AssertionError: correct answer rejected |
| nested_helper | 0 | compile_original | 0 | ZinniaException:   In method "kernel", line 3 to 4<br>        def correct(column):<br>        ^^^^^^^^^^^^^^^^^^^^<br>            return column - min(column)<br>    <br>InvalidCircuitStatementException: nested `def correct` cannot be defined inside a circuit body. It captures variables from the outer function. Move it to module level and pass any captured values as arguments, decorating it with @zk_chip. |
| nested_helper | 1 | compile_original | 0 | ZinniaException:   In method "kernel", line 3 to 4<br>        def correct(column):<br>        ^^^^^^^^^^^^^^^^^^^^<br>            return column - min(column)<br>    <br>InvalidCircuitStatementException: nested `def correct` cannot be defined inside a circuit body. It captures variables from the outer function. Move it to module level and pass any captured values as arguments, decorating it with @zk_chip. |
| nested_helper | 2 | compile_original | 0 | ZinniaException:   In method "kernel", line 3 to 4<br>        def correct(column):<br>        ^^^^^^^^^^^^^^^^^^^^<br>            return column - min(column)<br>    <br>InvalidCircuitStatementException: nested `def correct` cannot be defined inside a circuit body. It captures variables from the outer function. Move it to module level and pass any captured values as arguments, decorating it with @zk_chip. |
| helper_closure | 0 | compile_original | 0 | ZinniaException:   In method "kernel", line 3 to 4<br>        def value(t, c):<br>        ^^^^^^^^^^^^^^^^<br>            return x[t, c] - min(x[:, c])<br>    <br>InvalidCircuitStatementException: nested `def value` cannot be defined inside a circuit body. It captures variables from the outer function. Move it to module level and pass any captured values as arguments, decorating it with @zk_chip. |
| helper_closure | 1 | compile_original | 0 | ZinniaException:   In method "kernel", line 3 to 4<br>        def value(t, c):<br>        ^^^^^^^^^^^^^^^^<br>            return x[t, c] - min(x[:, c])<br>    <br>InvalidCircuitStatementException: nested `def value` cannot be defined inside a circuit body. It captures variables from the outer function. Move it to module level and pass any captured values as arguments, decorating it with @zk_chip. |
| helper_closure | 2 | compile_original | 0 | ZinniaException:   In method "kernel", line 3 to 4<br>        def value(t, c):<br>        ^^^^^^^^^^^^^^^^<br>            return x[t, c] - min(x[:, c])<br>    <br>InvalidCircuitStatementException: nested `def value` cannot be defined inside a circuit body. It captures variables from the outer function. Move it to module level and pass any captured values as arguments, decorating it with @zk_chip. |
| subtract_out | 0 | execute | 2 | AssertionError: correct answer rejected |
| subtract_out | 2 | execute | 2 | AssertionError: correct answer rejected |
| min_out | 0 | execute | 1 | AssertionError: correct answer rejected |
| min_out | 1 | execute | 1 | AssertionError: correct answer rejected |
| min_out | 2 | execute | 1 | AssertionError: correct answer rejected |

## Passing forms

`axis_tuple`, `negative_axis`, `row_minimum`, `where_reduce`, `enumerated_columns`, `generator_min`, `list_append`, `column_stack`, `stack_axis`, `copy_inplace`, `min_initial`, `sorted_columns`, `argmin_take`, `while_loops`, `sentinel_loop`, `masked_min`, `list_conversion`.

All baseline results predate the centrally integrated repair. Any post-repair results must be recorded separately; these baseline files should not be overwritten.
