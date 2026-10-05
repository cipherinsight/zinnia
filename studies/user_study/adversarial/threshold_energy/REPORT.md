> Coordinated final validation: **75/75 profile checks and 25/25 Halo2 proofs pass**, and all 25 task programs pass the actual study worker. The task review below preserves the baseline and repair history. See [the final audit report](../README.md) and [result manifest](../final-manifest.json) for the verified build and complete evidence.

# THRESHOLD_ENERGY adversarial audit

Twenty-five independently authored ordinary Python/NumPy implementations are preserved without algorithm rewrites. Only input type/shape annotations were injected by the shared validator. Local helpers are deliberately retained: the task naturally permits them, and a compiler rejection does not make that programming approach invalid.

Task: for each row, sum `v*v` for channels satisfying the **strict** predicate `v > floor[c]`; return a vector with one entry per row. Empty selection contributes zero. Tested profiles: `(N,C)=(3,4),(1,1),(16,4)`. Each profile uses 7 targeted fixtures and 12 seeded random fixtures. Inputs include negative readings, threshold equality, all-active/all-inactive selections, and domain endpoints.

## Initial baseline results

- All 25 programs agree with the independent NumPy/scalar oracle on all 57 inputs: **1,425 Python checks**.
- **12/25 programs**, equivalently **36/75 program/profile checks**, pass original compilation, instrumented compilation, all 19 compiled fixture checks, and a wrong-answer negative control per profile.
- Those successful combinations account for 684 compiled positive checks and 36 negative controls. Another 20 positive checks succeeded before semantic failures, for **704 compiled positive checks total**.
- Ten programs are rejected at original compilation. Three further programs compile but fail instrumented output shape/value validation. These are retained as compiler gaps, not exclusions or invalid solutions.
- `initial_compile.json` records the original compile-only smoke results; `baseline.json` records full validation, errors, timing, source hashes, and environment fingerprint. At the initial baseline stage no compiler repair had yet been made. Subsequent repairs and validation are described below.

| Program | Natural approach | Initial full result |
|---|---|---|
| `mask_product.py` | broadcast mask / multiply / sum | Pass: 3/3 profiles |
| `sum_where.py` | NumPy reduction with where and initial | Fail: execute |
| `where_square.py` | zero inactive readings before squaring | Pass: 3/3 profiles |
| `add_reduce.py` | ufunc reduction | Pass: 3/3 profiles |
| `einsum_masked.py` | einsum of masked readings | Pass: 3/3 profiles |
| `row_filtered_dot.py` | Boolean selection followed by dot | Fail: compile_original |
| `row_filtered_sum.py` | Boolean selection followed by sum | Pass: 3/3 profiles |
| `generator_filter.py` | generator with filter clause | Pass: 3/3 profiles |
| `helper_closure.py` | local helper captures floor | Fail: compile_original |
| `helper_explicit.py` | local helper with explicit argument | Fail: compile_original |
| `map_lambda.py` | map lambda over rows | Fail: compile_original |
| `filter_lambda.py` | filter pairs then map contribution | Fail: compile_original |
| `apply_along_axis.py` | apply row helper along channel axis | Fail: compile_original |
| `compress_rows.py` | compress selects channel values | Fail: compile_original |
| `nonzero_indices.py` | nonzero channel index iteration | Fail: execute |
| `flatnonzero_take.py` | flatnonzero and take gather | Fail: compile_original |
| `masked_copyto.py` | copyto inactive positions | Fail: compile_original |
| `boolean_assignment.py` | Boolean assignment on copy | Pass: 3/3 profiles |
| `channel_accumulation.py` | column iteration accumulates output vector | Pass: 3/3 profiles |
| `continue_loop.py` | nested loops with continue | Pass: 3/3 profiles |
| `while_indices.py` | static while loops with conditional scalar | Pass: 3/3 profiles |
| `list_filtered_append.py` | dynamic filtered list append and reduction | Pass: 3/3 profiles |
| `transpose_builtin_sum.py` | transpose channels and built-in sum | Fail: compile_checked |
| `masked_matmul.py` | masked rows times all-ones vector | Pass: 3/3 profiles |
| `reduce_keepdims_squeeze.py` | sum tuple axis and squeeze | Fail: compile_original |

## Concrete compiler gaps and repair direction

1. **Ignored reduction mask** — `sum_where.py`: `np.sum(x ** 2, axis=1, where=x > floor, initial=0)` compiles but includes inactive terms. Both the native fast path at `src/ir_gen/named_attr/mod.rs` (near line 85) and reduction dispatch in `numpy_funcs.rs` (near line 900) read axis/keepdims but omit `where`/`initial`. Respect mask broadcasting and the operation identity, then fold initial exactly once. Validate nonzero initial in compiler tests too.
2. **Bounded iteration executes padding** — `nonzero_indices.py`: dynamic loop handling at `src/ir_gen/visitors.rs` (near line 950) explicitly iterates the static maximum without gating the body by runtime length. Padded channel index zero contributes `row[0]**2`, including when the true selection is empty. Fixture 4 fails on every shape after the first four fixtures passed. Gate the complete body, including scalar mutations, array writes, and appends, by `k < runtime_length`; padding cannot be assigned to the caller as a semantic responsibility.
3. **Array start loses array addition semantics** — `transpose_builtin_sum.py`: original compilation succeeds but instrumented output length becomes `2*N`. `named_attr` materializes ndarray arguments as legacy lists; built-in sum at `src/ir_gen/named_attr/builtins.rs` (near line 161) then uses generic add on `start` and result (near line 202), concatenating lists. Preserve ndarray identity and perform elementwise addition for an array start.
4. **Local helpers** — both pure explicit arguments and captured `floor` helpers are rejected. `ZKCircuit.from_source` bypasses the existing `autolift_nested_defs` used by the decorator path; its error wrongly asserts that the explicit helper captures outer variables. A general callable lowering or safe compiler-owned inlining should preserve lexical scope and evaluate arguments once. Captured values must reflect Python call-time closure semantics, not silently freeze at definition time.
5. **Functional forms** — lambda expressions in map/filter are rejected before callable handling. `apply_along_axis` also requires a callable row helper. Lower named/lambda callables and their capture environment rather than requiring participants to rewrite sensible functional code.
6. **Dynamic vector dot** — `row_filtered_dot.py` fails explicitly at `numpy_funcs.rs` near line 977. For rank-one operands, enforce runtime length agreement and lower multiplication plus dynamic sum. Existing `value_ops::apply_binary_op` and `array_ops::reduce` provide the required operations. The empty-selection result is zero.
7. **Dynamic take indices** — `flatnonzero_take.py`: `static_array/sorting.rs::flatten_static_indices` only accepts constant integer indices. Gather using bounded dynamic index storage while preserving its runtime length and gating bounds constraints for inactive padding. Merely gathering the full capacity produces wrong energies when the first reading is nonzero.
8. **Selection APIs** — `np.compress` and `np.copyto` are not implemented. Compress should preserve bounded selected length; copyto should honor mask broadcasting and mutate the provided destination storage (including aliases/views as applicable).
9. **Shape method** — `ndarray.squeeze(axis=1)` is rejected, despite the axis being publicly known to have length one. Add both method/function dispatch using existing shape metadata and validate requested axes have size one.

## Interpretation

The task directly invites filtering and reduction. Its main mathematical behavior is well covered by 25 distinct forms, but the initial compiler cannot honestly claim compatibility with the full sensible surface explored here. The dynamic iteration and ignored-mask cases are correctness failures, not merely absent conveniences. Shared compiler repairs should re-run this unchanged inventory and retain this baseline.

This is empirical adversarial coverage, not a proof that every conceivable sensible program is accepted. At the initial baseline stage only the mock backend and wrong-answer negative controls were used; subsequent proof validation is described below.

Reproduce:

```sh
.venv/bin/python -m studies.user_study.adversarial.check --task threshold_energy --directory studies/user_study/adversarial/threshold_energy --output studies/user_study/adversarial/threshold_energy/after.json
```

## Post-repair validation

All **25 unchanged candidates now pass all 3 profiles**: **75/75 complete checks**, comprising **1,425 compiled positive cases** and **75 wrong-answer negative controls**. Each candidate/profile source SHA-256 matches its baseline entry. The complete results and environment fingerprint are in `../results/threshold_energy-after.json`.

The coordinating agent integrated shared callable lowering, reduction masks, runtime-aware iteration, ndarray start sums, and NumPy method/function handling. This reviewer implemented `src/ir_gen/named_attr/dynamic_array_compat.rs`: rank-one bounded dot/inner with equal logical lengths, compress with runtime condition lengths, and take preserving index runtime length and normalizing negative indices. Gather bounds constrain only active indices, and a one-hot selection network avoids invalid inactive memory reads. Native static-array views are read through their payload API before entering the dynamic representation.

The additional `_dynamic_api_probes.py` suite passes **12/12 probes**: 155 positive cases across 10 variants, 10 wrong-answer controls, and explicit rejection of out-of-bounds indices and mismatched dot lengths. It covers static/dynamic sources, empty selections, negative indices, shorter compress masks, and unequal public capacities with equal logical lengths. Results are preserved in `dynamic_api_probes.json`; the initial failure snapshot is `dynamic_api_probes_initial.json`.

The new module intentionally implements rank-one bounded operations; it does not establish arbitrary-rank support or complete NumPy API compatibility. The original 25-program gate is unchanged, and the successful results remain empirical coverage rather than a universality proof.

Real **Halo2 proof generation and verification pass for all 25 candidates** on profile 0, fixture 6. Each also rejects the wrong-answer mock negative control. This is one representative proof input per candidate, not an exhaustive proof-input sweep. Results: `../results/threshold_energy-halo2.json`.
