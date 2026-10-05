> Coordinated final validation: **75/75 profile checks and 25/25 Halo2 proofs pass**, and all 25 task programs pass the actual study worker. The task review below preserves the baseline and repair history. See [the final audit report](../README.md) and [result manifest](../final-manifest.json) for the verified build and complete evidence.

# Distances adversarial review

This inventory contains 25 independent ordinary Python/NumPy implementations of squared pairwise Euclidean distance. It covers unequal numbers of rows, shape unpacking, singleton-axis broadcasting, Gram identities through matmul/dot/einsum/tensordot/inner, coordinate and row iteration, comprehensions, helpers, repeated rows, ufunc reductions, output buffers, and while loops. Every candidate is retained unchanged, including rejected implementations. No task restriction or redesign is justified: these are mathematically valid implementations of the stated task.

## Baseline

Run from the project root:

```sh
.venv/bin/python -m studies.user_study.adversarial.check --task distances --directory studies/user_study/adversarial/distances --output studies/user_study/adversarial/distances/baseline.json
```

The baseline passed **57/75** complete profile checks (19/25 implementations pass on all three profiles). **All 1,425 NumPy oracle cases passed**, covering 19 fixtures on each profile for every candidate. The passing profile checks include compilation of the original and instrumented source, mock execution of all 19 fixtures, and rejection of an incorrect answer. There were 1,091 successful compiled fixture executions including partial executions of failing candidates. The machine-readable baseline includes source hashes and an environment/extension fingerprint. These results describe the pre-repair extension; no shared compiler files were changed by this reviewer.

## Rejected correct candidates

| Candidate | Failure across all 3 profiles | Diagnosis |
| --- | --- | --- |
| `global_helper.py` | Original compilation: `InvalidCircuitInputException: Circuit input must be annotated` at helper argument `b` | The provided entry-point name is not enough to keep an ordinary preceding module-level helper from being processed as circuit input. Support ordinary helper definitions and resolve the requested `kernel`. |
| `local_helper.py` | Original compilation: `InvalidCircuitStatementException: nested def distance cannot be defined inside a circuit body` | The local helper is a pure function with explicit arguments and captures nothing, despite the diagnostic claiming it captures outer variables. Local pure helpers are natural here. |
| `gram_inner.py` | Original compilation: `IR generation failed: np.inner is not implemented` | Rectangular pairwise row inner products are a direct use of `np.inner`. Lower to contraction of both last axes, equivalently `tensordot(..., axes=([-1],[-1]))` for these arrays. |
| `ndindex_pairs.py` | Original compilation: `IR generation failed: np.ndindex is not implemented` | The output dimensions are statically known. Enumerating their Cartesian product is a natural allocation-and-fill pattern. |
| `reshape_broadcast.py` | Instrumented compilation: `IRBuilder::ensure_ptr: cannot materialize None as a scalar IR pointer` | **The problem is `np.square`, not reshape or shape unpacking.** `np_square` in `src/ops/static_ndarray_ops/elementwise.rs` calls `elementwise_binary`, whose fallback sends a StaticArray to `apply_scalar_binary_op`. Original compilation silently accepts the invalid return; instrumented output checks expose it. |
| `output_buffer.py` | Both compilations succeed, execution rejects a correct answer | Both `np.square(delta, out=delta)` and `np.sum(delta, axis=2, out=result)` independently fail to update their output arrays. Zero/equal-input fixtures can pass despite the error; nontrivial fixtures expose it. |

## Isolation evidence

Diagnostic files start with `_isolate_` so they do not enter the 25-candidate inventory:

- `_isolate_square_to_mul.py`: changing only `np.square(delta)` to `delta * delta` makes the reshape implementation pass the tested original/checked compilation, execution, and negative control.
- `_isolate_reshape_to_broadcast.py`: replacing reshape with ordinary singleton-axis slicing leaves the same `None` failure.
- `_isolate_underscore_renamed.py`: renaming the unused unpacking target leaves the same failure.
- `_isolate_square_out_only.py`: normal sum plus `np.square(..., out=...)` fails compiled execution after the first two fixtures, while all 19 Python fixtures pass.
- `_isolate_sum_out_only.py`: ordinary multiplication plus `np.sum(..., out=...)` independently fails compiled execution after the first two fixtures, while all 19 Python fixtures pass.

The parent integration agent received the exact failures and diagnoses for automatic shared compiler repairs. Re-run the unmodified inventory after rebuilding; preserve `baseline.json` as the before-repair evidence and write later results to a separate file.
