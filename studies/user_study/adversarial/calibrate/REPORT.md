> Coordinated final validation: **72/72 profile checks and 24/24 Halo2 proofs pass**, and all 24 task programs pass the actual study worker. The task review below preserves the baseline and repair history. See [the final audit report](../README.md) and [result manifest](../final-manifest.json) for the verified build and complete evidence.

# CALIBRATE adversarial review

The baseline accepts 12 of 24 independently designed plausible implementations and rejects 12. All 24 are correct under ordinary Python/NumPy on all tested inputs. Three rejected forms compile successfully but compute results inconsistent with the oracle; these are semantic compiler defects, not syntax limitations.

Scope: one top-level `kernel(x)` with supplied `np`, local helpers permitted, positive-length integer vector, each element in [-512,4096], output `clip(2*x+5,0,255)` with the original shape. No task redesign is recommended: clamping naturally invites each of the alternatives below. Restricting participants to the supported subset would conceal real gaps.

## Reproduction and coverage

Run from the repository root:

```
.venv/bin/python -m studies.user_study.adversarial.calibrate._probe
.venv/bin/python -m studies.user_study.adversarial.check --task calibrate --directory studies/user_study/adversarial/calibrate --output studies/user_study/adversarial/calibrate/shared_baseline.json
```

`baseline.json` records the original 72 subprocess results and environment fingerprint; do not overwrite it during post-repair validation. The private worker initially freezes all 24 candidate files before observing compiler feedback. It injects only shape annotations using AST, checks Python against the independent oracle, compiles the original and instrumented functions, then checks assertions with mock proofs and corrupts the expected output for a negative control. Nested helper returns are deliberately left intact by the instrumentation.

Profiles are N=6, N=1, N=32. Each profile uses the shared deterministic fixtures plus 12 seeded random vectors and eight extra constant vectors covering -512, -3, -2, 0, 124, 125, 126, 4096; there are 27 Python comparisons per profile. Every accepted implementation passes all 27 compiled comparisons and one negative control per profile. Every rejected implementation passes all 27 Python comparisons per profile. Each candidate/profile runs in a fresh subprocess with a 60-second timeout. No subprocess timed out. The shared harness independently checks its 19 cases per profile with a 45-second timeout. This is empirical coverage, not proof that every conceivable implementation works; this baseline uses mock proofs, not Halo2 proofs.

## Rejected implementations and repairs

| Candidate | Baseline failure | Diagnosis / sensible repair |
| --- | --- | --- |
| `ufunc_out.py` | Compiles; correct results rejected after first fixture | `np.maximum(y,0,out=y)` and `np.minimum(y,255,out=y)` silently discard `out`. `define_np_minmax` in `src/ops/np_like/mod.rs` exposes only x1/x2. Implement common ufunc `out` mutation with alias preservation at the NumPy dispatch layer. |
| `continue_loop.py` | Compiles; correct results rejected after first fixture | Witness-conditioned `continue` updates scope liveness, but subsequent StaticArray writes are not guarded by that liveness. `visit_assign` in `src/ir_gen/visitors.rs` does not update the builder write predicate; `rebind_subscript_target` assumes same-segment writes were already predicated. `visit_cond` installs only the branching condition, not the full current loop/return condition. Guard mutations with the complete active condition, recalculated after control transfers. |
| `abs_identity.py` | Compiles; rejects first fixture | Two interacting defects: integer `np.abs` promotes to float via `define_np_unary_math!(...,ir_abs_f)`, unlike NumPy; then `src/helpers/value_ops.rs` float dispatch maps both `mod` and `floor_div` to `ir_mod_i(lhs,rhs)`. The mathematically valid clamp identity `(255+abs(y)-abs(y-255))//2` thus returns zero for positive interior values. Preserve integer abs through the existing `np_abs_value` helper, and repair floor division separately. Mock `floor_div_f` itself currently delegates true division and also deserves a proper floor implementation. |
| `local_helper.py` | Nested `def clamp` rejected as capturing variables | This helper captures nothing. `ZKCircuit.from_source` bypasses `autolift_nested_defs`, which is used by the decorator path. Apply the same helper lowering on the source entry point; the current capture diagnostic is false here. |
| `map_lambda.py` | Lambda unsupported | Support finite statically enumerable map with scalar lambda callback, including ordinary builtin min/max inside callback. A source-level lambda-to-helper lowering could reuse chip calls. |
| `vectorize.py` | Lambda unsupported before API dispatch | Implement scalar callback vectorization for finite shaped inputs and infer the result dtype; accepting lambda alone does not establish vectorize support. |
| `piecewise.py` | Lambda unsupported before API dispatch | Piecewise is a direct natural clamp representation. Handle scalar alternatives and default callable; do not claim API support merely after lambda parsing is repaired. |
| `fromiter.py` | `np.fromiter is not implemented` | Materialize the fixed-count generator through existing comprehension/array lowering, honoring count and dtype. |
| `select.py` | `np.select is not implemented` | Lower first-true condition selection by reverse folding `np.where` over broadcast alternatives and default. |
| `copyto.py` | `np.copyto is not implemented` | Broadcast source/mask and update destination under `where`, retaining aliases. |
| `putmask.py` | `np.putmask is not implemented` | Add masked in-place update. General NumPy putmask uses flattened-position cycling of values; scalar values in this candidate are the straightforward case. |
| `median_stack.py` | `np.median is not implemented` | Median of zero, y, and 255 is exactly clipping. Implement static-axis order-statistic reduction with NumPy's even-length averaging and dtype behavior if general support is claimed; this candidate uses three rows and casts to integer. |

`abs_diagnostic.json` records additional scalar assertions. `np.abs([-10])` itself accepts 10, while the composed identity at y=5 and y=255 accepts zero and rejects the correct output. This narrows the immediate wrong-value failure to the floor-div dispatch, while integer promotion is independently an incorrect dtype choice.

## Accepted baseline implementations

`clip`, `minmax`, `where`, `mask_update`, `indexed_if`, `while`, `comprehension`, `append`, `clip_method`, `fmin_fmax`, `boolean_arithmetic`, and `enumerate_ternary` pass all three profiles including positive and negative checks. Candidates are retained as standalone source files, including every failure. `inventory.json` has per-candidate hashes and baseline outcomes. Compiler repairs and final revalidation are the parent agent's responsibility; no shared compiler files were edited by this reviewer.
