# CirC acceptance study — batch5

54 programs; **23 accepted after both fixture checks** (16 as written, 7 with minimal rewrites).

Every program has `result.json`, an unchanged `kernel_source.py` snapshot, and `ATTEMPTS.md`. Executable attempts have `.zok` files and compile logs. All accepted attempts include A/B `.pin`, interpreter logs, and decoded NumPy comparison JSON.

## Method

Used the supplied stock CirC and zxi binaries without rebuilding. All kernel compilations ran through `memrun.sh`, `MEMRUN_MAX=8G`, and `timeout 300`; TMPDIR was this batch’s `tmp/`. Inputs are private and return values public. The standard-library path is explicit in `study.py`. No Zinnia IR, circuits, sizes, or acceptance outcomes were used.

Reals are rounded to fixed-point integers at scale 2^32. Native u64 values use two’s-complement signed interpretation; signed comparisons bias the sign bit. Static shapes remain multidimensional; rewritten dynamic outputs carry a logical length or dimensions and padding. NumPy is executed from the original function bodies with decorators and annotations removed, using the supplied virtual environment. Both fixtures use atol=rtol=1e-6; integer and boolean results are also checked exactly. For in-place kernels, all relevant mutated buffers are public outputs and checked.

Wide Q32 multiplication needs an intermediate wider than u64 followed by rescaling. Six full native-`integer` kernel translations were compiled and rejected by the R1CS backend (`Unhandled variable operator return...`). An additional probe with a field output also rejects native integer inputs (`Unhandled variable operator a`). No existing fixed-point library was found in the supplied stdlib inventory. Those kernels are marked `needs_gadget: division` because repairing this requires the prohibited custom wide fixed-point arithmetic/rescaling, rather than silently overflowing products or replacing integer quotients with field inverses. Native integer division by a bin count, which does not require a widened Q32-by-Q32 quotient, is used successfully for case0414.

For structurally unexpressible programs and missing gadgets, no placeholder or weakened kernel was compiled: the source construct, line, and rewrite analysis are recorded. Thus absence of a compile log for these cases is intentional. The patch-reshape rewrite was attempted and terminated under the 8 GB cap (exit 137, after IR optimization began); no constraint count or fixture correctness is claimed for it.

## Counts

| Category | Programs |
|---|---:|
| `accepted_as_written` | 16 |
| `accepted_with_rewrite` | 7 |
| `needs_gadget: division` | 8 |
| `needs_gadget: exp` | 1 |
| `needs_gadget: matrix inverse` | 1 |
| `needs_gadget: power` | 1 |
| `needs_gadget: sin` | 1 |
| `needs_gadget: sort` | 3 |
| `needs_gadget: sqrt` | 5 |
| `not_expressible: dynamic loop` | 4 |
| `not_expressible: dynamic shape` | 6 |
| `timeout_or_oom` | 1 |

## Programs

| Program | Category | Constraints | Reason |
|---|---|---:|---|
| [case0299](case0299/result.json) | `not_expressible: dynamic shape` | — | The 200 input integers have no declared value-range bound; max(a)-min(a)+1 is not bounded by their count. |
| [case0309](case0309/result.json) | `accepted_as_written` | 9304 | Row-major argmax with first-tie selection. |
| [case0317](case0317/result.json) | `accepted_as_written` | 65000 | Copy the fixed 10x20x5 array. |
| [case0323](case0323/result.json) | `needs_gadget: sin` | — | No native/stdlib Q32 sine; arbitrary signed degrees cannot be substituted by fixture constants. |
| [case0329](case0329/result.json) | `needs_gadget: power` | — | The exponent is a private, signed, unconstrained integer; negative exponents require real reciprocals. Native field exponentiation does not implement fixed-point real powers. |
| [case0336](case0336/result.json) | `accepted_as_written` | 520 | Static flip and diagonal gather. |
| [case0342](case0342/result.json) | `accepted_as_written` | 1950 | Exactly 30 statically determined list appends. |
| [case0349](case0349/result.json) | `accepted_as_written` | 1 | The signature fixes the second dimension at six. |
| [case0358](case0358/result.json) | `needs_gadget: sort` | — | Sorting is required to preserve the data-dependent reordering of b; the brief prohibits writing a sorting gadget. |
| [case0364](case0364/result.json) | `accepted_as_written` | 339581 | Native dynamic gather for scalar insertion; result is 101x10. |
| [case0373](case0373/result.json) | `accepted_with_rewrite` | 50 | Remove unused sorting; ECDF ordinate output depends only on length 50. |
| [case0384](case0384/result.json) | `needs_gadget: matrix inverse` | — | Signed private n includes negative powers, requiring matrix inversion and real division; no native fixed-point matrix-power/inverse routine exists. Repeated powers also have no declared exponent bound. |
| [case0390](case0390/result.json) | `timeout_or_oom` | — | 20,000-element padded patch scatter exceeded the 8 GB cap. |
| [case0400](case0400/result.json) | `accepted_with_rewrite` | 1 | CirC rejects empty arrays; use a padding slot with known zero logical length. |
| [case0407](case0407/result.json) | `not_expressible: dynamic shape` | — | The 100 map entries have no declared upper bound; bincount length is max(adjusted accmap)+1, not 100. |
| [case0414](case0414/result.json) | `accepted_with_rewrite` | 13629 | Eight-bin capacity; guarded sums and native integer division by bin size. |
| [case0420](case0420/result.json) | `needs_gadget: division` | — | Full Q32 products require wider intermediates than native u64. The faithful native arbitrary-integer candidate fails R1CS lowering; no custom wide multiplication/rescaling/division gadget was introduced. See probes/native_integer_field_output.compile.log and probes/stdlib_inventory.json. |
| [case0429](case0429/result.json) | `needs_gadget: sqrt` | — | Standard deviation needs a square root of the variance; no built-in fixed-point square-root route exists. |
| [case0436](case0436/result.json) | `accepted_as_written` | 7980 | Row maxima and elementwise equality mask. |
| [case0442](case0442/result.json) | `accepted_with_rewrite` | 8000 | Stable masked compaction into capacity 20, with logical length. |
| [case0451](case0451/result.json) | `accepted_as_written` | 4000 | Fixed all-zero tensor. |
| [case0458](case0458/result.json) | `needs_gadget: sqrt` | — | Euclidean norms require a square root; replacing them by squared distances changes the output. |
| [case0472](case0472/result.json) | `accepted_as_written` | 98280 | Static slice retains the leading dimension of length one. |
| [case0483](case0483/result.json) | `not_expressible: dynamic shape` | — | degree is unconstrained; even with 100 samples, NumPy can return degree+1 coefficients for higher-degree underdetermined fits. Padding has no domain-implied capacity. |
| [case0496](case0496/result.json) | `not_expressible: dynamic shape` | — | The two requested shape values have no declared upper bounds; input shape 42x10 supplies only lower bounds for valid padding. |
| [case0502](case0502/result.json) | `accepted_as_written` | 78923 | Native dynamic gather with normalized negative indices. |
| [case0509](case0509/result.json) | `accepted_with_rewrite` | 106832 | Bounded min/max and padded 10x10 crop with logical dimensions. |
| [npbench_compute_compute](npbench_compute_compute/result.json) | `accepted_as_written` | 151747 | Signed clipping, integer multiplication and addition. |
| [npbench_deep_learning_resnet_resnet](npbench_deep_learning_resnet_resnet/result.json) | `needs_gadget: sqrt` | — | Batch normalization needs standard deviation, another sqrt(std+eps), and real division; cannot omit normalization or supply its values as witnesses. |
| [npbench_polybench_adi_adi](npbench_polybench_adi_adi/result.json) | `not_expressible: dynamic loop` | — | TSTEPS != 0 gives no upper bound. The N <= 1024 precondition does not bound the time loop. Fixed-point division is an additional blocker. |
| [npbench_polybench_covariance2_covariance2](npbench_polybench_covariance2_covariance2/result.json) | `needs_gadget: division` | — | Full Q32 products require wider intermediates than native u64. The faithful native arbitrary-integer candidate fails R1CS lowering; no custom wide multiplication/rescaling/division gadget was introduced. See probes/native_integer_field_output.compile.log and probes/stdlib_inventory.json. |
| [npbench_polybench_floyd_warshall_floyd_warshall](npbench_polybench_floyd_warshall_floyd_warshall/result.json) | `accepted_as_written` | 811008 | Static 16-step loop; snapshots preserve NumPy simultaneous updates. |
| [npbench_polybench_jacobi_1d_jacobi_1d](npbench_polybench_jacobi_1d_jacobi_1d/result.json) | `not_expressible: dynamic loop` | — | TSTEPS is an unconstrained integer and is independent of the 16-element buffer size. |
| [npbench_polybench_mvt_mvt](npbench_polybench_mvt_mvt/result.json) | `needs_gadget: division` | — | Full Q32 products require wider intermediates than native u64. The faithful native arbitrary-integer candidate fails R1CS lowering; no custom wide multiplication/rescaling/division gadget was introduced. See probes/native_integer_field_output.compile.log and probes/stdlib_inventory.json. |
| [npbench_polybench_trisolv_trisolv](npbench_polybench_trisolv_trisolv/result.json) | `needs_gadget: division` | — | Triangular solve divides each computed real numerator by private L[i,i]; native field division is modular inverse, not Q32 real division. |
| [npbench_weather_stencils_hdiff_hdiff](npbench_weather_stencils_hdiff_hdiff/result.json) | `needs_gadget: division` | — | Full Q32 products require wider intermediates than native u64. The faithful native arbitrary-integer candidate fails R1CS lowering; no custom wide multiplication/rescaling/division gadget was introduced. See probes/native_integer_field_output.compile.log and probes/stdlib_inventory.json. |
| [pythran_another_quicksort](pythran_another_quicksort/result.json) | `needs_gadget: sort` | — | The first attempt has value-dependent partition loops, variable-length recursive subarrays and recursion (lines 23-56). The input capacity is ten, but replacing recursion by a bespoke bounded sorting implementation would be a forbidden sort gadget. No such replacement was made. |
| [pythran_calculate_u](pythran_calculate_u/result.json) | `not_expressible: dynamic loop` | — | dt != 0 gives neither a positive-step guarantee nor a bound on (t_stop-t)/dt; do not infer a cap from fixture values. Fixed-point division is an additional blocker. |
| [pythran_clip](pythran_clip/result.json) | `needs_gadget: sqrt` | — | Complex absolute value needs sqrt(real^2+imag^2), and limit1 divides by that magnitude; complex numbers could otherwise be represented by real/imaginary pairs. |
| [pythran_create_grid](pythran_create_grid/result.json) | `accepted_as_written` | 1792 | Broadcast and reshape at fixed input size. |
| [pythran_diffusion](pythran_diffusion/result.json) | `not_expressible: dynamic loop` | — | iterNum has no upper-bound precondition, independently of the fixed 16x16 grid. |
| [pythran_fbcorr](pythran_fbcorr/result.json) | `needs_gadget: division` | — | Full Q32 products require wider intermediates than native u64. The faithful native arbitrary-integer candidate fails R1CS lowering; no custom wide multiplication/rescaling/division gadget was introduced. See probes/native_integer_field_output.compile.log and probes/stdlib_inventory.json. |
| [pythran_goodexpomeasure](pythran_goodexpomeasure/result.json) | `needs_gadget: exp` | — | Needs fixed-point exp and division by private sigma; cannot precompute from fixtures. |
| [pythran_histogram](pythran_histogram/result.json) | `not_expressible: dynamic shape` | — | bin_width has no declared upper bound; 16 samples do not limit the number of bins. Real division is also needed. |
| [pythran_lapl2d](pythran_lapl2d/result.json) | `accepted_with_rewrite` | 1967264 | Guarded 1,024-step loop from the explicit precondition; retain both buffers. |
| [pythran_lttb_pt](pythran_lttb_pt/result.json) | `needs_gadget: division` | — | Valid n_out lies in [3,16], giving capacity 16 and at most 14 bins. Native integer division can compute bin means, but triangle areas multiply Q32 differences; full-width products and their Q32 rescaling need a custom wide fixed-point arithmetic gadget after native arbitrary-integer lowering fails. Guarded padding alone cannot supply that arithmetic. |
| [pythran_matmul](pythran_matmul/result.json) | `needs_gadget: division` | — | Full Q32 products require wider intermediates than native u64. The faithful native arbitrary-integer candidate fails R1CS lowering; no custom wide multiplication/rescaling/division gadget was introduced. See probes/native_integer_field_output.compile.log and probes/stdlib_inventory.json. |
| [pythran_multitype](pythran_multitype/result.json) | `accepted_as_written` | 134 | Private boolean branch; preserve both tuple outputs in a struct. |
| [pythran_periodic_bc](pythran_periodic_bc/result.json) | `accepted_as_written` | 1179 | Static boundary slice assignment. |
| [pythran_projection_simplex](pythran_projection_simplex/result.json) | `needs_gadget: sort` | — | Sort is required; capacity ten bounds conditional selections but does not eliminate sorting and real division. |
| [pythran_reverse_cumsum](pythran_reverse_cumsum/result.json) | `accepted_as_written` | 1253 | Reverse, cumulative sum, then reverse. |
| [pythran_sobelfilter](pythran_sobelfilter/result.json) | `accepted_with_rewrite` | 720927 | Bounds follow from valid 16-element accesses; collapse repeated zero-column writes. |
| [pythran_sumarray3d](pythran_sumarray3d/result.json) | `needs_gadget: sqrt` | — | Both returned arrays must be retained; the first requires inverse square roots. Returning only sum_array3d would weaken the program. |
| [pythran_wave_simulation](pythran_wave_simulation/result.json) | `not_expressible: dynamic shape` | — | PARTICLE_COUNT has no declared upper bound. The fixed two-coordinate return does not bound intermediate particle arrays. sqrt/division are additional blockers. |

## Reproduction and verification limits

Run `/home/zhantong/Projects/zinnia-project/wt/literal/.venv/bin/python study.py <program>` from this directory to regenerate and run an executable candidate (omit the program to run all candidates). Run the same Python on `blocked.py` to regenerate structural/gadget analyses, and on `finalize.py` to audit files and regenerate this report. `run_more.py` is a convenience runner. All compiler commands are also recorded in the per-attempt JSON.

All 23 finally accepted attempts passed A and B; these are fixture correctness checks, not exhaustive proofs over every numerical input. The numerical representation and rounding convention above apply throughout. No correctness is claimed for rejected, structurally unexpressible, gadget-blocked, or resource-limited attempts. Padding is decoded only according to the documented logical shape. No fixture value was used as a loop bound, array capacity, exponent, or private-data replacement.

BATCH_DONE
