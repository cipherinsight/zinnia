# Batch 4 CirC acceptance results

54 programs: 17 accepted as written, 7 accepted with minimal rewrite, 30 not accepted. All 24 accepted programs pass fixtures A and B (48 comparisons).

Each executable translation was compiled with the supplied stock CirC binary using `--language zsharp <file> r1cs --action count`, `MEMRUN_MAX=8G` through the supplied `memrun.sh`, and `timeout 300`. `TMPDIR` is this batch's `tmp/`. Per-program `command.json`, compile logs, source snapshots, fixtures, `.pin` inputs, interpreter logs, comparison JSON, and `result.json` preserve the evidence. Binary hashes and numerical settings are in `provenance.json`.

Inputs are private and returned values are public. Reals use signed Q32 integers (scale 2^32), encoded in two's-complement u64; signed comparisons use sign-bit bias. Products use stock field multiplication and bit conversion to retain the middle 64 bits after scaling, with sign restored. This is the ordinary fixed-point representation, not a handwritten division, square-root, transcendental, or sorting circuit. Integer-count division in case0419 uses native u64 division without fixed-point rescaling. Complex values in case0413 are real/imaginary Q32 pairs. Standard Q32 range and quantization limitations apply.

Private scalar shape parameters remain private even when fixture notes call them constants. Dynamic-output rewrites publish logical length/dimensions along with padded contents; the correctness decoder restores the NumPy output shape. For functions returning None while mutating arrays (SYRK and FDTD), all mutated arrays are public outputs and are compared. Reference execution strips decorators and annotations and executes the original function body with NumPy; it does not call Zinnia or inspect its IR, circuits, sizes, or acceptance results.

## Counts

| Category | Programs |
|---|---:|
| `accepted_as_written` | 17 |
| `accepted_with_rewrite` | 7 |
| `circ_error` | 1 |
| `needs_gadget: division` | 5 |
| `needs_gadget: exp` | 2 |
| `needs_gadget: least_squares` | 1 |
| `needs_gadget: matrix_rank` | 1 |
| `needs_gadget: sort` | 3 |
| `needs_gadget: sqrt` | 6 |
| `needs_gadget: tanh` | 1 |
| `not_expressible: dynamic loop` | 3 |
| `not_expressible: dynamic shape` | 7 |

## Programs

| Program | Category | Constraints | Reason |
|---|---|---:|---|
| [case0298](case0298/result.json) | `needs_gadget: sort` | — | kernel.py:11 np.unique sorts values and returns inverse ranks; requires a sorting/unique gadget. |
| [case0304](case0304/result.json) | `accepted_with_rewrite` | 3432 | kernel.py:12-13 private-dependent slice and reshape. |
| [case0314](case0314/result.json) | `accepted_as_written` | 72041 | Direct static-shape translation. |
| [case0322](case0322/result.json) | `accepted_with_rewrite` | 762 | kernel.py:11 value-dependent argwhere length. |
| [case0328](case0328/result.json) | `not_expressible: dynamic shape` | — | kernel.py:12 np.pad output length is unbounded length; signature has no upper bound. |
| [case0335](case0335/result.json) | `accepted_as_written` | 19700 | Direct static-shape translation. |
| [case0341](case0341/result.json) | `accepted_as_written` | 1950 | kernel.py:11-13 list length is statically 30 (5x6), translated as a fixed array. |
| [case0348](case0348/result.json) | `needs_gadget: matrix_rank` | — | kernel.py:18 np.linalg.matrix_rank requires numerical rank/SVD machinery. |
| [case0357](case0357/result.json) | `needs_gadget: sort` | — | kernel.py:11 np.argsort requires sort indices; no stock Z# argsort. |
| [case0363](case0363/result.json) | `accepted_as_written` | 64535 | Direct static-shape translation. |
| [case0370](case0370/result.json) | `accepted_as_written` | 4358 | Direct static-shape translation. |
| [case0383](case0383/result.json) | `needs_gadget: sort` | — | kernel.py:11 np.argsort requires sort indices. |
| [case0389](case0389/result.json) | `accepted_with_rewrite` | 92543 | kernel.py:12 private-dependent reshape dimensions. |
| [case0399](case0399/result.json) | `accepted_as_written` | 15427 | Direct static-shape translation. |
| [case0406](case0406/result.json) | `not_expressible: dynamic shape` | — | kernel.py:12 output length max(index)+1 has no bound in declared input domain; unique also needs sorting. |
| [case0413](case0413/result.json) | `accepted_with_rewrite` | 2842 | kernel.py:11 boolean filter has value-dependent length. |
| [case0419](case0419/result.json) | `accepted_with_rewrite` | 58469 | kernel.py:12 reshape/bin count is private-dependent; capacity 8x6 from input shape. |
| [case0428](case0428/result.json) | `needs_gadget: sqrt` | — | kernel.py:11 np.std requires fixed-point square root (and mean division). |
| [case0435](case0435/result.json) | `accepted_as_written` | 592 | Direct static-shape translation. |
| [case0441](case0441/result.json) | `accepted_as_written` | 335 | Direct static-shape translation. |
| [case0450](case0450/result.json) | `accepted_as_written` | 74695 | Direct static-shape translation. |
| [case0457](case0457/result.json) | `needs_gadget: sqrt` | — | kernel.py:11 np.linalg.norm requires fixed-point square root. |
| [case0471](case0471/result.json) | `accepted_as_written` | 147420 | Direct static-shape translation. |
| [case0482](case0482/result.json) | `needs_gadget: least_squares` | — | kernel.py:11 np.polyfit requires numerical least-squares/SVD and fixed-point division. |
| [case0495](case0495/result.json) | `not_expressible: dynamic shape` | — | kernel.py:11 private shape controls unbounded padded output dimensions. |
| [case0501](case0501/result.json) | `accepted_as_written` | 78923 | Direct static-shape translation. |
| [case0508](case0508/result.json) | `accepted_with_rewrite` | 123734 | kernel.py:11 argwhere and line 13 crop have data-dependent shapes. |
| [npbench_channel_flow_channel_flow](npbench_channel_flow_channel_flow/result.json) | `not_expressible: dynamic loop` | — | kernel.py:36 convergence while udiff>0.001 has no domain-implied iteration bound (nit bounds only the inner solver). |
| [npbench_deep_learning_mlp_mlp](npbench_deep_learning_mlp_mlp/result.json) | `needs_gadget: exp` | — | kernel.py:15 softmax np.exp requires fixed-point exp; line 17 also requires division. |
| [npbench_nbody_nbody](npbench_nbody_nbody/result.json) | `not_expressible: dynamic shape` | — | kernel.py:61-62 energy arrays have length Nt+1 and line 65 loop uses unbounded Nt; bound on N does not bound Nt. |
| [npbench_polybench_correlation_correlation](npbench_polybench_correlation_correlation/result.json) | `needs_gadget: sqrt` | — | kernel.py:11,14 standard deviation and sqrt(float_n) require square root and division. |
| [npbench_polybench_fdtd_2d_fdtd_2d](npbench_polybench_fdtd_2d_fdtd_2d/result.json) | `accepted_with_rewrite` | 700060 | kernel.py:10 private-dependent time loop. |
| [npbench_polybench_heat_3d_heat_3d](npbench_polybench_heat_3d_heat_3d/result.json) | `not_expressible: dynamic loop` | — | kernel.py:10 range(1,TSTEPS) has no declared upper bound and no index-implied bound on TSTEPS. |
| [npbench_polybench_ludcmp_ludcmp](npbench_polybench_ludcmp_ludcmp/result.json) | `needs_gadget: division` | — | kernel.py:15,21 division by private matrix entries requires fixed-point division. |
| [npbench_polybench_syrk_syrk](npbench_polybench_syrk_syrk/result.json) | `accepted_as_written` | 337201 | Returns the mutated C array (Python function has an implicit None return). |
| [npbench_stockham_fft_stockham_fft](npbench_stockham_fft_stockham_fft/result.json) | `not_expressible: dynamic shape` | — | kernel.py:11-12 allocate R by R with unbounded R when K=0 (valid, skips all FFT iterations); complex exp also needs a gadget. |
| [pythran_allpairs_distances_loops](pythran_allpairs_distances_loops/result.json) | `not_expressible: dynamic shape` | — | kernel.py:19-20 ones((500,d)), ones((200,d)) use unbounded d; do not algebraically replace the kernel by zeros. |
| [pythran_bubble_sort](pythran_bubble_sort/result.json) | `accepted_as_written` | 10890 | Source bubble-sort loops, no invented sorting network; break is unreachable because every inner loop executes. |
| [pythran_check_mask](pythran_check_mask/result.json) | `accepted_as_written` | 1398 | Direct static-shape translation. |
| [pythran_convnet](pythran_convnet/result.json) | `needs_gadget: tanh` | — | kernel.py:18 tanh and helper line 10 exp/division require gadgets. |
| [pythran_deuxd_convolution](pythran_deuxd_convolution/result.json) | `accepted_as_written` | 162556 | Preserves the source clamp bound sw=5, including its unusual use instead of image dimensions. |
| [pythran_fannkuch](pythran_fannkuch/result.json) | `circ_error` | — | Bounded 1024!-step rewrite rejected: `0 is not a constant integer` for an integer-typed loop bound. |
| [pythran_gauss](pythran_gauss/result.json) | `needs_gadget: division` | — | kernel.py:32 reciprocal of a private pivot requires fixed-point division. |
| [pythran_hasting](pythran_hasting/result.json) | `needs_gadget: division` | — | kernel.py:11-13 private denominators require fixed-point division. |
| [pythran_lap](pythran_lap/result.json) | `needs_gadget: division` | — | kernel.py:25 lap/dh2 requires fixed-point division by private dh2. |
| [pythran_log_likelihood](pythran_log_likelihood/result.json) | `needs_gadget: division` | — | kernel.py:11 private fixed-point division; lines 12-14 also require exp/log. |
| [pythran_mandel](pythran_mandel/result.json) | `not_expressible: dynamic loop` | — | kernel.py:18 while loop has private max_iteration without upper bound; size<=1024 bounds only image shape. |
| [pythran_multi_export](pythran_multi_export/result.json) | `accepted_as_written` | 65 | Direct static-shape translation. |
| [pythran_pairwise](pythran_pairwise/result.json) | `needs_gadget: sqrt` | — | kernel.py:18 np.sqrt(d) requires fixed-point square root. |
| [pythran_primes_sieve2](pythran_primes_sieve2/result.json) | `needs_gadget: sqrt` | — | kernel.py:19 n**0.5 requires square root; do not replace the source algorithm by a different sieve. |
| [pythran_repeating](pythran_repeating/result.json) | `not_expressible: dynamic shape` | — | kernel.py:11 output length 10*(1+nvar_y) has no upper bound in the signature. |
| [pythran_smoothing](pythran_smoothing/result.json) | `accepted_as_written` | 12372 | Direct static-shape translation. |
| [pythran_sum_primes](pythran_sum_primes/result.json) | `needs_gadget: sqrt` | — | kernel.py:15 math.sqrt(n) and ceil require a square-root gadget. |
| [pythran_vibr_energy](pythran_vibr_energy/result.json) | `needs_gadget: exp` | — | kernel.py:10 numpy.exp requires fixed-point exponential. |

## Rewrites and verification limits

Every rewrite and its domain-bound argument is listed in the corresponding `result.json` and `attempts.md`. Static append enumeration in case0341 and the source bubble-sort loops are translated directly: case0341 has exactly 30 values, and bubble_sort's break is unreachable because every inner loop executes and sets swap_test=True unconditionally. No sorting network was substituted.

Non-expressible and gadget-dependent attempts have source-line explanations instead of fabricated `.zok` stubs or invented compiler errors. Their A/B correctness fields are null because no allowed executable translation exists. Gadget requirements were identified from source operations and the stock Z# frontend/standard-library capabilities. These are not measured compiler rejections. There are 29 such programs and one measured compiler rejection; none timed out or exhausted memory.

Fannkuch has an explicit n<=1024 contract, so it is not classified as unbounded. Its minimal rewrite pads the permutation arrays to 1024, bounds permutation/prefix-reversal enumeration by 1024!, and masks early exits. CirC rejects the arbitrary-precision integer loop bound before unrolling (`0 is not a constant integer`). It was not reduced to the fixture values n=4 or n=3, and its A/B results were not verified.

Correctness establishes agreement on fixtures A and B at relative+absolute tolerance 1e-6, not a proof over every input or of Q32 overflow freedom. case0399 A has maximum absolute error approximately 2.16e-4 but passes the specified combined tolerance because the corresponding outputs are large. Source-domain reasoning for each padding/loop capacity is separate from fixture testing. No accepted result has a failed or missing fixture check.

Reproduce all final attempts and fixture checks with `/home/zhantong/Projects/zinnia-project/wt/literal/.venv/bin/python study.py`; pass program names to rerun a subset. Run `finish.py` with the same Python afterward to audit artifacts and regenerate this report. Development logs at batch root may contain corrected intermediate failures; per-program final logs and `result.json` are authoritative.

BATCH_DONE
