# CirC acceptance study — batch0

Completed 55 programs: **24 accepted**, all checked against both NumPy fixtures at relative + absolute tolerance 1e-6. No accepted program remains unchecked or mismatching.

## Method

Read BRIEF.md before any program work. Used only kernel source, signatures, preconditions and fixtures to choose translations; did not inspect Zinnia IR, circuits, sizes or use its outcomes. All work and temporary files are inside this batch directory.

Compiler: `/home/zhantong/.cache/zkcc-reconfirm/circ-stock/target/release/examples/circ`, language `zsharp`, backend `r1cs --action count`. Every executable attempt ran through the requested `memrun.sh` with `MEMRUN_MAX=8G`, `MemorySwapMax=0`, and `timeout 300`. `ZSHARP_STDLIB_PATH` selects the installed ZoKrates stdlib; `TMPDIR` is this batch's `tmp/`. Commands, exit statuses, durations and compiler diagnostics are in each result/log. No build or remote machine was used.

Inputs are private and return values public. Reals are rounded to integer multiples of 2^-32. Copies, indexing, linear operations and comparisons use centered signed field encoding; signed comparisons shift both operands by half the field modulus. Native `integer` arithmetic was tried for signed Q32 products and quotients. Those full translations reached CirC's R1CS backend and failed on integer operators. Modular field inversion was not substituted for fixed-point numerical division, nor were values narrowed to unsigned or fixture-only ranges. The missing signed fixed-point quotient/rescaling route is recorded as `needs_gadget: division` under the brief's precedence. No handwritten sort, search, division, square-root, transcendental or arithmetic circuit gadget was added.

For non-expressible shapes/loops and missing gadgets, `.zok` files are explicit explanatory records, **not executable stubs**. These cases have no compilation count; the result names the source line and why no permitted full translation can be emitted. Bounded rewrites that still need a forbidden gadget are documented but not presented as compiled programs. In particular, the study does not claim a compiler run for those exceptions.

Padded outputs include public lengths/dimensions (and zero padding); correctness decodes that representation back to the original output before comparison. Constant-output programs retain their original private input signatures. In-place kernels preserve their mutated arrays as observable outputs. The empty 3x0 array is represented by padding and public shape after CirC rejected a native empty array. For `case0385`, the static view permutation is expanded directly into an array constructor, avoiding the interpreter's memory blowup on sequential array stores; all 20,000 elements and the 5000x2x2 result shape are retained. For `pythran_emin`, the original million iterations are retained; the rewrite only uses the built-in integer-to-field cast on constant outputs.

## Counts

| Category | Programs |
| --- | ---: |
| accepted_as_written | 12 |
| accepted_with_rewrite | 12 |
| needs_gadget: complex pow and linear solve | 1 |
| needs_gadget: cos | 1 |
| needs_gadget: division | 14 |
| needs_gadget: exp | 1 |
| needs_gadget: fixed-point multiplication | 1 |
| needs_gadget: pow | 1 |
| needs_gadget: sort | 2 |
| needs_gadget: sqrt | 4 |
| not_expressible: dynamic loop | 2 |
| not_expressible: dynamic shape | 3 |
| timeout_or_oom | 1 |
| **Total** | **55** |

## Per-program results

| Program | Category | Constraints | Reason |
| --- | --- | ---: | --- |
| case0291 | accepted_as_written | 4 | A/B pass. Static-shape translation using native field operations. |
| case0300 | needs_gadget: sort | — | needs gadget: sort |
| case0310 | accepted_as_written | 72002 | A/B pass. Static-shape translation using native field operations. |
| case0318 | accepted_as_written | 3599 | A/B pass. Native dynamic scatter and gather; negative NumPy indices normalized by array length. Undefined np.empty entries for non-permutations remain outside deterministic valid executions. |
| case0324 | needs_gadget: cos | — | needs gadget: cos |
| case0330 | needs_gadget: pow | — | needs gadget: pow |
| case0337 | accepted_as_written | 0 | A/B pass. Static-shape translation using native field operations. |
| case0343 | accepted_as_written | 0 | A/B pass. List append count is syntactically fixed at 30, independent of input values. |
| case0353 | accepted_with_rewrite | 25068 | A/B pass. Pad to two rows, compact with guards and return public length; filtering two input rows can retain at most two. |
| case0359 | accepted_as_written | 0 | A/B pass. Static-shape translation using native field operations. |
| case0365 | accepted_with_rewrite | 51674 | A/B pass. Replace insertion scatter with a guarded gather into the known 101 output positions; inserting one scalar into 100 elements always yields length 101. |
| case0374 | needs_gadget: sort | — | needs gadget: sort |
| case0385 | accepted_as_written | 0 | A/B pass. Static reshape/swapaxes/transpose indexing expanded directly into a literal array constructor; avoids interpreter memory blowup from sequential stores. All 20,000 values retained. |
| case0391 | accepted_with_rewrite | 13046 | A/B pass. Clamp signed slice endpoints as NumPy does; pad output to 20x10 and return width, since slicing cannot exceed ten columns. |
| case0401 | accepted_with_rewrite | 5 | A/B pass. Represent the empty 3x0 array with 3x1 zero padding and public dimensions [3,0]. |
| case0408 | not_expressible: dynamic shape | — | No domain-implied finite bound. The signature permits arbitrarily large nonnegative group labels; 100 labels do not bound the largest label or dense output length. Unique and min-mask lengths are bounded by 100, but the returned dense array is not. |
| case0415 | accepted_with_rewrite | 18661 | A/B pass. Pad at most eight bins and return length; enumerate the eight possible nonempty bin widths with guards (larger positive widths produce no bins). |
| case0422 | needs_gadget: division | — | Signed fixed-point division (including rescaling a Q32 product) requires a gadget once native integer R1CS lowering fails. |
| case0430 | needs_gadget: sqrt | — | needs gadget: sqrt |
| case0437 | accepted_as_written | 61340 | A/B pass. Static-shape translation using native field operations. |
| case0443 | accepted_with_rewrite | 14922 | A/B pass. Use capacity 20, stable guarded compaction and public length; selection cannot exceed the 20 input entries. |
| case0452 | needs_gadget: division | — | Signed fixed-point division (including rescaling a Q32 product) requires a gadget once native integer R1CS lowering fails. |
| case0462 | accepted_with_rewrite | 14859 | A/B pass. Use capacity 20, stable guarded compaction and public length; selection cannot exceed the 20 input entries. |
| case0477 | accepted_as_written | 12 | A/B pass. Static-shape translation using native field operations. |
| case0485 | needs_gadget: division | — | Signed fixed-point division (including rescaling a Q32 product) requires a gadget once native integer R1CS lowering fails. |
| case0497 | not_expressible: dynamic shape | — | No domain-implied finite bound. The declared integer target dimensions have no upper bound; padding is valid for arbitrarily larger dimensions, so input shape 42x10 gives no output capacity. |
| case0503 | accepted_as_written | 108667 | A/B pass. Static-shape translation using native field operations. |
| case0510 | accepted_with_rewrite | 39036 | A/B pass. Use ten-entry row/column presence masks and guarded extrema; pad the cropped result to 10x10 with height and width, bounded by the input shape. |
| npbench_contour_integral_contour_integral | needs_gadget: complex pow and linear solve | — | needs gadget: complex pow and linear solve; The bounded rewrite still requires complex exponentiation and a matrix inverse/linear solve, including real division. |
| npbench_deep_learning_softmax_softmax | needs_gadget: exp | — | needs gadget: exp |
| npbench_polybench_atax_atax | needs_gadget: division | — | Signed fixed-point division (including rescaling a Q32 product) requires a gadget once native integer R1CS lowering fails. |
| npbench_polybench_covariance_covariance | needs_gadget: division | — | needs gadget: division; kernel.py:14 divides by private float_n - 1.0; fixed-point division remains necessary. |
| npbench_polybench_gemm_gemm | needs_gadget: division | — | Signed fixed-point division (including rescaling a Q32 product) requires a gadget once native integer R1CS lowering fails. |
| npbench_polybench_jacobi_2d_jacobi_2d | not_expressible: dynamic loop | — | No domain-implied finite bound. TSTEPS is unrestricted by the signature; fixed array extents do not bound the number of stencil time steps. |
| npbench_polybench_nussinov_nussinov | accepted_with_rewrite | 1310320 | A/B pass. Pad table to 14x14 and guard loops using N; valid executions require 0 <= N <= 14 because for N > 14 the recurrence reads seq[N-1]. |
| npbench_polybench_trmm_trmm | needs_gadget: division | — | Signed fixed-point division (including rescaling a Q32 product) requires a gadget once native integer R1CS lowering fails. |
| npbench_weather_stencils_vadv_vadv | needs_gadget: division | — | needs gadget: division |
| pythran_approximated_callgraph | timeout_or_oom | — | Running frontend Running IR optimizations |
| pythran_caxpy | needs_gadget: fixed-point multiplication | — | needs gadget: fixed-point multiplication; Signed products of Q32 components require scaling without overflow; native integer R1CS lowering fails and a custom arithmetic gadget is prohibited. |
| pythran_clip2 | needs_gadget: sqrt | — | needs gadget: sqrt; Complex absolute value requires sqrt and helper normalization needs fixed-point division. |
| pythran_cronbach | needs_gadget: division | — | needs gadget: division |
| pythran_emin | accepted_with_rewrite | 4 | A/B pass. Cast the constant integer outputs to field using the built-in integer_to_field; preserve all 1,000,000 iterations and Q32 arithmetic. |
| pythran_fdtd | not_expressible: dynamic loop | — | No domain-implied finite bound. steps is unrestricted by the signature; fixed 16x16 arrays do not bound the number of repeated time steps. |
| pythran_grouping | accepted_with_rewrite | 3002 | A/B pass. Pad index array to six and both outputs to five with public group count; five entries imply at most five runs. |
| pythran_hyantes_core | not_expressible: dynamic shape | — | No domain-implied finite bound. range_x and range_y have no upper bounds; only nonzero step and range_ are required. Arbitrarily large positive grids are valid. Trigonometry and division are additional blockers. |
| pythran_lapl3d | needs_gadget: division | — | Signed fixed-point division (including rescaling a Q32 product) requires a gadget once native integer R1CS lowering fails. |
| pythran_lu | needs_gadget: division | — | needs gadget: division |
| pythran_matrix_class_distance | needs_gadget: sqrt | — | needs gadget: sqrt; The norm over ten columns needs a real sqrt gadget. |
| pythran_multitype | accepted_as_written | 3 | A/B pass. Static-shape translation using native field operations. |
| pythran_periodic_dist | needs_gadget: sqrt | — | needs gadget: sqrt |
| pythran_pselect | accepted_with_rewrite | 3 | A/B pass. Inline the two possible callees under their original guards; allocate the list at its fixed two-element length. |
| pythran_rosen | needs_gadget: division | — | Signed fixed-point division (including rescaling a Q32 product) requires a gadget once native integer R1CS lowering fails. |
| pythran_specialconvolve | accepted_as_written | 85 | A/B pass. Static-shape translation using native field operations. |
| pythran_tax | needs_gadget: division | — | needs gadget: division; kernel.py:16 divides the tax sum by private np.sum(d), requiring fixed-point division; products also require Q32 scaling. |
| pythran_wdist | needs_gadget: division | — | needs gadget: division |

## Verification and reproduction

Each accepted program has fixture copies, A/B `.pin` inputs, interpreter logs, NumPy expected outputs, decoded actual outputs and explicit comparison statuses in `result.json`. NumPy references run the original function bodies after stripping only Zinnia decorators and annotations; no Zinnia compilation occurs. `np.in1d` uses `np.isin` on NumPy versions that removed the former. `compare.py` checks structure and values, not just flattened numbers. Equal NaNs are allowed by the comparator, though no accepted fixture depends on a NaN route.

From this directory, use `/home/zhantong/Projects/zinnia-project/wt/literal/.venv/bin/python reproduce.py` to rerun the final executable translations and both attempts where executable, regenerate the exception records, compare accepted results, and rebuild this report. Run scripts do not rebuild or edit CirC. `build.py`, `extend.py` and `fixes.py` retain translation construction history; the final per-program `.zok` files are authoritative and reproduction does not overwrite them. Some `.loop_expansion.zok` files and `original_attempt.json` files document earlier translation/resource diagnostics.

No numerical correctness claim is made for programs without an accepted executable attempt. Missing-gadget conclusions rely on the documented native frontend/stdlib capabilities and, for native signed-integer arithmetic, the saved actual backend rejection. `not_expressible` conclusions concern the declared domain, not the two fixture values. A finite input array does not bound arbitrary requested padding or time-step counts. Resource failures are measured under the specified per-process limits; a SIGKILL/137 under the memory-capped scope is classified as out of memory.

BATCH_DONE
