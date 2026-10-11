# CirC acceptance repair

Repaired **77 entries across 71 program names**, including all 11 entries of the five multi-function programs. `merged.json` contains exactly **327 unique (program, function) results**, in `programs.json` order. Repair results take precedence; the remaining 250 results come from the matching batch entry. Batch directories were not edited.

Every executable candidate was compiled with the stock CirC binary, `MEMRUN_MAX=8G`, `memrun.sh`, and `timeout 300`; temporary files stayed under `repair/tmp`. Real encoding remains Q32. Every repaired accepted result passes NumPy comparison on both fixture A and fixture B at 1e-6 relative plus absolute tolerance. Native arithmetic replaces the old arbitrary-integer backend operations and division/multiplication gadget labels.

The source, inputs, decoded outputs, logs, exact commands and checksums are recorded per entry; `provenance.json` identifies the result used for every merged row. See `README.md` for reproduction. Native unsigned quotient/remainder and bit casts are ordinary Z# operations; no manually asserted division constraints, auxiliary witness arguments, sorting networks, or transcendental implementations were added.

Fannkuch uses the declared entry `fannkuch2`, which discards a pure helper result and returns None. Its minimal rewrite removes that unused call while preserving the n=0 error with an assertion. Full bounded native candidates remain archived. Fibonacci uses twelve u64 limbs, preserving the entire n<=1024 domain.

## Category counts

| Category | Old affected entries | Repaired entries | Final merged entries |
|---|---:|---:|---:|
| accepted_as_written | 3 | 44 | 117 |
| accepted_with_rewrite | 0 | 14 | 62 |
| circ_error | 12 | 0 | 0 |
| needs_gadget: asin | 0 | 0 | 1 |
| needs_gadget: complex pow and linear solve | 0 | 0 | 1 |
| needs_gadget: cos | 0 | 0 | 1 |
| needs_gadget: division | 36 | 0 | 0 |
| needs_gadget: exp | 0 | 0 | 6 |
| needs_gadget: exp/log | 0 | 1 | 1 |
| needs_gadget: fixed-point multiplication | 9 | 0 | 0 |
| needs_gadget: fixed-point multiplication/rescaling | 12 | 0 | 0 |
| needs_gadget: gcd | 0 | 0 | 1 |
| needs_gadget: least_squares | 0 | 0 | 1 |
| needs_gadget: log | 0 | 0 | 2 |
| needs_gadget: matrix inverse | 0 | 0 | 1 |
| needs_gadget: matrix_rank | 0 | 0 | 1 |
| needs_gadget: pow | 0 | 0 | 1 |
| needs_gadget: power | 0 | 0 | 1 |
| needs_gadget: search | 0 | 0 | 1 |
| needs_gadget: sin | 0 | 0 | 4 |
| needs_gadget: sort | 0 | 0 | 15 |
| needs_gadget: sqrt | 0 | 1 | 28 |
| needs_gadget: tanh | 0 | 0 | 3 |
| not_expressible: dynamic loop | 0 | 0 | 18 |
| not_expressible: dynamic shape | 2 | 2 | 38 |
| timeout_or_oom | 3 | 15 | 22 |
| unsupported_feature: integer bitwise | 0 | 0 | 1 |

## Per-entry changes

| Program / function | Old category | New category | Constraints | A/B | Why |
|---|---|---|---:|---|---|
| [case0454 / kernel](case0454/result.json) | needs_gadget: division | accepted_as_written | 311860 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [case0485 / kernel](case0485/result.json) | needs_gadget: division | accepted_as_written | 209007 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [npbench_polybench_gemm_gemm / kernel](npbench_polybench_gemm_gemm/result.json) | needs_gadget: division | accepted_as_written | 2722120 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [npbench_polybench_k3mm_k3mm / kernel](npbench_polybench_k3mm_k3mm/result.json) | needs_gadget: fixed-point multiplication/rescaling | timeout_or_oom | — | pass/pass | Native-arithmetic timeout under 300 s / 8 GB; no arithmetic gadget blocker. |
| [npbench_weather_stencils_vadv_vadv / vadv](npbench_weather_stencils_vadv_vadv/result.json) | needs_gadget: division | timeout_or_oom | — | pass/pass | Native-arithmetic timeout under 300 s / 8 GB; no arithmetic gadget blocker. |
| [pythran_ln / ln](pythran_ln/result.json) | circ_error | accepted_as_written | 153968 | pass/pass | Replace the arbitrary-integer/field-remainder encoding with native unsigned arithmetic and wide products; A/B pass. |
| [case0334 / kernel](case0334/result.json) | circ_error | accepted_as_written | 33150 | pass/pass | Replace the arbitrary-integer/field-remainder encoding with native unsigned arithmetic and wide products; A/B pass. |
| [case0416 / kernel](case0416/result.json) | needs_gadget: division | accepted_with_rewrite | 41445 | pass/pass | Pad each row to six means and return the logical column count; at most six nonempty bins fit in each six-element row. A/B pass. |
| [npbench_polybench_covariance2_covariance2 / kernel](npbench_polybench_covariance2_covariance2/result.json) | needs_gadget: division | accepted_as_written | 1818782 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [npbench_polybench_gemver_gemver / kernel](npbench_polybench_gemver_gemver/result.json) | needs_gadget: fixed-point multiplication | accepted_as_written | 1017598 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [npbench_polybench_lu_lu / kernel](npbench_polybench_lu_lu/result.json) | needs_gadget: division | accepted_as_written | 2355337 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [npbench_polybench_trisolv_trisolv / kernel](npbench_polybench_trisolv_trisolv/result.json) | needs_gadget: division | accepted_as_written | 364307 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [pythran_allpairs / sqr_dists_loops](pythran_allpairs__sqr_dists_loops/result.json) | needs_gadget: fixed-point multiplication | accepted_as_written | 3282944 | pass/pass | Independent sqr_dists_loops translation and fixture. Native signed Q32 arithmetic compiles; both fixtures pass. |
| [pythran_allpairs / sqr_dists](pythran_allpairs__sqr_dists/result.json) | needs_gadget: fixed-point multiplication/rescaling | accepted_as_written | 3282944 | pass/pass | Independent sqr_dists translation and fixture. Native signed Q32 arithmetic compiles; both fixtures pass. |
| [pythran_d2q9_nxnyns / one_time_step](pythran_d2q9_nxnyns/result.json) | needs_gadget: fixed-point multiplication/rescaling | timeout_or_oom | — | pass/pass | Native-arithmetic out of memory under 300 s / 8 GB; no arithmetic gadget blocker. |
| [pythran_fannkuch / fannkuch2](pythran_fannkuch/result.json) | circ_error | accepted_with_rewrite | 196 | pass/pass | Delete the pure, unused fannkuch(n) call in fannkuch2 (kernel.py:44). It returns a value discarded by the entry and mutates only locally allocated lists; the observable result is None for every successful input.; Represent None by a false public presence tag (CirC rejects a literal zero-length array) and preserve the original n=0 IndexError as an assertion failure. For 1<=n<=1024 the permutation enumeration terminates and has no externally visible effects. A/B pass. |
| [pythran_log_likelihood / log_likelihood](pythran_log_likelihood/result.json) | needs_gadget: division | needs_gadget: exp/log | — | not_run/not_run | kernel.py:12,14 use numpy.exp(-s) and numpy.log(pdfs). Native unsigned division handles the arithmetic, but these transcendental calls have no native route. |
| [pythran_tax / tax](pythran_tax/result.json) | needs_gadget: division | accepted_with_rewrite | 62175 | pass/pass | Replace four value-dependent filtered segments by ten guarded entries each; each segment is a subset of the ten inputs. A/B pass. |
| [case0417 / kernel](case0417/result.json) | needs_gadget: division | accepted_with_rewrite | 14475 | pass/pass | Pad output to eight bins with a logical count; replace dynamic bin slices by guarded scans of the eight input values. A/B pass. |
| [npbench_deep_learning_conv2d_bias_conv2d / conv2d_bias](npbench_deep_learning_conv2d_bias_conv2d/result.json) | needs_gadget: fixed-point multiplication/rescaling | timeout_or_oom | — | out_of_memory/out_of_memory | Native-arithmetic out of memory under 300 s / 8 GB; no arithmetic gadget blocker. |
| [npbench_polybench_covariance_covariance / kernel](npbench_polybench_covariance_covariance/result.json) | needs_gadget: division | accepted_with_rewrite | 2768213 | pass/pass | Pad cov to 14x14 and return M; valid executions have 0<=M<=14 because nonempty assignments use the fourteen columns of data. A/B pass. |
| [npbench_polybench_gesummv_gesummv / kernel](npbench_polybench_gesummv_gesummv/result.json) | needs_gadget: fixed-point multiplication/rescaling | accepted_as_written | 376924 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [npbench_polybench_ludcmp_ludcmp / kernel](npbench_polybench_ludcmp_ludcmp/result.json) | needs_gadget: division | accepted_as_written | 2157378 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [npbench_polybench_trmm_trmm / kernel](npbench_polybench_trmm_trmm/result.json) | needs_gadget: division | accepted_as_written | 381802 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [pythran_deriv / deriv](pythran_deriv/result.json) | circ_error | accepted_with_rewrite | 15542 | pass/pass | Use guarded iterations 1..3 and 2..4; dg[n] forces n<=3 on valid calls and n>=0 is explicitly required.; Represent complex arithmetic by pairs of Q32 real fields using native multiply/add and constant rescaling. A/B pass. |
| [pythran_fbcorr / fbcorr](pythran_fbcorr/result.json) | needs_gadget: division | timeout_or_oom | — | pass/out_of_memory | Native-arithmetic out of memory under 300 s / 8 GB; no arithmetic gadget blocker. |
| [pythran_lttb_pt / downsample](pythran_lttb_pt/result.json) | needs_gadget: division | accepted_with_rewrite | 1157115 | pass/pass | Valid returns require 3<=n_out<=16 by explicit source exceptions. Pad the output to sixteen points and guard at most fourteen array_split bins; preserve larger first bins, mean and first argmax tie-breaking. A/B pass. |
| [pythran_train_eq / train_eq2](pythran_train_eq__train_eq2/result.json) | not_expressible: dynamic shape | not_expressible: dynamic shape | — | not_run/not_run | Function train_eq2: err=np.zeros(TrSyms) and range(TrSyms) have no domain-implied bound. os=0 permits a valid ten-tap slice for arbitrarily many TrSyms despite the fixed E shape; output err has TrSyms entries. The explicit arithmetic does not remove this blocker. |
| [pythran_train_eq / train_eq](pythran_train_eq__train_eq/result.json) | not_expressible: dynamic shape | not_expressible: dynamic shape | — | not_run/not_run | Function train_eq: err=np.zeros(TrSyms) and range(TrSyms) have no domain-implied bound. os=0 permits a valid ten-tap slice for arbitrarily many TrSyms despite the fixed E shape; output err has TrSyms entries. The explicit arithmetic does not remove this blocker. |
| [case0345 / kernel](case0345/result.json) | needs_gadget: fixed-point multiplication | accepted_as_written | 17145 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [case0388 / kernel](case0388/result.json) | timeout_or_oom | timeout_or_oom | — | interpreter_error/interpreter_error | Native-arithmetic out of memory under 300 s / 8 GB; no arithmetic gadget blocker. |
| [case0418 / kernel](case0418/result.json) | circ_error | accepted_with_rewrite | 31794 | pass/pass | Pad each row to six means and return the logical column count; at most six nonempty bins fit in each six-element row. A/B pass. |
| [npbench_deep_learning_lenet_lenet / lenet5](npbench_deep_learning_lenet_lenet/result.json) | circ_error | timeout_or_oom | — | out_of_memory/out_of_memory | Native-arithmetic timeout under 300 s / 8 GB; no arithmetic gadget blocker. |
| [npbench_polybench_mvt_mvt / kernel](npbench_polybench_mvt_mvt/result.json) | needs_gadget: division | accepted_as_written | 162752 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [pythran_caxpy / CAXPY](pythran_caxpy/result.json) | needs_gadget: fixed-point multiplication | timeout_or_oom | — | pass/pass | Native-arithmetic out of memory under 300 s / 8 GB; no arithmetic gadget blocker. |
| [pythran_lap / laplacian](pythran_lap/result.json) | needs_gadget: division | accepted_as_written | 263142 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [pythran_lu / factorMatrix1](pythran_lu__factorMatrix1/result.json) | needs_gadget: division | accepted_as_written | 195808 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [pythran_lu / factorMatrix0](pythran_lu__factorMatrix0/result.json) | needs_gadget: division | accepted_as_written | 206398 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [pythran_quicksort / quicksort](pythran_quicksort/result.json) | timeout_or_oom | timeout_or_oom | — | pass/pass | Native-arithmetic timeout under 300 s / 8 GB; no arithmetic gadget blocker. |
| [case0346 / kernel](case0346/result.json) | needs_gadget: fixed-point multiplication/rescaling | accepted_as_written | 29598 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [case0439 / kernel](case0439/result.json) | needs_gadget: fixed-point multiplication/rescaling | accepted_as_written | 437600 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [npbench_polybench_atax_atax / kernel](npbench_polybench_atax_atax/result.json) | needs_gadget: division | accepted_as_written | 130466 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [npbench_polybench_doitgen_doitgen / kernel](npbench_polybench_doitgen_doitgen/result.json) | needs_gadget: fixed-point multiplication/rescaling | timeout_or_oom | — | pass/pass | Native-arithmetic out of memory under 300 s / 8 GB; no arithmetic gadget blocker. |
| [npbench_scattering_self_energies_scattering_self_energies / scattering_self_energies](npbench_scattering_self_energies_scattering_self_energies/result.json) | needs_gadget: fixed-point multiplication/rescaling | timeout_or_oom | — | out_of_memory/out_of_memory | Native-arithmetic timeout under 300 s / 8 GB; no arithmetic gadget blocker. |
| [pythran_fft / fft](pythran_fft/result.json) | needs_gadget: fixed-point multiplication | accepted_as_written | 34972 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [pythran_harris / harris](pythran_harris/result.json) | circ_error | accepted_as_written | 41185 | pass/pass | Replace the arbitrary-integer/field-remainder encoding with native unsigned arithmetic and wide products; A/B pass. |
| [case0347 / kernel](case0347/result.json) | needs_gadget: division | accepted_as_written | 121735 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [case0420 / kernel](case0420/result.json) | needs_gadget: division | accepted_as_written | 1574 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [npbench_cavity_flow_cavity_flow / cavity_flow](npbench_cavity_flow_cavity_flow/result.json) | needs_gadget: division | timeout_or_oom | — | interpreter_error/interpreter_error | Native-arithmetic out of memory under 300 s / 8 GB; no arithmetic gadget blocker. |
| [npbench_polybench_bicg_bicg / kernel](npbench_polybench_bicg_bicg/result.json) | needs_gadget: fixed-point multiplication | accepted_as_written | 131324 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [npbench_polybench_durbin_durbin / kernel](npbench_polybench_durbin_durbin/result.json) | needs_gadget: division | accepted_as_written | 317211 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [npbench_spmv_spmv / spmv](npbench_spmv_spmv/result.json) | circ_error | accepted_with_rewrite | 76041 | pass/pass | Normalize Python slice endpoints into [0,16] and replace each slice dot product with 16 guarded multiply-accumulate steps. The source arrays have 16 entries. A/B pass. |
| [pythran_cdotc / CDOTC](pythran_cdotc/result.json) | needs_gadget: fixed-point multiplication/rescaling | accepted_with_rewrite | 2473354 | pass/pass | Guard 1024 iterations with I<N.; Encode optional None/complex return with a public presence tag. A/B pass. |
| [pythran_fibo / test](pythran_fibo/result.json) | circ_error | accepted_with_rewrite | 3069624 | pass/pass | Use 1024 guarded iterations from the explicit n<=1024 contract; twelve u64 limbs preserve the entire Fibonacci result (F1026 < 2^768). Replace repeated recursion by the same Fibonacci recurrence. A/B pass. |
| [pythran_hasting / fweb](pythran_hasting/result.json) | needs_gadget: division | accepted_as_written | 35320 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [pythran_lapl3d / laplacien](pythran_lapl3d/result.json) | needs_gadget: division | timeout_or_oom | — | pass/pass | Native-arithmetic out of memory under 300 s / 8 GB; no arithmetic gadget blocker. |
| [pythran_multitype / check2](pythran_multitype__check2/result.json) | accepted_as_written | accepted_as_written | 134 | pass/pass | Independent check2 translation and fixture. Native signed Q32 arithmetic compiles; both fixtures pass. |
| [pythran_multitype / check](pythran_multitype__check/result.json) | accepted_as_written | accepted_as_written | 200 | pass/pass | Independent check translation and fixture. Native signed Q32 arithmetic compiles; both fixtures pass. |
| [pythran_multitype / times](pythran_multitype__times/result.json) | accepted_as_written | accepted_as_written | 261 | pass/pass | Independent times translation and fixture. Native signed Q32 arithmetic compiles; both fixtures pass. |
| [pythran_spectraldns / add_pressure_diffusion](pythran_spectraldns__add_pressure_diffusion/result.json) | needs_gadget: fixed-point multiplication | accepted_as_written | 620188 | pass/pass | Independent add_pressure_diffusion translation and fixture. Native signed Q32 arithmetic compiles; both fixtures pass. |
| [pythran_spectraldns / cross2](pythran_spectraldns__cross2/result.json) | needs_gadget: fixed-point multiplication/rescaling | accepted_as_written | 305340 | pass/pass | Independent cross2 translation and fixture. Native signed Q32 arithmetic compiles; both fixtures pass. |
| [case0422 / kernel](case0422/result.json) | needs_gadget: division | accepted_as_written | 518400 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [case0452 / kernel](case0452/result.json) | needs_gadget: division | accepted_as_written | 308580 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [npbench_polybench_symm_symm / kernel](npbench_polybench_symm_symm/result.json) | needs_gadget: fixed-point multiplication/rescaling | accepted_as_written | 443692 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [pythran_cdotu / CDOTU](pythran_cdotu/result.json) | circ_error | accepted_with_rewrite | 2345284 | pass/pass | Use 1024 guarded iterations, from the declared precondition, including repeated accesses when INCX or INCY is zero.; Represent complex numbers by real/imaginary Q32 fields and expose a presence tag for the N<=0 return.; Normalize valid negative array indices against length four and guard inactive accesses. A/B pass. |
| [pythran_euclidean_distance_square / euclidean_distance_square](pythran_euclidean_distance_square/result.json) | needs_gadget: fixed-point multiplication | accepted_as_written | 11490 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [pythran_fibo_seq / fibo](pythran_fibo_seq/result.json) | circ_error | accepted_with_rewrite | 3072540 | pass/pass | Use 1024 guarded iterations from the explicit n<=1024 contract; twelve u64 limbs preserve the entire Fibonacci result (F1026 < 2^768). A/B pass. |
| [pythran_laplace / calc](pythran_laplace/result.json) | timeout_or_oom | timeout_or_oom | — | out_of_memory/out_of_memory | Native-arithmetic out of memory under 300 s / 8 GB; no arithmetic gadget blocker. |
| [pythran_rosen / rosen](pythran_rosen/result.json) | needs_gadget: division | accepted_as_written | 19921 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [pythran_wdist / slow_wdist](pythran_wdist/result.json) | needs_gadget: division | needs_gadget: sqrt | — | not_run/not_run | kernel.py:16 uses np.sqrt((wdiff ** 2).sum()); native Q32 division repairs the division subexpression but cannot supply square root. All loops already have static bounds. |
| [npbench_polybench_k2mm_k2mm / kernel](npbench_polybench_k2mm_k2mm/result.json) | needs_gadget: fixed-point multiplication | timeout_or_oom | — | pass/pass | Native-arithmetic timeout under 300 s / 8 GB; no arithmetic gadget blocker. |
| [npbench_polybench_syr2k_syr2k / kernel](npbench_polybench_syr2k_syr2k/result.json) | circ_error | accepted_as_written | 496073 | pass/pass | Replace the arbitrary-integer/field-remainder encoding with native unsigned arithmetic and wide products; A/B pass. |
| [npbench_weather_stencils_hdiff_hdiff / hdiff](npbench_weather_stencils_hdiff_hdiff/result.json) | needs_gadget: division | accepted_as_written | 2249570 | pass/pass | Use operand signs and nonzero tests for sign-only products to avoid u64 overflow; native Q32 arithmetic compiles and A/B pass. |
| [pythran_cronbach / cronbach](pythran_cronbach/result.json) | needs_gadget: division | accepted_as_written | 157585 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [pythran_gauss / gauss](pythran_gauss/result.json) | needs_gadget: division | accepted_with_rewrite | 188287 | pass/pass | Bound active iterations by four: any n>4 accesses beyond the declared 4x4 matrix and is not a valid successful input. Guard updates and the reciprocal on inactive iterations.; Store the mutable 4x4 matrix in a flat sixteen-element buffer after the frontend succeeds but tuple-update optimization crashes; preserve every row/column operation. A/B pass. |
| [pythran_matmul / matrix_multiply](pythran_matmul/result.json) | needs_gadget: division | accepted_as_written | 2835456 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |
| [pythran_weights / weights](pythran_weights/result.json) | needs_gadget: division | accepted_as_written | 8794 | pass/pass | Native signed Q32 arithmetic compiles; both fixtures pass. |

## Verification limits

Resource failures and non-expressible/gadget cases are not counted as accepted. Interpreter evaluation does not override failed circuit compilation. The original unmodified batch results outside this repair scope were merged without revalidation.

A/B could not both be verified for: pythran_log_likelihood/log_likelihood (not_run/not_run); npbench_deep_learning_conv2d_bias_conv2d/conv2d_bias (out_of_memory/out_of_memory); pythran_fbcorr/fbcorr (pass/out_of_memory); pythran_train_eq/train_eq2 (not_run/not_run); pythran_train_eq/train_eq (not_run/not_run); case0388/kernel (interpreter_error/interpreter_error); npbench_deep_learning_lenet_lenet/lenet5 (out_of_memory/out_of_memory); npbench_scattering_self_energies_scattering_self_energies/scattering_self_energies (out_of_memory/out_of_memory); npbench_cavity_flow_cavity_flow/cavity_flow (interpreter_error/interpreter_error); pythran_laplace/calc (out_of_memory/out_of_memory); pythran_wdist/slow_wdist (not_run/not_run).

REPAIR_DONE
