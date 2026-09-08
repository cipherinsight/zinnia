# Benchmark Source Pool Filtering

Generated: 2026-05-19

This directory is initialized from canonical upstream NPBench and Pythran source programs. Each included program occupies exactly one subdirectory with exactly one faithful source copy named `kernel.py`. No function signatures or program bodies were modified during initialization.

## Source Universe

- NPBench canonical NumPy kernels: `54` from the upstream NPBench benchmark tree used during initialization
- Pythran raw test-case files: `131` from the upstream Pythran test-case tree used during initialization
- Pythran selected kernel families: `126`; when multiple implementations exist for the same kernel, the smallest source file is selected
- Raw file total before same-kernel de-duplication: `185`
- Selected source-family total before filtering: `180`

## Filtering Policy

Control flow, loop count, and source body size are not filters. Large or control-heavy programs remain in scope and should be tested faithfully.

Programs are excluded only when they are not faithful standalone kernel sources for this benchmark pool:

- try/except control flow
- external I/O in the kernel
- randomness or nondeterministic input generation inside the kernel
- non-NumPy external imports needed by the kernel
- no top-level callable kernel

Unsupported NumPy/operator families are not excluded. They are kept as `attempt_classify` programs so the porting pass can record the exact outcome.

## Counts

| Status | Total | NPBench | Pythran |
| --- | ---: | ---: | ---: |
| direct faithful candidates | `158` | `49` | `109` |
| attempt/classify candidates | `11` | `5` | `6` |
| early excluded | `11` | `0` | `11` |
| **Included** | `169` | `54` | `115` |
| **All** | `180` | `54` | `126` |

Included source pool after same-kernel de-duplication: `169 / 180`. Raw upstream file pool before de-duplication: `185`.


## Kernel Extraction Rule

The copied `kernel.py` files are trimmed source extracts, not full upstream files. The extraction keeps:

- selected kernel function(s) with original name, signature, and body text
- transitive helper functions/classes called by those kernels
- module constants referenced by retained code
- imports needed by retained code

The extraction removes unrelated top-level material only: runners, benchmark harnesses, data generators, unused alternate kernels, and scaffold code. It must not rename parameters, normalize signatures, simplify algorithms, replace unsupported operations, or otherwise rewrite the retained kernel/helper bodies.

Validation after extraction: `169` program folders, `169` `kernel.py` files, exactly one `kernel.py` per folder, and all `kernel.py` files pass Python parsing/bytecode compilation.

## Included Programs

| Folder | Status | Upstream source |
| --- | --- | --- |
| `npbench_azimint_hist_azimint_hist` | `attempt_classify` | `npbench:azimint_hist/azimint_hist_numpy.py` |
| `npbench_azimint_naive_azimint_naive` | `direct` | `npbench:azimint_naive/azimint_naive_numpy.py` |
| `npbench_cavity_flow_cavity_flow` | `direct` | `npbench:cavity_flow/cavity_flow_numpy.py` |
| `npbench_channel_flow_channel_flow` | `direct` | `npbench:channel_flow/channel_flow_numpy.py` |
| `npbench_compute_compute` | `direct` | `npbench:compute/compute_numpy.py` |
| `npbench_contour_integral_contour_integral` | `attempt_classify` | `npbench:contour_integral/contour_integral_numpy.py` |
| `npbench_crc16_crc16` | `direct` | `npbench:crc16/crc16_numpy.py` |
| `npbench_deep_learning_conv2d_bias_conv2d` | `direct` | `npbench:deep_learning/conv2d_bias/conv2d_numpy.py` |
| `npbench_deep_learning_lenet_lenet` | `direct` | `npbench:deep_learning/lenet/lenet_numpy.py` |
| `npbench_deep_learning_mlp_mlp` | `direct` | `npbench:deep_learning/mlp/mlp_numpy.py` |
| `npbench_deep_learning_resnet_resnet` | `direct` | `npbench:deep_learning/resnet/resnet_numpy.py` |
| `npbench_deep_learning_softmax_softmax` | `direct` | `npbench:deep_learning/softmax/softmax_numpy.py` |
| `npbench_go_fast_go_fast` | `direct` | `npbench:go_fast/go_fast_numpy.py` |
| `npbench_mandelbrot1_mandelbrot1` | `direct` | `npbench:mandelbrot1/mandelbrot1_numpy.py` |
| `npbench_mandelbrot2_mandelbrot2` | `direct` | `npbench:mandelbrot2/mandelbrot2_numpy.py` |
| `npbench_nbody_nbody` | `direct` | `npbench:nbody/nbody_numpy.py` |
| `npbench_polybench_adi_adi` | `direct` | `npbench:polybench/adi/adi_numpy.py` |
| `npbench_polybench_atax_atax` | `direct` | `npbench:polybench/atax/atax_numpy.py` |
| `npbench_polybench_bicg_bicg` | `direct` | `npbench:polybench/bicg/bicg_numpy.py` |
| `npbench_polybench_cholesky_cholesky` | `direct` | `npbench:polybench/cholesky/cholesky_numpy.py` |
| `npbench_polybench_cholesky2_cholesky2` | `attempt_classify` | `npbench:polybench/cholesky2/cholesky2_numpy.py` |
| `npbench_polybench_correlation_correlation` | `direct` | `npbench:polybench/correlation/correlation_numpy.py` |
| `npbench_polybench_covariance_covariance` | `direct` | `npbench:polybench/covariance/covariance_numpy.py` |
| `npbench_polybench_covariance2_covariance2` | `direct` | `npbench:polybench/covariance2/covariance2_numpy.py` |
| `npbench_polybench_deriche_deriche` | `direct` | `npbench:polybench/deriche/deriche_numpy.py` |
| `npbench_polybench_doitgen_doitgen` | `direct` | `npbench:polybench/doitgen/doitgen_numpy.py` |
| `npbench_polybench_durbin_durbin` | `direct` | `npbench:polybench/durbin/durbin_numpy.py` |
| `npbench_polybench_fdtd_2d_fdtd_2d` | `direct` | `npbench:polybench/fdtd_2d/fdtd_2d_numpy.py` |
| `npbench_polybench_floyd_warshall_floyd_warshall` | `direct` | `npbench:polybench/floyd_warshall/floyd_warshall_numpy.py` |
| `npbench_polybench_gemm_gemm` | `direct` | `npbench:polybench/gemm/gemm_numpy.py` |
| `npbench_polybench_gemver_gemver` | `attempt_classify` | `npbench:polybench/gemver/gemver_numpy.py` |
| `npbench_polybench_gesummv_gesummv` | `direct` | `npbench:polybench/gesummv/gesummv_numpy.py` |
| `npbench_polybench_gramschmidt_gramschmidt` | `direct` | `npbench:polybench/gramschmidt/gramschmidt_numpy.py` |
| `npbench_polybench_heat_3d_heat_3d` | `direct` | `npbench:polybench/heat_3d/heat_3d_numpy.py` |
| `npbench_polybench_jacobi_1d_jacobi_1d` | `direct` | `npbench:polybench/jacobi_1d/jacobi_1d_numpy.py` |
| `npbench_polybench_jacobi_2d_jacobi_2d` | `direct` | `npbench:polybench/jacobi_2d/jacobi_2d_numpy.py` |
| `npbench_polybench_k2mm_k2mm` | `direct` | `npbench:polybench/k2mm/k2mm_numpy.py` |
| `npbench_polybench_k3mm_k3mm` | `direct` | `npbench:polybench/k3mm/k3mm_numpy.py` |
| `npbench_polybench_lu_lu` | `direct` | `npbench:polybench/lu/lu_numpy.py` |
| `npbench_polybench_ludcmp_ludcmp` | `direct` | `npbench:polybench/ludcmp/ludcmp_numpy.py` |
| `npbench_polybench_mvt_mvt` | `direct` | `npbench:polybench/mvt/mvt_numpy.py` |
| `npbench_polybench_nussinov_nussinov` | `direct` | `npbench:polybench/nussinov/nussinov_numpy.py` |
| `npbench_polybench_seidel_2d_seidel_2d` | `direct` | `npbench:polybench/seidel_2d/seidel_2d_numpy.py` |
| `npbench_polybench_symm_symm` | `direct` | `npbench:polybench/symm/symm_numpy.py` |
| `npbench_polybench_syr2k_syr2k` | `direct` | `npbench:polybench/syr2k/syr2k_numpy.py` |
| `npbench_polybench_syrk_syrk` | `direct` | `npbench:polybench/syrk/syrk_numpy.py` |
| `npbench_polybench_trisolv_trisolv` | `direct` | `npbench:polybench/trisolv/trisolv_numpy.py` |
| `npbench_polybench_trmm_trmm` | `direct` | `npbench:polybench/trmm/trmm_numpy.py` |
| `npbench_pythran_arc_distance_arc_distance` | `direct` | `npbench:pythran/arc_distance/arc_distance_numpy.py` |
| `npbench_scattering_self_energies_scattering_self_energies` | `direct` | `npbench:scattering_self_energies/scattering_self_energies_numpy.py` |
| `npbench_spmv_spmv` | `direct` | `npbench:spmv/spmv_numpy.py` |
| `npbench_stockham_fft_stockham_fft` | `direct` | `npbench:stockham_fft/stockham_fft_numpy.py` |
| `npbench_weather_stencils_hdiff_hdiff` | `attempt_classify` | `npbench:weather_stencils/hdiff/hdiff_numpy.py` |
| `npbench_weather_stencils_vadv_vadv` | `direct` | `npbench:weather_stencils/vadv/vadv_numpy.py` |
| `pythran_allpairs` | `direct` | `pythran:allpairs.py` |
| `pythran_allpairs_distances` | `direct` | `pythran:allpairs_distances.py` |
| `pythran_allpairs_distances_loops` | `direct` | `pythran:allpairs_distances_loops.py` |
| `pythran_another_quicksort` | `direct` | `pythran:another_quicksort.py` |
| `pythran_approximated_callgraph` | `direct` | `pythran:approximated_callgraph.py` |
| `pythran_arc_distance` | `direct` | `pythran:arc_distance.py` |
| `pythran_babylonian` | `direct` | `pythran:babylonian.py` |
| `pythran_blacksholes` | `direct` | `pythran:blacksholes.py` |
| `pythran_bubble_sort` | `direct` | `pythran:bubble_sort.py` |
| `pythran_build_system` | `direct` | `pythran:build_system.py` |
| `pythran_calculate_u` | `direct` | `pythran:calculate_u.py` |
| `pythran_caxpy` | `direct` | `pythran:caxpy.py` |
| `pythran_ccopy` | `direct` | `pythran:ccopy.py` |
| `pythran_cdotc` | `direct` | `pythran:cdotc.py` |
| `pythran_cdotu` | `direct` | `pythran:cdotu.py` |
| `pythran_check_mask` | `direct` | `pythran:check_mask.py` |
| `pythran_clip` | `direct` | `pythran:clip.py` |
| `pythran_clip2` | `direct` | `pythran:clip2.py` |
| `pythran_collatz` | `attempt_classify` | `pythran:collatz.py` |
| `pythran_collatz_modified` | `direct` | `pythran:collatz_modified.py` |
| `pythran_comp_unrolling` | `direct` | `pythran:comp_unrolling.py` |
| `pythran_compute_subpix_2d_gaussian2` | `attempt_classify` | `pythran:compute_subpix_2d_gaussian2.py` |
| `pythran_convnet` | `direct` | `pythran:convnet.py` |
| `pythran_create_grid` | `direct` | `pythran:create_grid.py` |
| `pythran_cronbach` | `direct` | `pythran:cronbach.py` |
| `pythran_crotg` | `direct` | `pythran:crotg.py` |
| `pythran_d2q9_nxnyns` | `direct` | `pythran:d2q9_nxnyns.py` |
| `pythran_deriv` | `direct` | `pythran:deriv.py` |
| `pythran_descent` | `direct` | `pythran:descent.py` |
| `pythran_deuxd_convolution` | `direct` | `pythran:deuxd_convolution.py` |
| `pythran_diffusion` | `direct` | `pythran:diffusion_numpy.py` |
| `pythran_emin` | `direct` | `pythran:emin.py` |
| `pythran_euclidean_distance_square` | `direct` | `pythran:euclidean_distance_square.py` |
| `pythran_euler_challenge13` | `direct` | `pythran:euler_challenge13.py` |
| `pythran_euler_challenge14` | `direct` | `pythran:euler_challenge14.py` |
| `pythran_extrema` | `direct` | `pythran:extrema.py` |
| `pythran_factorize_naive` | `direct` | `pythran:factorize_naive.py` |
| `pythran_fannkuch` | `direct` | `pythran:fannkuch.py` |
| `pythran_fbcorr` | `direct` | `pythran:fbcorr_numpy.py` |
| `pythran_fdtd` | `direct` | `pythran:fdtd.py` |
| `pythran_fft` | `direct` | `pythran:fft.py` |
| `pythran_fibo` | `direct` | `pythran:fibo.py` |
| `pythran_fibo_seq` | `direct` | `pythran:fibo_seq.py` |
| `pythran_gauss` | `direct` | `pythran:gauss.py` |
| `pythran_goodexpomeasure` | `direct` | `pythran:goodExpoMeasure.py` |
| `pythran_grouping` | `attempt_classify` | `pythran:grouping.py` |
| `pythran_growcut` | `direct` | `pythran:growcut.py` |
| `pythran_guerre` | `direct` | `pythran:guerre.py` |
| `pythran_harris` | `direct` | `pythran:harris.py` |
| `pythran_hasting` | `direct` | `pythran:hasting.py` |
| `pythran_histogram` | `direct` | `pythran:histogram.py` |
| `pythran_hyantes_core` | `direct` | `pythran:hyantes_core.py` |
| `pythran_insertion_sort` | `direct` | `pythran:insertion_sort.py` |
| `pythran_ising` | `direct` | `pythran:ising.py` |
| `pythran_l2norm` | `direct` | `pythran:l2norm.py` |
| `pythran_lap` | `direct` | `pythran:lap.py` |
| `pythran_lapl2d` | `direct` | `pythran:lapl2d.py` |
| `pythran_lapl3d` | `direct` | `pythran:lapl3d.py` |
| `pythran_laplace` | `direct` | `pythran:laplace.py` |
| `pythran_ldpc_decoder` | `direct` | `pythran:ldpc_decoder.py` |
| `pythran_ln` | `direct` | `pythran:ln.py` |
| `pythran_log_likelihood` | `direct` | `pythran:log_likelihood.py` |
| `pythran_lttb_pt` | `direct` | `pythran:lttb_pt.py` |
| `pythran_lu` | `direct` | `pythran:lu.py` |
| `pythran_make_circle_two_points` | `direct` | `pythran:make_circle_two_points.py` |
| `pythran_make_decision` | `direct` | `pythran:make_decision.py` |
| `pythran_mandel` | `direct` | `pythran:mandel.py` |
| `pythran_matmul` | `direct` | `pythran:matmul.py` |
| `pythran_matrix_class_distance` | `direct` | `pythran:matrix_class_distance.py` |
| `pythran_monte_carlo` | `direct` | `pythran:monte_carlo.py` |
| `pythran_morphology` | `direct` | `pythran:morphology.py` |
| `pythran_mulmod` | `direct` | `pythran:mulmod.py` |
| `pythran_multi_export` | `direct` | `pythran:multi_export.py` |
| `pythran_multitype` | `direct` | `pythran:multitype.py` |
| `pythran_nd_local_maxima` | `direct` | `pythran:nd_local_maxima.py` |
| `pythran_nqueens` | `direct` | `pythran:nqueens.py` |
| `pythran_pairwise` | `direct` | `pythran:pairwise_numpy.py` |
| `pythran_periodic_bc` | `direct` | `pythran:periodic_bc.py` |
| `pythran_periodic_dist` | `direct` | `pythran:periodic_dist.py` |
| `pythran_perm` | `direct` | `pythran:perm.py` |
| `pythran_pivot` | `direct` | `pythran:pivot.py` |
| `pythran_primes_sieve` | `direct` | `pythran:primes_sieve.py` |
| `pythran_primes_sieve2` | `direct` | `pythran:primes_sieve2.py` |
| `pythran_projection_simplex` | `attempt_classify` | `pythran:projection_simplex.py` |
| `pythran_pselect` | `direct` | `pythran:pselect.py` |
| `pythran_pythagorean_triples` | `direct` | `pythran:pythagorean_triples.py` |
| `pythran_queens_numba` | `direct` | `pythran:queens_numba.py` |
| `pythran_quicksort` | `direct` | `pythran:quicksort.py` |
| `pythran_ramsurf` | `direct` | `pythran:ramsurf.py` |
| `pythran_rand_mat_stat` | `direct` | `pythran:rand_mat_stat.py` |
| `pythran_rc4` | `direct` | `pythran:rc4.py` |
| `pythran_repeating` | `direct` | `pythran:repeating.py` |
| `pythran_reverse_cumsum` | `attempt_classify` | `pythran:reverse_cumsum.py` |
| `pythran_roman_decode` | `direct` | `pythran:roman_decode.py` |
| `pythran_rosen` | `direct` | `pythran:rosen.py` |
| `pythran_scrabble` | `direct` | `pythran:scrabble.py` |
| `pythran_sexy_primes` | `direct` | `pythran:sexy_primes.py` |
| `pythran_shallow_water` | `direct` | `pythran:shallow_water.py` |
| `pythran_slowparts` | `direct` | `pythran:slowparts.py` |
| `pythran_smoothing` | `direct` | `pythran:smoothing.py` |
| `pythran_sobelfilter` | `direct` | `pythran:sobelfilter.py` |
| `pythran_specialconvolve` | `direct` | `pythran:specialconvolve.py` |
| `pythran_spectraldns` | `direct` | `pythran:spectralDNS.py` |
| `pythran_stone` | `direct` | `pythran:stone.py` |
| `pythran_sum_primes` | `direct` | `pythran:sum_primes.py` |
| `pythran_sumarray3d` | `direct` | `pythran:sumarray3d.py` |
| `pythran_tax` | `direct` | `pythran:tax.py` |
| `pythran_train_eq` | `direct` | `pythran:train_eq.py` |
| `pythran_train_equalizer` | `direct` | `pythran:train_equalizer.py` |
| `pythran_value_index` | `attempt_classify` | `pythran:value_index.py` |
| `pythran_vibr_energy` | `direct` | `pythran:vibr_energy.py` |
| `pythran_wave_simulation` | `direct` | `pythran:wave_simulation.py` |
| `pythran_wdist` | `direct` | `pythran:wdist.py` |
| `pythran_weights` | `direct` | `pythran:weights.py` |
| `pythran_zero` | `direct` | `pythran:zero.py` |

## Early Exclusions

| Source | Reason |
| --- | --- |
| `pythran:arc_distance_list.py` | `randomness` |
| `pythran:brownian.py` | `randomness` |
| `pythran:count_perm_sig.py` | `randomness` |
| `pythran:empirical.py` | `external_io_call` |
| `pythran:frequent_itemsets.py` | `external_io_call` |
| `pythran:grayscott.py` | `randomness` |
| `pythran:julia_pure_python.py` | `external_non_numpy_import:time` |
| `pythran:kmeans.py` | `randomness` |
| `pythran:loopy_jacob.py` | `try_except` |
| `pythran:monte_carlo_pricer.py` | `randomness` |
| `pythran:pi_buffon.py` | `randomness` |
