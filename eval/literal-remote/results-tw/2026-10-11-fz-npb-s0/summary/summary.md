# Epoch 2026-10-11-fz-npb-s0

Ratios are Zinnia / baseline of per-case medians; below 1 means Zinnia is faster.
Only cases with status `ok` (proved, verified, outputs equal to NumPy, tampered proof rejected) enter a comparison.

`zinnia_noir` `end_to_end_s` witness step: noir-execute (28 cases); `end_to_end_nargo_s` uses `nargo execute`.
`noir_hw` `end_to_end_s` witness step: noir-execute (28 cases); `end_to_end_nargo_s` uses `nargo execute`.
`noir_lit` `end_to_end_s` witness step: noir-execute (28 cases); `end_to_end_nargo_s` uses `nargo execute`.

## Coverage

| system | cases | ok | other statuses |
|---|---|---|---|
| zinnia_halo2 | 28 | 28 | - |
| zinnia_noir | 28 | 28 | - |
| halo2_hw | 28 | 28 | - |
| noir_hw | 28 | 28 | - |
| halo2_lit | 28 | 28 | - |
| noir_lit | 28 | 28 | - |

## Headline: all-supported subset without copy-only cases

27 cases completed by every system that do more than copy inputs: npbench_compute_compute__compute, npbench_deep_learning_mlp_mlp__mlp, npbench_go_fast_go_fast__go_fast, npbench_polybench_cholesky_cholesky__kernel, npbench_polybench_durbin_durbin__kernel, npbench_polybench_gemver_gemver__kernel, npbench_polybench_k2mm_k2mm__kernel, npbench_polybench_ludcmp_ludcmp__kernel, npbench_polybench_syr2k_syr2k__kernel, npbench_polybench_trmm_trmm__kernel, npbench_weather_stencils_hdiff_hdiff__hdiff, pythran_allpairs__sqr_dists_loops, pythran_ccopy__CCOPY, pythran_cronbach__cronbach, pythran_deuxd_convolution__conv, pythran_fibo_seq__fibo, pythran_hasting__fweb, pythran_l2norm__l2_norm, pythran_ln__ln, pythran_lu__factorMatrix1, pythran_multitype__times, pythran_periodic_dist__dist, pythran_pythagorean_triples__next_pythagorean_triples, pythran_smoothing__smoothing, pythran_spectraldns__cross2, pythran_value_index__tri, pythran_weights__weights

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 27 | 0.953 | [0.898, 1.006] | 17 / 10 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 27 | 0.968 | [0.914, 1.020] | 16 / 11 |
| zinnia_halo2 | halo2_hw | verify_s | 27 | 0.913 | [0.865, 0.963] | 20 / 7 |
| zinnia_noir | noir_hw | prove_core_s | 27 | 1.042 | [1.007, 1.077] | 11 / 16 |
| zinnia_noir | noir_hw | end_to_end_s | 27 | 1.048 | [1.015, 1.081] | 9 / 18 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 27 | 1.139 | [1.062, 1.232] | 8 / 19 |
| zinnia_noir | noir_hw | verify_s | 27 | 0.984 | [0.905, 1.074] | 14 / 13 |
| zinnia_halo2 | noir_hw | prove_core_s | 27 | 2.005 | [1.538, 2.568] | 4 / 23 |
| zinnia_halo2 | noir_hw | end_to_end_s | 27 | 1.880 | [1.456, 2.380] | 4 / 23 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 27 | 1.069 | [0.767, 1.457] | 11 / 16 |
| zinnia_halo2 | noir_hw | verify_s | 27 | 0.982 | [0.771, 1.250] | 16 / 11 |
| zinnia_halo2 | halo2_lit | prove_core_s | 27 | 0.617 | [0.459, 0.763] | 26 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 27 | 0.627 | [0.466, 0.775] | 26 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 27 | 0.699 | [0.559, 0.814] | 26 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 27 | 1.592 | [1.120, 2.182] | 6 / 21 |
| zinnia_halo2 | noir_lit | end_to_end_s | 27 | 1.493 | [1.053, 2.036] | 6 / 21 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 27 | 0.881 | [0.598, 1.251] | 12 / 15 |
| zinnia_halo2 | noir_lit | verify_s | 27 | 1.009 | [0.787, 1.283] | 12 / 15 |
| zinnia_noir | halo2_hw | prove_core_s | 27 | 0.495 | [0.393, 0.637] | 22 / 5 |
| zinnia_noir | halo2_hw | end_to_end_s | 27 | 0.540 | [0.435, 0.684] | 22 / 5 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 27 | 1.031 | [0.798, 1.372] | 16 / 11 |
| zinnia_noir | halo2_hw | verify_s | 27 | 0.915 | [0.722, 1.166] | 16 / 11 |
| zinnia_noir | halo2_lit | prove_core_s | 27 | 0.321 | [0.210, 0.465] | 23 / 4 |
| zinnia_noir | halo2_lit | end_to_end_s | 27 | 0.349 | [0.231, 0.501] | 23 / 4 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 27 | 0.668 | [0.431, 1.000] | 18 / 9 |
| zinnia_noir | halo2_lit | verify_s | 27 | 0.701 | [0.493, 0.963] | 19 / 8 |
| zinnia_noir | noir_lit | prove_core_s | 27 | 0.827 | [0.591, 1.008] | 15 / 12 |
| zinnia_noir | noir_lit | end_to_end_s | 27 | 0.832 | [0.593, 1.016] | 14 / 13 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 27 | 0.938 | [0.685, 1.144] | 12 / 15 |
| zinnia_noir | noir_lit | verify_s | 27 | 1.012 | [0.926, 1.106] | 12 / 15 |

## Copy-only cases (reported apart)

1 all-supported cases that only copy inputs: pythran_multi_export__a

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 1 | 0.955 | [0.955, 0.955] | 1 / 0 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 1 | 0.956 | [0.956, 0.956] | 1 / 0 |
| zinnia_halo2 | halo2_hw | verify_s | 1 | 1.297 | [1.297, 1.297] | 0 / 1 |
| zinnia_noir | noir_hw | prove_core_s | 1 | 0.878 | [0.878, 0.878] | 1 / 0 |
| zinnia_noir | noir_hw | end_to_end_s | 1 | 0.886 | [0.886, 0.886] | 1 / 0 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 1 | 0.907 | [0.907, 0.907] | 1 / 0 |
| zinnia_noir | noir_hw | verify_s | 1 | 1.034 | [1.034, 1.034] | 0 / 1 |
| zinnia_halo2 | noir_hw | prove_core_s | 1 | 0.245 | [0.245, 0.245] | 1 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_s | 1 | 0.242 | [0.242, 0.242] | 1 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 1 | 0.121 | [0.121, 0.121] | 1 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 1 | 0.407 | [0.407, 0.407] | 1 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 1 | 0.890 | [0.890, 0.890] | 1 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 1 | 0.891 | [0.891, 0.891] | 1 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 1 | 1.191 | [1.191, 1.191] | 0 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 1 | 0.250 | [0.250, 0.250] | 1 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_s | 1 | 0.247 | [0.247, 0.247] | 1 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 1 | 0.125 | [0.125, 0.125] | 1 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 1 | 0.409 | [0.409, 0.409] | 1 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 1 | 3.419 | [3.419, 3.419] | 0 / 1 |
| zinnia_noir | halo2_hw | end_to_end_s | 1 | 3.500 | [3.500, 3.500] | 0 / 1 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 1 | 7.191 | [7.191, 7.191] | 0 / 1 |
| zinnia_noir | halo2_hw | verify_s | 1 | 3.293 | [3.293, 3.293] | 0 / 1 |
| zinnia_noir | halo2_lit | prove_core_s | 1 | 3.187 | [3.187, 3.187] | 0 / 1 |
| zinnia_noir | halo2_lit | end_to_end_s | 1 | 3.261 | [3.261, 3.261] | 0 / 1 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 1 | 6.700 | [6.700, 6.700] | 0 / 1 |
| zinnia_noir | halo2_lit | verify_s | 1 | 3.022 | [3.022, 3.022] | 0 / 1 |
| zinnia_noir | noir_lit | prove_core_s | 1 | 0.894 | [0.894, 0.894] | 1 / 0 |
| zinnia_noir | noir_lit | end_to_end_s | 1 | 0.906 | [0.906, 0.906] | 1 / 0 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 1 | 0.941 | [0.941, 0.941] | 1 / 0 |
| zinnia_noir | noir_lit | verify_s | 1 | 1.039 | [1.039, 1.039] | 0 / 1 |

## Group npbench: headline cases

11 headline cases of group npbench: npbench_compute_compute__compute, npbench_deep_learning_mlp_mlp__mlp, npbench_go_fast_go_fast__go_fast, npbench_polybench_cholesky_cholesky__kernel, npbench_polybench_durbin_durbin__kernel, npbench_polybench_gemver_gemver__kernel, npbench_polybench_k2mm_k2mm__kernel, npbench_polybench_ludcmp_ludcmp__kernel, npbench_polybench_syr2k_syr2k__kernel, npbench_polybench_trmm_trmm__kernel, npbench_weather_stencils_hdiff_hdiff__hdiff

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 11 | 0.963 | [0.921, 1.003] | 8 / 3 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 11 | 0.977 | [0.934, 1.019] | 7 / 4 |
| zinnia_halo2 | halo2_hw | verify_s | 11 | 0.929 | [0.866, 0.996] | 8 / 3 |
| zinnia_noir | noir_hw | prove_core_s | 11 | 1.073 | [1.006, 1.141] | 4 / 7 |
| zinnia_noir | noir_hw | end_to_end_s | 11 | 1.075 | [1.017, 1.137] | 4 / 7 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 11 | 1.165 | [1.064, 1.303] | 2 / 9 |
| zinnia_noir | noir_hw | verify_s | 11 | 0.976 | [0.878, 1.078] | 5 / 6 |
| zinnia_halo2 | noir_hw | prove_core_s | 11 | 2.746 | [2.166, 3.442] | 0 / 11 |
| zinnia_halo2 | noir_hw | end_to_end_s | 11 | 2.551 | [2.041, 3.137] | 0 / 11 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 11 | 1.602 | [1.197, 2.112] | 2 / 9 |
| zinnia_halo2 | noir_hw | verify_s | 11 | 1.175 | [0.912, 1.529] | 5 / 6 |
| zinnia_halo2 | halo2_lit | prove_core_s | 11 | 0.788 | [0.734, 0.844] | 11 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 11 | 0.799 | [0.746, 0.855] | 11 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 11 | 0.785 | [0.731, 0.838] | 11 / 0 |
| zinnia_halo2 | noir_lit | prove_core_s | 11 | 2.499 | [2.031, 3.124] | 0 / 11 |
| zinnia_halo2 | noir_lit | end_to_end_s | 11 | 2.351 | [1.941, 2.887] | 0 / 11 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 11 | 1.511 | [1.158, 1.985] | 2 / 9 |
| zinnia_halo2 | noir_lit | verify_s | 11 | 1.239 | [0.987, 1.582] | 3 / 8 |
| zinnia_noir | halo2_hw | prove_core_s | 11 | 0.376 | [0.320, 0.443] | 11 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 11 | 0.412 | [0.358, 0.477] | 11 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 11 | 0.710 | [0.607, 0.855] | 9 / 2 |
| zinnia_noir | halo2_hw | verify_s | 11 | 0.773 | [0.619, 0.961] | 8 / 3 |
| zinnia_noir | halo2_lit | prove_core_s | 11 | 0.308 | [0.248, 0.384] | 11 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 11 | 0.337 | [0.276, 0.414] | 11 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 11 | 0.581 | [0.464, 0.745] | 9 / 2 |
| zinnia_noir | halo2_lit | verify_s | 11 | 0.653 | [0.525, 0.797] | 10 / 1 |
| zinnia_noir | noir_lit | prove_core_s | 11 | 0.976 | [0.876, 1.089] | 5 / 6 |
| zinnia_noir | noir_lit | end_to_end_s | 11 | 0.991 | [0.892, 1.100] | 5 / 6 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 11 | 1.099 | [0.988, 1.246] | 4 / 7 |
| zinnia_noir | noir_lit | verify_s | 11 | 1.030 | [0.904, 1.181] | 5 / 6 |

## Group pythran: headline cases

16 headline cases of group pythran: pythran_allpairs__sqr_dists_loops, pythran_ccopy__CCOPY, pythran_cronbach__cronbach, pythran_deuxd_convolution__conv, pythran_fibo_seq__fibo, pythran_hasting__fweb, pythran_l2norm__l2_norm, pythran_ln__ln, pythran_lu__factorMatrix1, pythran_multitype__times, pythran_periodic_dist__dist, pythran_pythagorean_triples__next_pythagorean_triples, pythran_smoothing__smoothing, pythran_spectraldns__cross2, pythran_value_index__tri, pythran_weights__weights

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 16 | 0.947 | [0.861, 1.032] | 9 / 7 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 16 | 0.962 | [0.878, 1.046] | 9 / 7 |
| zinnia_halo2 | halo2_hw | verify_s | 16 | 0.901 | [0.837, 0.974] | 12 / 4 |
| zinnia_noir | noir_hw | prove_core_s | 16 | 1.021 | [0.986, 1.055] | 7 / 9 |
| zinnia_noir | noir_hw | end_to_end_s | 16 | 1.030 | [0.994, 1.064] | 5 / 11 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 16 | 1.121 | [1.016, 1.255] | 6 / 10 |
| zinnia_noir | noir_hw | verify_s | 16 | 0.990 | [0.873, 1.127] | 9 / 7 |
| zinnia_halo2 | noir_hw | prove_core_s | 16 | 1.615 | [1.108, 2.342] | 4 / 12 |
| zinnia_halo2 | noir_hw | end_to_end_s | 16 | 1.524 | [1.058, 2.178] | 4 / 12 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 16 | 0.810 | [0.505, 1.276] | 9 / 7 |
| zinnia_halo2 | noir_hw | verify_s | 16 | 0.867 | [0.610, 1.250] | 11 / 5 |
| zinnia_halo2 | halo2_lit | prove_core_s | 16 | 0.522 | [0.329, 0.740] | 15 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 16 | 0.530 | [0.335, 0.750] | 15 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 16 | 0.645 | [0.452, 0.829] | 15 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 16 | 1.168 | [0.692, 1.896] | 6 / 10 |
| zinnia_halo2 | noir_lit | end_to_end_s | 16 | 1.092 | [0.646, 1.756] | 6 / 10 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 16 | 0.608 | [0.352, 1.033] | 10 / 6 |
| zinnia_halo2 | noir_lit | verify_s | 16 | 0.876 | [0.606, 1.277] | 9 / 7 |
| zinnia_noir | halo2_hw | prove_core_s | 16 | 0.599 | [0.418, 0.867] | 11 / 5 |
| zinnia_noir | halo2_hw | end_to_end_s | 16 | 0.650 | [0.465, 0.923] | 11 / 5 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 16 | 1.332 | [0.904, 2.014] | 7 / 9 |
| zinnia_noir | halo2_hw | verify_s | 16 | 1.029 | [0.711, 1.474] | 8 / 8 |
| zinnia_noir | halo2_lit | prove_core_s | 16 | 0.330 | [0.165, 0.600] | 12 / 4 |
| zinnia_noir | halo2_lit | end_to_end_s | 16 | 0.358 | [0.181, 0.640] | 12 / 4 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 16 | 0.734 | [0.358, 1.393] | 9 / 7 |
| zinnia_noir | halo2_lit | verify_s | 16 | 0.736 | [0.423, 1.217] | 9 / 7 |
| zinnia_noir | noir_lit | prove_core_s | 16 | 0.738 | [0.431, 1.000] | 10 / 6 |
| zinnia_noir | noir_lit | end_to_end_s | 16 | 0.738 | [0.428, 1.003] | 9 / 7 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 16 | 0.842 | [0.510, 1.148] | 8 / 8 |
| zinnia_noir | noir_lit | verify_s | 16 | 1.000 | [0.886, 1.125] | 7 / 9 |

## All-supported subset

28 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): npbench_compute_compute__compute, npbench_deep_learning_mlp_mlp__mlp, npbench_go_fast_go_fast__go_fast, npbench_polybench_cholesky_cholesky__kernel, npbench_polybench_durbin_durbin__kernel, npbench_polybench_gemver_gemver__kernel, npbench_polybench_k2mm_k2mm__kernel, npbench_polybench_ludcmp_ludcmp__kernel, npbench_polybench_syr2k_syr2k__kernel, npbench_polybench_trmm_trmm__kernel, npbench_weather_stencils_hdiff_hdiff__hdiff, pythran_allpairs__sqr_dists_loops, pythran_ccopy__CCOPY, pythran_cronbach__cronbach, pythran_deuxd_convolution__conv, pythran_fibo_seq__fibo, pythran_hasting__fweb, pythran_l2norm__l2_norm, pythran_ln__ln, pythran_lu__factorMatrix1, pythran_multi_export__a, pythran_multitype__times, pythran_periodic_dist__dist, pythran_pythagorean_triples__next_pythagorean_triples, pythran_smoothing__smoothing, pythran_spectraldns__cross2, pythran_value_index__tri, pythran_weights__weights

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 28 | 0.953 | [0.898, 1.004] | 18 / 10 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 28 | 0.968 | [0.914, 1.018] | 17 / 11 |
| zinnia_halo2 | halo2_hw | verify_s | 28 | 0.924 | [0.874, 0.979] | 20 / 8 |
| zinnia_noir | noir_hw | prove_core_s | 28 | 1.035 | [1.000, 1.073] | 12 / 16 |
| zinnia_noir | noir_hw | end_to_end_s | 28 | 1.042 | [1.009, 1.077] | 10 / 18 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 28 | 1.130 | [1.053, 1.222] | 9 / 19 |
| zinnia_noir | noir_hw | verify_s | 28 | 0.986 | [0.909, 1.073] | 14 / 14 |
| zinnia_halo2 | noir_hw | prove_core_s | 28 | 1.860 | [1.380, 2.443] | 5 / 23 |
| zinnia_halo2 | noir_hw | end_to_end_s | 28 | 1.747 | [1.312, 2.270] | 5 / 23 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 28 | 0.989 | [0.694, 1.378] | 12 / 16 |
| zinnia_halo2 | noir_hw | verify_s | 28 | 0.951 | [0.747, 1.212] | 17 / 11 |
| zinnia_halo2 | halo2_lit | prove_core_s | 28 | 0.625 | [0.467, 0.769] | 27 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 28 | 0.635 | [0.474, 0.780] | 27 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 28 | 0.712 | [0.568, 0.831] | 26 / 2 |
| zinnia_halo2 | noir_lit | prove_core_s | 28 | 1.490 | [1.016, 2.073] | 7 / 21 |
| zinnia_halo2 | noir_lit | end_to_end_s | 28 | 1.400 | [0.957, 1.937] | 7 / 21 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 28 | 0.822 | [0.545, 1.186] | 13 / 15 |
| zinnia_halo2 | noir_lit | verify_s | 28 | 0.977 | [0.767, 1.244] | 13 / 15 |
| zinnia_noir | halo2_hw | prove_core_s | 28 | 0.531 | [0.413, 0.698] | 22 / 6 |
| zinnia_noir | halo2_hw | end_to_end_s | 28 | 0.577 | [0.456, 0.749] | 22 / 6 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 28 | 1.105 | [0.840, 1.495] | 16 / 12 |
| zinnia_noir | halo2_hw | verify_s | 28 | 0.958 | [0.752, 1.222] | 16 / 12 |
| zinnia_noir | halo2_lit | prove_core_s | 28 | 0.348 | [0.225, 0.514] | 23 / 5 |
| zinnia_noir | halo2_lit | end_to_end_s | 28 | 0.378 | [0.247, 0.552] | 23 / 5 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 28 | 0.725 | [0.465, 1.098] | 18 / 10 |
| zinnia_noir | halo2_lit | verify_s | 28 | 0.739 | [0.522, 1.010] | 19 / 9 |
| zinnia_noir | noir_lit | prove_core_s | 28 | 0.829 | [0.599, 1.003] | 16 / 12 |
| zinnia_noir | noir_lit | end_to_end_s | 28 | 0.835 | [0.602, 1.011] | 15 / 13 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 28 | 0.938 | [0.689, 1.136] | 13 / 15 |
| zinnia_noir | noir_lit | verify_s | 28 | 1.013 | [0.931, 1.106] | 12 / 16 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 28 | 0.953 | [0.898, 1.004] | 18 / 10 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 28 | 0.968 | [0.914, 1.018] | 17 / 11 |
| zinnia_halo2 | halo2_hw | verify_s | 28 | 0.924 | [0.874, 0.979] | 20 / 8 |
| zinnia_noir | noir_hw | prove_core_s | 28 | 1.035 | [1.000, 1.073] | 12 / 16 |
| zinnia_noir | noir_hw | end_to_end_s | 28 | 1.042 | [1.009, 1.077] | 10 / 18 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 28 | 1.130 | [1.053, 1.222] | 9 / 19 |
| zinnia_noir | noir_hw | verify_s | 28 | 0.986 | [0.909, 1.073] | 14 / 14 |
| zinnia_halo2 | noir_hw | prove_core_s | 28 | 1.860 | [1.380, 2.443] | 5 / 23 |
| zinnia_halo2 | noir_hw | end_to_end_s | 28 | 1.747 | [1.312, 2.270] | 5 / 23 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 28 | 0.989 | [0.694, 1.378] | 12 / 16 |
| zinnia_halo2 | noir_hw | verify_s | 28 | 0.951 | [0.747, 1.212] | 17 / 11 |
| zinnia_halo2 | halo2_lit | prove_core_s | 28 | 0.625 | [0.467, 0.769] | 27 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 28 | 0.635 | [0.474, 0.780] | 27 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 28 | 0.712 | [0.568, 0.831] | 26 / 2 |
| zinnia_halo2 | noir_lit | prove_core_s | 28 | 1.490 | [1.016, 2.073] | 7 / 21 |
| zinnia_halo2 | noir_lit | end_to_end_s | 28 | 1.400 | [0.957, 1.937] | 7 / 21 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 28 | 0.822 | [0.545, 1.186] | 13 / 15 |
| zinnia_halo2 | noir_lit | verify_s | 28 | 0.977 | [0.767, 1.244] | 13 / 15 |
| zinnia_noir | halo2_hw | prove_core_s | 28 | 0.531 | [0.413, 0.698] | 22 / 6 |
| zinnia_noir | halo2_hw | end_to_end_s | 28 | 0.577 | [0.456, 0.749] | 22 / 6 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 28 | 1.105 | [0.840, 1.495] | 16 / 12 |
| zinnia_noir | halo2_hw | verify_s | 28 | 0.958 | [0.752, 1.222] | 16 / 12 |
| zinnia_noir | halo2_lit | prove_core_s | 28 | 0.348 | [0.225, 0.514] | 23 / 5 |
| zinnia_noir | halo2_lit | end_to_end_s | 28 | 0.378 | [0.247, 0.552] | 23 / 5 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 28 | 0.725 | [0.465, 1.098] | 18 / 10 |
| zinnia_noir | halo2_lit | verify_s | 28 | 0.739 | [0.522, 1.010] | 19 / 9 |
| zinnia_noir | noir_lit | prove_core_s | 28 | 0.829 | [0.599, 1.003] | 16 / 12 |
| zinnia_noir | noir_lit | end_to_end_s | 28 | 0.835 | [0.602, 1.011] | 15 / 13 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 28 | 0.938 | [0.689, 1.136] | 13 / 15 |
| zinnia_noir | noir_lit | verify_s | 28 | 1.013 | [0.931, 1.106] | 12 / 16 |

## Excluded cases

(none)

## Cases without a program

No program for the case on that system (e.g. no handwritten baseline yet); not a failure.

(none)

## Failures by system

