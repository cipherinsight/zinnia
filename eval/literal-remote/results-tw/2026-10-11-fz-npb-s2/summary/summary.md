# Epoch 2026-10-11-fz-npb-s2

Ratios are Zinnia / baseline of per-case medians; below 1 means Zinnia is faster.
Only cases with status `ok` (proved, verified, outputs equal to NumPy, tampered proof rejected) enter a comparison.

`zinnia_noir` `end_to_end_s` witness step: noir-execute (27 cases); `end_to_end_nargo_s` uses `nargo execute`.
`noir_hw` `end_to_end_s` witness step: noir-execute (27 cases); `end_to_end_nargo_s` uses `nargo execute`.
`noir_lit` `end_to_end_s` witness step: noir-execute (27 cases); `end_to_end_nargo_s` uses `nargo execute`.

## Coverage

| system | cases | ok | other statuses |
|---|---|---|---|
| zinnia_halo2 | 27 | 27 | - |
| zinnia_noir | 27 | 27 | - |
| halo2_hw | 27 | 27 | - |
| noir_hw | 27 | 27 | - |
| halo2_lit | 27 | 27 | - |
| noir_lit | 27 | 27 | - |

## Headline: all-supported subset without copy-only cases

25 cases completed by every system that do more than copy inputs: npbench_deep_learning_conv2d_bias_conv2d__conv2d_bias, npbench_deep_learning_softmax_softmax__softmax, npbench_polybench_bicg_bicg__kernel, npbench_polybench_deriche_deriche__kernel, npbench_polybench_gemm_gemm__kernel, npbench_polybench_gramschmidt_gramschmidt__kernel, npbench_polybench_lu_lu__kernel, npbench_polybench_symm_symm__kernel, npbench_polybench_trisolv_trisolv__kernel, npbench_scattering_self_energies_scattering_self_energies__scattering_self_energies, pythran_allpairs__sqr_dists, pythran_bubble_sort__bubble_sort, pythran_d2q9_nxnyns__one_time_step, pythran_fbcorr__fbcorr, pythran_harris__harris, pythran_ising__ising_step, pythran_lapl3d__laplacien, pythran_lu__factorMatrix0, pythran_matmul__matrix_multiply, pythran_multitype__check2, pythran_projection_simplex__projection_simplex, pythran_rosen__rosen, pythran_spectraldns__add_pressure_diffusion, pythran_tax__tax, pythran_wdist__slow_wdist

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 25 | 0.928 | [0.879, 0.977] | 17 / 8 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 25 | 0.946 | [0.895, 0.997] | 15 / 10 |
| zinnia_halo2 | halo2_hw | verify_s | 25 | 0.982 | [0.902, 1.087] | 18 / 7 |
| zinnia_noir | noir_hw | prove_core_s | 25 | 0.970 | [0.926, 1.008] | 12 / 13 |
| zinnia_noir | noir_hw | end_to_end_s | 25 | 0.986 | [0.942, 1.025] | 9 / 16 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 25 | 1.116 | [1.038, 1.205] | 8 / 17 |
| zinnia_noir | noir_hw | verify_s | 25 | 0.975 | [0.927, 1.029] | 17 / 8 |
| zinnia_halo2 | noir_hw | prove_core_s | 25 | 2.209 | [1.620, 2.955] | 2 / 23 |
| zinnia_halo2 | noir_hw | end_to_end_s | 25 | 2.034 | [1.520, 2.653] | 2 / 23 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 25 | 1.221 | [0.833, 1.751] | 11 / 14 |
| zinnia_halo2 | noir_hw | verify_s | 25 | 1.364 | [0.994, 1.982] | 10 / 15 |
| zinnia_halo2 | halo2_lit | prove_core_s | 25 | 0.750 | [0.667, 0.831] | 23 / 2 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 25 | 0.764 | [0.680, 0.845] | 23 / 2 |
| zinnia_halo2 | halo2_lit | verify_s | 25 | 0.867 | [0.788, 0.953] | 18 / 7 |
| zinnia_halo2 | noir_lit | prove_core_s | 25 | 1.939 | [1.349, 2.723] | 4 / 21 |
| zinnia_halo2 | noir_lit | end_to_end_s | 25 | 1.772 | [1.250, 2.446] | 4 / 21 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 25 | 1.081 | [0.721, 1.600] | 12 / 13 |
| zinnia_halo2 | noir_lit | verify_s | 25 | 1.360 | [0.992, 1.968] | 10 / 15 |
| zinnia_noir | halo2_hw | prove_core_s | 25 | 0.408 | [0.312, 0.543] | 24 / 1 |
| zinnia_noir | halo2_hw | end_to_end_s | 25 | 0.459 | [0.362, 0.594] | 24 / 1 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 25 | 0.865 | [0.646, 1.186] | 14 / 11 |
| zinnia_noir | halo2_hw | verify_s | 25 | 0.702 | [0.511, 0.946] | 15 / 10 |
| zinnia_noir | halo2_lit | prove_core_s | 25 | 0.330 | [0.240, 0.458] | 24 / 1 |
| zinnia_noir | halo2_lit | end_to_end_s | 25 | 0.371 | [0.279, 0.502] | 24 / 1 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 25 | 0.699 | [0.499, 0.995] | 15 / 10 |
| zinnia_noir | halo2_lit | verify_s | 25 | 0.619 | [0.417, 0.882] | 15 / 10 |
| zinnia_noir | noir_lit | prove_core_s | 25 | 0.852 | [0.684, 0.980] | 13 / 12 |
| zinnia_noir | noir_lit | end_to_end_s | 25 | 0.859 | [0.684, 0.992] | 13 / 12 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 25 | 0.988 | [0.798, 1.154] | 11 / 14 |
| zinnia_noir | noir_lit | verify_s | 25 | 0.972 | [0.923, 1.027] | 17 / 8 |

## Copy-only cases (reported apart)

2 all-supported cases that only copy inputs: pythran_create_grid__create_grid, pythran_periodic_bc__periodic_bc

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 2 | 0.992 | [0.942, 1.044] | 1 / 1 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 2 | 0.995 | [0.946, 1.047] | 1 / 1 |
| zinnia_halo2 | halo2_hw | verify_s | 2 | 0.834 | [0.789, 0.882] | 2 / 0 |
| zinnia_noir | noir_hw | prove_core_s | 2 | 0.996 | [0.942, 1.054] | 1 / 1 |
| zinnia_noir | noir_hw | end_to_end_s | 2 | 0.993 | [0.938, 1.051] | 1 / 1 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 2 | 0.988 | [0.978, 0.997] | 2 / 0 |
| zinnia_noir | noir_hw | verify_s | 2 | 0.951 | [0.827, 1.092] | 1 / 1 |
| zinnia_halo2 | noir_hw | prove_core_s | 2 | 0.790 | [0.519, 1.200] | 1 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_s | 2 | 0.766 | [0.505, 1.162] | 1 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 2 | 0.354 | [0.233, 0.538] | 2 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 2 | 0.496 | [0.299, 0.825] | 2 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 2 | 1.021 | [0.973, 1.072] | 1 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 2 | 1.025 | [0.979, 1.074] | 1 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 2 | 0.870 | [0.719, 1.052] | 1 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 2 | 0.766 | [0.553, 1.061] | 1 / 1 |
| zinnia_halo2 | noir_lit | end_to_end_s | 2 | 0.745 | [0.541, 1.024] | 1 / 1 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 2 | 0.347 | [0.233, 0.519] | 2 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 2 | 0.428 | [0.288, 0.635] | 2 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 2 | 1.251 | [0.917, 1.707] | 1 / 1 |
| zinnia_noir | halo2_hw | end_to_end_s | 2 | 1.290 | [0.947, 1.758] | 1 / 1 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 2 | 2.772 | [1.937, 3.967] | 0 / 2 |
| zinnia_noir | halo2_hw | verify_s | 2 | 1.598 | [1.168, 2.185] | 0 / 2 |
| zinnia_noir | halo2_lit | prove_core_s | 2 | 1.289 | [0.941, 1.765] | 1 / 1 |
| zinnia_noir | halo2_lit | end_to_end_s | 2 | 1.329 | [0.972, 1.818] | 1 / 1 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 2 | 2.856 | [1.988, 4.103] | 0 / 2 |
| zinnia_noir | halo2_lit | verify_s | 2 | 1.666 | [1.393, 1.991] | 0 / 2 |
| zinnia_noir | noir_lit | prove_core_s | 2 | 0.966 | [0.931, 1.002] | 1 / 1 |
| zinnia_noir | noir_lit | end_to_end_s | 2 | 0.965 | [0.926, 1.006] | 1 / 1 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 2 | 0.968 | [0.961, 0.976] | 2 / 0 |
| zinnia_noir | noir_lit | verify_s | 2 | 0.820 | [0.798, 0.841] | 2 / 0 |

## Group npbench: headline cases

10 headline cases of group npbench: npbench_deep_learning_conv2d_bias_conv2d__conv2d_bias, npbench_deep_learning_softmax_softmax__softmax, npbench_polybench_bicg_bicg__kernel, npbench_polybench_deriche_deriche__kernel, npbench_polybench_gemm_gemm__kernel, npbench_polybench_gramschmidt_gramschmidt__kernel, npbench_polybench_lu_lu__kernel, npbench_polybench_symm_symm__kernel, npbench_polybench_trisolv_trisolv__kernel, npbench_scattering_self_energies_scattering_self_energies__scattering_self_energies

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 10 | 0.873 | [0.785, 0.964] | 8 / 2 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 10 | 0.892 | [0.802, 0.985] | 8 / 2 |
| zinnia_halo2 | halo2_hw | verify_s | 10 | 0.967 | [0.804, 1.235] | 8 / 2 |
| zinnia_noir | noir_hw | prove_core_s | 10 | 1.013 | [0.989, 1.039] | 3 / 7 |
| zinnia_noir | noir_hw | end_to_end_s | 10 | 1.031 | [1.013, 1.051] | 2 / 8 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 10 | 1.163 | [1.075, 1.280] | 0 / 10 |
| zinnia_noir | noir_hw | verify_s | 10 | 0.966 | [0.898, 1.033] | 6 / 4 |
| zinnia_halo2 | noir_hw | prove_core_s | 10 | 3.091 | [2.251, 4.170] | 0 / 10 |
| zinnia_halo2 | noir_hw | end_to_end_s | 10 | 2.771 | [2.104, 3.567] | 0 / 10 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 10 | 1.835 | [1.254, 2.629] | 2 / 8 |
| zinnia_halo2 | noir_hw | verify_s | 10 | 1.909 | [1.129, 4.068] | 2 / 8 |
| zinnia_halo2 | halo2_lit | prove_core_s | 10 | 0.760 | [0.670, 0.848] | 10 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 10 | 0.776 | [0.684, 0.865] | 10 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 10 | 0.836 | [0.726, 0.975] | 8 / 2 |
| zinnia_halo2 | noir_lit | prove_core_s | 10 | 3.016 | [2.201, 4.063] | 0 / 10 |
| zinnia_halo2 | noir_lit | end_to_end_s | 10 | 2.723 | [2.061, 3.515] | 0 / 10 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 10 | 1.794 | [1.227, 2.568] | 2 / 8 |
| zinnia_halo2 | noir_lit | verify_s | 10 | 1.902 | [1.128, 4.012] | 2 / 8 |
| zinnia_noir | halo2_hw | prove_core_s | 10 | 0.286 | [0.211, 0.397] | 10 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 10 | 0.332 | [0.256, 0.440] | 10 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 10 | 0.565 | [0.404, 0.784] | 8 / 2 |
| zinnia_noir | halo2_hw | verify_s | 10 | 0.490 | [0.280, 0.776] | 9 / 1 |
| zinnia_noir | halo2_lit | prove_core_s | 10 | 0.249 | [0.165, 0.380] | 10 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 10 | 0.289 | [0.202, 0.418] | 10 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 10 | 0.491 | [0.323, 0.745] | 8 / 2 |
| zinnia_noir | halo2_lit | verify_s | 10 | 0.423 | [0.193, 0.769] | 8 / 2 |
| zinnia_noir | noir_lit | prove_core_s | 10 | 0.988 | [0.916, 1.049] | 5 / 5 |
| zinnia_noir | noir_lit | end_to_end_s | 10 | 1.013 | [0.948, 1.068] | 4 / 6 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 10 | 1.137 | [1.026, 1.280] | 2 / 8 |
| zinnia_noir | noir_lit | verify_s | 10 | 0.963 | [0.908, 1.014] | 6 / 4 |

## Group pythran: headline cases

15 headline cases of group pythran: pythran_allpairs__sqr_dists, pythran_bubble_sort__bubble_sort, pythran_d2q9_nxnyns__one_time_step, pythran_fbcorr__fbcorr, pythran_harris__harris, pythran_ising__ising_step, pythran_lapl3d__laplacien, pythran_lu__factorMatrix0, pythran_matmul__matrix_multiply, pythran_multitype__check2, pythran_projection_simplex__projection_simplex, pythran_rosen__rosen, pythran_spectraldns__add_pressure_diffusion, pythran_tax__tax, pythran_wdist__slow_wdist

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 15 | 0.967 | [0.925, 1.010] | 9 / 6 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 15 | 0.984 | [0.938, 1.033] | 7 / 8 |
| zinnia_halo2 | halo2_hw | verify_s | 15 | 0.993 | [0.931, 1.065] | 10 / 5 |
| zinnia_noir | noir_hw | prove_core_s | 15 | 0.943 | [0.877, 1.002] | 9 / 6 |
| zinnia_noir | noir_hw | end_to_end_s | 15 | 0.958 | [0.890, 1.019] | 7 / 8 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 15 | 1.086 | [0.979, 1.211] | 8 / 7 |
| zinnia_noir | noir_hw | verify_s | 15 | 0.981 | [0.917, 1.066] | 11 / 4 |
| zinnia_halo2 | noir_hw | prove_core_s | 15 | 1.765 | [1.162, 2.652] | 2 / 13 |
| zinnia_halo2 | noir_hw | end_to_end_s | 15 | 1.655 | [1.111, 2.419] | 2 / 13 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 15 | 0.930 | [0.552, 1.565] | 9 / 6 |
| zinnia_halo2 | noir_hw | verify_s | 15 | 1.091 | [0.794, 1.547] | 8 / 7 |
| zinnia_halo2 | halo2_lit | prove_core_s | 15 | 0.744 | [0.617, 0.861] | 13 / 2 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 15 | 0.757 | [0.629, 0.874] | 13 / 2 |
| zinnia_halo2 | halo2_lit | verify_s | 15 | 0.888 | [0.779, 1.001] | 10 / 5 |
| zinnia_halo2 | noir_lit | prove_core_s | 15 | 1.444 | [0.874, 2.333] | 4 / 11 |
| zinnia_halo2 | noir_lit | end_to_end_s | 15 | 1.330 | [0.813, 2.099] | 4 / 11 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 15 | 0.772 | [0.444, 1.335] | 10 / 5 |
| zinnia_halo2 | noir_lit | verify_s | 15 | 1.087 | [0.791, 1.559] | 8 / 7 |
| zinnia_noir | halo2_hw | prove_core_s | 15 | 0.516 | [0.361, 0.743] | 14 / 1 |
| zinnia_noir | halo2_hw | end_to_end_s | 15 | 0.570 | [0.416, 0.792] | 14 / 1 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 15 | 1.148 | [0.780, 1.703] | 6 / 9 |
| zinnia_noir | halo2_hw | verify_s | 15 | 0.893 | [0.630, 1.240] | 6 / 9 |
| zinnia_noir | halo2_lit | prove_core_s | 15 | 0.397 | [0.258, 0.610] | 14 / 1 |
| zinnia_noir | halo2_lit | end_to_end_s | 15 | 0.438 | [0.294, 0.655] | 14 / 1 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 15 | 0.883 | [0.554, 1.403] | 7 / 8 |
| zinnia_noir | halo2_lit | verify_s | 15 | 0.798 | [0.555, 1.144] | 7 / 8 |
| zinnia_noir | noir_lit | prove_core_s | 15 | 0.771 | [0.540, 0.962] | 8 / 7 |
| zinnia_noir | noir_lit | end_to_end_s | 15 | 0.770 | [0.532, 0.966] | 9 / 6 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 15 | 0.901 | [0.634, 1.139] | 9 / 6 |
| zinnia_noir | noir_lit | verify_s | 15 | 0.978 | [0.906, 1.065] | 11 / 4 |

## All-supported subset

27 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): npbench_deep_learning_conv2d_bias_conv2d__conv2d_bias, npbench_deep_learning_softmax_softmax__softmax, npbench_polybench_bicg_bicg__kernel, npbench_polybench_deriche_deriche__kernel, npbench_polybench_gemm_gemm__kernel, npbench_polybench_gramschmidt_gramschmidt__kernel, npbench_polybench_lu_lu__kernel, npbench_polybench_symm_symm__kernel, npbench_polybench_trisolv_trisolv__kernel, npbench_scattering_self_energies_scattering_self_energies__scattering_self_energies, pythran_allpairs__sqr_dists, pythran_bubble_sort__bubble_sort, pythran_create_grid__create_grid, pythran_d2q9_nxnyns__one_time_step, pythran_fbcorr__fbcorr, pythran_harris__harris, pythran_ising__ising_step, pythran_lapl3d__laplacien, pythran_lu__factorMatrix0, pythran_matmul__matrix_multiply, pythran_multitype__check2, pythran_periodic_bc__periodic_bc, pythran_projection_simplex__projection_simplex, pythran_rosen__rosen, pythran_spectraldns__add_pressure_diffusion, pythran_tax__tax, pythran_wdist__slow_wdist

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 27 | 0.933 | [0.886, 0.978] | 18 / 9 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 27 | 0.950 | [0.901, 0.996] | 16 / 11 |
| zinnia_halo2 | halo2_hw | verify_s | 27 | 0.971 | [0.894, 1.068] | 20 / 7 |
| zinnia_noir | noir_hw | prove_core_s | 27 | 0.972 | [0.931, 1.008] | 13 / 14 |
| zinnia_noir | noir_hw | end_to_end_s | 27 | 0.987 | [0.945, 1.023] | 10 / 17 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 27 | 1.106 | [1.034, 1.185] | 10 / 17 |
| zinnia_noir | noir_hw | verify_s | 27 | 0.973 | [0.927, 1.027] | 18 / 9 |
| zinnia_halo2 | noir_hw | prove_core_s | 27 | 2.047 | [1.525, 2.712] | 3 / 24 |
| zinnia_halo2 | noir_hw | end_to_end_s | 27 | 1.892 | [1.433, 2.452] | 3 / 24 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 27 | 1.114 | [0.777, 1.574] | 13 / 14 |
| zinnia_halo2 | noir_hw | verify_s | 27 | 1.266 | [0.929, 1.821] | 12 / 15 |
| zinnia_halo2 | halo2_lit | prove_core_s | 27 | 0.768 | [0.686, 0.848] | 24 / 3 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 27 | 0.781 | [0.698, 0.861] | 24 / 3 |
| zinnia_halo2 | halo2_lit | verify_s | 27 | 0.867 | [0.792, 0.948] | 19 / 8 |
| zinnia_halo2 | noir_lit | prove_core_s | 27 | 1.810 | [1.275, 2.502] | 5 / 22 |
| zinnia_halo2 | noir_lit | end_to_end_s | 27 | 1.661 | [1.187, 2.254] | 5 / 22 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 27 | 0.994 | [0.676, 1.440] | 14 / 13 |
| zinnia_halo2 | noir_lit | verify_s | 27 | 1.248 | [0.915, 1.796] | 12 / 15 |
| zinnia_noir | halo2_hw | prove_core_s | 27 | 0.443 | [0.339, 0.586] | 25 / 2 |
| zinnia_noir | halo2_hw | end_to_end_s | 27 | 0.495 | [0.391, 0.638] | 25 / 2 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 27 | 0.943 | [0.706, 1.278] | 14 / 13 |
| zinnia_noir | halo2_hw | verify_s | 27 | 0.746 | [0.549, 0.990] | 15 / 12 |
| zinnia_noir | halo2_lit | prove_core_s | 27 | 0.365 | [0.267, 0.506] | 25 / 2 |
| zinnia_noir | halo2_lit | end_to_end_s | 27 | 0.408 | [0.307, 0.552] | 25 / 2 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 27 | 0.776 | [0.556, 1.097] | 15 / 12 |
| zinnia_noir | halo2_lit | verify_s | 27 | 0.666 | [0.459, 0.927] | 15 / 12 |
| zinnia_noir | noir_lit | prove_core_s | 27 | 0.860 | [0.699, 0.979] | 14 / 13 |
| zinnia_noir | noir_lit | end_to_end_s | 27 | 0.867 | [0.701, 0.991] | 14 / 13 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 27 | 0.987 | [0.806, 1.138] | 13 / 14 |
| zinnia_noir | noir_lit | verify_s | 27 | 0.960 | [0.911, 1.013] | 19 / 8 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 27 | 0.933 | [0.886, 0.978] | 18 / 9 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 27 | 0.950 | [0.901, 0.996] | 16 / 11 |
| zinnia_halo2 | halo2_hw | verify_s | 27 | 0.971 | [0.894, 1.068] | 20 / 7 |
| zinnia_noir | noir_hw | prove_core_s | 27 | 0.972 | [0.931, 1.008] | 13 / 14 |
| zinnia_noir | noir_hw | end_to_end_s | 27 | 0.987 | [0.945, 1.023] | 10 / 17 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 27 | 1.106 | [1.034, 1.185] | 10 / 17 |
| zinnia_noir | noir_hw | verify_s | 27 | 0.973 | [0.927, 1.027] | 18 / 9 |
| zinnia_halo2 | noir_hw | prove_core_s | 27 | 2.047 | [1.525, 2.712] | 3 / 24 |
| zinnia_halo2 | noir_hw | end_to_end_s | 27 | 1.892 | [1.433, 2.452] | 3 / 24 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 27 | 1.114 | [0.777, 1.574] | 13 / 14 |
| zinnia_halo2 | noir_hw | verify_s | 27 | 1.266 | [0.929, 1.821] | 12 / 15 |
| zinnia_halo2 | halo2_lit | prove_core_s | 27 | 0.768 | [0.686, 0.848] | 24 / 3 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 27 | 0.781 | [0.698, 0.861] | 24 / 3 |
| zinnia_halo2 | halo2_lit | verify_s | 27 | 0.867 | [0.792, 0.948] | 19 / 8 |
| zinnia_halo2 | noir_lit | prove_core_s | 27 | 1.810 | [1.275, 2.502] | 5 / 22 |
| zinnia_halo2 | noir_lit | end_to_end_s | 27 | 1.661 | [1.187, 2.254] | 5 / 22 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 27 | 0.994 | [0.676, 1.440] | 14 / 13 |
| zinnia_halo2 | noir_lit | verify_s | 27 | 1.248 | [0.915, 1.796] | 12 / 15 |
| zinnia_noir | halo2_hw | prove_core_s | 27 | 0.443 | [0.339, 0.586] | 25 / 2 |
| zinnia_noir | halo2_hw | end_to_end_s | 27 | 0.495 | [0.391, 0.638] | 25 / 2 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 27 | 0.943 | [0.706, 1.278] | 14 / 13 |
| zinnia_noir | halo2_hw | verify_s | 27 | 0.746 | [0.549, 0.990] | 15 / 12 |
| zinnia_noir | halo2_lit | prove_core_s | 27 | 0.365 | [0.267, 0.506] | 25 / 2 |
| zinnia_noir | halo2_lit | end_to_end_s | 27 | 0.408 | [0.307, 0.552] | 25 / 2 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 27 | 0.776 | [0.556, 1.097] | 15 / 12 |
| zinnia_noir | halo2_lit | verify_s | 27 | 0.666 | [0.459, 0.927] | 15 / 12 |
| zinnia_noir | noir_lit | prove_core_s | 27 | 0.860 | [0.699, 0.979] | 14 / 13 |
| zinnia_noir | noir_lit | end_to_end_s | 27 | 0.867 | [0.701, 0.991] | 14 / 13 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 27 | 0.987 | [0.806, 1.138] | 13 / 14 |
| zinnia_noir | noir_lit | verify_s | 27 | 0.960 | [0.911, 1.013] | 19 / 8 |

## Excluded cases

(none)

## Cases without a program

No program for the case on that system (e.g. no handwritten baseline yet); not a failure.

(none)

## Failures by system

