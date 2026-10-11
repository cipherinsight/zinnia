# Epoch 2026-10-11-fz-npb-s1

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

26 cases completed by every system that do more than copy inputs: npbench_crc16_crc16__crc16, npbench_deep_learning_resnet_resnet__resnet_basicblock, npbench_polybench_atax_atax__kernel, npbench_polybench_covariance2_covariance2__kernel, npbench_polybench_floyd_warshall_floyd_warshall__kernel, npbench_polybench_gesummv_gesummv__kernel, npbench_polybench_k3mm_k3mm__kernel, npbench_polybench_mvt_mvt__kernel, npbench_polybench_syrk_syrk__kernel, npbench_pythran_arc_distance_arc_distance__arc_distance, npbench_weather_stencils_vadv_vadv__vadv, pythran_arc_distance__arc_distance, pythran_clip__clip, pythran_crotg__CROTG, pythran_euclidean_distance_square__euclidean_distance_square, pythran_goodexpomeasure__goodExpoMeasure, pythran_insertion_sort__insertion_sort, pythran_lap__laplacian, pythran_log_likelihood__log_likelihood, pythran_make_decision__md, pythran_multitype__check, pythran_pairwise__pairwise, pythran_reverse_cumsum__reverse_cumsum, pythran_specialconvolve__specialconvolve, pythran_sumarray3d__summation, pythran_vibr_energy__calculate_vibr_energy

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 26 | 0.886 | [0.809, 0.954] | 20 / 6 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 26 | 0.897 | [0.820, 0.966] | 17 / 9 |
| zinnia_halo2 | halo2_hw | verify_s | 26 | 0.909 | [0.842, 0.983] | 18 / 8 |
| zinnia_noir | noir_hw | prove_core_s | 26 | 0.937 | [0.849, 1.017] | 15 / 11 |
| zinnia_noir | noir_hw | end_to_end_s | 26 | 0.946 | [0.859, 1.025] | 15 / 11 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 26 | 1.014 | [0.951, 1.081] | 12 / 14 |
| zinnia_noir | noir_hw | verify_s | 26 | 1.046 | [0.970, 1.132] | 12 / 14 |
| zinnia_halo2 | noir_hw | prove_core_s | 26 | 2.000 | [1.502, 2.600] | 4 / 22 |
| zinnia_halo2 | noir_hw | end_to_end_s | 26 | 1.845 | [1.410, 2.364] | 4 / 22 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 26 | 1.128 | [0.797, 1.545] | 9 / 17 |
| zinnia_halo2 | noir_hw | verify_s | 26 | 1.224 | [0.952, 1.585] | 13 / 13 |
| zinnia_halo2 | halo2_lit | prove_core_s | 26 | 0.761 | [0.657, 0.861] | 23 / 3 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 26 | 0.771 | [0.665, 0.871] | 21 / 5 |
| zinnia_halo2 | halo2_lit | verify_s | 26 | 0.909 | [0.824, 0.996] | 13 / 13 |
| zinnia_halo2 | noir_lit | prove_core_s | 26 | 1.940 | [1.446, 2.547] | 5 / 21 |
| zinnia_halo2 | noir_lit | end_to_end_s | 26 | 1.789 | [1.352, 2.316] | 5 / 21 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 26 | 1.098 | [0.773, 1.523] | 11 / 15 |
| zinnia_halo2 | noir_lit | verify_s | 26 | 1.148 | [0.880, 1.504] | 13 / 13 |
| zinnia_noir | halo2_hw | prove_core_s | 26 | 0.415 | [0.298, 0.585] | 22 / 4 |
| zinnia_noir | halo2_hw | end_to_end_s | 26 | 0.460 | [0.338, 0.634] | 22 / 4 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 26 | 0.806 | [0.575, 1.145] | 16 / 10 |
| zinnia_noir | halo2_hw | verify_s | 26 | 0.777 | [0.569, 1.045] | 15 / 11 |
| zinnia_noir | halo2_lit | prove_core_s | 26 | 0.357 | [0.256, 0.508] | 23 / 3 |
| zinnia_noir | halo2_lit | end_to_end_s | 26 | 0.395 | [0.290, 0.550] | 23 / 3 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 26 | 0.693 | [0.497, 0.992] | 17 / 9 |
| zinnia_noir | halo2_lit | verify_s | 26 | 0.777 | [0.581, 1.033] | 15 / 11 |
| zinnia_noir | noir_lit | prove_core_s | 26 | 0.909 | [0.817, 0.994] | 17 / 9 |
| zinnia_noir | noir_lit | end_to_end_s | 26 | 0.917 | [0.823, 1.004] | 15 / 11 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 26 | 0.986 | [0.917, 1.060] | 13 / 13 |
| zinnia_noir | noir_lit | verify_s | 26 | 0.982 | [0.924, 1.043] | 18 / 8 |

## Copy-only cases (reported apart)

1 all-supported cases that only copy inputs: pythran_perm__permutations

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 1 | 0.928 | [0.928, 0.928] | 1 / 0 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 1 | 0.930 | [0.930, 0.930] | 1 / 0 |
| zinnia_halo2 | halo2_hw | verify_s | 1 | 0.867 | [0.867, 0.867] | 1 / 0 |
| zinnia_noir | noir_hw | prove_core_s | 1 | 1.145 | [1.145, 1.145] | 0 / 1 |
| zinnia_noir | noir_hw | end_to_end_s | 1 | 1.138 | [1.138, 1.138] | 0 / 1 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 1 | 1.109 | [1.109, 1.109] | 0 / 1 |
| zinnia_noir | noir_hw | verify_s | 1 | 0.787 | [0.787, 0.787] | 1 / 0 |
| zinnia_halo2 | noir_hw | prove_core_s | 1 | 0.347 | [0.347, 0.347] | 1 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_s | 1 | 0.341 | [0.341, 0.341] | 1 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 1 | 0.165 | [0.165, 0.165] | 1 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 1 | 0.297 | [0.297, 0.297] | 1 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 1 | 0.957 | [0.957, 0.957] | 1 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 1 | 0.959 | [0.959, 0.959] | 1 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 1 | 1.220 | [1.220, 1.220] | 0 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 1 | 0.365 | [0.365, 0.365] | 1 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_s | 1 | 0.360 | [0.360, 0.360] | 1 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 1 | 0.165 | [0.165, 0.165] | 1 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 1 | 0.360 | [0.360, 0.360] | 1 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 1 | 3.062 | [3.062, 3.062] | 0 / 1 |
| zinnia_noir | halo2_hw | end_to_end_s | 1 | 3.107 | [3.107, 3.107] | 0 / 1 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 1 | 6.253 | [6.253, 6.253] | 0 / 1 |
| zinnia_noir | halo2_hw | verify_s | 1 | 2.293 | [2.293, 2.293] | 0 / 1 |
| zinnia_noir | halo2_lit | prove_core_s | 1 | 3.158 | [3.158, 3.158] | 0 / 1 |
| zinnia_noir | halo2_lit | end_to_end_s | 1 | 3.204 | [3.204, 3.204] | 0 / 1 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 1 | 6.449 | [6.449, 6.449] | 0 / 1 |
| zinnia_noir | halo2_lit | verify_s | 1 | 3.227 | [3.227, 3.227] | 0 / 1 |
| zinnia_noir | noir_lit | prove_core_s | 1 | 1.203 | [1.203, 1.203] | 0 / 1 |
| zinnia_noir | noir_lit | end_to_end_s | 1 | 1.201 | [1.201, 1.201] | 0 / 1 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 1 | 1.106 | [1.106, 1.106] | 0 / 1 |
| zinnia_noir | noir_lit | verify_s | 1 | 0.951 | [0.951, 0.951] | 1 / 0 |

## Group npbench: headline cases

11 headline cases of group npbench: npbench_crc16_crc16__crc16, npbench_deep_learning_resnet_resnet__resnet_basicblock, npbench_polybench_atax_atax__kernel, npbench_polybench_covariance2_covariance2__kernel, npbench_polybench_floyd_warshall_floyd_warshall__kernel, npbench_polybench_gesummv_gesummv__kernel, npbench_polybench_k3mm_k3mm__kernel, npbench_polybench_mvt_mvt__kernel, npbench_polybench_syrk_syrk__kernel, npbench_pythran_arc_distance_arc_distance__arc_distance, npbench_weather_stencils_vadv_vadv__vadv

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 11 | 0.852 | [0.704, 0.968] | 10 / 1 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 11 | 0.868 | [0.717, 0.986] | 8 / 3 |
| zinnia_halo2 | halo2_hw | verify_s | 11 | 0.848 | [0.755, 0.949] | 8 / 3 |
| zinnia_noir | noir_hw | prove_core_s | 11 | 0.839 | [0.695, 0.981] | 6 / 5 |
| zinnia_noir | noir_hw | end_to_end_s | 11 | 0.852 | [0.705, 0.994] | 6 / 5 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 11 | 1.008 | [0.879, 1.153] | 4 / 7 |
| zinnia_noir | noir_hw | verify_s | 11 | 1.052 | [0.939, 1.198] | 5 / 6 |
| zinnia_halo2 | noir_hw | prove_core_s | 11 | 2.728 | [2.106, 3.527] | 0 / 11 |
| zinnia_halo2 | noir_hw | end_to_end_s | 11 | 2.459 | [1.943, 3.115] | 0 / 11 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 11 | 1.716 | [1.266, 2.297] | 2 / 9 |
| zinnia_halo2 | noir_hw | verify_s | 11 | 1.698 | [1.203, 2.439] | 3 / 8 |
| zinnia_halo2 | halo2_lit | prove_core_s | 11 | 0.742 | [0.608, 0.878] | 9 / 2 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 11 | 0.755 | [0.619, 0.893] | 9 / 2 |
| zinnia_halo2 | halo2_lit | verify_s | 11 | 0.795 | [0.678, 0.930] | 8 / 3 |
| zinnia_halo2 | noir_lit | prove_core_s | 11 | 2.658 | [2.079, 3.352] | 0 / 11 |
| zinnia_halo2 | noir_lit | end_to_end_s | 11 | 2.421 | [1.928, 3.023] | 0 / 11 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 11 | 1.702 | [1.264, 2.264] | 3 / 8 |
| zinnia_halo2 | noir_lit | verify_s | 11 | 1.615 | [1.125, 2.295] | 3 / 8 |
| zinnia_noir | halo2_hw | prove_core_s | 11 | 0.262 | [0.175, 0.379] | 11 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 11 | 0.300 | [0.208, 0.421] | 11 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 11 | 0.510 | [0.353, 0.724] | 9 / 2 |
| zinnia_noir | halo2_hw | verify_s | 11 | 0.525 | [0.355, 0.764] | 8 / 3 |
| zinnia_noir | halo2_lit | prove_core_s | 11 | 0.228 | [0.153, 0.331] | 11 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 11 | 0.262 | [0.183, 0.366] | 11 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 11 | 0.444 | [0.311, 0.633] | 9 / 2 |
| zinnia_noir | halo2_lit | verify_s | 11 | 0.492 | [0.348, 0.694] | 9 / 2 |
| zinnia_noir | noir_lit | prove_core_s | 11 | 0.818 | [0.684, 0.938] | 9 / 2 |
| zinnia_noir | noir_lit | end_to_end_s | 11 | 0.838 | [0.697, 0.969] | 7 / 4 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 11 | 1.000 | [0.877, 1.135] | 4 / 7 |
| zinnia_noir | noir_lit | verify_s | 11 | 1.000 | [0.899, 1.115] | 6 / 5 |

## Group pythran: headline cases

15 headline cases of group pythran: pythran_arc_distance__arc_distance, pythran_clip__clip, pythran_crotg__CROTG, pythran_euclidean_distance_square__euclidean_distance_square, pythran_goodexpomeasure__goodExpoMeasure, pythran_insertion_sort__insertion_sort, pythran_lap__laplacian, pythran_log_likelihood__log_likelihood, pythran_make_decision__md, pythran_multitype__check, pythran_pairwise__pairwise, pythran_reverse_cumsum__reverse_cumsum, pythran_specialconvolve__specialconvolve, pythran_sumarray3d__summation, pythran_vibr_energy__calculate_vibr_energy

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 15 | 0.912 | [0.846, 0.982] | 10 / 5 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 15 | 0.920 | [0.854, 0.991] | 9 / 6 |
| zinnia_halo2 | halo2_hw | verify_s | 15 | 0.956 | [0.866, 1.051] | 10 / 5 |
| zinnia_noir | noir_hw | prove_core_s | 15 | 1.016 | [0.954, 1.083] | 9 / 6 |
| zinnia_noir | noir_hw | end_to_end_s | 15 | 1.022 | [0.965, 1.085] | 9 / 6 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 15 | 1.018 | [0.968, 1.077] | 8 / 7 |
| zinnia_noir | noir_hw | verify_s | 15 | 1.042 | [0.944, 1.149] | 7 / 8 |
| zinnia_halo2 | noir_hw | prove_core_s | 15 | 1.593 | [1.061, 2.334] | 4 / 11 |
| zinnia_halo2 | noir_hw | end_to_end_s | 15 | 1.494 | [1.013, 2.141] | 4 / 11 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 15 | 0.829 | [0.514, 1.312] | 7 / 8 |
| zinnia_halo2 | noir_hw | verify_s | 15 | 0.962 | [0.724, 1.309] | 10 / 5 |
| zinnia_halo2 | halo2_lit | prove_core_s | 15 | 0.776 | [0.626, 0.907] | 14 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 15 | 0.783 | [0.631, 0.915] | 12 / 3 |
| zinnia_halo2 | halo2_lit | verify_s | 15 | 1.003 | [0.912, 1.086] | 5 / 10 |
| zinnia_halo2 | noir_lit | prove_core_s | 15 | 1.540 | [1.008, 2.306] | 5 / 10 |
| zinnia_halo2 | noir_lit | end_to_end_s | 15 | 1.433 | [0.957, 2.092] | 5 / 10 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 15 | 0.796 | [0.491, 1.275] | 8 / 7 |
| zinnia_halo2 | noir_lit | verify_s | 15 | 0.894 | [0.656, 1.253] | 10 / 5 |
| zinnia_noir | halo2_hw | prove_core_s | 15 | 0.581 | [0.383, 0.894] | 11 / 4 |
| zinnia_noir | halo2_hw | end_to_end_s | 15 | 0.629 | [0.426, 0.944] | 11 / 4 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 15 | 1.129 | [0.721, 1.785] | 7 / 8 |
| zinnia_noir | halo2_hw | verify_s | 15 | 1.036 | [0.688, 1.489] | 7 / 8 |
| zinnia_noir | halo2_lit | prove_core_s | 15 | 0.495 | [0.321, 0.780] | 12 / 3 |
| zinnia_noir | halo2_lit | end_to_end_s | 15 | 0.535 | [0.356, 0.828] | 12 / 3 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 15 | 0.961 | [0.609, 1.554] | 8 / 7 |
| zinnia_noir | halo2_lit | verify_s | 15 | 1.086 | [0.766, 1.489] | 6 / 9 |
| zinnia_noir | noir_lit | prove_core_s | 15 | 0.982 | [0.874, 1.071] | 8 / 7 |
| zinnia_noir | noir_lit | end_to_end_s | 15 | 0.980 | [0.869, 1.070] | 8 / 7 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 15 | 0.977 | [0.895, 1.052] | 9 / 6 |
| zinnia_noir | noir_lit | verify_s | 15 | 0.969 | [0.906, 1.042] | 12 / 3 |

## All-supported subset

27 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): npbench_crc16_crc16__crc16, npbench_deep_learning_resnet_resnet__resnet_basicblock, npbench_polybench_atax_atax__kernel, npbench_polybench_covariance2_covariance2__kernel, npbench_polybench_floyd_warshall_floyd_warshall__kernel, npbench_polybench_gesummv_gesummv__kernel, npbench_polybench_k3mm_k3mm__kernel, npbench_polybench_mvt_mvt__kernel, npbench_polybench_syrk_syrk__kernel, npbench_pythran_arc_distance_arc_distance__arc_distance, npbench_weather_stencils_vadv_vadv__vadv, pythran_arc_distance__arc_distance, pythran_clip__clip, pythran_crotg__CROTG, pythran_euclidean_distance_square__euclidean_distance_square, pythran_goodexpomeasure__goodExpoMeasure, pythran_insertion_sort__insertion_sort, pythran_lap__laplacian, pythran_log_likelihood__log_likelihood, pythran_make_decision__md, pythran_multitype__check, pythran_pairwise__pairwise, pythran_perm__permutations, pythran_reverse_cumsum__reverse_cumsum, pythran_specialconvolve__specialconvolve, pythran_sumarray3d__summation, pythran_vibr_energy__calculate_vibr_energy

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 27 | 0.888 | [0.811, 0.952] | 21 / 6 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 27 | 0.898 | [0.821, 0.963] | 18 / 9 |
| zinnia_halo2 | halo2_hw | verify_s | 27 | 0.907 | [0.844, 0.977] | 19 / 8 |
| zinnia_noir | noir_hw | prove_core_s | 27 | 0.944 | [0.856, 1.023] | 15 / 12 |
| zinnia_noir | noir_hw | end_to_end_s | 27 | 0.953 | [0.865, 1.029] | 15 / 12 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 27 | 1.017 | [0.954, 1.082] | 12 / 15 |
| zinnia_noir | noir_hw | verify_s | 27 | 1.035 | [0.959, 1.121] | 13 / 14 |
| zinnia_halo2 | noir_hw | prove_core_s | 27 | 1.874 | [1.379, 2.485] | 5 / 22 |
| zinnia_halo2 | noir_hw | end_to_end_s | 27 | 1.733 | [1.298, 2.264] | 5 / 22 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 27 | 1.050 | [0.731, 1.470] | 10 / 17 |
| zinnia_halo2 | noir_hw | verify_s | 27 | 1.161 | [0.894, 1.518] | 14 / 13 |
| zinnia_halo2 | halo2_lit | prove_core_s | 27 | 0.768 | [0.666, 0.862] | 24 / 3 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 27 | 0.777 | [0.675, 0.872] | 22 / 5 |
| zinnia_halo2 | halo2_lit | verify_s | 27 | 0.919 | [0.837, 1.005] | 13 / 14 |
| zinnia_halo2 | noir_lit | prove_core_s | 27 | 1.823 | [1.338, 2.438] | 6 / 21 |
| zinnia_halo2 | noir_lit | end_to_end_s | 27 | 1.686 | [1.255, 2.221] | 6 / 21 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 27 | 1.023 | [0.711, 1.445] | 12 / 15 |
| zinnia_halo2 | noir_lit | verify_s | 27 | 1.100 | [0.843, 1.445] | 14 / 13 |
| zinnia_noir | halo2_hw | prove_core_s | 27 | 0.447 | [0.318, 0.640] | 22 / 5 |
| zinnia_noir | halo2_hw | end_to_end_s | 27 | 0.494 | [0.359, 0.692] | 22 / 5 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 27 | 0.870 | [0.614, 1.260] | 16 / 11 |
| zinnia_noir | halo2_hw | verify_s | 27 | 0.809 | [0.597, 1.090] | 15 / 12 |
| zinnia_noir | halo2_lit | prove_core_s | 27 | 0.387 | [0.273, 0.563] | 23 / 4 |
| zinnia_noir | halo2_lit | end_to_end_s | 27 | 0.427 | [0.309, 0.608] | 23 / 4 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 27 | 0.753 | [0.530, 1.111] | 17 / 10 |
| zinnia_noir | halo2_lit | verify_s | 27 | 0.819 | [0.612, 1.096] | 15 / 12 |
| zinnia_noir | noir_lit | prove_core_s | 27 | 0.918 | [0.827, 1.002] | 17 / 10 |
| zinnia_noir | noir_lit | end_to_end_s | 27 | 0.927 | [0.833, 1.012] | 15 / 12 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 27 | 0.991 | [0.922, 1.061] | 13 / 14 |
| zinnia_noir | noir_lit | verify_s | 27 | 0.981 | [0.924, 1.042] | 19 / 8 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 27 | 0.888 | [0.811, 0.952] | 21 / 6 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 27 | 0.898 | [0.821, 0.963] | 18 / 9 |
| zinnia_halo2 | halo2_hw | verify_s | 27 | 0.907 | [0.844, 0.977] | 19 / 8 |
| zinnia_noir | noir_hw | prove_core_s | 27 | 0.944 | [0.856, 1.023] | 15 / 12 |
| zinnia_noir | noir_hw | end_to_end_s | 27 | 0.953 | [0.865, 1.029] | 15 / 12 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 27 | 1.017 | [0.954, 1.082] | 12 / 15 |
| zinnia_noir | noir_hw | verify_s | 27 | 1.035 | [0.959, 1.121] | 13 / 14 |
| zinnia_halo2 | noir_hw | prove_core_s | 27 | 1.874 | [1.379, 2.485] | 5 / 22 |
| zinnia_halo2 | noir_hw | end_to_end_s | 27 | 1.733 | [1.298, 2.264] | 5 / 22 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 27 | 1.050 | [0.731, 1.470] | 10 / 17 |
| zinnia_halo2 | noir_hw | verify_s | 27 | 1.161 | [0.894, 1.518] | 14 / 13 |
| zinnia_halo2 | halo2_lit | prove_core_s | 27 | 0.768 | [0.666, 0.862] | 24 / 3 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 27 | 0.777 | [0.675, 0.872] | 22 / 5 |
| zinnia_halo2 | halo2_lit | verify_s | 27 | 0.919 | [0.837, 1.005] | 13 / 14 |
| zinnia_halo2 | noir_lit | prove_core_s | 27 | 1.823 | [1.338, 2.438] | 6 / 21 |
| zinnia_halo2 | noir_lit | end_to_end_s | 27 | 1.686 | [1.255, 2.221] | 6 / 21 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 27 | 1.023 | [0.711, 1.445] | 12 / 15 |
| zinnia_halo2 | noir_lit | verify_s | 27 | 1.100 | [0.843, 1.445] | 14 / 13 |
| zinnia_noir | halo2_hw | prove_core_s | 27 | 0.447 | [0.318, 0.640] | 22 / 5 |
| zinnia_noir | halo2_hw | end_to_end_s | 27 | 0.494 | [0.359, 0.692] | 22 / 5 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 27 | 0.870 | [0.614, 1.260] | 16 / 11 |
| zinnia_noir | halo2_hw | verify_s | 27 | 0.809 | [0.597, 1.090] | 15 / 12 |
| zinnia_noir | halo2_lit | prove_core_s | 27 | 0.387 | [0.273, 0.563] | 23 / 4 |
| zinnia_noir | halo2_lit | end_to_end_s | 27 | 0.427 | [0.309, 0.608] | 23 / 4 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 27 | 0.753 | [0.530, 1.111] | 17 / 10 |
| zinnia_noir | halo2_lit | verify_s | 27 | 0.819 | [0.612, 1.096] | 15 / 12 |
| zinnia_noir | noir_lit | prove_core_s | 27 | 0.918 | [0.827, 1.002] | 17 / 10 |
| zinnia_noir | noir_lit | end_to_end_s | 27 | 0.927 | [0.833, 1.012] | 15 / 12 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 27 | 0.991 | [0.922, 1.061] | 13 / 14 |
| zinnia_noir | noir_lit | verify_s | 27 | 0.981 | [0.924, 1.042] | 19 / 8 |

## Excluded cases

(none)

## Cases without a program

No program for the case on that system (e.g. no handwritten baseline yet); not a failure.

(none)

## Failures by system

