# Epoch 2026-10-09-tw1-npb-s4

Ratios are Zinnia / baseline of per-case medians; below 1 means Zinnia is faster.
Only cases with status `ok` (proved, verified, outputs equal to NumPy, tampered proof rejected) enter a comparison.

`zinnia_noir` `end_to_end_s` witness step: noir-execute (10 cases); `end_to_end_nargo_s` uses `nargo execute`.
`noir_hw` `end_to_end_s` witness step: noir-execute (10 cases); `end_to_end_nargo_s` uses `nargo execute`.
`noir_lit` `end_to_end_s` witness step: noir-execute (10 cases); `end_to_end_nargo_s` uses `nargo execute`.

## Coverage

| system | cases | ok | other statuses |
|---|---|---|---|
| zinnia_halo2 | 10 | 10 | - |
| zinnia_noir | 10 | 10 | - |
| halo2_hw | 10 | 10 | - |
| noir_hw | 10 | 10 | - |
| halo2_lit | 10 | 10 | - |
| noir_lit | 10 | 10 | - |

## Headline: all-supported subset without copy-only cases

10 cases completed by every system that do more than copy inputs: npbench_deep_learning_softmax_softmax__softmax, npbench_polybench_gemm_gemm__kernel, npbench_polybench_mvt_mvt__kernel, npbench_weather_stencils_hdiff_hdiff__hdiff, pythran_cronbach__cronbach, pythran_harris__harris, pythran_log_likelihood__log_likelihood, pythran_multitype__times, pythran_rosen__rosen, pythran_vibr_energy__calculate_vibr_energy

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 10 | 0.866 | [0.733, 1.000] | 6 / 4 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 10 | 0.880 | [0.747, 1.017] | 6 / 4 |
| zinnia_halo2 | halo2_hw | verify_s | 10 | 0.952 | [0.855, 1.049] | 6 / 4 |
| zinnia_noir | noir_hw | prove_core_s | 10 | 1.045 | [0.946, 1.159] | 4 / 6 |
| zinnia_noir | noir_hw | end_to_end_s | 10 | 1.049 | [0.961, 1.151] | 5 / 5 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 10 | 1.104 | [1.005, 1.229] | 4 / 6 |
| zinnia_noir | noir_hw | verify_s | 10 | 1.067 | [0.976, 1.170] | 2 / 8 |
| zinnia_halo2 | noir_hw | prove_core_s | 10 | 1.892 | [1.299, 2.684] | 1 / 9 |
| zinnia_halo2 | noir_hw | end_to_end_s | 10 | 1.764 | [1.228, 2.455] | 1 / 9 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 10 | 0.983 | [0.571, 1.611] | 6 / 4 |
| zinnia_halo2 | noir_hw | verify_s | 10 | 1.062 | [0.733, 1.578] | 6 / 4 |
| zinnia_halo2 | halo2_lit | prove_core_s | 10 | 0.824 | [0.714, 0.922] | 8 / 2 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 10 | 0.837 | [0.729, 0.935] | 8 / 2 |
| zinnia_halo2 | halo2_lit | verify_s | 10 | 0.833 | [0.728, 0.945] | 8 / 2 |
| zinnia_halo2 | noir_lit | prove_core_s | 10 | 1.855 | [1.283, 2.646] | 1 / 9 |
| zinnia_halo2 | noir_lit | end_to_end_s | 10 | 1.725 | [1.220, 2.398] | 1 / 9 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 10 | 0.970 | [0.558, 1.612] | 6 / 4 |
| zinnia_halo2 | noir_lit | verify_s | 10 | 0.942 | [0.616, 1.463] | 6 / 4 |
| zinnia_noir | halo2_hw | prove_core_s | 10 | 0.478 | [0.311, 0.713] | 9 / 1 |
| zinnia_noir | halo2_hw | end_to_end_s | 10 | 0.524 | [0.354, 0.757] | 8 / 2 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 10 | 0.988 | [0.594, 1.661] | 5 / 5 |
| zinnia_noir | halo2_hw | verify_s | 10 | 0.956 | [0.590, 1.485] | 4 / 6 |
| zinnia_noir | halo2_lit | prove_core_s | 10 | 0.455 | [0.299, 0.673] | 9 / 1 |
| zinnia_noir | halo2_lit | end_to_end_s | 10 | 0.498 | [0.343, 0.714] | 9 / 1 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 10 | 0.940 | [0.576, 1.561] | 5 / 5 |
| zinnia_noir | halo2_lit | verify_s | 10 | 0.837 | [0.515, 1.305] | 5 / 5 |
| zinnia_noir | noir_lit | prove_core_s | 10 | 1.024 | [0.972, 1.091] | 6 / 4 |
| zinnia_noir | noir_lit | end_to_end_s | 10 | 1.026 | [0.976, 1.088] | 6 / 4 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 10 | 1.089 | [0.985, 1.228] | 4 / 6 |
| zinnia_noir | noir_lit | verify_s | 10 | 0.946 | [0.855, 1.033] | 5 / 5 |

## Copy-only cases (reported apart)

0 all-supported cases that only copy inputs: (none)

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 0 | - | - | - |
| zinnia_halo2 | halo2_hw | end_to_end_s | 0 | - | - | - |
| zinnia_halo2 | halo2_hw | verify_s | 0 | - | - | - |
| zinnia_noir | noir_hw | prove_core_s | 0 | - | - | - |
| zinnia_noir | noir_hw | end_to_end_s | 0 | - | - | - |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 0 | - | - | - |
| zinnia_noir | noir_hw | verify_s | 0 | - | - | - |
| zinnia_halo2 | noir_hw | prove_core_s | 0 | - | - | - |
| zinnia_halo2 | noir_hw | end_to_end_s | 0 | - | - | - |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 0 | - | - | - |
| zinnia_halo2 | noir_hw | verify_s | 0 | - | - | - |
| zinnia_halo2 | halo2_lit | prove_core_s | 0 | - | - | - |
| zinnia_halo2 | halo2_lit | end_to_end_s | 0 | - | - | - |
| zinnia_halo2 | halo2_lit | verify_s | 0 | - | - | - |
| zinnia_halo2 | noir_lit | prove_core_s | 0 | - | - | - |
| zinnia_halo2 | noir_lit | end_to_end_s | 0 | - | - | - |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 0 | - | - | - |
| zinnia_halo2 | noir_lit | verify_s | 0 | - | - | - |
| zinnia_noir | halo2_hw | prove_core_s | 0 | - | - | - |
| zinnia_noir | halo2_hw | end_to_end_s | 0 | - | - | - |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 0 | - | - | - |
| zinnia_noir | halo2_hw | verify_s | 0 | - | - | - |
| zinnia_noir | halo2_lit | prove_core_s | 0 | - | - | - |
| zinnia_noir | halo2_lit | end_to_end_s | 0 | - | - | - |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 0 | - | - | - |
| zinnia_noir | halo2_lit | verify_s | 0 | - | - | - |
| zinnia_noir | noir_lit | prove_core_s | 0 | - | - | - |
| zinnia_noir | noir_lit | end_to_end_s | 0 | - | - | - |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 0 | - | - | - |
| zinnia_noir | noir_lit | verify_s | 0 | - | - | - |

## Group npbench: headline cases

4 headline cases of group npbench: npbench_deep_learning_softmax_softmax__softmax, npbench_polybench_gemm_gemm__kernel, npbench_polybench_mvt_mvt__kernel, npbench_weather_stencils_hdiff_hdiff__hdiff

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 4 | 0.817 | [0.583, 1.071] | 2 / 2 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 4 | 0.839 | [0.602, 1.102] | 2 / 2 |
| zinnia_halo2 | halo2_hw | verify_s | 4 | 0.892 | [0.733, 1.020] | 2 / 2 |
| zinnia_noir | noir_hw | prove_core_s | 4 | 1.037 | [0.863, 1.246] | 2 / 2 |
| zinnia_noir | noir_hw | end_to_end_s | 4 | 1.061 | [0.907, 1.240] | 2 / 2 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 4 | 1.240 | [1.063, 1.446] | 1 / 3 |
| zinnia_noir | noir_hw | verify_s | 4 | 0.974 | [0.845, 1.060] | 1 / 3 |
| zinnia_halo2 | noir_hw | prove_core_s | 4 | 2.953 | [2.042, 3.747] | 0 / 4 |
| zinnia_halo2 | noir_hw | end_to_end_s | 4 | 2.684 | [1.977, 3.376] | 0 / 4 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 4 | 1.951 | [1.195, 2.738] | 1 / 3 |
| zinnia_halo2 | noir_hw | verify_s | 4 | 1.784 | [0.969, 2.976] | 1 / 3 |
| zinnia_halo2 | halo2_lit | prove_core_s | 4 | 0.781 | [0.563, 0.976] | 3 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 4 | 0.802 | [0.581, 0.999] | 3 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 4 | 0.756 | [0.608, 0.941] | 4 / 0 |
| zinnia_halo2 | noir_lit | prove_core_s | 4 | 2.985 | [1.860, 4.037] | 0 / 4 |
| zinnia_halo2 | noir_lit | end_to_end_s | 4 | 2.676 | [1.753, 3.554] | 0 / 4 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 4 | 1.994 | [1.139, 2.874] | 1 / 3 |
| zinnia_halo2 | noir_lit | verify_s | 4 | 1.651 | [0.841, 2.912] | 1 / 3 |
| zinnia_noir | halo2_hw | prove_core_s | 4 | 0.287 | [0.157, 0.553] | 4 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 4 | 0.332 | [0.193, 0.607] | 4 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 4 | 0.533 | [0.268, 1.037] | 3 / 1 |
| zinnia_noir | halo2_hw | verify_s | 4 | 0.487 | [0.255, 0.893] | 3 / 1 |
| zinnia_noir | halo2_lit | prove_core_s | 4 | 0.274 | [0.151, 0.508] | 4 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 4 | 0.317 | [0.191, 0.558] | 4 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 4 | 0.510 | [0.265, 0.993] | 3 / 1 |
| zinnia_noir | halo2_lit | verify_s | 4 | 0.413 | [0.224, 0.692] | 4 / 0 |
| zinnia_noir | noir_lit | prove_core_s | 4 | 1.048 | [0.970, 1.185] | 2 / 2 |
| zinnia_noir | noir_lit | end_to_end_s | 4 | 1.057 | [0.989, 1.170] | 2 / 2 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 4 | 1.267 | [1.072, 1.498] | 0 / 4 |
| zinnia_noir | noir_lit | verify_s | 4 | 0.902 | [0.730, 1.025] | 2 / 2 |

## Group pythran: headline cases

6 headline cases of group pythran: pythran_cronbach__cronbach, pythran_harris__harris, pythran_log_likelihood__log_likelihood, pythran_multitype__times, pythran_rosen__rosen, pythran_vibr_energy__calculate_vibr_energy

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 6 | 0.901 | [0.790, 1.039] | 4 / 2 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 6 | 0.909 | [0.798, 1.051] | 4 / 2 |
| zinnia_halo2 | halo2_hw | verify_s | 6 | 0.994 | [0.897, 1.112] | 4 / 2 |
| zinnia_noir | noir_hw | prove_core_s | 6 | 1.050 | [0.942, 1.187] | 2 / 4 |
| zinnia_noir | noir_hw | end_to_end_s | 6 | 1.042 | [0.942, 1.169] | 3 / 3 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 6 | 1.021 | [0.947, 1.104] | 3 / 3 |
| zinnia_noir | noir_hw | verify_s | 6 | 1.133 | [1.025, 1.262] | 1 / 5 |
| zinnia_halo2 | noir_hw | prove_core_s | 6 | 1.407 | [0.904, 2.125] | 1 / 5 |
| zinnia_halo2 | noir_hw | end_to_end_s | 6 | 1.333 | [0.868, 1.988] | 1 / 5 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 6 | 0.623 | [0.324, 1.061] | 5 / 1 |
| zinnia_halo2 | noir_hw | verify_s | 6 | 0.752 | [0.556, 0.980] | 5 / 1 |
| zinnia_halo2 | halo2_lit | prove_core_s | 6 | 0.853 | [0.774, 0.931] | 5 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 6 | 0.861 | [0.783, 0.939] | 5 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 6 | 0.888 | [0.776, 1.007] | 4 / 2 |
| zinnia_halo2 | noir_lit | prove_core_s | 6 | 1.351 | [0.927, 1.890] | 1 / 5 |
| zinnia_halo2 | noir_lit | end_to_end_s | 6 | 1.287 | [0.891, 1.772] | 1 / 5 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 6 | 0.600 | [0.321, 0.983] | 5 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 6 | 0.648 | [0.447, 0.890] | 5 / 1 |
| zinnia_noir | halo2_hw | prove_core_s | 6 | 0.672 | [0.465, 0.969] | 5 / 1 |
| zinnia_noir | halo2_hw | end_to_end_s | 6 | 0.710 | [0.497, 1.014] | 4 / 2 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 6 | 1.490 | [0.891, 2.653] | 2 / 4 |
| zinnia_noir | halo2_hw | verify_s | 6 | 1.499 | [1.081, 2.104] | 1 / 5 |
| zinnia_noir | halo2_lit | prove_core_s | 6 | 0.637 | [0.469, 0.884] | 5 / 1 |
| zinnia_noir | halo2_lit | end_to_end_s | 6 | 0.673 | [0.501, 0.928] | 5 / 1 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 6 | 1.412 | [0.908, 2.477] | 2 / 4 |
| zinnia_noir | halo2_lit | verify_s | 6 | 1.339 | [0.999, 1.890] | 1 / 5 |
| zinnia_noir | noir_lit | prove_core_s | 6 | 1.008 | [0.953, 1.080] | 4 / 2 |
| zinnia_noir | noir_lit | end_to_end_s | 6 | 1.006 | [0.951, 1.075] | 4 / 2 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 6 | 0.984 | [0.932, 1.052] | 4 / 2 |
| zinnia_noir | noir_lit | verify_s | 6 | 0.977 | [0.897, 1.072] | 3 / 3 |

## All-supported subset

10 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): npbench_deep_learning_softmax_softmax__softmax, npbench_polybench_gemm_gemm__kernel, npbench_polybench_mvt_mvt__kernel, npbench_weather_stencils_hdiff_hdiff__hdiff, pythran_cronbach__cronbach, pythran_harris__harris, pythran_log_likelihood__log_likelihood, pythran_multitype__times, pythran_rosen__rosen, pythran_vibr_energy__calculate_vibr_energy

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 10 | 0.866 | [0.733, 1.000] | 6 / 4 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 10 | 0.880 | [0.747, 1.017] | 6 / 4 |
| zinnia_halo2 | halo2_hw | verify_s | 10 | 0.952 | [0.855, 1.049] | 6 / 4 |
| zinnia_noir | noir_hw | prove_core_s | 10 | 1.045 | [0.946, 1.159] | 4 / 6 |
| zinnia_noir | noir_hw | end_to_end_s | 10 | 1.049 | [0.961, 1.151] | 5 / 5 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 10 | 1.104 | [1.005, 1.229] | 4 / 6 |
| zinnia_noir | noir_hw | verify_s | 10 | 1.067 | [0.976, 1.170] | 2 / 8 |
| zinnia_halo2 | noir_hw | prove_core_s | 10 | 1.892 | [1.299, 2.684] | 1 / 9 |
| zinnia_halo2 | noir_hw | end_to_end_s | 10 | 1.764 | [1.228, 2.455] | 1 / 9 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 10 | 0.983 | [0.571, 1.611] | 6 / 4 |
| zinnia_halo2 | noir_hw | verify_s | 10 | 1.062 | [0.733, 1.578] | 6 / 4 |
| zinnia_halo2 | halo2_lit | prove_core_s | 10 | 0.824 | [0.714, 0.922] | 8 / 2 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 10 | 0.837 | [0.729, 0.935] | 8 / 2 |
| zinnia_halo2 | halo2_lit | verify_s | 10 | 0.833 | [0.728, 0.945] | 8 / 2 |
| zinnia_halo2 | noir_lit | prove_core_s | 10 | 1.855 | [1.283, 2.646] | 1 / 9 |
| zinnia_halo2 | noir_lit | end_to_end_s | 10 | 1.725 | [1.220, 2.398] | 1 / 9 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 10 | 0.970 | [0.558, 1.612] | 6 / 4 |
| zinnia_halo2 | noir_lit | verify_s | 10 | 0.942 | [0.616, 1.463] | 6 / 4 |
| zinnia_noir | halo2_hw | prove_core_s | 10 | 0.478 | [0.311, 0.713] | 9 / 1 |
| zinnia_noir | halo2_hw | end_to_end_s | 10 | 0.524 | [0.354, 0.757] | 8 / 2 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 10 | 0.988 | [0.594, 1.661] | 5 / 5 |
| zinnia_noir | halo2_hw | verify_s | 10 | 0.956 | [0.590, 1.485] | 4 / 6 |
| zinnia_noir | halo2_lit | prove_core_s | 10 | 0.455 | [0.299, 0.673] | 9 / 1 |
| zinnia_noir | halo2_lit | end_to_end_s | 10 | 0.498 | [0.343, 0.714] | 9 / 1 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 10 | 0.940 | [0.576, 1.561] | 5 / 5 |
| zinnia_noir | halo2_lit | verify_s | 10 | 0.837 | [0.515, 1.305] | 5 / 5 |
| zinnia_noir | noir_lit | prove_core_s | 10 | 1.024 | [0.972, 1.091] | 6 / 4 |
| zinnia_noir | noir_lit | end_to_end_s | 10 | 1.026 | [0.976, 1.088] | 6 / 4 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 10 | 1.089 | [0.985, 1.228] | 4 / 6 |
| zinnia_noir | noir_lit | verify_s | 10 | 0.946 | [0.855, 1.033] | 5 / 5 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 10 | 0.866 | [0.733, 1.000] | 6 / 4 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 10 | 0.880 | [0.747, 1.017] | 6 / 4 |
| zinnia_halo2 | halo2_hw | verify_s | 10 | 0.952 | [0.855, 1.049] | 6 / 4 |
| zinnia_noir | noir_hw | prove_core_s | 10 | 1.045 | [0.946, 1.159] | 4 / 6 |
| zinnia_noir | noir_hw | end_to_end_s | 10 | 1.049 | [0.961, 1.151] | 5 / 5 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 10 | 1.104 | [1.005, 1.229] | 4 / 6 |
| zinnia_noir | noir_hw | verify_s | 10 | 1.067 | [0.976, 1.170] | 2 / 8 |
| zinnia_halo2 | noir_hw | prove_core_s | 10 | 1.892 | [1.299, 2.684] | 1 / 9 |
| zinnia_halo2 | noir_hw | end_to_end_s | 10 | 1.764 | [1.228, 2.455] | 1 / 9 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 10 | 0.983 | [0.571, 1.611] | 6 / 4 |
| zinnia_halo2 | noir_hw | verify_s | 10 | 1.062 | [0.733, 1.578] | 6 / 4 |
| zinnia_halo2 | halo2_lit | prove_core_s | 10 | 0.824 | [0.714, 0.922] | 8 / 2 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 10 | 0.837 | [0.729, 0.935] | 8 / 2 |
| zinnia_halo2 | halo2_lit | verify_s | 10 | 0.833 | [0.728, 0.945] | 8 / 2 |
| zinnia_halo2 | noir_lit | prove_core_s | 10 | 1.855 | [1.283, 2.646] | 1 / 9 |
| zinnia_halo2 | noir_lit | end_to_end_s | 10 | 1.725 | [1.220, 2.398] | 1 / 9 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 10 | 0.970 | [0.558, 1.612] | 6 / 4 |
| zinnia_halo2 | noir_lit | verify_s | 10 | 0.942 | [0.616, 1.463] | 6 / 4 |
| zinnia_noir | halo2_hw | prove_core_s | 10 | 0.478 | [0.311, 0.713] | 9 / 1 |
| zinnia_noir | halo2_hw | end_to_end_s | 10 | 0.524 | [0.354, 0.757] | 8 / 2 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 10 | 0.988 | [0.594, 1.661] | 5 / 5 |
| zinnia_noir | halo2_hw | verify_s | 10 | 0.956 | [0.590, 1.485] | 4 / 6 |
| zinnia_noir | halo2_lit | prove_core_s | 10 | 0.455 | [0.299, 0.673] | 9 / 1 |
| zinnia_noir | halo2_lit | end_to_end_s | 10 | 0.498 | [0.343, 0.714] | 9 / 1 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 10 | 0.940 | [0.576, 1.561] | 5 / 5 |
| zinnia_noir | halo2_lit | verify_s | 10 | 0.837 | [0.515, 1.305] | 5 / 5 |
| zinnia_noir | noir_lit | prove_core_s | 10 | 1.024 | [0.972, 1.091] | 6 / 4 |
| zinnia_noir | noir_lit | end_to_end_s | 10 | 1.026 | [0.976, 1.088] | 6 / 4 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 10 | 1.089 | [0.985, 1.228] | 4 / 6 |
| zinnia_noir | noir_lit | verify_s | 10 | 0.946 | [0.855, 1.033] | 5 / 5 |

## Excluded cases

- npbench_azimint_hist_azimint_hist__azimint_hist: rejected_compiler_limitation (np.histogram of runtime data): ZinniaException: IR generation failed: histogram: first slice supports compile-time-data histogram only; runtime/private values are not supported
- npbench_azimint_naive_azimint_naive__azimint_naive: rejected_no_provable_bound (array shape from a runtime value): ZinniaException: IR generation failed: shape element at axis 0 must be a compile-time constant int
- npbench_cavity_flow_cavity_flow__cavity_flow: rejected_no_provable_bound (array shape from a runtime value): ZinniaException: IR generation failed: shape element at axis 0 must be a compile-time constant int
- npbench_channel_flow_channel_flow__channel_flow: rejected_resource_limit (compile time over 300 s): timeout after 300s
- npbench_contour_integral_contour_integral__contour_integral: rejected_no_provable_bound (array shape from a runtime value): ZinniaException: IR generation failed: shape element at axis 0 must be a compile-time constant int
- npbench_deep_learning_lenet_lenet__lenet5: rejected_compiler_limitation (operation on a runtime-shaped array): ZinniaException: IR generation failed: this operation is not supported on an array with a runtime shape (the result of a boolean mask, nonzero/argwhere/unique, or a slice with runtime bounds)
- npbench_deep_learning_resnet_resnet__resnet_basicblock: rejected_resource_limit (compile time over 300 s): timeout after 300s
- npbench_mandelbrot1_mandelbrot1__mandelbrot: rejected_no_provable_bound (np.linspace num from a runtime value): ZinniaException: IR generation failed: linspace `num` must be a compile-time constant int
- npbench_mandelbrot2_mandelbrot2__mandelbrot: rejected_unsupported_construct (assignment to an Attribute): ZinniaException:   In method "mandelbrot", line 10
        Xi.shape = Yi.shape = C.shape = xn * yn
        ^^^^^^^^
InvalidAssignStatementException: Unsupported assignment target `Attribute`; only nam
- npbench_nbody_nbody__nbody: rejected_compiler_limitation (precondition of `pow_f` neither proved nor checkable in the circuit): ZinniaException: IR generation failed: op `pow_f` requires precondition `BoolComb { op: Or, operands: [Cmp { op: Ne, lhs: Var(Value(ValueId(48545))), rhs: LitFloat(ContractFloat(0.0)) }, Cmp { op: Ge,
- npbench_polybench_adi_adi__kernel: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- npbench_polybench_cholesky2_cholesky2__kernel: rejected_compiler_limitation (np.linalg.cholesky (result-returning linear algebra)): ZinniaException: IR generation failed: np.linalg.cholesky: implicit result-returning linear algebra is deferred because @zk_external currently returns one scalar, not an array witness; supply an expli
- npbench_polybench_correlation_correlation__kernel: rejected_no_provable_bound (array shape from a runtime value): ZinniaException: IR generation failed: shape element at axis 0 must be a compile-time constant int
- npbench_polybench_covariance_covariance__kernel: rejected_no_provable_bound (array shape from a runtime value): ZinniaException: IR generation failed: shape element at axis 0 must be a compile-time constant int
- npbench_polybench_doitgen_doitgen__kernel: rejected_compiler_limitation (operation on a runtime-shaped array): ZinniaException: IR generation failed: this operation is not supported on an array with a runtime shape (the result of a boolean mask, nonzero/argwhere/unique, or a slice with runtime bounds)
- npbench_polybench_fdtd_2d_fdtd_2d__kernel: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- npbench_polybench_floyd_warshall_floyd_warshall__kernel: rejected_resource_limit (compile time over 300 s): timeout after 300s
- npbench_polybench_heat_3d_heat_3d__kernel: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- npbench_polybench_jacobi_1d_jacobi_1d__kernel: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- npbench_polybench_jacobi_2d_jacobi_2d__kernel: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- npbench_polybench_nussinov_nussinov__kernel: rejected_no_provable_bound (array shape from a runtime value): ZinniaException: IR generation failed: shape element at axis 0 must be a compile-time constant int
- npbench_polybench_seidel_2d_seidel_2d__kernel: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- npbench_spmv_spmv__spmv: rejected_compiler_limitation (matmul of a runtime-shaped array): ZinniaException: IR generation failed: DynamicNDArray matmul is not yet supported for bounded-dynamic arrays; use static arrays or keep matmul outside the bounded-dynamic surface
- npbench_stockham_fft_stockham_fft__stockham_fft: rejected_unsupported_construct (np.mgrid): ZinniaException: IR generation failed: np.mgrid is not implemented
- pythran_allpairs_distances__allpairs_distances: rejected_no_provable_bound (np.arange length from a runtime value): ZinniaException: IR generation failed: range stop must be a compile-time constant int
- pythran_allpairs_distances_loops__allpairs_distances_loops: rejected_no_provable_bound (array shape from a runtime value): ZinniaException: IR generation failed: shape element at axis 1 must be a compile-time constant int
- pythran_another_quicksort__QuickSort: rejected_no_provable_bound (while-loop trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: the `while` condition was not proven false within loop_limit=1000 unrolled iterations. Bound the 
- pythran_approximated_callgraph__approximated_callgraph: rejected_compiler_limitation (list() of a runtime-shaped array): ZinniaException: IR generation failed: list() of an array with a runtime shape is not supported: its length is not a compile-time number
- pythran_babylonian__is_square: rejected_unsupported_construct (set): ZinniaException:   In method "is_square", line 5
        seen = {x}
               ^^^
UnsupportedLangFeatureException: Set literals are not supported inside @zk_circuit because circuits have no set t
- pythran_blacksholes__BlackScholes: rejected_resource_limit (compile time over 300 s): timeout after 300s
- pythran_calculate_u__timeloop: rejected_resource_limit (compile time over 300 s): timeout after 300s
- pythran_caxpy__CAXPY: rejected_resource_limit (compile time over 300 s): timeout after 300s
- pythran_ccopy__CCOPY: rejected_resource_limit (compile time over 300 s): timeout after 300s
- pythran_cdotc__CDOTC: rejected_resource_limit (compile time over 300 s): timeout after 300s
- pythran_cdotu__CDOTU: rejected_resource_limit (compile time over 300 s): timeout after 300s
- pythran_check_mask__check_mask: rejected_unsupported_construct (np.bitwise_and): ZinniaException: IR generation failed: np.bitwise_and is not implemented
- pythran_clip2__clip: rejected_compiler_limitation (shape element from a bounded runtime value where a constant is needed): ZinniaException: IR generation failed: shape element at axis 0 must be a compile-time constant int (it is a runtime value in [0, 2]; a runtime size is not supported here)
- pythran_collatz__collatz: rejected_compiler_limitation (operation on a runtime-shaped array): ZinniaException: IR generation failed: this operation is not supported on an array with a runtime shape (the result of a boolean mask, nonzero/argwhere/unique, or a slice with runtime bounds)
- pythran_collatz_modified__collatz_modified: rejected_no_provable_bound (while-loop trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: the `while` condition was not proven false within loop_limit=1000 unrolled iterations. Bound the 
- pythran_compute_subpix_2d_gaussian2__compute_subpix_2d_gaussian2: rejected_compiler_limitation (np.nonzero of a runtime-shaped array): ZinniaException: IR generation failed: nonzero: DynamicNDArray input is deferred; use static numeric arrays for now
- pythran_convnet__convnet: rejected_compiler_limitation (precondition of `div_f` neither proved nor checkable in the circuit): ZinniaException: IR generation failed: op `div_f` requires precondition `Cmp { op: Ne, lhs: Var(Value(ValueId(2013))), rhs: LitFloat(ContractFloat(0.0)) }` could not be lowered to a witness constraint
- pythran_deriv__deriv: rejected_resource_limit (compile time over 300 s): timeout after 300s
- pythran_diffusion__diffuseNumpy: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- pythran_emin__run: rejected_resource_limit (compile time over 300 s): timeout after 300s
- pythran_euler_challenge14__euler14: rejected_unsupported_construct (dict): ZinniaException:   In method "euler14", line 5
        lengths = {1: 0}
                  ^^^^^^
UnsupportedLangFeatureException: Dict literals are not supported inside @zk_circuit because mapping typ
- pythran_factorize_naive__factorize_naive: rejected_resource_limit (compile time over 300 s): timeout after 300s
- pythran_fannkuch__fannkuch2: rejected_compiler_limitation (range() with a runtime bound used as a value): ZinniaException: IR generation failed: range() with a runtime argument is only supported as the iterable of a `for` loop or comprehension (got a Integer that is not a compile-time constant)
- pythran_fdtd__fdtd: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- pythran_fft__fft: rejected_unsupported_construct (call of `fft` (no builtin, chip or numpy function of that name in scope)): ZinniaException: IR generation failed: Unknown function `fft` in @zk_circuit. If `fft` is a helper you defined, decorate it with @zk_chip (or @zk_external for off-circuit witness functions). no @zk_ch
- pythran_fibo__test: rejected_compiler_limitation (recursion deeper than the inlining limit (16)): ZinniaException: IR generation failed: RecursionLimitExceededError: recursion limit (16) exceeded while inlining @zk_chip `rfibo`. Each recursive call into a chip is unrolled at compile time; if your 
- pythran_gauss__gauss: rejected_compiler_limitation (chip that returns on some paths only): ZinniaException: IR generation failed: Chip return inference does not support partial returns without an explicit return annotation
- pythran_grouping__grouping: rejected_compiler_limitation (np.concatenate/np.stack of nested lists): ZinniaException: IR generation failed: concatenate/stack: unsupported element type List { elements: [Integer] }
- pythran_growcut__test: rejected_resource_limit (compile memory over 8G): out of memory after 56s
- pythran_guerre__guerre: rejected_resource_limit (compile time over 300 s): timeout after 300s
- pythran_histogram__histogram: rejected_no_provable_bound (list repetition with a runtime count): ZinniaException: IR generation failed: sequence repetition `seq * n` with a runtime count `n` is not supported: a list's length must be known at compile time (use np.zeros/np.full with a bounded shape
- pythran_hyantes_core__run: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- pythran_lapl2d__lapl2d: rejected_compiler_limitation (array name bound to different arrays on runtime paths, then written): ZinniaException: IR generation failed: Reading `In` is not supported here: a name that refers to this array on some paths and to another array on others (bound under a runtime condition) was used to m
- pythran_laplace__calc: rejected_no_provable_bound (list repetition with a runtime count): ZinniaException: IR generation failed: sequence repetition `seq * n` with a runtime count `n` is not supported: a list's length must be known at compile time (use np.zeros/np.full with a bounded shape
- pythran_ldpc_decoder__Decoding_logBP: rejected_unsupported_construct (method or attribute `.dot`): ZinniaException: IR generation failed: Named attribute `X.dot` not yet implemented in Rust IR generator
- pythran_lttb_pt__downsample: rejected_compiler_limitation (split sections from a bounded runtime value where a constant is needed): ZinniaException: IR generation failed: split sections must be a compile-time constant int (it is a runtime value in [-2, 1022]; a runtime size is not supported here)
- pythran_make_circle_two_points___make_circle_one_point: rejected_unsupported_construct (`is` / `is not`): ZinniaException:   In method "make_circle_two_points", line 13
            if c is None:
               ^^^^^^^^^
UnsupportedOperatorException: `is` / `is not` compare object identity, which circuit v
- pythran_mandel__mandel: rejected_resource_limit (compile memory over 8G): out of memory after 24s
- pythran_matrix_class_distance__matrix_class_distance: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- pythran_monte_carlo__montecarlo_integration: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- pythran_morphology__dilate_decompose_loops: rejected_no_provable_bound (range() trip count (proven bound too large to unroll)): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: the proven bound on the number of iterations of range() is 9223372036854775806, more than the 167
- pythran_mulmod__gf2mulmod: rejected_resource_limit (compile time over 300 s): timeout after 300s
- pythran_nd_local_maxima__local_maxima: rejected_unsupported_construct (unannotated entry parameter): ZinniaException:   In method "local_maxima", line 2
    def local_maxima(data: Float[Array, '8 6 4 2'], mode=wrap):
                                                    ^^^^
InvalidCircuitInputExceptio
- pythran_nqueens__n_queens: rejected_unsupported_construct (`is` / `is not`): ZinniaException:   In method "permutations", line 6
        if r is None:
           ^^^^^^^^^
UnsupportedOperatorException: `is` / `is not` compare object identity, which circuit values do not have; 
- pythran_pivot__pivot: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- pythran_primes_sieve2__get_primes7: rejected_compiler_limitation (range() with a runtime bound used as a value): ZinniaException: IR generation failed: range() with a runtime argument is only supported as the iterable of a `for` loop or comprehension (got a Integer that is not a compile-time constant)
- pythran_primes_sieve__primes_sieve: rejected_no_provable_bound (list repetition with a runtime count): ZinniaException: IR generation failed: sequence repetition `seq * n` with a runtime count `n` is not supported: a list's length must be known at compile time (use np.zeros/np.full with a bounded shape
- pythran_pselect__pselect: rejected_unsupported_construct (function used as a value): ZinniaException: IR generation failed: functions as values are not supported: `sel0` is used without being called (circuits have no function values; call it directly)
- pythran_queens_numba__solve: rejected_unsupported_construct (list mutated through a second name (aliasing)): ZinniaException: IR generation failed: mutating this list through `queens_x` is not supported: the same list is also reachable through `queens_x` (Python shares a list on assignment, argument passing 
- pythran_quicksort__quicksort: rejected_no_provable_bound (while-loop trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: the `while` condition was not proven false within loop_limit=1000 unrolled iterations. Bound the 
- pythran_repeating__repeating: rejected_no_provable_bound (array shape from a runtime value): ZinniaException: IR generation failed: shape element at axis 0 must be a compile-time constant int
- pythran_sexy_primes__primes_below: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- pythran_shallow_water__run: rejected_no_provable_bound (array shape from a runtime value): ZinniaException: IR generation failed: shape element at axis 0 must be a compile-time constant int
- pythran_slowparts__slowparts: rejected_no_provable_bound (array shape from a runtime value): ZinniaException: IR generation failed: shape element at axis 0 must be a compile-time constant int
- pythran_sobelfilter__sobelFilter: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- pythran_stone__whetstone: rejected_no_provable_bound (range() trip count): ZinniaException: IR generation failed: loop bound depends on a private value with no provable maximum: cannot bound the number of iterations of range() because its start or stop is a runtime value wit
- pythran_sum_primes__sum_primes: rejected_unsupported_construct (np.ceil): ZinniaException: IR generation failed: np.ceil is not implemented
- pythran_train_eq__train_eq: rejected_no_provable_bound (array shape from a runtime value): ZinniaException: IR generation failed: shape element at axis 0 must be a compile-time constant int
- pythran_train_eq__train_eq2: rejected_no_provable_bound (array shape from a runtime value): ZinniaException: IR generation failed: shape element at axis 0 must be a compile-time constant int
- pythran_wave_simulation__test: rejected_no_provable_bound (array shape from a runtime value): ZinniaException: IR generation failed: shape element at axis 1 must be a compile-time constant int
- pythran_zero__zero: rejected_resource_limit (compile memory over 8G): out of memory after 22s

## Cases without a program

No program for the case on that system (e.g. no handwritten baseline yet); not a failure.

(none)

## Failures by system

