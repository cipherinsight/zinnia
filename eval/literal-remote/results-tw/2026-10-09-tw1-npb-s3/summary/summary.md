# Epoch 2026-10-09-tw1-npb-s3

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

9 cases completed by every system that do more than copy inputs: npbench_deep_learning_mlp_mlp__mlp, npbench_polybench_durbin_durbin__kernel, npbench_polybench_ludcmp_ludcmp__kernel, npbench_scattering_self_energies_scattering_self_energies__scattering_self_energies, pythran_goodexpomeasure__goodExpoMeasure, pythran_ln__ln, pythran_multitype__check2, pythran_reverse_cumsum__reverse_cumsum, pythran_value_index__tri

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 9 | 0.862 | [0.738, 0.998] | 7 / 2 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 9 | 0.874 | [0.749, 1.012] | 7 / 2 |
| zinnia_halo2 | halo2_hw | verify_s | 9 | 1.068 | [0.878, 1.371] | 4 / 5 |
| zinnia_noir | noir_hw | prove_core_s | 9 | 0.977 | [0.905, 1.027] | 4 / 5 |
| zinnia_noir | noir_hw | end_to_end_s | 9 | 0.981 | [0.909, 1.030] | 3 / 6 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 9 | 1.041 | [0.948, 1.167] | 4 / 5 |
| zinnia_noir | noir_hw | verify_s | 9 | 1.146 | [1.025, 1.300] | 2 / 7 |
| zinnia_halo2 | noir_hw | prove_core_s | 9 | 1.434 | [0.792, 2.720] | 3 / 6 |
| zinnia_halo2 | noir_hw | end_to_end_s | 9 | 1.325 | [0.762, 2.341] | 3 / 6 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 9 | 0.790 | [0.394, 1.639] | 6 / 3 |
| zinnia_halo2 | noir_hw | verify_s | 9 | 1.367 | [0.681, 3.470] | 5 / 4 |
| zinnia_halo2 | halo2_lit | prove_core_s | 9 | 0.651 | [0.493, 0.838] | 8 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 9 | 0.660 | [0.501, 0.847] | 8 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 9 | 0.949 | [0.851, 1.065] | 5 / 4 |
| zinnia_halo2 | noir_lit | prove_core_s | 9 | 1.480 | [0.819, 2.724] | 3 / 6 |
| zinnia_halo2 | noir_lit | end_to_end_s | 9 | 1.369 | [0.787, 2.373] | 3 / 6 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 9 | 0.803 | [0.401, 1.645] | 6 / 3 |
| zinnia_halo2 | noir_lit | verify_s | 9 | 1.300 | [0.592, 3.740] | 6 / 3 |
| zinnia_noir | halo2_hw | prove_core_s | 9 | 0.587 | [0.320, 1.061] | 7 / 2 |
| zinnia_noir | halo2_hw | end_to_end_s | 9 | 0.647 | [0.375, 1.119] | 7 / 2 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 9 | 1.152 | [0.609, 2.178] | 4 / 5 |
| zinnia_noir | halo2_hw | verify_s | 9 | 0.896 | [0.397, 1.750] | 3 / 6 |
| zinnia_noir | halo2_lit | prove_core_s | 9 | 0.443 | [0.217, 0.912] | 7 / 2 |
| zinnia_noir | halo2_lit | end_to_end_s | 9 | 0.489 | [0.255, 0.959] | 7 / 2 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 9 | 0.870 | [0.420, 1.858] | 5 / 4 |
| zinnia_noir | halo2_lit | verify_s | 9 | 0.795 | [0.289, 1.666] | 3 / 6 |
| zinnia_noir | noir_lit | prove_core_s | 9 | 1.008 | [0.955, 1.072] | 5 / 4 |
| zinnia_noir | noir_lit | end_to_end_s | 9 | 1.013 | [0.956, 1.080] | 5 / 4 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 9 | 1.058 | [0.980, 1.163] | 4 / 5 |
| zinnia_noir | noir_lit | verify_s | 9 | 1.090 | [1.010, 1.203] | 2 / 7 |

## Copy-only cases (reported apart)

1 all-supported cases that only copy inputs: pythran_create_grid__create_grid

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 1 | 0.952 | [0.952, 0.952] | 1 / 0 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 1 | 0.955 | [0.955, 0.955] | 1 / 0 |
| zinnia_halo2 | halo2_hw | verify_s | 1 | 1.586 | [1.586, 1.586] | 0 / 1 |
| zinnia_noir | noir_hw | prove_core_s | 1 | 0.948 | [0.948, 0.948] | 1 / 0 |
| zinnia_noir | noir_hw | end_to_end_s | 1 | 0.931 | [0.931, 0.931] | 1 / 0 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 1 | 0.937 | [0.937, 0.937] | 1 / 0 |
| zinnia_noir | noir_hw | verify_s | 1 | 1.023 | [1.023, 1.023] | 0 / 1 |
| zinnia_halo2 | noir_hw | prove_core_s | 1 | 0.989 | [0.989, 0.989] | 1 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_s | 1 | 0.937 | [0.937, 0.937] | 1 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 1 | 0.459 | [0.459, 0.459] | 1 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 1 | 0.901 | [0.901, 0.901] | 1 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 1 | 0.866 | [0.866, 0.866] | 1 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 1 | 0.868 | [0.868, 0.868] | 1 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 1 | 1.188 | [1.188, 1.188] | 0 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 1 | 1.021 | [1.021, 1.021] | 0 / 1 |
| zinnia_halo2 | noir_lit | end_to_end_s | 1 | 0.989 | [0.989, 0.989] | 1 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 1 | 0.462 | [0.462, 0.462] | 1 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 1 | 0.844 | [0.844, 0.844] | 1 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 1 | 0.914 | [0.914, 0.914] | 1 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 1 | 0.949 | [0.949, 0.949] | 1 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 1 | 1.948 | [1.948, 1.948] | 0 / 1 |
| zinnia_noir | halo2_hw | verify_s | 1 | 1.800 | [1.800, 1.800] | 0 / 1 |
| zinnia_noir | halo2_lit | prove_core_s | 1 | 0.831 | [0.831, 0.831] | 1 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 1 | 0.863 | [0.863, 0.863] | 1 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 1 | 1.770 | [1.770, 1.770] | 0 / 1 |
| zinnia_noir | halo2_lit | verify_s | 1 | 1.349 | [1.349, 1.349] | 0 / 1 |
| zinnia_noir | noir_lit | prove_core_s | 1 | 0.980 | [0.980, 0.980] | 1 / 0 |
| zinnia_noir | noir_lit | end_to_end_s | 1 | 0.983 | [0.983, 0.983] | 1 / 0 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 1 | 0.943 | [0.943, 0.943] | 1 / 0 |
| zinnia_noir | noir_lit | verify_s | 1 | 0.958 | [0.958, 0.958] | 1 / 0 |

## Group npbench: headline cases

4 headline cases of group npbench: npbench_deep_learning_mlp_mlp__mlp, npbench_polybench_durbin_durbin__kernel, npbench_polybench_ludcmp_ludcmp__kernel, npbench_scattering_self_energies_scattering_self_energies__scattering_self_energies

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 4 | 0.943 | [0.742, 1.170] | 3 / 1 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 4 | 0.956 | [0.751, 1.192] | 3 / 1 |
| zinnia_halo2 | halo2_hw | verify_s | 4 | 1.308 | [0.971, 2.000] | 1 / 3 |
| zinnia_noir | noir_hw | prove_core_s | 4 | 1.003 | [0.982, 1.025] | 2 / 2 |
| zinnia_noir | noir_hw | end_to_end_s | 4 | 1.014 | [0.987, 1.049] | 1 / 3 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 4 | 1.118 | [0.999, 1.375] | 1 / 3 |
| zinnia_noir | noir_hw | verify_s | 4 | 1.226 | [1.030, 1.459] | 0 / 4 |
| zinnia_halo2 | noir_hw | prove_core_s | 4 | 2.466 | [1.392, 5.294] | 0 / 4 |
| zinnia_halo2 | noir_hw | end_to_end_s | 4 | 2.215 | [1.338, 4.188] | 0 / 4 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 4 | 1.437 | [0.728, 3.373] | 2 / 2 |
| zinnia_halo2 | noir_hw | verify_s | 4 | 2.556 | [0.996, 13.596] | 1 / 3 |
| zinnia_halo2 | halo2_lit | prove_core_s | 4 | 0.782 | [0.599, 0.943] | 4 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 4 | 0.793 | [0.610, 0.954] | 4 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 4 | 0.942 | [0.775, 1.144] | 2 / 2 |
| zinnia_halo2 | noir_lit | prove_core_s | 4 | 2.470 | [1.480, 5.140] | 0 / 4 |
| zinnia_halo2 | noir_lit | end_to_end_s | 4 | 2.213 | [1.404, 4.083] | 0 / 4 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 4 | 1.429 | [0.736, 3.279] | 2 / 2 |
| zinnia_halo2 | noir_lit | verify_s | 4 | 2.402 | [0.808, 17.574] | 2 / 2 |
| zinnia_noir | halo2_hw | prove_core_s | 4 | 0.383 | [0.207, 0.626] | 4 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 4 | 0.437 | [0.274, 0.659] | 4 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 4 | 0.744 | [0.450, 1.133] | 3 / 1 |
| zinnia_noir | halo2_hw | verify_s | 4 | 0.627 | [0.162, 1.455] | 2 / 2 |
| zinnia_noir | halo2_lit | prove_core_s | 4 | 0.318 | [0.114, 0.610] | 4 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 4 | 0.363 | [0.151, 0.644] | 4 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 4 | 0.617 | [0.247, 1.183] | 2 / 2 |
| zinnia_noir | halo2_lit | verify_s | 4 | 0.452 | [0.068, 1.458] | 2 / 2 |
| zinnia_noir | noir_lit | prove_core_s | 4 | 1.005 | [0.940, 1.124] | 3 / 1 |
| zinnia_noir | noir_lit | end_to_end_s | 4 | 1.012 | [0.944, 1.119] | 3 / 1 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 4 | 1.112 | [0.981, 1.318] | 1 / 3 |
| zinnia_noir | noir_lit | verify_s | 4 | 1.152 | [0.990, 1.403] | 1 / 3 |

## Group pythran: headline cases

5 headline cases of group pythran: pythran_goodexpomeasure__goodExpoMeasure, pythran_ln__ln, pythran_multitype__check2, pythran_reverse_cumsum__reverse_cumsum, pythran_value_index__tri

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 5 | 0.803 | [0.651, 0.963] | 4 / 1 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 5 | 0.814 | [0.665, 0.973] | 4 / 1 |
| zinnia_halo2 | halo2_hw | verify_s | 5 | 0.909 | [0.765, 1.053] | 3 / 2 |
| zinnia_noir | noir_hw | prove_core_s | 5 | 0.956 | [0.835, 1.041] | 2 / 3 |
| zinnia_noir | noir_hw | end_to_end_s | 5 | 0.955 | [0.838, 1.032] | 2 / 3 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 5 | 0.983 | [0.883, 1.080] | 3 / 2 |
| zinnia_noir | noir_hw | verify_s | 5 | 1.085 | [0.987, 1.280] | 2 / 3 |
| zinnia_halo2 | noir_hw | prove_core_s | 5 | 0.930 | [0.458, 2.098] | 3 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_s | 5 | 0.878 | [0.449, 1.897] | 3 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 5 | 0.489 | [0.209, 1.249] | 4 / 1 |
| zinnia_halo2 | noir_hw | verify_s | 5 | 0.829 | [0.458, 1.801] | 4 / 1 |
| zinnia_halo2 | halo2_lit | prove_core_s | 5 | 0.562 | [0.380, 0.831] | 4 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 5 | 0.570 | [0.386, 0.838] | 4 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 5 | 0.954 | [0.854, 1.081] | 3 / 2 |
| zinnia_halo2 | noir_lit | prove_core_s | 5 | 0.982 | [0.466, 2.204] | 3 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_s | 5 | 0.932 | [0.458, 1.957] | 3 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 5 | 0.506 | [0.212, 1.301] | 4 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 5 | 0.795 | [0.397, 1.870] | 4 / 1 |
| zinnia_noir | halo2_hw | prove_core_s | 5 | 0.826 | [0.320, 1.925] | 3 / 2 |
| zinnia_noir | halo2_hw | end_to_end_s | 5 | 0.886 | [0.371, 1.979] | 3 / 2 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 5 | 1.635 | [0.579, 4.238] | 1 / 4 |
| zinnia_noir | halo2_hw | verify_s | 5 | 1.190 | [0.459, 2.468] | 1 / 4 |
| zinnia_noir | halo2_lit | prove_core_s | 5 | 0.578 | [0.220, 1.726] | 3 / 2 |
| zinnia_noir | halo2_lit | end_to_end_s | 5 | 0.620 | [0.246, 1.770] | 3 / 2 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 5 | 1.145 | [0.398, 3.789] | 3 / 2 |
| zinnia_noir | halo2_lit | verify_s | 5 | 1.250 | [0.621, 2.348] | 1 / 4 |
| zinnia_noir | noir_lit | prove_core_s | 5 | 1.010 | [0.938, 1.093] | 2 / 3 |
| zinnia_noir | noir_lit | end_to_end_s | 5 | 1.014 | [0.931, 1.107] | 2 / 3 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 5 | 1.018 | [0.941, 1.112] | 3 / 2 |
| zinnia_noir | noir_lit | verify_s | 5 | 1.042 | [0.980, 1.097] | 1 / 4 |

## All-supported subset

10 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): npbench_deep_learning_mlp_mlp__mlp, npbench_polybench_durbin_durbin__kernel, npbench_polybench_ludcmp_ludcmp__kernel, npbench_scattering_self_energies_scattering_self_energies__scattering_self_energies, pythran_create_grid__create_grid, pythran_goodexpomeasure__goodExpoMeasure, pythran_ln__ln, pythran_multitype__check2, pythran_reverse_cumsum__reverse_cumsum, pythran_value_index__tri

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 10 | 0.871 | [0.755, 0.995] | 8 / 2 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 10 | 0.882 | [0.765, 1.009] | 8 / 2 |
| zinnia_halo2 | halo2_hw | verify_s | 10 | 1.112 | [0.910, 1.404] | 4 / 6 |
| zinnia_noir | noir_hw | prove_core_s | 10 | 0.974 | [0.907, 1.020] | 5 / 5 |
| zinnia_noir | noir_hw | end_to_end_s | 10 | 0.976 | [0.910, 1.023] | 4 / 6 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 10 | 1.030 | [0.946, 1.146] | 5 / 5 |
| zinnia_noir | noir_hw | verify_s | 10 | 1.133 | [1.024, 1.271] | 2 / 8 |
| zinnia_halo2 | noir_hw | prove_core_s | 10 | 1.382 | [0.799, 2.491] | 4 / 6 |
| zinnia_halo2 | noir_hw | end_to_end_s | 10 | 1.280 | [0.767, 2.175] | 4 / 6 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 10 | 0.748 | [0.394, 1.481] | 7 / 3 |
| zinnia_halo2 | noir_hw | verify_s | 10 | 1.311 | [0.694, 3.157] | 6 / 4 |
| zinnia_halo2 | halo2_lit | prove_core_s | 10 | 0.670 | [0.515, 0.847] | 9 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 10 | 0.678 | [0.523, 0.856] | 9 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 10 | 0.970 | [0.870, 1.086] | 5 / 5 |
| zinnia_halo2 | noir_lit | prove_core_s | 10 | 1.426 | [0.827, 2.518] | 3 / 7 |
| zinnia_halo2 | noir_lit | end_to_end_s | 10 | 1.325 | [0.798, 2.215] | 4 / 6 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 10 | 0.760 | [0.400, 1.484] | 7 / 3 |
| zinnia_halo2 | noir_lit | verify_s | 10 | 1.245 | [0.607, 3.368] | 7 / 3 |
| zinnia_noir | halo2_hw | prove_core_s | 10 | 0.614 | [0.350, 1.051] | 8 / 2 |
| zinnia_noir | halo2_hw | end_to_end_s | 10 | 0.673 | [0.409, 1.100] | 8 / 2 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 10 | 1.214 | [0.675, 2.170] | 4 / 6 |
| zinnia_noir | halo2_hw | verify_s | 10 | 0.960 | [0.442, 1.775] | 3 / 7 |
| zinnia_noir | halo2_lit | prove_core_s | 10 | 0.472 | [0.238, 0.918] | 8 / 2 |
| zinnia_noir | halo2_lit | end_to_end_s | 10 | 0.517 | [0.279, 0.962] | 8 / 2 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 10 | 0.934 | [0.466, 1.892] | 5 / 5 |
| zinnia_noir | halo2_lit | verify_s | 10 | 0.838 | [0.324, 1.644] | 3 / 7 |
| zinnia_noir | noir_lit | prove_core_s | 10 | 1.005 | [0.957, 1.063] | 6 / 4 |
| zinnia_noir | noir_lit | end_to_end_s | 10 | 1.010 | [0.959, 1.070] | 6 / 4 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 10 | 1.046 | [0.973, 1.144] | 5 / 5 |
| zinnia_noir | noir_lit | verify_s | 10 | 1.076 | [1.001, 1.182] | 3 / 7 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 10 | 0.871 | [0.755, 0.995] | 8 / 2 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 10 | 0.882 | [0.765, 1.009] | 8 / 2 |
| zinnia_halo2 | halo2_hw | verify_s | 10 | 1.112 | [0.910, 1.404] | 4 / 6 |
| zinnia_noir | noir_hw | prove_core_s | 10 | 0.974 | [0.907, 1.020] | 5 / 5 |
| zinnia_noir | noir_hw | end_to_end_s | 10 | 0.976 | [0.910, 1.023] | 4 / 6 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 10 | 1.030 | [0.946, 1.146] | 5 / 5 |
| zinnia_noir | noir_hw | verify_s | 10 | 1.133 | [1.024, 1.271] | 2 / 8 |
| zinnia_halo2 | noir_hw | prove_core_s | 10 | 1.382 | [0.799, 2.491] | 4 / 6 |
| zinnia_halo2 | noir_hw | end_to_end_s | 10 | 1.280 | [0.767, 2.175] | 4 / 6 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 10 | 0.748 | [0.394, 1.481] | 7 / 3 |
| zinnia_halo2 | noir_hw | verify_s | 10 | 1.311 | [0.694, 3.157] | 6 / 4 |
| zinnia_halo2 | halo2_lit | prove_core_s | 10 | 0.670 | [0.515, 0.847] | 9 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 10 | 0.678 | [0.523, 0.856] | 9 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 10 | 0.970 | [0.870, 1.086] | 5 / 5 |
| zinnia_halo2 | noir_lit | prove_core_s | 10 | 1.426 | [0.827, 2.518] | 3 / 7 |
| zinnia_halo2 | noir_lit | end_to_end_s | 10 | 1.325 | [0.798, 2.215] | 4 / 6 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 10 | 0.760 | [0.400, 1.484] | 7 / 3 |
| zinnia_halo2 | noir_lit | verify_s | 10 | 1.245 | [0.607, 3.368] | 7 / 3 |
| zinnia_noir | halo2_hw | prove_core_s | 10 | 0.614 | [0.350, 1.051] | 8 / 2 |
| zinnia_noir | halo2_hw | end_to_end_s | 10 | 0.673 | [0.409, 1.100] | 8 / 2 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 10 | 1.214 | [0.675, 2.170] | 4 / 6 |
| zinnia_noir | halo2_hw | verify_s | 10 | 0.960 | [0.442, 1.775] | 3 / 7 |
| zinnia_noir | halo2_lit | prove_core_s | 10 | 0.472 | [0.238, 0.918] | 8 / 2 |
| zinnia_noir | halo2_lit | end_to_end_s | 10 | 0.517 | [0.279, 0.962] | 8 / 2 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 10 | 0.934 | [0.466, 1.892] | 5 / 5 |
| zinnia_noir | halo2_lit | verify_s | 10 | 0.838 | [0.324, 1.644] | 3 / 7 |
| zinnia_noir | noir_lit | prove_core_s | 10 | 1.005 | [0.957, 1.063] | 6 / 4 |
| zinnia_noir | noir_lit | end_to_end_s | 10 | 1.010 | [0.959, 1.070] | 6 / 4 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 10 | 1.046 | [0.973, 1.144] | 5 / 5 |
| zinnia_noir | noir_lit | verify_s | 10 | 1.076 | [1.001, 1.182] | 3 / 7 |

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

