# Epoch 2026-10-09-tw1-npb-s1

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

9 cases completed by every system that do more than copy inputs: npbench_crc16_crc16__crc16, npbench_polybench_covariance2_covariance2__kernel, npbench_polybench_k3mm_k3mm__kernel, npbench_polybench_trmm_trmm__kernel, pythran_bubble_sort__bubble_sort, pythran_fbcorr__fbcorr, pythran_lap__laplacian, pythran_projection_simplex__projection_simplex, pythran_sumarray3d__summation

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 9 | 0.913 | [0.742, 1.072] | 5 / 4 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 9 | 0.927 | [0.753, 1.088] | 5 / 4 |
| zinnia_halo2 | halo2_hw | verify_s | 9 | 0.978 | [0.894, 1.053] | 4 / 5 |
| zinnia_noir | noir_hw | prove_core_s | 9 | 0.917 | [0.811, 0.999] | 5 / 4 |
| zinnia_noir | noir_hw | end_to_end_s | 9 | 0.928 | [0.812, 1.010] | 4 / 5 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 9 | 1.025 | [0.893, 1.152] | 4 / 5 |
| zinnia_noir | noir_hw | verify_s | 9 | 1.013 | [0.895, 1.165] | 5 / 4 |
| zinnia_halo2 | noir_hw | prove_core_s | 9 | 2.223 | [1.545, 3.309] | 0 / 9 |
| zinnia_halo2 | noir_hw | end_to_end_s | 9 | 2.050 | [1.477, 2.928] | 0 / 9 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 9 | 1.300 | [0.819, 2.097] | 3 / 6 |
| zinnia_halo2 | noir_hw | verify_s | 9 | 1.417 | [0.986, 2.119] | 2 / 7 |
| zinnia_halo2 | halo2_lit | prove_core_s | 9 | 0.757 | [0.684, 0.838] | 9 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 9 | 0.768 | [0.694, 0.851] | 9 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 9 | 0.834 | [0.720, 0.941] | 9 / 0 |
| zinnia_halo2 | noir_lit | prove_core_s | 9 | 2.267 | [1.524, 3.514] | 0 / 9 |
| zinnia_halo2 | noir_lit | end_to_end_s | 9 | 2.068 | [1.449, 3.054] | 1 / 8 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 9 | 1.308 | [0.810, 2.170] | 4 / 5 |
| zinnia_halo2 | noir_lit | verify_s | 9 | 1.369 | [0.954, 2.071] | 3 / 6 |
| zinnia_noir | halo2_hw | prove_core_s | 9 | 0.376 | [0.232, 0.588] | 9 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 9 | 0.419 | [0.271, 0.628] | 9 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 9 | 0.731 | [0.451, 1.159] | 5 / 4 |
| zinnia_noir | halo2_hw | verify_s | 9 | 0.699 | [0.454, 1.050] | 6 / 3 |
| zinnia_noir | halo2_lit | prove_core_s | 9 | 0.312 | [0.216, 0.444] | 9 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 9 | 0.348 | [0.254, 0.474] | 9 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 9 | 0.606 | [0.422, 0.874] | 6 / 3 |
| zinnia_noir | halo2_lit | verify_s | 9 | 0.597 | [0.379, 0.921] | 6 / 3 |
| zinnia_noir | noir_lit | prove_core_s | 9 | 0.935 | [0.832, 1.019] | 4 / 5 |
| zinnia_noir | noir_lit | end_to_end_s | 9 | 0.936 | [0.820, 1.031] | 3 / 6 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 9 | 1.031 | [0.891, 1.190] | 3 / 6 |
| zinnia_noir | noir_lit | verify_s | 9 | 0.979 | [0.850, 1.117] | 4 / 5 |

## Copy-only cases (reported apart)

1 all-supported cases that only copy inputs: pythran_multi_export__a

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 1 | 0.897 | [0.897, 0.897] | 1 / 0 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 1 | 0.900 | [0.900, 0.900] | 1 / 0 |
| zinnia_halo2 | halo2_hw | verify_s | 1 | 1.151 | [1.151, 1.151] | 0 / 1 |
| zinnia_noir | noir_hw | prove_core_s | 1 | 0.993 | [0.993, 0.993] | 1 / 0 |
| zinnia_noir | noir_hw | end_to_end_s | 1 | 0.991 | [0.991, 0.991] | 1 / 0 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 1 | 0.962 | [0.962, 0.962] | 1 / 0 |
| zinnia_noir | noir_hw | verify_s | 1 | 1.136 | [1.136, 1.136] | 0 / 1 |
| zinnia_halo2 | noir_hw | prove_core_s | 1 | 0.306 | [0.306, 0.306] | 1 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_s | 1 | 0.300 | [0.300, 0.300] | 1 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 1 | 0.133 | [0.133, 0.133] | 1 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 1 | 0.425 | [0.425, 0.425] | 1 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 1 | 0.967 | [0.967, 0.967] | 1 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 1 | 0.969 | [0.969, 0.969] | 1 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 1 | 0.966 | [0.966, 0.966] | 1 / 0 |
| zinnia_halo2 | noir_lit | prove_core_s | 1 | 0.306 | [0.306, 0.306] | 1 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_s | 1 | 0.298 | [0.298, 0.298] | 1 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 1 | 0.129 | [0.129, 0.129] | 1 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 1 | 0.378 | [0.378, 0.378] | 1 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 1 | 2.915 | [2.915, 2.915] | 0 / 1 |
| zinnia_noir | halo2_hw | end_to_end_s | 1 | 2.974 | [2.974, 2.974] | 0 / 1 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 1 | 6.507 | [6.507, 6.507] | 0 / 1 |
| zinnia_noir | halo2_hw | verify_s | 1 | 3.079 | [3.079, 3.079] | 0 / 1 |
| zinnia_noir | halo2_lit | prove_core_s | 1 | 3.142 | [3.142, 3.142] | 0 / 1 |
| zinnia_noir | halo2_lit | end_to_end_s | 1 | 3.202 | [3.202, 3.202] | 0 / 1 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 1 | 7.006 | [7.006, 7.006] | 0 / 1 |
| zinnia_noir | halo2_lit | verify_s | 1 | 2.586 | [2.586, 2.586] | 0 / 1 |
| zinnia_noir | noir_lit | prove_core_s | 1 | 0.994 | [0.994, 0.994] | 1 / 0 |
| zinnia_noir | noir_lit | end_to_end_s | 1 | 0.986 | [0.986, 0.986] | 1 / 0 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 1 | 0.935 | [0.935, 0.935] | 1 / 0 |
| zinnia_noir | noir_lit | verify_s | 1 | 1.012 | [1.012, 1.012] | 0 / 1 |

## Group npbench: headline cases

4 headline cases of group npbench: npbench_crc16_crc16__crc16, npbench_polybench_covariance2_covariance2__kernel, npbench_polybench_k3mm_k3mm__kernel, npbench_polybench_trmm_trmm__kernel

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 4 | 0.978 | [0.941, 1.035] | 3 / 1 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 4 | 0.996 | [0.961, 1.048] | 3 / 1 |
| zinnia_halo2 | halo2_hw | verify_s | 4 | 0.918 | [0.787, 0.998] | 3 / 1 |
| zinnia_noir | noir_hw | prove_core_s | 4 | 0.874 | [0.680, 1.010] | 3 / 1 |
| zinnia_noir | noir_hw | end_to_end_s | 4 | 0.873 | [0.654, 1.018] | 2 / 2 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 4 | 1.000 | [0.756, 1.191] | 1 / 3 |
| zinnia_noir | noir_hw | verify_s | 4 | 1.006 | [0.819, 1.257] | 2 / 2 |
| zinnia_halo2 | noir_hw | prove_core_s | 4 | 2.216 | [1.812, 2.536] | 0 / 4 |
| zinnia_halo2 | noir_hw | end_to_end_s | 4 | 2.073 | [1.738, 2.290] | 0 / 4 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 4 | 1.448 | [1.167, 1.723] | 0 / 4 |
| zinnia_halo2 | noir_hw | verify_s | 4 | 1.556 | [1.197, 2.255] | 0 / 4 |
| zinnia_halo2 | halo2_lit | prove_core_s | 4 | 0.731 | [0.628, 0.868] | 4 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 4 | 0.745 | [0.642, 0.879] | 4 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 4 | 0.745 | [0.586, 0.922] | 4 / 0 |
| zinnia_halo2 | noir_lit | prove_core_s | 4 | 2.246 | [1.786, 2.626] | 0 / 4 |
| zinnia_halo2 | noir_lit | end_to_end_s | 4 | 2.087 | [1.681, 2.382] | 0 / 4 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 4 | 1.421 | [1.100, 1.752] | 1 / 3 |
| zinnia_halo2 | noir_lit | verify_s | 4 | 1.376 | [1.095, 1.943] | 0 / 4 |
| zinnia_noir | halo2_hw | prove_core_s | 4 | 0.386 | [0.279, 0.520] | 4 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 4 | 0.420 | [0.304, 0.563] | 4 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 4 | 0.688 | [0.465, 0.961] | 3 / 1 |
| zinnia_noir | halo2_hw | verify_s | 4 | 0.593 | [0.375, 0.917] | 4 / 0 |
| zinnia_noir | halo2_lit | prove_core_s | 4 | 0.288 | [0.239, 0.347] | 4 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 4 | 0.314 | [0.264, 0.373] | 4 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 4 | 0.514 | [0.405, 0.631] | 4 / 0 |
| zinnia_noir | halo2_lit | verify_s | 4 | 0.481 | [0.304, 0.762] | 4 / 0 |
| zinnia_noir | noir_lit | prove_core_s | 4 | 0.886 | [0.706, 1.011] | 2 / 2 |
| zinnia_noir | noir_lit | end_to_end_s | 4 | 0.879 | [0.684, 1.028] | 2 / 2 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 4 | 0.981 | [0.761, 1.181] | 1 / 3 |
| zinnia_noir | noir_lit | verify_s | 4 | 0.889 | [0.720, 1.062] | 2 / 2 |

## Group pythran: headline cases

5 headline cases of group pythran: pythran_bubble_sort__bubble_sort, pythran_fbcorr__fbcorr, pythran_lap__laplacian, pythran_projection_simplex__projection_simplex, pythran_sumarray3d__summation

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 5 | 0.864 | [0.600, 1.162] | 2 / 3 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 5 | 0.875 | [0.611, 1.176] | 2 / 3 |
| zinnia_halo2 | halo2_hw | verify_s | 5 | 1.029 | [0.933, 1.112] | 1 / 4 |
| zinnia_noir | noir_hw | prove_core_s | 5 | 0.953 | [0.880, 1.018] | 2 / 3 |
| zinnia_noir | noir_hw | end_to_end_s | 5 | 0.974 | [0.925, 1.024] | 2 / 3 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 5 | 1.046 | [0.930, 1.215] | 3 / 2 |
| zinnia_noir | noir_hw | verify_s | 5 | 1.019 | [0.902, 1.217] | 3 / 2 |
| zinnia_halo2 | noir_hw | prove_core_s | 5 | 2.228 | [1.163, 4.511] | 0 / 5 |
| zinnia_halo2 | noir_hw | end_to_end_s | 5 | 2.032 | [1.127, 3.831] | 0 / 5 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 5 | 1.192 | [0.548, 2.922] | 3 / 2 |
| zinnia_halo2 | noir_hw | verify_s | 5 | 1.314 | [0.762, 2.545] | 2 / 3 |
| zinnia_halo2 | halo2_lit | prove_core_s | 5 | 0.778 | [0.693, 0.874] | 5 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 5 | 0.787 | [0.698, 0.888] | 5 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 5 | 0.913 | [0.816, 0.981] | 5 / 0 |
| zinnia_halo2 | noir_lit | prove_core_s | 5 | 2.284 | [1.139, 5.067] | 0 / 5 |
| zinnia_halo2 | noir_lit | end_to_end_s | 5 | 2.052 | [1.100, 4.148] | 1 / 4 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 5 | 1.224 | [0.543, 3.142] | 3 / 2 |
| zinnia_halo2 | noir_lit | verify_s | 5 | 1.363 | [0.738, 2.897] | 3 / 2 |
| zinnia_noir | halo2_hw | prove_core_s | 5 | 0.369 | [0.165, 0.792] | 5 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 5 | 0.419 | [0.198, 0.826] | 5 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 5 | 0.768 | [0.330, 1.629] | 2 / 3 |
| zinnia_noir | halo2_hw | verify_s | 5 | 0.797 | [0.401, 1.417] | 2 / 3 |
| zinnia_noir | halo2_lit | prove_core_s | 5 | 0.333 | [0.172, 0.609] | 5 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 5 | 0.377 | [0.219, 0.637] | 5 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 5 | 0.691 | [0.371, 1.254] | 2 / 3 |
| zinnia_noir | halo2_lit | verify_s | 5 | 0.708 | [0.338, 1.299] | 2 / 3 |
| zinnia_noir | noir_lit | prove_core_s | 5 | 0.976 | [0.886, 1.050] | 2 / 3 |
| zinnia_noir | noir_lit | end_to_end_s | 5 | 0.984 | [0.885, 1.056] | 1 / 4 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 5 | 1.074 | [0.913, 1.306] | 2 / 3 |
| zinnia_noir | noir_lit | verify_s | 5 | 1.057 | [0.911, 1.234] | 2 / 3 |

## All-supported subset

10 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): npbench_crc16_crc16__crc16, npbench_polybench_covariance2_covariance2__kernel, npbench_polybench_k3mm_k3mm__kernel, npbench_polybench_trmm_trmm__kernel, pythran_bubble_sort__bubble_sort, pythran_fbcorr__fbcorr, pythran_lap__laplacian, pythran_multi_export__a, pythran_projection_simplex__projection_simplex, pythran_sumarray3d__summation

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 10 | 0.911 | [0.751, 1.053] | 6 / 4 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 10 | 0.924 | [0.763, 1.068] | 6 / 4 |
| zinnia_halo2 | halo2_hw | verify_s | 10 | 0.994 | [0.912, 1.070] | 4 / 6 |
| zinnia_noir | noir_hw | prove_core_s | 10 | 0.924 | [0.830, 0.998] | 6 / 4 |
| zinnia_noir | noir_hw | end_to_end_s | 10 | 0.934 | [0.830, 1.009] | 5 / 5 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 10 | 1.019 | [0.900, 1.134] | 5 / 5 |
| zinnia_noir | noir_hw | verify_s | 10 | 1.025 | [0.914, 1.165] | 5 / 5 |
| zinnia_halo2 | noir_hw | prove_core_s | 10 | 1.823 | [1.084, 2.951] | 1 / 9 |
| zinnia_halo2 | noir_hw | end_to_end_s | 10 | 1.692 | [1.031, 2.633] | 1 / 9 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 10 | 1.035 | [0.562, 1.832] | 4 / 6 |
| zinnia_halo2 | noir_hw | verify_s | 10 | 1.256 | [0.846, 1.899] | 3 / 7 |
| zinnia_halo2 | halo2_lit | prove_core_s | 10 | 0.775 | [0.699, 0.859] | 10 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 10 | 0.786 | [0.709, 0.869] | 10 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 10 | 0.847 | [0.740, 0.945] | 10 / 0 |
| zinnia_halo2 | noir_lit | prove_core_s | 10 | 1.856 | [1.084, 3.100] | 1 / 9 |
| zinnia_halo2 | noir_lit | end_to_end_s | 10 | 1.704 | [1.028, 2.727] | 2 / 8 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 10 | 1.037 | [0.554, 1.885] | 5 / 5 |
| zinnia_halo2 | noir_lit | verify_s | 10 | 1.204 | [0.798, 1.864] | 4 / 6 |
| zinnia_noir | halo2_hw | prove_core_s | 10 | 0.462 | [0.263, 0.820] | 9 / 1 |
| zinnia_noir | halo2_hw | end_to_end_s | 10 | 0.510 | [0.304, 0.871] | 9 / 1 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 10 | 0.910 | [0.511, 1.657] | 5 / 5 |
| zinnia_noir | halo2_hw | verify_s | 10 | 0.811 | [0.513, 1.290] | 6 / 4 |
| zinnia_noir | halo2_lit | prove_core_s | 10 | 0.393 | [0.240, 0.699] | 9 / 1 |
| zinnia_noir | halo2_lit | end_to_end_s | 10 | 0.434 | [0.278, 0.745] | 9 / 1 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 10 | 0.774 | [0.467, 1.415] | 6 / 4 |
| zinnia_noir | halo2_lit | verify_s | 10 | 0.691 | [0.431, 1.116] | 6 / 4 |
| zinnia_noir | noir_lit | prove_core_s | 10 | 0.941 | [0.848, 1.016] | 5 / 5 |
| zinnia_noir | noir_lit | end_to_end_s | 10 | 0.941 | [0.837, 1.026] | 4 / 6 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 10 | 1.021 | [0.895, 1.164] | 4 / 6 |
| zinnia_noir | noir_lit | verify_s | 10 | 0.982 | [0.864, 1.109] | 4 / 6 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 10 | 0.911 | [0.751, 1.053] | 6 / 4 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 10 | 0.924 | [0.763, 1.068] | 6 / 4 |
| zinnia_halo2 | halo2_hw | verify_s | 10 | 0.994 | [0.912, 1.070] | 4 / 6 |
| zinnia_noir | noir_hw | prove_core_s | 10 | 0.924 | [0.830, 0.998] | 6 / 4 |
| zinnia_noir | noir_hw | end_to_end_s | 10 | 0.934 | [0.830, 1.009] | 5 / 5 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 10 | 1.019 | [0.900, 1.134] | 5 / 5 |
| zinnia_noir | noir_hw | verify_s | 10 | 1.025 | [0.914, 1.165] | 5 / 5 |
| zinnia_halo2 | noir_hw | prove_core_s | 10 | 1.823 | [1.084, 2.951] | 1 / 9 |
| zinnia_halo2 | noir_hw | end_to_end_s | 10 | 1.692 | [1.031, 2.633] | 1 / 9 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 10 | 1.035 | [0.562, 1.832] | 4 / 6 |
| zinnia_halo2 | noir_hw | verify_s | 10 | 1.256 | [0.846, 1.899] | 3 / 7 |
| zinnia_halo2 | halo2_lit | prove_core_s | 10 | 0.775 | [0.699, 0.859] | 10 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 10 | 0.786 | [0.709, 0.869] | 10 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 10 | 0.847 | [0.740, 0.945] | 10 / 0 |
| zinnia_halo2 | noir_lit | prove_core_s | 10 | 1.856 | [1.084, 3.100] | 1 / 9 |
| zinnia_halo2 | noir_lit | end_to_end_s | 10 | 1.704 | [1.028, 2.727] | 2 / 8 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 10 | 1.037 | [0.554, 1.885] | 5 / 5 |
| zinnia_halo2 | noir_lit | verify_s | 10 | 1.204 | [0.798, 1.864] | 4 / 6 |
| zinnia_noir | halo2_hw | prove_core_s | 10 | 0.462 | [0.263, 0.820] | 9 / 1 |
| zinnia_noir | halo2_hw | end_to_end_s | 10 | 0.510 | [0.304, 0.871] | 9 / 1 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 10 | 0.910 | [0.511, 1.657] | 5 / 5 |
| zinnia_noir | halo2_hw | verify_s | 10 | 0.811 | [0.513, 1.290] | 6 / 4 |
| zinnia_noir | halo2_lit | prove_core_s | 10 | 0.393 | [0.240, 0.699] | 9 / 1 |
| zinnia_noir | halo2_lit | end_to_end_s | 10 | 0.434 | [0.278, 0.745] | 9 / 1 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 10 | 0.774 | [0.467, 1.415] | 6 / 4 |
| zinnia_noir | halo2_lit | verify_s | 10 | 0.691 | [0.431, 1.116] | 6 / 4 |
| zinnia_noir | noir_lit | prove_core_s | 10 | 0.941 | [0.848, 1.016] | 5 / 5 |
| zinnia_noir | noir_lit | end_to_end_s | 10 | 0.941 | [0.837, 1.026] | 4 / 6 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 10 | 1.021 | [0.895, 1.164] | 4 / 6 |
| zinnia_noir | noir_lit | verify_s | 10 | 0.982 | [0.864, 1.109] | 4 / 6 |

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

