# Epoch 2026-10-09-tw1-npb-s0

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

9 cases completed by every system that do more than copy inputs: npbench_compute_compute__compute, npbench_polybench_cholesky_cholesky__kernel, npbench_polybench_k2mm_k2mm__kernel, npbench_polybench_trisolv_trisolv__kernel, pythran_arc_distance__arc_distance, pythran_euclidean_distance_square__euclidean_distance_square, pythran_l2norm__l2_norm, pythran_matmul__matrix_multiply, pythran_spectraldns__cross2

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 9 | 0.924 | [0.882, 0.960] | 9 / 0 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 9 | 0.937 | [0.893, 0.974] | 9 / 0 |
| zinnia_halo2 | halo2_hw | verify_s | 9 | 0.932 | [0.902, 0.966] | 8 / 1 |
| zinnia_noir | noir_hw | prove_core_s | 9 | 0.999 | [0.965, 1.047] | 6 / 3 |
| zinnia_noir | noir_hw | end_to_end_s | 9 | 1.007 | [0.975, 1.051] | 7 / 2 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 9 | 1.068 | [1.016, 1.124] | 3 / 6 |
| zinnia_noir | noir_hw | verify_s | 9 | 0.918 | [0.808, 1.018] | 5 / 4 |
| zinnia_halo2 | noir_hw | prove_core_s | 9 | 2.316 | [1.640, 3.111] | 1 / 8 |
| zinnia_halo2 | noir_hw | end_to_end_s | 9 | 2.194 | [1.578, 2.891] | 1 / 8 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 9 | 1.321 | [0.867, 1.898] | 2 / 7 |
| zinnia_halo2 | noir_hw | verify_s | 9 | 1.089 | [0.813, 1.460] | 5 / 4 |
| zinnia_halo2 | halo2_lit | prove_core_s | 9 | 0.829 | [0.768, 0.892] | 8 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 9 | 0.840 | [0.780, 0.903] | 8 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 9 | 0.897 | [0.838, 0.964] | 7 / 2 |
| zinnia_halo2 | noir_lit | prove_core_s | 9 | 2.403 | [1.710, 3.209] | 1 / 8 |
| zinnia_halo2 | noir_lit | end_to_end_s | 9 | 2.248 | [1.627, 2.949] | 1 / 8 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 9 | 1.352 | [0.893, 1.945] | 2 / 7 |
| zinnia_halo2 | noir_lit | verify_s | 9 | 1.107 | [0.845, 1.442] | 5 / 4 |
| zinnia_noir | halo2_hw | prove_core_s | 9 | 0.399 | [0.292, 0.576] | 8 / 1 |
| zinnia_noir | halo2_hw | end_to_end_s | 9 | 0.430 | [0.322, 0.605] | 8 / 1 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 9 | 0.758 | [0.550, 1.116] | 7 / 2 |
| zinnia_noir | halo2_hw | verify_s | 9 | 0.785 | [0.572, 1.110] | 7 / 2 |
| zinnia_noir | halo2_lit | prove_core_s | 9 | 0.358 | [0.258, 0.530] | 8 / 1 |
| zinnia_noir | halo2_lit | end_to_end_s | 9 | 0.386 | [0.285, 0.558] | 8 / 1 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 9 | 0.680 | [0.489, 1.026] | 7 / 2 |
| zinnia_noir | halo2_lit | verify_s | 9 | 0.756 | [0.539, 1.109] | 7 / 2 |
| zinnia_noir | noir_lit | prove_core_s | 9 | 1.037 | [0.980, 1.102] | 4 / 5 |
| zinnia_noir | noir_lit | end_to_end_s | 9 | 1.032 | [0.975, 1.094] | 4 / 5 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 9 | 1.093 | [1.023, 1.168] | 3 / 6 |
| zinnia_noir | noir_lit | verify_s | 9 | 0.933 | [0.803, 1.079] | 6 / 3 |

## Copy-only cases (reported apart)

1 all-supported cases that only copy inputs: pythran_perm__permutations

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 1 | 0.947 | [0.947, 0.947] | 1 / 0 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 1 | 0.949 | [0.949, 0.949] | 1 / 0 |
| zinnia_halo2 | halo2_hw | verify_s | 1 | 0.916 | [0.916, 0.916] | 1 / 0 |
| zinnia_noir | noir_hw | prove_core_s | 1 | 0.997 | [0.997, 0.997] | 1 / 0 |
| zinnia_noir | noir_hw | end_to_end_s | 1 | 0.999 | [0.999, 0.999] | 1 / 0 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 1 | 0.940 | [0.940, 0.940] | 1 / 0 |
| zinnia_noir | noir_hw | verify_s | 1 | 0.882 | [0.882, 0.882] | 1 / 0 |
| zinnia_halo2 | noir_hw | prove_core_s | 1 | 0.347 | [0.347, 0.347] | 1 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_s | 1 | 0.343 | [0.343, 0.343] | 1 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 1 | 0.159 | [0.159, 0.159] | 1 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 1 | 0.335 | [0.335, 0.335] | 1 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 1 | 0.834 | [0.834, 0.834] | 1 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 1 | 0.837 | [0.837, 0.837] | 1 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 1 | 1.161 | [1.161, 1.161] | 0 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 1 | 0.340 | [0.340, 0.340] | 1 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_s | 1 | 0.334 | [0.334, 0.334] | 1 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 1 | 0.158 | [0.158, 0.158] | 1 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 1 | 0.393 | [0.393, 0.393] | 1 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 1 | 2.719 | [2.719, 2.719] | 0 / 1 |
| zinnia_noir | halo2_hw | end_to_end_s | 1 | 2.763 | [2.763, 2.763] | 0 / 1 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 1 | 5.629 | [5.629, 5.629] | 0 / 1 |
| zinnia_noir | halo2_hw | verify_s | 1 | 2.413 | [2.413, 2.413] | 0 / 1 |
| zinnia_noir | halo2_lit | prove_core_s | 1 | 2.395 | [2.395, 2.395] | 0 / 1 |
| zinnia_noir | halo2_lit | end_to_end_s | 1 | 2.435 | [2.435, 2.435] | 0 / 1 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 1 | 4.961 | [4.961, 4.961] | 0 / 1 |
| zinnia_noir | halo2_lit | verify_s | 1 | 3.057 | [3.057, 3.057] | 0 / 1 |
| zinnia_noir | noir_lit | prove_core_s | 1 | 0.977 | [0.977, 0.977] | 1 / 0 |
| zinnia_noir | noir_lit | end_to_end_s | 1 | 0.973 | [0.973, 0.973] | 1 / 0 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 1 | 0.936 | [0.936, 0.936] | 1 / 0 |
| zinnia_noir | noir_lit | verify_s | 1 | 1.034 | [1.034, 1.034] | 0 / 1 |

## Group npbench: headline cases

4 headline cases of group npbench: npbench_compute_compute__compute, npbench_polybench_cholesky_cholesky__kernel, npbench_polybench_k2mm_k2mm__kernel, npbench_polybench_trisolv_trisolv__kernel

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 4 | 0.915 | [0.841, 0.957] | 4 / 0 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 4 | 0.928 | [0.853, 0.974] | 4 / 0 |
| zinnia_halo2 | halo2_hw | verify_s | 4 | 0.912 | [0.885, 0.954] | 4 / 0 |
| zinnia_noir | noir_hw | prove_core_s | 4 | 0.981 | [0.941, 1.024] | 2 / 2 |
| zinnia_noir | noir_hw | end_to_end_s | 4 | 0.983 | [0.949, 1.019] | 3 / 1 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 4 | 1.049 | [0.980, 1.122] | 2 / 2 |
| zinnia_noir | noir_hw | verify_s | 4 | 0.877 | [0.695, 1.025] | 2 / 2 |
| zinnia_halo2 | noir_hw | prove_core_s | 4 | 2.236 | [1.647, 2.817] | 0 / 4 |
| zinnia_halo2 | noir_hw | end_to_end_s | 4 | 2.144 | [1.590, 2.674] | 0 / 4 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 4 | 1.321 | [0.856, 1.914] | 1 / 3 |
| zinnia_halo2 | noir_hw | verify_s | 4 | 1.064 | [0.781, 1.587] | 3 / 1 |
| zinnia_halo2 | halo2_lit | prove_core_s | 4 | 0.797 | [0.697, 0.912] | 4 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 4 | 0.809 | [0.709, 0.923] | 4 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 4 | 0.869 | [0.798, 0.946] | 4 / 0 |
| zinnia_halo2 | noir_lit | prove_core_s | 4 | 2.385 | [1.810, 3.124] | 0 / 4 |
| zinnia_halo2 | noir_lit | end_to_end_s | 4 | 2.261 | [1.732, 2.923] | 0 / 4 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 4 | 1.392 | [0.946, 2.007] | 1 / 3 |
| zinnia_halo2 | noir_lit | verify_s | 4 | 1.170 | [0.968, 1.547] | 2 / 2 |
| zinnia_noir | halo2_hw | prove_core_s | 4 | 0.401 | [0.284, 0.561] | 4 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 4 | 0.425 | [0.309, 0.585] | 4 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 4 | 0.737 | [0.498, 1.108] | 3 / 1 |
| zinnia_noir | halo2_hw | verify_s | 4 | 0.752 | [0.516, 1.096] | 3 / 1 |
| zinnia_noir | halo2_lit | prove_core_s | 4 | 0.350 | [0.285, 0.499] | 4 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 4 | 0.371 | [0.305, 0.525] | 4 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 4 | 0.642 | [0.508, 0.984] | 3 / 1 |
| zinnia_noir | halo2_lit | verify_s | 4 | 0.717 | [0.518, 0.991] | 3 / 1 |
| zinnia_noir | noir_lit | prove_core_s | 4 | 1.047 | [0.942, 1.131] | 1 / 3 |
| zinnia_noir | noir_lit | end_to_end_s | 4 | 1.037 | [0.929, 1.119] | 1 / 3 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 4 | 1.105 | [0.996, 1.225] | 1 / 3 |
| zinnia_noir | noir_lit | verify_s | 4 | 0.965 | [0.706, 1.279] | 2 / 2 |

## Group pythran: headline cases

5 headline cases of group pythran: pythran_arc_distance__arc_distance, pythran_euclidean_distance_square__euclidean_distance_square, pythran_l2norm__l2_norm, pythran_matmul__matrix_multiply, pythran_spectraldns__cross2

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 5 | 0.931 | [0.879, 0.973] | 5 / 0 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 5 | 0.944 | [0.890, 0.986] | 5 / 0 |
| zinnia_halo2 | halo2_hw | verify_s | 5 | 0.948 | [0.904, 0.994] | 4 / 1 |
| zinnia_noir | noir_hw | prove_core_s | 5 | 1.014 | [0.972, 1.091] | 4 / 1 |
| zinnia_noir | noir_hw | end_to_end_s | 5 | 1.026 | [0.992, 1.096] | 4 / 1 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 5 | 1.085 | [1.016, 1.159] | 1 / 4 |
| zinnia_noir | noir_hw | verify_s | 5 | 0.951 | [0.828, 1.052] | 3 / 2 |
| zinnia_halo2 | noir_hw | prove_core_s | 5 | 2.382 | [1.375, 3.787] | 1 / 4 |
| zinnia_halo2 | noir_hw | end_to_end_s | 5 | 2.234 | [1.316, 3.446] | 1 / 4 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 5 | 1.322 | [0.698, 2.292] | 1 / 4 |
| zinnia_halo2 | noir_hw | verify_s | 5 | 1.109 | [0.703, 1.682] | 2 / 3 |
| zinnia_halo2 | halo2_lit | prove_core_s | 5 | 0.854 | [0.801, 0.930] | 4 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 5 | 0.866 | [0.811, 0.940] | 4 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 5 | 0.920 | [0.834, 1.016] | 3 / 2 |
| zinnia_halo2 | noir_lit | prove_core_s | 5 | 2.418 | [1.410, 3.795] | 1 / 4 |
| zinnia_halo2 | noir_lit | end_to_end_s | 5 | 2.238 | [1.340, 3.415] | 1 / 4 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 5 | 1.320 | [0.705, 2.326] | 1 / 4 |
| zinnia_halo2 | noir_lit | verify_s | 5 | 1.058 | [0.699, 1.566] | 3 / 2 |
| zinnia_noir | halo2_hw | prove_core_s | 5 | 0.396 | [0.243, 0.707] | 4 / 1 |
| zinnia_noir | halo2_hw | end_to_end_s | 5 | 0.434 | [0.278, 0.744] | 4 / 1 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 5 | 0.775 | [0.485, 1.396] | 4 / 1 |
| zinnia_noir | halo2_hw | verify_s | 5 | 0.813 | [0.509, 1.313] | 4 / 1 |
| zinnia_noir | halo2_lit | prove_core_s | 5 | 0.364 | [0.215, 0.689] | 4 / 1 |
| zinnia_noir | halo2_lit | end_to_end_s | 5 | 0.398 | [0.247, 0.729] | 4 / 1 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 5 | 0.711 | [0.431, 1.367] | 4 / 1 |
| zinnia_noir | halo2_lit | verify_s | 5 | 0.789 | [0.461, 1.392] | 4 / 1 |
| zinnia_noir | noir_lit | prove_core_s | 5 | 1.029 | [0.973, 1.127] | 3 / 2 |
| zinnia_noir | noir_lit | end_to_end_s | 5 | 1.028 | [0.974, 1.116] | 3 / 2 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 5 | 1.083 | [1.005, 1.175] | 2 / 3 |
| zinnia_noir | noir_lit | verify_s | 5 | 0.908 | [0.783, 1.034] | 4 / 1 |

## All-supported subset

10 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): npbench_compute_compute__compute, npbench_polybench_cholesky_cholesky__kernel, npbench_polybench_k2mm_k2mm__kernel, npbench_polybench_trisolv_trisolv__kernel, pythran_arc_distance__arc_distance, pythran_euclidean_distance_square__euclidean_distance_square, pythran_l2norm__l2_norm, pythran_matmul__matrix_multiply, pythran_perm__permutations, pythran_spectraldns__cross2

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 10 | 0.926 | [0.888, 0.959] | 10 / 0 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 10 | 0.938 | [0.899, 0.972] | 10 / 0 |
| zinnia_halo2 | halo2_hw | verify_s | 10 | 0.930 | [0.904, 0.960] | 9 / 1 |
| zinnia_noir | noir_hw | prove_core_s | 10 | 0.999 | [0.967, 1.042] | 7 / 3 |
| zinnia_noir | noir_hw | end_to_end_s | 10 | 1.006 | [0.976, 1.045] | 8 / 2 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 10 | 1.055 | [1.003, 1.110] | 4 / 6 |
| zinnia_noir | noir_hw | verify_s | 10 | 0.914 | [0.817, 1.003] | 6 / 4 |
| zinnia_halo2 | noir_hw | prove_core_s | 10 | 1.916 | [1.162, 2.882] | 2 / 8 |
| zinnia_halo2 | noir_hw | end_to_end_s | 10 | 1.822 | [1.122, 2.694] | 2 / 8 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 10 | 1.069 | [0.601, 1.726] | 3 / 7 |
| zinnia_halo2 | noir_hw | verify_s | 10 | 0.968 | [0.680, 1.346] | 6 / 4 |
| zinnia_halo2 | halo2_lit | prove_core_s | 10 | 0.829 | [0.774, 0.887] | 9 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 10 | 0.840 | [0.785, 0.897] | 9 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 10 | 0.921 | [0.854, 1.001] | 7 / 3 |
| zinnia_halo2 | noir_lit | prove_core_s | 10 | 1.976 | [1.187, 2.973] | 2 / 8 |
| zinnia_halo2 | noir_lit | end_to_end_s | 10 | 1.858 | [1.135, 2.746] | 2 / 8 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 10 | 1.090 | [0.613, 1.760] | 3 / 7 |
| zinnia_halo2 | noir_lit | verify_s | 10 | 0.998 | [0.725, 1.342] | 6 / 4 |
| zinnia_noir | halo2_hw | prove_core_s | 10 | 0.483 | [0.317, 0.810] | 8 / 2 |
| zinnia_noir | halo2_hw | end_to_end_s | 10 | 0.518 | [0.348, 0.850] | 8 / 2 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 10 | 0.926 | [0.597, 1.590] | 7 / 3 |
| zinnia_noir | halo2_hw | verify_s | 10 | 0.878 | [0.621, 1.287] | 7 / 3 |
| zinnia_noir | halo2_lit | prove_core_s | 10 | 0.432 | [0.281, 0.729] | 8 / 2 |
| zinnia_noir | halo2_lit | end_to_end_s | 10 | 0.464 | [0.308, 0.766] | 8 / 2 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 10 | 0.829 | [0.528, 1.436] | 7 / 3 |
| zinnia_noir | halo2_lit | verify_s | 10 | 0.870 | [0.587, 1.353] | 7 / 3 |
| zinnia_noir | noir_lit | prove_core_s | 10 | 1.031 | [0.979, 1.090] | 5 / 5 |
| zinnia_noir | noir_lit | end_to_end_s | 10 | 1.026 | [0.974, 1.082] | 5 / 5 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 10 | 1.076 | [1.008, 1.151] | 4 / 6 |
| zinnia_noir | noir_lit | verify_s | 10 | 0.942 | [0.820, 1.078] | 6 / 4 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 10 | 0.926 | [0.888, 0.959] | 10 / 0 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 10 | 0.938 | [0.899, 0.972] | 10 / 0 |
| zinnia_halo2 | halo2_hw | verify_s | 10 | 0.930 | [0.904, 0.960] | 9 / 1 |
| zinnia_noir | noir_hw | prove_core_s | 10 | 0.999 | [0.967, 1.042] | 7 / 3 |
| zinnia_noir | noir_hw | end_to_end_s | 10 | 1.006 | [0.976, 1.045] | 8 / 2 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 10 | 1.055 | [1.003, 1.110] | 4 / 6 |
| zinnia_noir | noir_hw | verify_s | 10 | 0.914 | [0.817, 1.003] | 6 / 4 |
| zinnia_halo2 | noir_hw | prove_core_s | 10 | 1.916 | [1.162, 2.882] | 2 / 8 |
| zinnia_halo2 | noir_hw | end_to_end_s | 10 | 1.822 | [1.122, 2.694] | 2 / 8 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 10 | 1.069 | [0.601, 1.726] | 3 / 7 |
| zinnia_halo2 | noir_hw | verify_s | 10 | 0.968 | [0.680, 1.346] | 6 / 4 |
| zinnia_halo2 | halo2_lit | prove_core_s | 10 | 0.829 | [0.774, 0.887] | 9 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 10 | 0.840 | [0.785, 0.897] | 9 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 10 | 0.921 | [0.854, 1.001] | 7 / 3 |
| zinnia_halo2 | noir_lit | prove_core_s | 10 | 1.976 | [1.187, 2.973] | 2 / 8 |
| zinnia_halo2 | noir_lit | end_to_end_s | 10 | 1.858 | [1.135, 2.746] | 2 / 8 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 10 | 1.090 | [0.613, 1.760] | 3 / 7 |
| zinnia_halo2 | noir_lit | verify_s | 10 | 0.998 | [0.725, 1.342] | 6 / 4 |
| zinnia_noir | halo2_hw | prove_core_s | 10 | 0.483 | [0.317, 0.810] | 8 / 2 |
| zinnia_noir | halo2_hw | end_to_end_s | 10 | 0.518 | [0.348, 0.850] | 8 / 2 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 10 | 0.926 | [0.597, 1.590] | 7 / 3 |
| zinnia_noir | halo2_hw | verify_s | 10 | 0.878 | [0.621, 1.287] | 7 / 3 |
| zinnia_noir | halo2_lit | prove_core_s | 10 | 0.432 | [0.281, 0.729] | 8 / 2 |
| zinnia_noir | halo2_lit | end_to_end_s | 10 | 0.464 | [0.308, 0.766] | 8 / 2 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 10 | 0.829 | [0.528, 1.436] | 7 / 3 |
| zinnia_noir | halo2_lit | verify_s | 10 | 0.870 | [0.587, 1.353] | 7 / 3 |
| zinnia_noir | noir_lit | prove_core_s | 10 | 1.031 | [0.979, 1.090] | 5 / 5 |
| zinnia_noir | noir_lit | end_to_end_s | 10 | 1.026 | [0.974, 1.082] | 5 / 5 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 10 | 1.076 | [1.008, 1.151] | 4 / 6 |
| zinnia_noir | noir_lit | verify_s | 10 | 0.942 | [0.820, 1.078] | 6 / 4 |

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

