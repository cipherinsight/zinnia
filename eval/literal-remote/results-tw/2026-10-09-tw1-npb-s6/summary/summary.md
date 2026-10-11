# Epoch 2026-10-09-tw1-npb-s6

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

9 cases completed by every system that do more than copy inputs: npbench_polybench_atax_atax__kernel, npbench_polybench_gesummv_gesummv__kernel, npbench_polybench_syr2k_syr2k__kernel, pythran_allpairs__sqr_dists, pythran_d2q9_nxnyns__one_time_step, pythran_insertion_sort__insertion_sort, pythran_lu__factorMatrix1, pythran_specialconvolve__specialconvolve, pythran_weights__weights

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 9 | 0.911 | [0.850, 0.974] | 7 / 2 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 9 | 0.927 | [0.864, 0.991] | 6 / 3 |
| zinnia_halo2 | halo2_hw | verify_s | 9 | 0.986 | [0.892, 1.103] | 6 / 3 |
| zinnia_noir | noir_hw | prove_core_s | 9 | 0.999 | [0.975, 1.021] | 4 / 5 |
| zinnia_noir | noir_hw | end_to_end_s | 9 | 1.011 | [0.984, 1.036] | 2 / 7 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 9 | 1.111 | [0.991, 1.281] | 4 / 5 |
| zinnia_noir | noir_hw | verify_s | 9 | 0.977 | [0.865, 1.115] | 6 / 3 |
| zinnia_halo2 | noir_hw | prove_core_s | 9 | 1.997 | [1.242, 3.251] | 2 / 7 |
| zinnia_halo2 | noir_hw | end_to_end_s | 9 | 1.880 | [1.198, 2.977] | 2 / 7 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 9 | 1.118 | [0.632, 2.015] | 5 / 4 |
| zinnia_halo2 | noir_hw | verify_s | 9 | 1.069 | [0.737, 1.634] | 5 / 4 |
| zinnia_halo2 | halo2_lit | prove_core_s | 9 | 0.658 | [0.457, 0.875] | 7 / 2 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 9 | 0.669 | [0.466, 0.889] | 7 / 2 |
| zinnia_halo2 | halo2_lit | verify_s | 9 | 0.788 | [0.679, 0.893] | 8 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 9 | 1.793 | [1.073, 3.000] | 3 / 6 |
| zinnia_halo2 | noir_lit | end_to_end_s | 9 | 1.683 | [1.021, 2.756] | 3 / 6 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 9 | 1.051 | [0.584, 1.909] | 5 / 4 |
| zinnia_halo2 | noir_lit | verify_s | 9 | 1.079 | [0.787, 1.490] | 3 / 6 |
| zinnia_noir | halo2_hw | prove_core_s | 9 | 0.456 | [0.286, 0.709] | 8 / 1 |
| zinnia_noir | halo2_hw | end_to_end_s | 9 | 0.499 | [0.326, 0.749] | 8 / 1 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 9 | 0.921 | [0.578, 1.428] | 4 / 5 |
| zinnia_noir | halo2_hw | verify_s | 9 | 0.901 | [0.613, 1.299] | 5 / 4 |
| zinnia_noir | halo2_lit | prove_core_s | 9 | 0.329 | [0.200, 0.544] | 9 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 9 | 0.360 | [0.227, 0.575] | 9 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 9 | 0.665 | [0.415, 1.070] | 5 / 4 |
| zinnia_noir | halo2_lit | verify_s | 9 | 0.720 | [0.481, 1.068] | 6 / 3 |
| zinnia_noir | noir_lit | prove_core_s | 9 | 0.897 | [0.777, 0.998] | 6 / 3 |
| zinnia_noir | noir_lit | end_to_end_s | 9 | 0.906 | [0.778, 1.010] | 5 / 4 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 9 | 1.044 | [0.907, 1.206] | 3 / 6 |
| zinnia_noir | noir_lit | verify_s | 9 | 0.986 | [0.865, 1.106] | 3 / 6 |

## Copy-only cases (reported apart)

1 all-supported cases that only copy inputs: pythran_periodic_bc__periodic_bc

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 1 | 0.883 | [0.883, 0.883] | 1 / 0 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 1 | 0.887 | [0.887, 0.887] | 1 / 0 |
| zinnia_halo2 | halo2_hw | verify_s | 1 | 1.060 | [1.060, 1.060] | 0 / 1 |
| zinnia_noir | noir_hw | prove_core_s | 1 | 0.967 | [0.967, 0.967] | 1 / 0 |
| zinnia_noir | noir_hw | end_to_end_s | 1 | 0.979 | [0.979, 0.979] | 1 / 0 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 1 | 0.925 | [0.925, 0.925] | 1 / 0 |
| zinnia_noir | noir_hw | verify_s | 1 | 0.954 | [0.954, 0.954] | 1 / 0 |
| zinnia_halo2 | noir_hw | prove_core_s | 1 | 0.541 | [0.541, 0.541] | 1 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_s | 1 | 0.532 | [0.532, 0.532] | 1 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 1 | 0.236 | [0.236, 0.236] | 1 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 1 | 0.394 | [0.394, 0.394] | 1 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 1 | 1.028 | [1.028, 1.028] | 0 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 1 | 1.033 | [1.033, 1.033] | 0 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 1 | 1.127 | [1.127, 1.127] | 0 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 1 | 0.540 | [0.540, 0.540] | 1 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_s | 1 | 0.520 | [0.520, 0.520] | 1 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 1 | 0.229 | [0.229, 0.229] | 1 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 1 | 0.395 | [0.395, 0.395] | 1 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 1 | 1.578 | [1.578, 1.578] | 0 / 1 |
| zinnia_noir | halo2_hw | end_to_end_s | 1 | 1.632 | [1.632, 1.632] | 0 / 1 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 1 | 3.485 | [3.485, 3.485] | 0 / 1 |
| zinnia_noir | halo2_hw | verify_s | 1 | 2.567 | [2.567, 2.567] | 0 / 1 |
| zinnia_noir | halo2_lit | prove_core_s | 1 | 1.837 | [1.837, 1.837] | 0 / 1 |
| zinnia_noir | halo2_lit | end_to_end_s | 1 | 1.900 | [1.900, 1.900] | 0 / 1 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 1 | 4.056 | [4.056, 4.056] | 0 / 1 |
| zinnia_noir | halo2_lit | verify_s | 1 | 2.729 | [2.729, 2.729] | 0 / 1 |
| zinnia_noir | noir_lit | prove_core_s | 1 | 0.965 | [0.965, 0.965] | 1 / 0 |
| zinnia_noir | noir_lit | end_to_end_s | 1 | 0.956 | [0.956, 0.956] | 1 / 0 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 1 | 0.899 | [0.899, 0.899] | 1 / 0 |
| zinnia_noir | noir_lit | verify_s | 1 | 0.957 | [0.957, 0.957] | 1 / 0 |

## Group npbench: headline cases

3 headline cases of group npbench: npbench_polybench_atax_atax__kernel, npbench_polybench_gesummv_gesummv__kernel, npbench_polybench_syr2k_syr2k__kernel

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 3 | 0.925 | [0.916, 0.933] | 3 / 0 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 3 | 0.941 | [0.933, 0.950] | 3 / 0 |
| zinnia_halo2 | halo2_hw | verify_s | 3 | 0.910 | [0.852, 0.950] | 3 / 0 |
| zinnia_noir | noir_hw | prove_core_s | 3 | 1.023 | [1.004, 1.041] | 0 / 3 |
| zinnia_noir | noir_hw | end_to_end_s | 3 | 1.037 | [1.013, 1.054] | 0 / 3 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 3 | 1.080 | [0.971, 1.192] | 1 / 2 |
| zinnia_noir | noir_hw | verify_s | 3 | 0.923 | [0.737, 1.086] | 2 / 1 |
| zinnia_halo2 | noir_hw | prove_core_s | 3 | 2.791 | [1.524, 4.357] | 0 / 3 |
| zinnia_halo2 | noir_hw | end_to_end_s | 3 | 2.638 | [1.493, 4.004] | 0 / 3 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 3 | 1.600 | [0.742, 2.874] | 1 / 2 |
| zinnia_halo2 | noir_hw | verify_s | 3 | 1.054 | [0.610, 1.614] | 1 / 2 |
| zinnia_halo2 | halo2_lit | prove_core_s | 3 | 0.834 | [0.623, 1.024] | 2 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 3 | 0.848 | [0.633, 1.042] | 2 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 3 | 0.844 | [0.726, 0.956] | 3 / 0 |
| zinnia_halo2 | noir_lit | prove_core_s | 3 | 2.736 | [1.514, 4.160] | 0 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_s | 3 | 2.576 | [1.483, 3.834] | 0 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 3 | 1.588 | [0.776, 2.773] | 1 / 2 |
| zinnia_halo2 | noir_lit | verify_s | 3 | 1.159 | [0.633, 1.722] | 1 / 2 |
| zinnia_noir | halo2_hw | prove_core_s | 3 | 0.339 | [0.217, 0.637] | 3 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 3 | 0.370 | [0.245, 0.670] | 3 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 3 | 0.635 | [0.389, 1.243] | 2 / 1 |
| zinnia_noir | halo2_hw | verify_s | 3 | 0.798 | [0.434, 1.518] | 2 / 1 |
| zinnia_noir | halo2_lit | prove_core_s | 3 | 0.306 | [0.147, 0.620] | 3 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 3 | 0.333 | [0.165, 0.653] | 3 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 3 | 0.572 | [0.262, 1.211] | 2 / 1 |
| zinnia_noir | halo2_lit | verify_s | 3 | 0.739 | [0.332, 1.704] | 2 / 1 |
| zinnia_noir | noir_lit | prove_core_s | 3 | 1.003 | [0.978, 1.034] | 2 / 1 |
| zinnia_noir | noir_lit | end_to_end_s | 3 | 1.013 | [0.992, 1.046] | 1 / 2 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 3 | 1.072 | [1.015, 1.150] | 0 / 3 |
| zinnia_noir | noir_lit | verify_s | 3 | 1.015 | [0.787, 1.179] | 1 / 2 |

## Group pythran: headline cases

6 headline cases of group pythran: pythran_allpairs__sqr_dists, pythran_d2q9_nxnyns__one_time_step, pythran_insertion_sort__insertion_sort, pythran_lu__factorMatrix1, pythran_specialconvolve__specialconvolve, pythran_weights__weights

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 6 | 0.904 | [0.822, 0.999] | 4 / 2 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 6 | 0.921 | [0.833, 1.014] | 3 / 3 |
| zinnia_halo2 | halo2_hw | verify_s | 6 | 1.026 | [0.887, 1.199] | 3 / 3 |
| zinnia_noir | noir_hw | prove_core_s | 6 | 0.987 | [0.958, 1.016] | 4 / 2 |
| zinnia_noir | noir_hw | end_to_end_s | 6 | 0.999 | [0.966, 1.032] | 2 / 4 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 6 | 1.126 | [0.953, 1.376] | 3 / 3 |
| zinnia_noir | noir_hw | verify_s | 6 | 1.005 | [0.856, 1.191] | 4 / 2 |
| zinnia_halo2 | noir_hw | prove_core_s | 6 | 1.689 | [0.923, 3.251] | 2 / 4 |
| zinnia_halo2 | noir_hw | end_to_end_s | 6 | 1.587 | [0.898, 2.949] | 2 / 4 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 6 | 0.935 | [0.445, 2.008] | 4 / 2 |
| zinnia_halo2 | noir_hw | verify_s | 6 | 1.076 | [0.649, 1.900] | 4 / 2 |
| zinnia_halo2 | halo2_lit | prove_core_s | 6 | 0.584 | [0.361, 0.866] | 5 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 6 | 0.595 | [0.367, 0.881] | 5 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 6 | 0.761 | [0.622, 0.910] | 5 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 6 | 1.452 | [0.765, 2.854] | 3 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_s | 6 | 1.361 | [0.728, 2.616] | 3 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 6 | 0.855 | [0.405, 1.837] | 4 / 2 |
| zinnia_halo2 | noir_lit | verify_s | 6 | 1.040 | [0.690, 1.571] | 2 / 4 |
| zinnia_noir | halo2_hw | prove_core_s | 6 | 0.528 | [0.298, 0.916] | 5 / 1 |
| zinnia_noir | halo2_hw | end_to_end_s | 6 | 0.579 | [0.342, 0.955] | 5 / 1 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 6 | 1.109 | [0.616, 1.870] | 2 / 4 |
| zinnia_noir | halo2_hw | verify_s | 6 | 0.958 | [0.597, 1.521] | 3 / 3 |
| zinnia_noir | halo2_lit | prove_core_s | 6 | 0.341 | [0.182, 0.655] | 6 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 6 | 0.374 | [0.211, 0.685] | 6 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 6 | 0.717 | [0.401, 1.308] | 3 / 3 |
| zinnia_noir | halo2_lit | verify_s | 6 | 0.711 | [0.450, 1.117] | 4 / 2 |
| zinnia_noir | noir_lit | prove_core_s | 6 | 0.848 | [0.705, 0.985] | 4 / 2 |
| zinnia_noir | noir_lit | end_to_end_s | 6 | 0.857 | [0.705, 0.999] | 4 / 2 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 6 | 1.030 | [0.835, 1.275] | 3 / 3 |
| zinnia_noir | noir_lit | verify_s | 6 | 0.972 | [0.818, 1.120] | 2 / 4 |

## All-supported subset

10 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): npbench_polybench_atax_atax__kernel, npbench_polybench_gesummv_gesummv__kernel, npbench_polybench_syr2k_syr2k__kernel, pythran_allpairs__sqr_dists, pythran_d2q9_nxnyns__one_time_step, pythran_insertion_sort__insertion_sort, pythran_lu__factorMatrix1, pythran_periodic_bc__periodic_bc, pythran_specialconvolve__specialconvolve, pythran_weights__weights

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 10 | 0.908 | [0.853, 0.965] | 8 / 2 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 10 | 0.923 | [0.867, 0.981] | 7 / 3 |
| zinnia_halo2 | halo2_hw | verify_s | 10 | 0.993 | [0.905, 1.105] | 6 / 4 |
| zinnia_noir | noir_hw | prove_core_s | 10 | 0.996 | [0.974, 1.016] | 5 / 5 |
| zinnia_noir | noir_hw | end_to_end_s | 10 | 1.008 | [0.984, 1.032] | 3 / 7 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 10 | 1.091 | [0.980, 1.244] | 5 / 5 |
| zinnia_noir | noir_hw | verify_s | 10 | 0.975 | [0.876, 1.098] | 7 / 3 |
| zinnia_halo2 | noir_hw | prove_core_s | 10 | 1.752 | [1.064, 2.920] | 3 / 7 |
| zinnia_halo2 | noir_hw | end_to_end_s | 10 | 1.657 | [1.032, 2.695] | 3 / 7 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 10 | 0.957 | [0.527, 1.762] | 6 / 4 |
| zinnia_halo2 | noir_hw | verify_s | 10 | 0.967 | [0.659, 1.489] | 6 / 4 |
| zinnia_halo2 | halo2_lit | prove_core_s | 10 | 0.688 | [0.491, 0.898] | 7 / 3 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 10 | 0.699 | [0.499, 0.912] | 7 / 3 |
| zinnia_halo2 | halo2_lit | verify_s | 10 | 0.817 | [0.705, 0.933] | 8 / 2 |
| zinnia_halo2 | noir_lit | prove_core_s | 10 | 1.590 | [0.950, 2.691] | 4 / 6 |
| zinnia_halo2 | noir_lit | end_to_end_s | 10 | 1.497 | [0.908, 2.484] | 4 / 6 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 10 | 0.902 | [0.493, 1.666] | 6 / 4 |
| zinnia_halo2 | noir_lit | verify_s | 10 | 0.975 | [0.692, 1.381] | 4 / 6 |
| zinnia_noir | halo2_hw | prove_core_s | 10 | 0.516 | [0.320, 0.818] | 8 / 2 |
| zinnia_noir | halo2_hw | end_to_end_s | 10 | 0.562 | [0.360, 0.863] | 8 / 2 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 10 | 1.052 | [0.646, 1.685] | 4 / 6 |
| zinnia_noir | halo2_hw | verify_s | 10 | 1.001 | [0.672, 1.472] | 5 / 5 |
| zinnia_noir | halo2_lit | prove_core_s | 10 | 0.391 | [0.228, 0.680] | 9 / 1 |
| zinnia_noir | halo2_lit | end_to_end_s | 10 | 0.425 | [0.256, 0.721] | 9 / 1 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 10 | 0.797 | [0.468, 1.375] | 5 / 5 |
| zinnia_noir | halo2_lit | verify_s | 10 | 0.823 | [0.534, 1.284] | 6 / 4 |
| zinnia_noir | noir_lit | prove_core_s | 10 | 0.904 | [0.793, 0.995] | 7 / 3 |
| zinnia_noir | noir_lit | end_to_end_s | 10 | 0.911 | [0.793, 1.004] | 6 / 4 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 10 | 1.028 | [0.902, 1.176] | 4 / 6 |
| zinnia_noir | noir_lit | verify_s | 10 | 0.983 | [0.875, 1.092] | 4 / 6 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 10 | 0.908 | [0.853, 0.965] | 8 / 2 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 10 | 0.923 | [0.867, 0.981] | 7 / 3 |
| zinnia_halo2 | halo2_hw | verify_s | 10 | 0.993 | [0.905, 1.105] | 6 / 4 |
| zinnia_noir | noir_hw | prove_core_s | 10 | 0.996 | [0.974, 1.016] | 5 / 5 |
| zinnia_noir | noir_hw | end_to_end_s | 10 | 1.008 | [0.984, 1.032] | 3 / 7 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 10 | 1.091 | [0.980, 1.244] | 5 / 5 |
| zinnia_noir | noir_hw | verify_s | 10 | 0.975 | [0.876, 1.098] | 7 / 3 |
| zinnia_halo2 | noir_hw | prove_core_s | 10 | 1.752 | [1.064, 2.920] | 3 / 7 |
| zinnia_halo2 | noir_hw | end_to_end_s | 10 | 1.657 | [1.032, 2.695] | 3 / 7 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 10 | 0.957 | [0.527, 1.762] | 6 / 4 |
| zinnia_halo2 | noir_hw | verify_s | 10 | 0.967 | [0.659, 1.489] | 6 / 4 |
| zinnia_halo2 | halo2_lit | prove_core_s | 10 | 0.688 | [0.491, 0.898] | 7 / 3 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 10 | 0.699 | [0.499, 0.912] | 7 / 3 |
| zinnia_halo2 | halo2_lit | verify_s | 10 | 0.817 | [0.705, 0.933] | 8 / 2 |
| zinnia_halo2 | noir_lit | prove_core_s | 10 | 1.590 | [0.950, 2.691] | 4 / 6 |
| zinnia_halo2 | noir_lit | end_to_end_s | 10 | 1.497 | [0.908, 2.484] | 4 / 6 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 10 | 0.902 | [0.493, 1.666] | 6 / 4 |
| zinnia_halo2 | noir_lit | verify_s | 10 | 0.975 | [0.692, 1.381] | 4 / 6 |
| zinnia_noir | halo2_hw | prove_core_s | 10 | 0.516 | [0.320, 0.818] | 8 / 2 |
| zinnia_noir | halo2_hw | end_to_end_s | 10 | 0.562 | [0.360, 0.863] | 8 / 2 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 10 | 1.052 | [0.646, 1.685] | 4 / 6 |
| zinnia_noir | halo2_hw | verify_s | 10 | 1.001 | [0.672, 1.472] | 5 / 5 |
| zinnia_noir | halo2_lit | prove_core_s | 10 | 0.391 | [0.228, 0.680] | 9 / 1 |
| zinnia_noir | halo2_lit | end_to_end_s | 10 | 0.425 | [0.256, 0.721] | 9 / 1 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 10 | 0.797 | [0.468, 1.375] | 5 / 5 |
| zinnia_noir | halo2_lit | verify_s | 10 | 0.823 | [0.534, 1.284] | 6 / 4 |
| zinnia_noir | noir_lit | prove_core_s | 10 | 0.904 | [0.793, 0.995] | 7 / 3 |
| zinnia_noir | noir_lit | end_to_end_s | 10 | 0.911 | [0.793, 1.004] | 6 / 4 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 10 | 1.028 | [0.902, 1.176] | 4 / 6 |
| zinnia_noir | noir_lit | verify_s | 10 | 0.983 | [0.875, 1.092] | 4 / 6 |

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

