# Epoch 2026-10-09-tw1-npb-s2

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

10 cases completed by every system that do more than copy inputs: npbench_deep_learning_conv2d_bias_conv2d__conv2d_bias, npbench_polybench_deriche_deriche__kernel, npbench_polybench_lu_lu__kernel, npbench_pythran_arc_distance_arc_distance__arc_distance, pythran_clip__clip, pythran_fibo_seq__fibo, pythran_lapl3d__laplacien, pythran_multitype__check, pythran_pythagorean_triples__next_pythagorean_triples, pythran_tax__tax

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 10 | 0.896 | [0.824, 0.971] | 7 / 3 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 10 | 0.916 | [0.841, 0.993] | 7 / 3 |
| zinnia_halo2 | halo2_hw | verify_s | 10 | 0.891 | [0.774, 1.002] | 7 / 3 |
| zinnia_noir | noir_hw | prove_core_s | 10 | 0.971 | [0.916, 1.027] | 8 / 2 |
| zinnia_noir | noir_hw | end_to_end_s | 10 | 0.987 | [0.943, 1.035] | 5 / 5 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 10 | 1.105 | [1.012, 1.229] | 4 / 6 |
| zinnia_noir | noir_hw | verify_s | 10 | 1.022 | [0.940, 1.121] | 4 / 6 |
| zinnia_halo2 | noir_hw | prove_core_s | 10 | 1.603 | [0.904, 2.655] | 3 / 7 |
| zinnia_halo2 | noir_hw | end_to_end_s | 10 | 1.490 | [0.857, 2.421] | 3 / 7 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 10 | 0.881 | [0.445, 1.626] | 4 / 6 |
| zinnia_halo2 | noir_hw | verify_s | 10 | 0.877 | [0.601, 1.208] | 5 / 5 |
| zinnia_halo2 | halo2_lit | prove_core_s | 10 | 0.763 | [0.674, 0.867] | 9 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 10 | 0.780 | [0.689, 0.884] | 9 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 10 | 0.832 | [0.722, 0.943] | 7 / 3 |
| zinnia_halo2 | noir_lit | prove_core_s | 10 | 1.540 | [0.870, 2.558] | 3 / 7 |
| zinnia_halo2 | noir_lit | end_to_end_s | 10 | 1.436 | [0.829, 2.331] | 3 / 7 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 10 | 0.859 | [0.436, 1.582] | 4 / 6 |
| zinnia_halo2 | noir_lit | verify_s | 10 | 0.865 | [0.609, 1.167] | 4 / 6 |
| zinnia_noir | halo2_hw | prove_core_s | 10 | 0.543 | [0.318, 1.009] | 7 / 3 |
| zinnia_noir | halo2_hw | end_to_end_s | 10 | 0.607 | [0.368, 1.088] | 7 / 3 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 10 | 1.148 | [0.656, 2.159] | 6 / 4 |
| zinnia_noir | halo2_hw | verify_s | 10 | 1.038 | [0.695, 1.612] | 6 / 4 |
| zinnia_noir | halo2_lit | prove_core_s | 10 | 0.462 | [0.266, 0.882] | 8 / 2 |
| zinnia_noir | halo2_lit | end_to_end_s | 10 | 0.517 | [0.306, 0.952] | 8 / 2 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 10 | 0.978 | [0.550, 1.882] | 7 / 3 |
| zinnia_noir | halo2_lit | verify_s | 10 | 0.971 | [0.643, 1.531] | 6 / 4 |
| zinnia_noir | noir_lit | prove_core_s | 10 | 0.933 | [0.874, 0.990] | 7 / 3 |
| zinnia_noir | noir_lit | end_to_end_s | 10 | 0.952 | [0.900, 1.003] | 7 / 3 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 10 | 1.077 | [0.986, 1.206] | 3 / 7 |
| zinnia_noir | noir_lit | verify_s | 10 | 1.009 | [0.903, 1.115] | 6 / 4 |

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

4 headline cases of group npbench: npbench_deep_learning_conv2d_bias_conv2d__conv2d_bias, npbench_polybench_deriche_deriche__kernel, npbench_polybench_lu_lu__kernel, npbench_pythran_arc_distance_arc_distance__arc_distance

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 4 | 0.896 | [0.768, 1.044] | 2 / 2 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 4 | 0.920 | [0.784, 1.080] | 2 / 2 |
| zinnia_halo2 | halo2_hw | verify_s | 4 | 0.811 | [0.636, 0.939] | 4 / 0 |
| zinnia_noir | noir_hw | prove_core_s | 4 | 0.968 | [0.948, 0.989] | 4 / 0 |
| zinnia_noir | noir_hw | end_to_end_s | 4 | 0.983 | [0.961, 1.016] | 3 / 1 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 4 | 1.143 | [1.007, 1.419] | 1 / 3 |
| zinnia_noir | noir_hw | verify_s | 4 | 0.966 | [0.850, 1.075] | 2 / 2 |
| zinnia_halo2 | noir_hw | prove_core_s | 4 | 2.969 | [2.498, 3.437] | 0 / 4 |
| zinnia_halo2 | noir_hw | end_to_end_s | 4 | 2.680 | [2.320, 3.043] | 0 / 4 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 4 | 1.863 | [1.487, 2.408] | 0 / 4 |
| zinnia_halo2 | noir_hw | verify_s | 4 | 1.316 | [1.081, 1.512] | 1 / 3 |
| zinnia_halo2 | halo2_lit | prove_core_s | 4 | 0.827 | [0.674, 1.030] | 3 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 4 | 0.849 | [0.694, 1.045] | 3 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 4 | 0.795 | [0.602, 0.966] | 3 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 4 | 2.998 | [2.466, 3.531] | 0 / 4 |
| zinnia_halo2 | noir_lit | end_to_end_s | 4 | 2.708 | [2.294, 3.116] | 0 / 4 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 4 | 1.884 | [1.487, 2.451] | 0 / 4 |
| zinnia_halo2 | noir_lit | verify_s | 4 | 1.251 | [1.098, 1.437] | 0 / 4 |
| zinnia_noir | halo2_hw | prove_core_s | 4 | 0.292 | [0.238, 0.390] | 4 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 4 | 0.337 | [0.279, 0.428] | 4 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 4 | 0.564 | [0.446, 0.722] | 4 / 0 |
| zinnia_noir | halo2_hw | verify_s | 4 | 0.595 | [0.471, 0.696] | 4 / 0 |
| zinnia_noir | halo2_lit | prove_core_s | 4 | 0.270 | [0.206, 0.385] | 4 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 4 | 0.311 | [0.240, 0.430] | 4 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 4 | 0.521 | [0.380, 0.744] | 4 / 0 |
| zinnia_noir | halo2_lit | verify_s | 4 | 0.584 | [0.452, 0.703] | 4 / 0 |
| zinnia_noir | noir_lit | prove_core_s | 4 | 0.978 | [0.935, 1.023] | 2 / 2 |
| zinnia_noir | noir_lit | end_to_end_s | 4 | 0.993 | [0.953, 1.035] | 2 / 2 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 4 | 1.156 | [1.014, 1.437] | 0 / 4 |
| zinnia_noir | noir_lit | verify_s | 4 | 0.918 | [0.746, 1.100] | 3 / 1 |

## Group pythran: headline cases

6 headline cases of group pythran: pythran_clip__clip, pythran_fibo_seq__fibo, pythran_lapl3d__laplacien, pythran_multitype__check, pythran_pythagorean_triples__next_pythagorean_triples, pythran_tax__tax

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 6 | 0.897 | [0.817, 0.980] | 5 / 1 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 6 | 0.913 | [0.834, 0.994] | 5 / 1 |
| zinnia_halo2 | halo2_hw | verify_s | 6 | 0.948 | [0.802, 1.075] | 3 / 3 |
| zinnia_noir | noir_hw | prove_core_s | 6 | 0.973 | [0.884, 1.069] | 4 / 2 |
| zinnia_noir | noir_hw | end_to_end_s | 6 | 0.991 | [0.922, 1.069] | 2 / 4 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 6 | 1.081 | [0.984, 1.211] | 3 / 3 |
| zinnia_noir | noir_hw | verify_s | 6 | 1.062 | [0.965, 1.204] | 2 / 4 |
| zinnia_halo2 | noir_hw | prove_core_s | 6 | 1.063 | [0.516, 2.209] | 3 / 3 |
| zinnia_halo2 | noir_hw | end_to_end_s | 6 | 1.008 | [0.500, 2.060] | 3 / 3 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 6 | 0.535 | [0.227, 1.263] | 4 / 2 |
| zinnia_halo2 | noir_hw | verify_s | 6 | 0.669 | [0.409, 1.041] | 4 / 2 |
| zinnia_halo2 | halo2_lit | prove_core_s | 6 | 0.723 | [0.626, 0.840] | 6 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 6 | 0.736 | [0.641, 0.845] | 6 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 6 | 0.858 | [0.737, 0.992] | 4 / 2 |
| zinnia_halo2 | noir_lit | prove_core_s | 6 | 0.988 | [0.488, 2.000] | 3 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_s | 6 | 0.941 | [0.475, 1.862] | 3 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 6 | 0.509 | [0.222, 1.168] | 4 / 2 |
| zinnia_halo2 | noir_lit | verify_s | 6 | 0.676 | [0.438, 1.030] | 4 / 2 |
| zinnia_noir | halo2_hw | prove_core_s | 6 | 0.821 | [0.367, 1.854] | 3 / 3 |
| zinnia_noir | halo2_hw | end_to_end_s | 6 | 0.897 | [0.428, 1.960] | 3 / 3 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 6 | 1.844 | [0.846, 4.035] | 2 / 4 |
| zinnia_noir | halo2_hw | verify_s | 6 | 1.505 | [0.863, 2.516] | 2 / 4 |
| zinnia_noir | halo2_lit | prove_core_s | 6 | 0.662 | [0.278, 1.622] | 4 / 2 |
| zinnia_noir | halo2_lit | end_to_end_s | 6 | 0.724 | [0.319, 1.686] | 4 / 2 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 6 | 1.487 | [0.642, 3.489] | 3 / 3 |
| zinnia_noir | halo2_lit | verify_s | 6 | 1.362 | [0.770, 2.410] | 2 / 4 |
| zinnia_noir | noir_lit | prove_core_s | 6 | 0.905 | [0.825, 0.987] | 5 / 1 |
| zinnia_noir | noir_lit | end_to_end_s | 6 | 0.925 | [0.854, 1.002] | 5 / 1 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 6 | 1.028 | [0.955, 1.153] | 3 / 3 |
| zinnia_noir | noir_lit | verify_s | 6 | 1.074 | [0.974, 1.189] | 3 / 3 |

## All-supported subset

10 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): npbench_deep_learning_conv2d_bias_conv2d__conv2d_bias, npbench_polybench_deriche_deriche__kernel, npbench_polybench_lu_lu__kernel, npbench_pythran_arc_distance_arc_distance__arc_distance, pythran_clip__clip, pythran_fibo_seq__fibo, pythran_lapl3d__laplacien, pythran_multitype__check, pythran_pythagorean_triples__next_pythagorean_triples, pythran_tax__tax

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 10 | 0.896 | [0.824, 0.971] | 7 / 3 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 10 | 0.916 | [0.841, 0.993] | 7 / 3 |
| zinnia_halo2 | halo2_hw | verify_s | 10 | 0.891 | [0.774, 1.002] | 7 / 3 |
| zinnia_noir | noir_hw | prove_core_s | 10 | 0.971 | [0.916, 1.027] | 8 / 2 |
| zinnia_noir | noir_hw | end_to_end_s | 10 | 0.987 | [0.943, 1.035] | 5 / 5 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 10 | 1.105 | [1.012, 1.229] | 4 / 6 |
| zinnia_noir | noir_hw | verify_s | 10 | 1.022 | [0.940, 1.121] | 4 / 6 |
| zinnia_halo2 | noir_hw | prove_core_s | 10 | 1.603 | [0.904, 2.655] | 3 / 7 |
| zinnia_halo2 | noir_hw | end_to_end_s | 10 | 1.490 | [0.857, 2.421] | 3 / 7 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 10 | 0.881 | [0.445, 1.626] | 4 / 6 |
| zinnia_halo2 | noir_hw | verify_s | 10 | 0.877 | [0.601, 1.208] | 5 / 5 |
| zinnia_halo2 | halo2_lit | prove_core_s | 10 | 0.763 | [0.674, 0.867] | 9 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 10 | 0.780 | [0.689, 0.884] | 9 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 10 | 0.832 | [0.722, 0.943] | 7 / 3 |
| zinnia_halo2 | noir_lit | prove_core_s | 10 | 1.540 | [0.870, 2.558] | 3 / 7 |
| zinnia_halo2 | noir_lit | end_to_end_s | 10 | 1.436 | [0.829, 2.331] | 3 / 7 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 10 | 0.859 | [0.436, 1.582] | 4 / 6 |
| zinnia_halo2 | noir_lit | verify_s | 10 | 0.865 | [0.609, 1.167] | 4 / 6 |
| zinnia_noir | halo2_hw | prove_core_s | 10 | 0.543 | [0.318, 1.009] | 7 / 3 |
| zinnia_noir | halo2_hw | end_to_end_s | 10 | 0.607 | [0.368, 1.088] | 7 / 3 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 10 | 1.148 | [0.656, 2.159] | 6 / 4 |
| zinnia_noir | halo2_hw | verify_s | 10 | 1.038 | [0.695, 1.612] | 6 / 4 |
| zinnia_noir | halo2_lit | prove_core_s | 10 | 0.462 | [0.266, 0.882] | 8 / 2 |
| zinnia_noir | halo2_lit | end_to_end_s | 10 | 0.517 | [0.306, 0.952] | 8 / 2 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 10 | 0.978 | [0.550, 1.882] | 7 / 3 |
| zinnia_noir | halo2_lit | verify_s | 10 | 0.971 | [0.643, 1.531] | 6 / 4 |
| zinnia_noir | noir_lit | prove_core_s | 10 | 0.933 | [0.874, 0.990] | 7 / 3 |
| zinnia_noir | noir_lit | end_to_end_s | 10 | 0.952 | [0.900, 1.003] | 7 / 3 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 10 | 1.077 | [0.986, 1.206] | 3 / 7 |
| zinnia_noir | noir_lit | verify_s | 10 | 1.009 | [0.903, 1.115] | 6 / 4 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 10 | 0.896 | [0.824, 0.971] | 7 / 3 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 10 | 0.916 | [0.841, 0.993] | 7 / 3 |
| zinnia_halo2 | halo2_hw | verify_s | 10 | 0.891 | [0.774, 1.002] | 7 / 3 |
| zinnia_noir | noir_hw | prove_core_s | 10 | 0.971 | [0.916, 1.027] | 8 / 2 |
| zinnia_noir | noir_hw | end_to_end_s | 10 | 0.987 | [0.943, 1.035] | 5 / 5 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 10 | 1.105 | [1.012, 1.229] | 4 / 6 |
| zinnia_noir | noir_hw | verify_s | 10 | 1.022 | [0.940, 1.121] | 4 / 6 |
| zinnia_halo2 | noir_hw | prove_core_s | 10 | 1.603 | [0.904, 2.655] | 3 / 7 |
| zinnia_halo2 | noir_hw | end_to_end_s | 10 | 1.490 | [0.857, 2.421] | 3 / 7 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 10 | 0.881 | [0.445, 1.626] | 4 / 6 |
| zinnia_halo2 | noir_hw | verify_s | 10 | 0.877 | [0.601, 1.208] | 5 / 5 |
| zinnia_halo2 | halo2_lit | prove_core_s | 10 | 0.763 | [0.674, 0.867] | 9 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 10 | 0.780 | [0.689, 0.884] | 9 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 10 | 0.832 | [0.722, 0.943] | 7 / 3 |
| zinnia_halo2 | noir_lit | prove_core_s | 10 | 1.540 | [0.870, 2.558] | 3 / 7 |
| zinnia_halo2 | noir_lit | end_to_end_s | 10 | 1.436 | [0.829, 2.331] | 3 / 7 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 10 | 0.859 | [0.436, 1.582] | 4 / 6 |
| zinnia_halo2 | noir_lit | verify_s | 10 | 0.865 | [0.609, 1.167] | 4 / 6 |
| zinnia_noir | halo2_hw | prove_core_s | 10 | 0.543 | [0.318, 1.009] | 7 / 3 |
| zinnia_noir | halo2_hw | end_to_end_s | 10 | 0.607 | [0.368, 1.088] | 7 / 3 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 10 | 1.148 | [0.656, 2.159] | 6 / 4 |
| zinnia_noir | halo2_hw | verify_s | 10 | 1.038 | [0.695, 1.612] | 6 / 4 |
| zinnia_noir | halo2_lit | prove_core_s | 10 | 0.462 | [0.266, 0.882] | 8 / 2 |
| zinnia_noir | halo2_lit | end_to_end_s | 10 | 0.517 | [0.306, 0.952] | 8 / 2 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 10 | 0.978 | [0.550, 1.882] | 7 / 3 |
| zinnia_noir | halo2_lit | verify_s | 10 | 0.971 | [0.643, 1.531] | 6 / 4 |
| zinnia_noir | noir_lit | prove_core_s | 10 | 0.933 | [0.874, 0.990] | 7 / 3 |
| zinnia_noir | noir_lit | end_to_end_s | 10 | 0.952 | [0.900, 1.003] | 7 / 3 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 10 | 1.077 | [0.986, 1.206] | 3 / 7 |
| zinnia_noir | noir_lit | verify_s | 10 | 1.009 | [0.903, 1.115] | 6 / 4 |

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

