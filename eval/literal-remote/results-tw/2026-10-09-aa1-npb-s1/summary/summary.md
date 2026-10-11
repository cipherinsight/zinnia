# Epoch 2026-10-09-aa1-npb-s1

Ratios are Zinnia / baseline of per-case medians; below 1 means Zinnia is faster.
Only cases with status `ok` (proved, verified, outputs equal to NumPy, tampered proof rejected) enter a comparison.

`zinnia_noir` `end_to_end_s` witness step: noir-execute (2 cases); `end_to_end_nargo_s` uses `nargo execute`.
`noir_hw` `end_to_end_s` witness step: noir-execute (2 cases); `end_to_end_nargo_s` uses `nargo execute`.
`noir_lit` `end_to_end_s` witness step: noir-execute (2 cases); `end_to_end_nargo_s` uses `nargo execute`.

## Coverage

| system | cases | ok | other statuses |
|---|---|---|---|
| zinnia_halo2 | 2 | 2 | - |
| zinnia_noir | 2 | 2 | - |
| halo2_hw | 2 | 2 | - |
| noir_hw | 2 | 2 | - |
| halo2_lit | 2 | 2 | - |
| noir_lit | 2 | 2 | - |

## Headline: all-supported subset without copy-only cases

2 cases completed by every system that do more than copy inputs: npbench_polybench_floyd_warshall_floyd_warshall__kernel, pythran_ccopy__CCOPY

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 2 | 0.768 | [0.709, 0.832] | 2 / 0 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 2 | 0.792 | [0.735, 0.853] | 2 / 0 |
| zinnia_halo2 | halo2_hw | verify_s | 2 | 0.731 | [0.604, 0.884] | 2 / 0 |
| zinnia_noir | noir_hw | prove_core_s | 2 | 0.938 | [0.765, 1.150] | 1 / 1 |
| zinnia_noir | noir_hw | end_to_end_s | 2 | 0.962 | [0.788, 1.175] | 1 / 1 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 2 | 1.463 | [1.020, 2.098] | 0 / 2 |
| zinnia_noir | noir_hw | verify_s | 2 | 0.950 | [0.891, 1.012] | 1 / 1 |
| zinnia_halo2 | noir_hw | prove_core_s | 2 | 3.226 | [2.938, 3.541] | 0 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_s | 2 | 2.851 | [2.563, 3.172] | 0 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 2 | 2.282 | [2.013, 2.587] | 0 / 2 |
| zinnia_halo2 | noir_hw | verify_s | 2 | 3.698 | [2.434, 5.619] | 0 / 2 |
| zinnia_halo2 | halo2_lit | prove_core_s | 2 | 0.539 | [0.384, 0.757] | 2 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 2 | 0.555 | [0.397, 0.774] | 2 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 2 | 0.666 | [0.442, 1.003] | 1 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 2 | 2.985 | [2.412, 3.693] | 0 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_s | 2 | 2.530 | [1.978, 3.236] | 0 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 2 | 2.075 | [1.623, 2.652] | 0 / 2 |
| zinnia_halo2 | noir_lit | verify_s | 2 | 3.617 | [2.624, 4.987] | 0 / 2 |
| zinnia_noir | halo2_hw | prove_core_s | 2 | 0.223 | [0.180, 0.278] | 2 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 2 | 0.267 | [0.212, 0.337] | 2 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 2 | 0.508 | [0.336, 0.766] | 2 / 0 |
| zinnia_noir | halo2_hw | verify_s | 2 | 0.188 | [0.140, 0.251] | 2 / 0 |
| zinnia_noir | halo2_lit | prove_core_s | 2 | 0.157 | [0.150, 0.163] | 2 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 2 | 0.187 | [0.182, 0.192] | 2 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 2 | 0.355 | [0.305, 0.414] | 2 / 0 |
| zinnia_noir | halo2_lit | verify_s | 2 | 0.171 | [0.159, 0.184] | 2 / 0 |
| zinnia_noir | noir_lit | prove_core_s | 2 | 0.868 | [0.797, 0.944] | 2 / 0 |
| zinnia_noir | noir_lit | end_to_end_s | 2 | 0.854 | [0.804, 0.907] | 2 / 0 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 2 | 1.330 | [1.045, 1.691] | 0 / 2 |
| zinnia_noir | noir_lit | verify_s | 2 | 0.929 | [0.791, 1.091] | 1 / 1 |

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

1 headline cases of group npbench: npbench_polybench_floyd_warshall_floyd_warshall__kernel

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 1 | 0.832 | [0.832, 0.832] | 1 / 0 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 1 | 0.853 | [0.853, 0.853] | 1 / 0 |
| zinnia_halo2 | halo2_hw | verify_s | 1 | 0.884 | [0.884, 0.884] | 1 / 0 |
| zinnia_noir | noir_hw | prove_core_s | 1 | 0.765 | [0.765, 0.765] | 1 / 0 |
| zinnia_noir | noir_hw | end_to_end_s | 1 | 0.788 | [0.788, 0.788] | 1 / 0 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 1 | 1.020 | [1.020, 1.020] | 0 / 1 |
| zinnia_noir | noir_hw | verify_s | 1 | 0.891 | [0.891, 0.891] | 1 / 0 |
| zinnia_halo2 | noir_hw | prove_core_s | 1 | 3.541 | [3.541, 3.541] | 0 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_s | 1 | 3.172 | [3.172, 3.172] | 0 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 1 | 2.587 | [2.587, 2.587] | 0 / 1 |
| zinnia_halo2 | noir_hw | verify_s | 1 | 5.619 | [5.619, 5.619] | 0 / 1 |
| zinnia_halo2 | halo2_lit | prove_core_s | 1 | 0.757 | [0.757, 0.757] | 1 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 1 | 0.774 | [0.774, 0.774] | 1 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 1 | 1.003 | [1.003, 1.003] | 0 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 1 | 3.693 | [3.693, 3.693] | 0 / 1 |
| zinnia_halo2 | noir_lit | end_to_end_s | 1 | 3.236 | [3.236, 3.236] | 0 / 1 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 1 | 2.652 | [2.652, 2.652] | 0 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 1 | 4.987 | [4.987, 4.987] | 0 / 1 |
| zinnia_noir | halo2_hw | prove_core_s | 1 | 0.180 | [0.180, 0.180] | 1 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 1 | 0.212 | [0.212, 0.212] | 1 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 1 | 0.336 | [0.336, 0.336] | 1 / 0 |
| zinnia_noir | halo2_hw | verify_s | 1 | 0.140 | [0.140, 0.140] | 1 / 0 |
| zinnia_noir | halo2_lit | prove_core_s | 1 | 0.163 | [0.163, 0.163] | 1 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 1 | 0.192 | [0.192, 0.192] | 1 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 1 | 0.305 | [0.305, 0.305] | 1 / 0 |
| zinnia_noir | halo2_lit | verify_s | 1 | 0.159 | [0.159, 0.159] | 1 / 0 |
| zinnia_noir | noir_lit | prove_core_s | 1 | 0.797 | [0.797, 0.797] | 1 / 0 |
| zinnia_noir | noir_lit | end_to_end_s | 1 | 0.804 | [0.804, 0.804] | 1 / 0 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 1 | 1.045 | [1.045, 1.045] | 0 / 1 |
| zinnia_noir | noir_lit | verify_s | 1 | 0.791 | [0.791, 0.791] | 1 / 0 |

## Group pythran: headline cases

1 headline cases of group pythran: pythran_ccopy__CCOPY

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 1 | 0.709 | [0.709, 0.709] | 1 / 0 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 1 | 0.735 | [0.735, 0.735] | 1 / 0 |
| zinnia_halo2 | halo2_hw | verify_s | 1 | 0.604 | [0.604, 0.604] | 1 / 0 |
| zinnia_noir | noir_hw | prove_core_s | 1 | 1.150 | [1.150, 1.150] | 0 / 1 |
| zinnia_noir | noir_hw | end_to_end_s | 1 | 1.175 | [1.175, 1.175] | 0 / 1 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 1 | 2.098 | [2.098, 2.098] | 0 / 1 |
| zinnia_noir | noir_hw | verify_s | 1 | 1.012 | [1.012, 1.012] | 0 / 1 |
| zinnia_halo2 | noir_hw | prove_core_s | 1 | 2.938 | [2.938, 2.938] | 0 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_s | 1 | 2.563 | [2.563, 2.563] | 0 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 1 | 2.013 | [2.013, 2.013] | 0 / 1 |
| zinnia_halo2 | noir_hw | verify_s | 1 | 2.434 | [2.434, 2.434] | 0 / 1 |
| zinnia_halo2 | halo2_lit | prove_core_s | 1 | 0.384 | [0.384, 0.384] | 1 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 1 | 0.397 | [0.397, 0.397] | 1 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 1 | 0.442 | [0.442, 0.442] | 1 / 0 |
| zinnia_halo2 | noir_lit | prove_core_s | 1 | 2.412 | [2.412, 2.412] | 0 / 1 |
| zinnia_halo2 | noir_lit | end_to_end_s | 1 | 1.978 | [1.978, 1.978] | 0 / 1 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 1 | 1.623 | [1.623, 1.623] | 0 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 1 | 2.624 | [2.624, 2.624] | 0 / 1 |
| zinnia_noir | halo2_hw | prove_core_s | 1 | 0.278 | [0.278, 0.278] | 1 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 1 | 0.337 | [0.337, 0.337] | 1 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 1 | 0.766 | [0.766, 0.766] | 1 / 0 |
| zinnia_noir | halo2_hw | verify_s | 1 | 0.251 | [0.251, 0.251] | 1 / 0 |
| zinnia_noir | halo2_lit | prove_core_s | 1 | 0.150 | [0.150, 0.150] | 1 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 1 | 0.182 | [0.182, 0.182] | 1 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 1 | 0.414 | [0.414, 0.414] | 1 / 0 |
| zinnia_noir | halo2_lit | verify_s | 1 | 0.184 | [0.184, 0.184] | 1 / 0 |
| zinnia_noir | noir_lit | prove_core_s | 1 | 0.944 | [0.944, 0.944] | 1 / 0 |
| zinnia_noir | noir_lit | end_to_end_s | 1 | 0.907 | [0.907, 0.907] | 1 / 0 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 1 | 1.691 | [1.691, 1.691] | 0 / 1 |
| zinnia_noir | noir_lit | verify_s | 1 | 1.091 | [1.091, 1.091] | 0 / 1 |

## All-supported subset

2 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): npbench_polybench_floyd_warshall_floyd_warshall__kernel, pythran_ccopy__CCOPY

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 2 | 0.768 | [0.709, 0.832] | 2 / 0 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 2 | 0.792 | [0.735, 0.853] | 2 / 0 |
| zinnia_halo2 | halo2_hw | verify_s | 2 | 0.731 | [0.604, 0.884] | 2 / 0 |
| zinnia_noir | noir_hw | prove_core_s | 2 | 0.938 | [0.765, 1.150] | 1 / 1 |
| zinnia_noir | noir_hw | end_to_end_s | 2 | 0.962 | [0.788, 1.175] | 1 / 1 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 2 | 1.463 | [1.020, 2.098] | 0 / 2 |
| zinnia_noir | noir_hw | verify_s | 2 | 0.950 | [0.891, 1.012] | 1 / 1 |
| zinnia_halo2 | noir_hw | prove_core_s | 2 | 3.226 | [2.938, 3.541] | 0 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_s | 2 | 2.851 | [2.563, 3.172] | 0 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 2 | 2.282 | [2.013, 2.587] | 0 / 2 |
| zinnia_halo2 | noir_hw | verify_s | 2 | 3.698 | [2.434, 5.619] | 0 / 2 |
| zinnia_halo2 | halo2_lit | prove_core_s | 2 | 0.539 | [0.384, 0.757] | 2 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 2 | 0.555 | [0.397, 0.774] | 2 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 2 | 0.666 | [0.442, 1.003] | 1 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 2 | 2.985 | [2.412, 3.693] | 0 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_s | 2 | 2.530 | [1.978, 3.236] | 0 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 2 | 2.075 | [1.623, 2.652] | 0 / 2 |
| zinnia_halo2 | noir_lit | verify_s | 2 | 3.617 | [2.624, 4.987] | 0 / 2 |
| zinnia_noir | halo2_hw | prove_core_s | 2 | 0.223 | [0.180, 0.278] | 2 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 2 | 0.267 | [0.212, 0.337] | 2 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 2 | 0.508 | [0.336, 0.766] | 2 / 0 |
| zinnia_noir | halo2_hw | verify_s | 2 | 0.188 | [0.140, 0.251] | 2 / 0 |
| zinnia_noir | halo2_lit | prove_core_s | 2 | 0.157 | [0.150, 0.163] | 2 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 2 | 0.187 | [0.182, 0.192] | 2 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 2 | 0.355 | [0.305, 0.414] | 2 / 0 |
| zinnia_noir | halo2_lit | verify_s | 2 | 0.171 | [0.159, 0.184] | 2 / 0 |
| zinnia_noir | noir_lit | prove_core_s | 2 | 0.868 | [0.797, 0.944] | 2 / 0 |
| zinnia_noir | noir_lit | end_to_end_s | 2 | 0.854 | [0.804, 0.907] | 2 / 0 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 2 | 1.330 | [1.045, 1.691] | 0 / 2 |
| zinnia_noir | noir_lit | verify_s | 2 | 0.929 | [0.791, 1.091] | 1 / 1 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 2 | 0.768 | [0.709, 0.832] | 2 / 0 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 2 | 0.792 | [0.735, 0.853] | 2 / 0 |
| zinnia_halo2 | halo2_hw | verify_s | 2 | 0.731 | [0.604, 0.884] | 2 / 0 |
| zinnia_noir | noir_hw | prove_core_s | 2 | 0.938 | [0.765, 1.150] | 1 / 1 |
| zinnia_noir | noir_hw | end_to_end_s | 2 | 0.962 | [0.788, 1.175] | 1 / 1 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 2 | 1.463 | [1.020, 2.098] | 0 / 2 |
| zinnia_noir | noir_hw | verify_s | 2 | 0.950 | [0.891, 1.012] | 1 / 1 |
| zinnia_halo2 | noir_hw | prove_core_s | 2 | 3.226 | [2.938, 3.541] | 0 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_s | 2 | 2.851 | [2.563, 3.172] | 0 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 2 | 2.282 | [2.013, 2.587] | 0 / 2 |
| zinnia_halo2 | noir_hw | verify_s | 2 | 3.698 | [2.434, 5.619] | 0 / 2 |
| zinnia_halo2 | halo2_lit | prove_core_s | 2 | 0.539 | [0.384, 0.757] | 2 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 2 | 0.555 | [0.397, 0.774] | 2 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 2 | 0.666 | [0.442, 1.003] | 1 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 2 | 2.985 | [2.412, 3.693] | 0 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_s | 2 | 2.530 | [1.978, 3.236] | 0 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 2 | 2.075 | [1.623, 2.652] | 0 / 2 |
| zinnia_halo2 | noir_lit | verify_s | 2 | 3.617 | [2.624, 4.987] | 0 / 2 |
| zinnia_noir | halo2_hw | prove_core_s | 2 | 0.223 | [0.180, 0.278] | 2 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 2 | 0.267 | [0.212, 0.337] | 2 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 2 | 0.508 | [0.336, 0.766] | 2 / 0 |
| zinnia_noir | halo2_hw | verify_s | 2 | 0.188 | [0.140, 0.251] | 2 / 0 |
| zinnia_noir | halo2_lit | prove_core_s | 2 | 0.157 | [0.150, 0.163] | 2 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 2 | 0.187 | [0.182, 0.192] | 2 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 2 | 0.355 | [0.305, 0.414] | 2 / 0 |
| zinnia_noir | halo2_lit | verify_s | 2 | 0.171 | [0.159, 0.184] | 2 / 0 |
| zinnia_noir | noir_lit | prove_core_s | 2 | 0.868 | [0.797, 0.944] | 2 / 0 |
| zinnia_noir | noir_lit | end_to_end_s | 2 | 0.854 | [0.804, 0.907] | 2 / 0 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 2 | 1.330 | [1.045, 1.691] | 0 / 2 |
| zinnia_noir | noir_lit | verify_s | 2 | 0.929 | [0.791, 1.091] | 1 / 1 |

## Excluded cases

(none)

## Cases without a program

No program for the case on that system (e.g. no handwritten baseline yet); not a failure.

(none)

## Failures by system

