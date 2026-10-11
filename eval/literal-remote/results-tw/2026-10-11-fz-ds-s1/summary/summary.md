# Epoch 2026-10-11-fz-ds-s1

Ratios are Zinnia / baseline of per-case medians; below 1 means Zinnia is faster.
Only cases with status `ok` (proved, verified, outputs equal to NumPy, tampered proof rejected) enter a comparison.

`zinnia_noir` `end_to_end_s` witness step: noir-execute (44 cases); `end_to_end_nargo_s` uses `nargo execute`.
`noir_hw` `end_to_end_s` witness step: noir-execute (44 cases); `end_to_end_nargo_s` uses `nargo execute`.
`noir_lit` `end_to_end_s` witness step: noir-execute (44 cases); `end_to_end_nargo_s` uses `nargo execute`.

## Coverage

| system | cases | ok | other statuses |
|---|---|---|---|
| zinnia_halo2 | 44 | 44 | - |
| zinnia_noir | 44 | 44 | - |
| halo2_hw | 44 | 44 | - |
| noir_hw | 44 | 44 | - |
| halo2_lit | 44 | 44 | - |
| noir_lit | 44 | 44 | - |

## Headline: all-supported subset without copy-only cases

33 cases completed by every system that do more than copy inputs: ds1000_case0298, ds1000_case0309, ds1000_case0312, ds1000_case0320, ds1000_case0323, ds1000_case0326, ds1000_case0334, ds1000_case0347, ds1000_case0355, ds1000_case0358, ds1000_case0365, ds1000_case0370, ds1000_case0375, ds1000_case0383, ds1000_case0399, ds1000_case0402, ds1000_case0412, ds1000_case0415, ds1000_case0418, ds1000_case0422, ds1000_case0430, ds1000_case0434, ds1000_case0437, ds1000_case0440, ds1000_case0443, ds1000_case0452, ds1000_case0456, ds1000_case0462, ds1000_case0481, ds1000_case0491, ds1000_case0502, ds1000_case0505, ds1000_case0509

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 33 | 0.931 | [0.876, 0.992] | 26 / 7 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 33 | 0.940 | [0.885, 1.003] | 25 / 8 |
| zinnia_halo2 | halo2_hw | verify_s | 33 | 0.967 | [0.914, 1.023] | 19 / 14 |
| zinnia_noir | noir_hw | prove_core_s | 33 | 0.985 | [0.949, 1.023] | 21 / 12 |
| zinnia_noir | noir_hw | end_to_end_s | 33 | 0.989 | [0.953, 1.028] | 19 / 14 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 33 | 0.986 | [0.965, 1.010] | 19 / 14 |
| zinnia_noir | noir_hw | verify_s | 33 | 0.989 | [0.914, 1.073] | 18 / 15 |
| zinnia_halo2 | noir_hw | prove_core_s | 33 | 0.987 | [0.871, 1.117] | 14 / 19 |
| zinnia_halo2 | noir_hw | end_to_end_s | 33 | 0.927 | [0.794, 1.069] | 16 / 17 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 33 | 0.451 | [0.385, 0.527] | 32 / 1 |
| zinnia_halo2 | noir_hw | verify_s | 33 | 0.605 | [0.545, 0.671] | 31 / 2 |
| zinnia_halo2 | halo2_lit | prove_core_s | 33 | 0.597 | [0.451, 0.741] | 31 / 2 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 33 | 0.603 | [0.456, 0.748] | 31 / 2 |
| zinnia_halo2 | halo2_lit | verify_s | 33 | 0.809 | [0.700, 0.906] | 27 / 6 |
| zinnia_halo2 | noir_lit | prove_core_s | 33 | 0.836 | [0.706, 0.980] | 19 / 14 |
| zinnia_halo2 | noir_lit | end_to_end_s | 33 | 0.778 | [0.641, 0.931] | 20 / 13 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 33 | 0.402 | [0.339, 0.476] | 32 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 33 | 0.631 | [0.568, 0.700] | 31 / 2 |
| zinnia_noir | halo2_hw | prove_core_s | 33 | 0.928 | [0.792, 1.081] | 20 / 13 |
| zinnia_noir | halo2_hw | end_to_end_s | 33 | 1.003 | [0.835, 1.231] | 19 / 14 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 33 | 2.055 | [1.716, 2.487] | 2 / 31 |
| zinnia_noir | halo2_hw | verify_s | 33 | 1.580 | [1.403, 1.765] | 4 / 29 |
| zinnia_noir | halo2_lit | prove_core_s | 33 | 0.596 | [0.427, 0.784] | 25 / 8 |
| zinnia_noir | halo2_lit | end_to_end_s | 33 | 0.644 | [0.456, 0.871] | 24 / 9 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 33 | 1.318 | [0.931, 1.775] | 8 / 25 |
| zinnia_noir | halo2_lit | verify_s | 33 | 1.321 | [1.085, 1.572] | 8 / 25 |
| zinnia_noir | noir_lit | prove_core_s | 33 | 0.834 | [0.716, 0.935] | 26 / 7 |
| zinnia_noir | noir_lit | end_to_end_s | 33 | 0.831 | [0.707, 0.936] | 26 / 7 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 33 | 0.880 | [0.781, 0.956] | 25 / 8 |
| zinnia_noir | noir_lit | verify_s | 33 | 1.031 | [0.966, 1.104] | 15 / 18 |

## Copy-only cases (reported apart)

11 all-supported cases that only copy inputs: ds1000_case0302, ds1000_case0317, ds1000_case0337, ds1000_case0340, ds1000_case0343, ds1000_case0361, ds1000_case0387, ds1000_case0392, ds1000_case0449, ds1000_case0468, ds1000_case0477

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 11 | 1.048 | [0.992, 1.100] | 1 / 10 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 11 | 1.062 | [1.002, 1.115] | 1 / 10 |
| zinnia_halo2 | halo2_hw | verify_s | 11 | 0.960 | [0.858, 1.074] | 6 / 5 |
| zinnia_noir | noir_hw | prove_core_s | 11 | 1.056 | [0.946, 1.182] | 5 / 6 |
| zinnia_noir | noir_hw | end_to_end_s | 11 | 1.046 | [0.944, 1.162] | 5 / 6 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 11 | 1.030 | [0.933, 1.162] | 6 / 5 |
| zinnia_noir | noir_hw | verify_s | 11 | 0.964 | [0.815, 1.125] | 5 / 6 |
| zinnia_halo2 | noir_hw | prove_core_s | 11 | 0.857 | [0.612, 1.328] | 9 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_s | 11 | 0.827 | [0.597, 1.247] | 9 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 11 | 0.407 | [0.267, 0.711] | 9 / 2 |
| zinnia_halo2 | noir_hw | verify_s | 11 | 0.483 | [0.388, 0.631] | 10 / 1 |
| zinnia_halo2 | halo2_lit | prove_core_s | 11 | 1.039 | [0.979, 1.096] | 3 / 8 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 11 | 1.053 | [0.989, 1.112] | 2 / 9 |
| zinnia_halo2 | halo2_lit | verify_s | 11 | 1.005 | [0.911, 1.109] | 5 / 6 |
| zinnia_halo2 | noir_lit | prove_core_s | 11 | 0.837 | [0.581, 1.316] | 9 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_s | 11 | 0.811 | [0.569, 1.249] | 9 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 11 | 0.408 | [0.266, 0.716] | 9 / 2 |
| zinnia_halo2 | noir_lit | verify_s | 11 | 0.541 | [0.463, 0.642] | 11 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 11 | 1.291 | [0.822, 1.911] | 2 / 9 |
| zinnia_noir | halo2_hw | end_to_end_s | 11 | 1.344 | [0.882, 1.953] | 2 / 9 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 11 | 2.685 | [1.686, 3.993] | 2 / 9 |
| zinnia_noir | halo2_hw | verify_s | 11 | 1.917 | [1.517, 2.385] | 1 / 10 |
| zinnia_noir | halo2_lit | prove_core_s | 11 | 1.281 | [0.820, 1.873] | 2 / 9 |
| zinnia_noir | halo2_lit | end_to_end_s | 11 | 1.333 | [0.880, 1.912] | 2 / 9 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 11 | 2.663 | [1.690, 3.911] | 2 / 9 |
| zinnia_noir | halo2_lit | verify_s | 11 | 2.007 | [1.591, 2.508] | 1 / 10 |
| zinnia_noir | noir_lit | prove_core_s | 11 | 1.031 | [0.937, 1.140] | 6 / 5 |
| zinnia_noir | noir_lit | end_to_end_s | 11 | 1.027 | [0.938, 1.129] | 6 / 5 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 11 | 1.032 | [0.940, 1.160] | 6 / 5 |
| zinnia_noir | noir_lit | verify_s | 11 | 1.081 | [0.940, 1.245] | 4 / 7 |

## All-supported subset

44 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): ds1000_case0298, ds1000_case0302, ds1000_case0309, ds1000_case0312, ds1000_case0317, ds1000_case0320, ds1000_case0323, ds1000_case0326, ds1000_case0334, ds1000_case0337, ds1000_case0340, ds1000_case0343, ds1000_case0347, ds1000_case0355, ds1000_case0358, ds1000_case0361, ds1000_case0365, ds1000_case0370, ds1000_case0375, ds1000_case0383, ds1000_case0387, ds1000_case0392, ds1000_case0399, ds1000_case0402, ds1000_case0412, ds1000_case0415, ds1000_case0418, ds1000_case0422, ds1000_case0430, ds1000_case0434, ds1000_case0437, ds1000_case0440, ds1000_case0443, ds1000_case0449, ds1000_case0452, ds1000_case0456, ds1000_case0462, ds1000_case0468, ds1000_case0477, ds1000_case0481, ds1000_case0491, ds1000_case0502, ds1000_case0505, ds1000_case0509

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 44 | 0.959 | [0.912, 1.008] | 27 / 17 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 44 | 0.969 | [0.921, 1.020] | 26 / 18 |
| zinnia_halo2 | halo2_hw | verify_s | 44 | 0.965 | [0.918, 1.015] | 25 / 19 |
| zinnia_noir | noir_hw | prove_core_s | 44 | 1.002 | [0.963, 1.044] | 26 / 18 |
| zinnia_noir | noir_hw | end_to_end_s | 44 | 1.003 | [0.965, 1.043] | 24 / 20 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 44 | 0.997 | [0.967, 1.034] | 25 / 19 |
| zinnia_noir | noir_hw | verify_s | 44 | 0.982 | [0.914, 1.056] | 23 / 21 |
| zinnia_halo2 | noir_hw | prove_core_s | 44 | 0.953 | [0.831, 1.097] | 23 / 21 |
| zinnia_halo2 | noir_hw | end_to_end_s | 44 | 0.901 | [0.775, 1.044] | 25 / 19 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 44 | 0.440 | [0.372, 0.524] | 41 / 3 |
| zinnia_halo2 | noir_hw | verify_s | 44 | 0.572 | [0.516, 0.633] | 41 / 3 |
| zinnia_halo2 | halo2_lit | prove_core_s | 44 | 0.686 | [0.550, 0.820] | 34 / 10 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 44 | 0.693 | [0.556, 0.828] | 33 / 11 |
| zinnia_halo2 | halo2_lit | verify_s | 44 | 0.854 | [0.763, 0.937] | 32 / 12 |
| zinnia_halo2 | noir_lit | prove_core_s | 44 | 0.836 | [0.715, 0.984] | 28 / 16 |
| zinnia_halo2 | noir_lit | end_to_end_s | 44 | 0.786 | [0.663, 0.931] | 29 / 15 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 44 | 0.404 | [0.341, 0.485] | 41 / 3 |
| zinnia_halo2 | noir_lit | verify_s | 44 | 0.607 | [0.554, 0.664] | 42 / 2 |
| zinnia_noir | halo2_hw | prove_core_s | 44 | 1.008 | [0.857, 1.185] | 22 / 22 |
| zinnia_noir | halo2_hw | end_to_end_s | 44 | 1.079 | [0.908, 1.295] | 21 / 23 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 44 | 2.197 | [1.844, 2.624] | 4 / 40 |
| zinnia_noir | halo2_hw | verify_s | 44 | 1.658 | [1.489, 1.842] | 5 / 39 |
| zinnia_noir | halo2_lit | prove_core_s | 44 | 0.721 | [0.546, 0.929] | 27 / 17 |
| zinnia_noir | halo2_lit | end_to_end_s | 44 | 0.772 | [0.578, 1.005] | 26 / 18 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 44 | 1.572 | [1.176, 2.044] | 10 / 34 |
| zinnia_noir | halo2_lit | verify_s | 44 | 1.467 | [1.243, 1.711] | 9 / 35 |
| zinnia_noir | noir_lit | prove_core_s | 44 | 0.879 | [0.781, 0.966] | 32 / 12 |
| zinnia_noir | noir_lit | end_to_end_s | 44 | 0.876 | [0.773, 0.965] | 32 / 12 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 44 | 0.916 | [0.834, 0.985] | 31 / 13 |
| zinnia_noir | noir_lit | verify_s | 44 | 1.043 | [0.982, 1.109] | 19 / 25 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 44 | 0.959 | [0.912, 1.008] | 27 / 17 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 44 | 0.969 | [0.921, 1.020] | 26 / 18 |
| zinnia_halo2 | halo2_hw | verify_s | 44 | 0.965 | [0.918, 1.015] | 25 / 19 |
| zinnia_noir | noir_hw | prove_core_s | 44 | 1.002 | [0.963, 1.044] | 26 / 18 |
| zinnia_noir | noir_hw | end_to_end_s | 44 | 1.003 | [0.965, 1.043] | 24 / 20 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 44 | 0.997 | [0.967, 1.034] | 25 / 19 |
| zinnia_noir | noir_hw | verify_s | 44 | 0.982 | [0.914, 1.056] | 23 / 21 |
| zinnia_halo2 | noir_hw | prove_core_s | 44 | 0.953 | [0.831, 1.097] | 23 / 21 |
| zinnia_halo2 | noir_hw | end_to_end_s | 44 | 0.901 | [0.775, 1.044] | 25 / 19 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 44 | 0.440 | [0.372, 0.524] | 41 / 3 |
| zinnia_halo2 | noir_hw | verify_s | 44 | 0.572 | [0.516, 0.633] | 41 / 3 |
| zinnia_halo2 | halo2_lit | prove_core_s | 44 | 0.686 | [0.550, 0.820] | 34 / 10 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 44 | 0.693 | [0.556, 0.828] | 33 / 11 |
| zinnia_halo2 | halo2_lit | verify_s | 44 | 0.854 | [0.763, 0.937] | 32 / 12 |
| zinnia_halo2 | noir_lit | prove_core_s | 44 | 0.836 | [0.715, 0.984] | 28 / 16 |
| zinnia_halo2 | noir_lit | end_to_end_s | 44 | 0.786 | [0.663, 0.931] | 29 / 15 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 44 | 0.404 | [0.341, 0.485] | 41 / 3 |
| zinnia_halo2 | noir_lit | verify_s | 44 | 0.607 | [0.554, 0.664] | 42 / 2 |
| zinnia_noir | halo2_hw | prove_core_s | 44 | 1.008 | [0.857, 1.185] | 22 / 22 |
| zinnia_noir | halo2_hw | end_to_end_s | 44 | 1.079 | [0.908, 1.295] | 21 / 23 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 44 | 2.197 | [1.844, 2.624] | 4 / 40 |
| zinnia_noir | halo2_hw | verify_s | 44 | 1.658 | [1.489, 1.842] | 5 / 39 |
| zinnia_noir | halo2_lit | prove_core_s | 44 | 0.721 | [0.546, 0.929] | 27 / 17 |
| zinnia_noir | halo2_lit | end_to_end_s | 44 | 0.772 | [0.578, 1.005] | 26 / 18 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 44 | 1.572 | [1.176, 2.044] | 10 / 34 |
| zinnia_noir | halo2_lit | verify_s | 44 | 1.467 | [1.243, 1.711] | 9 / 35 |
| zinnia_noir | noir_lit | prove_core_s | 44 | 0.879 | [0.781, 0.966] | 32 / 12 |
| zinnia_noir | noir_lit | end_to_end_s | 44 | 0.876 | [0.773, 0.965] | 32 / 12 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 44 | 0.916 | [0.834, 0.985] | 31 / 13 |
| zinnia_noir | noir_lit | verify_s | 44 | 1.043 | [0.982, 1.109] | 19 / 25 |

## Excluded cases

(none)

## Cases without a program

No program for the case on that system (e.g. no handwritten baseline yet); not a failure.

(none)

## Failures by system

