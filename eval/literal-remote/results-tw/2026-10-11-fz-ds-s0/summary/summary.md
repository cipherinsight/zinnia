# Epoch 2026-10-11-fz-ds-s0

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

33 cases completed by every system that do more than copy inputs: ds1000_case0291, ds1000_case0311, ds1000_case0314, ds1000_case0319, ds1000_case0322, ds1000_case0325, ds1000_case0330, ds1000_case0346, ds1000_case0353, ds1000_case0357, ds1000_case0364, ds1000_case0369, ds1000_case0374, ds1000_case0382, ds1000_case0398, ds1000_case0401, ds1000_case0409, ds1000_case0414, ds1000_case0417, ds1000_case0420, ds1000_case0429, ds1000_case0433, ds1000_case0436, ds1000_case0439, ds1000_case0442, ds1000_case0451, ds1000_case0454, ds1000_case0458, ds1000_case0480, ds1000_case0490, ds1000_case0501, ds1000_case0504, ds1000_case0508

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 33 | 0.903 | [0.808, 0.982] | 19 / 14 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 33 | 0.912 | [0.816, 0.990] | 19 / 14 |
| zinnia_halo2 | halo2_hw | verify_s | 33 | 0.959 | [0.899, 1.023] | 21 / 12 |
| zinnia_noir | noir_hw | prove_core_s | 33 | 1.000 | [0.965, 1.037] | 17 / 16 |
| zinnia_noir | noir_hw | end_to_end_s | 33 | 0.996 | [0.962, 1.032] | 17 / 16 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 33 | 0.986 | [0.956, 1.018] | 22 / 11 |
| zinnia_noir | noir_hw | verify_s | 33 | 0.995 | [0.936, 1.057] | 15 / 18 |
| zinnia_halo2 | noir_hw | prove_core_s | 33 | 1.005 | [0.829, 1.217] | 15 / 18 |
| zinnia_halo2 | noir_hw | end_to_end_s | 33 | 0.967 | [0.798, 1.167] | 16 / 17 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 33 | 0.464 | [0.356, 0.599] | 27 / 6 |
| zinnia_halo2 | noir_hw | verify_s | 33 | 0.600 | [0.510, 0.701] | 29 / 4 |
| zinnia_halo2 | halo2_lit | prove_core_s | 33 | 0.562 | [0.407, 0.738] | 28 / 5 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 33 | 0.567 | [0.412, 0.744] | 27 / 6 |
| zinnia_halo2 | halo2_lit | verify_s | 33 | 0.722 | [0.597, 0.846] | 29 / 4 |
| zinnia_halo2 | noir_lit | prove_core_s | 33 | 0.771 | [0.614, 0.954] | 19 / 14 |
| zinnia_halo2 | noir_lit | end_to_end_s | 33 | 0.729 | [0.573, 0.909] | 20 / 13 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 33 | 0.372 | [0.287, 0.478] | 28 / 5 |
| zinnia_halo2 | noir_lit | verify_s | 33 | 0.629 | [0.537, 0.729] | 29 / 4 |
| zinnia_noir | halo2_hw | prove_core_s | 33 | 0.900 | [0.701, 1.144] | 17 / 16 |
| zinnia_noir | halo2_hw | end_to_end_s | 33 | 0.940 | [0.735, 1.192] | 16 / 17 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 33 | 1.939 | [1.465, 2.583] | 7 / 26 |
| zinnia_noir | halo2_hw | verify_s | 33 | 1.590 | [1.341, 1.882] | 3 / 30 |
| zinnia_noir | halo2_lit | prove_core_s | 33 | 0.560 | [0.356, 0.843] | 19 / 14 |
| zinnia_noir | halo2_lit | end_to_end_s | 33 | 0.585 | [0.373, 0.879] | 19 / 14 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 33 | 1.207 | [0.748, 1.881] | 11 / 22 |
| zinnia_noir | halo2_lit | verify_s | 33 | 1.197 | [0.901, 1.553] | 10 / 23 |
| zinnia_noir | noir_lit | prove_core_s | 33 | 0.767 | [0.600, 0.934] | 21 / 12 |
| zinnia_noir | noir_lit | end_to_end_s | 33 | 0.751 | [0.575, 0.930] | 20 / 13 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 33 | 0.792 | [0.636, 0.940] | 24 / 9 |
| zinnia_noir | noir_lit | verify_s | 33 | 1.042 | [0.977, 1.113] | 15 / 18 |

## Copy-only cases (reported apart)

11 all-supported cases that only copy inputs: ds1000_case0301, ds1000_case0304, ds1000_case0336, ds1000_case0339, ds1000_case0342, ds1000_case0360, ds1000_case0386, ds1000_case0391, ds1000_case0448, ds1000_case0467, ds1000_case0472

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 11 | 0.980 | [0.906, 1.056] | 7 / 4 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 11 | 0.990 | [0.915, 1.066] | 6 / 5 |
| zinnia_halo2 | halo2_hw | verify_s | 11 | 1.014 | [0.904, 1.137] | 5 / 6 |
| zinnia_noir | noir_hw | prove_core_s | 11 | 0.974 | [0.905, 1.054] | 6 / 5 |
| zinnia_noir | noir_hw | end_to_end_s | 11 | 0.984 | [0.920, 1.059] | 6 / 5 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 11 | 0.967 | [0.908, 1.035] | 7 / 4 |
| zinnia_noir | noir_hw | verify_s | 11 | 1.068 | [0.932, 1.223] | 5 / 6 |
| zinnia_halo2 | noir_hw | prove_core_s | 11 | 0.818 | [0.572, 1.300] | 9 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_s | 11 | 0.784 | [0.561, 1.205] | 9 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 11 | 0.389 | [0.253, 0.683] | 9 / 2 |
| zinnia_halo2 | noir_hw | verify_s | 11 | 0.579 | [0.448, 0.778] | 9 / 2 |
| zinnia_halo2 | halo2_lit | prove_core_s | 11 | 0.980 | [0.882, 1.072] | 5 / 6 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 11 | 0.989 | [0.889, 1.084] | 5 / 6 |
| zinnia_halo2 | halo2_lit | verify_s | 11 | 0.985 | [0.902, 1.083] | 6 / 5 |
| zinnia_halo2 | noir_lit | prove_core_s | 11 | 0.892 | [0.616, 1.423] | 8 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_s | 11 | 0.843 | [0.598, 1.298] | 9 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 11 | 0.413 | [0.262, 0.745] | 9 / 2 |
| zinnia_halo2 | noir_lit | verify_s | 11 | 0.540 | [0.426, 0.708] | 9 / 2 |
| zinnia_noir | halo2_hw | prove_core_s | 11 | 1.168 | [0.711, 1.753] | 3 / 8 |
| zinnia_noir | halo2_hw | end_to_end_s | 11 | 1.243 | [0.789, 1.807] | 2 / 9 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 11 | 2.463 | [1.455, 3.761] | 2 / 9 |
| zinnia_noir | halo2_hw | verify_s | 11 | 1.870 | [1.440, 2.345] | 1 / 10 |
| zinnia_noir | halo2_lit | prove_core_s | 11 | 1.167 | [0.731, 1.704] | 2 / 9 |
| zinnia_noir | halo2_lit | end_to_end_s | 11 | 1.242 | [0.811, 1.757] | 2 / 9 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 11 | 2.462 | [1.502, 3.610] | 2 / 9 |
| zinnia_noir | halo2_lit | verify_s | 11 | 1.817 | [1.407, 2.232] | 1 / 10 |
| zinnia_noir | noir_lit | prove_core_s | 11 | 1.062 | [1.004, 1.122] | 3 / 8 |
| zinnia_noir | noir_lit | end_to_end_s | 11 | 1.059 | [1.009, 1.110] | 3 / 8 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 11 | 1.029 | [0.957, 1.116] | 6 / 5 |
| zinnia_noir | noir_lit | verify_s | 11 | 0.996 | [0.906, 1.096] | 6 / 5 |

## All-supported subset

44 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): ds1000_case0291, ds1000_case0301, ds1000_case0304, ds1000_case0311, ds1000_case0314, ds1000_case0319, ds1000_case0322, ds1000_case0325, ds1000_case0330, ds1000_case0336, ds1000_case0339, ds1000_case0342, ds1000_case0346, ds1000_case0353, ds1000_case0357, ds1000_case0360, ds1000_case0364, ds1000_case0369, ds1000_case0374, ds1000_case0382, ds1000_case0386, ds1000_case0391, ds1000_case0398, ds1000_case0401, ds1000_case0409, ds1000_case0414, ds1000_case0417, ds1000_case0420, ds1000_case0429, ds1000_case0433, ds1000_case0436, ds1000_case0439, ds1000_case0442, ds1000_case0448, ds1000_case0451, ds1000_case0454, ds1000_case0458, ds1000_case0467, ds1000_case0472, ds1000_case0480, ds1000_case0490, ds1000_case0501, ds1000_case0504, ds1000_case0508

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 44 | 0.922 | [0.847, 0.986] | 26 / 18 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 44 | 0.931 | [0.855, 0.994] | 25 / 19 |
| zinnia_halo2 | halo2_hw | verify_s | 44 | 0.972 | [0.918, 1.030] | 26 / 18 |
| zinnia_noir | noir_hw | prove_core_s | 44 | 0.994 | [0.961, 1.028] | 23 / 21 |
| zinnia_noir | noir_hw | end_to_end_s | 44 | 0.993 | [0.962, 1.026] | 23 / 21 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 44 | 0.981 | [0.954, 1.010] | 29 / 15 |
| zinnia_noir | noir_hw | verify_s | 44 | 1.013 | [0.955, 1.074] | 20 / 24 |
| zinnia_halo2 | noir_hw | prove_core_s | 44 | 0.954 | [0.801, 1.148] | 24 / 20 |
| zinnia_halo2 | noir_hw | end_to_end_s | 44 | 0.917 | [0.774, 1.096] | 25 / 19 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 44 | 0.444 | [0.353, 0.563] | 36 / 8 |
| zinnia_halo2 | noir_hw | verify_s | 44 | 0.595 | [0.519, 0.681] | 38 / 6 |
| zinnia_halo2 | halo2_lit | prove_core_s | 44 | 0.646 | [0.499, 0.801] | 33 / 11 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 44 | 0.652 | [0.504, 0.808] | 32 / 12 |
| zinnia_halo2 | halo2_lit | verify_s | 44 | 0.780 | [0.669, 0.887] | 35 / 9 |
| zinnia_halo2 | noir_lit | prove_core_s | 44 | 0.799 | [0.655, 0.976] | 27 / 17 |
| zinnia_halo2 | noir_lit | end_to_end_s | 44 | 0.756 | [0.618, 0.923] | 29 / 15 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 44 | 0.382 | [0.304, 0.485] | 37 / 7 |
| zinnia_halo2 | noir_lit | verify_s | 44 | 0.605 | [0.529, 0.690] | 38 / 6 |
| zinnia_noir | halo2_hw | prove_core_s | 44 | 0.960 | [0.768, 1.187] | 20 / 24 |
| zinnia_noir | halo2_hw | end_to_end_s | 44 | 1.008 | [0.814, 1.238] | 18 / 26 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 44 | 2.059 | [1.613, 2.611] | 9 / 35 |
| zinnia_noir | halo2_hw | verify_s | 44 | 1.656 | [1.434, 1.906] | 4 / 40 |
| zinnia_noir | halo2_lit | prove_core_s | 44 | 0.673 | [0.465, 0.937] | 21 / 23 |
| zinnia_noir | halo2_lit | end_to_end_s | 44 | 0.706 | [0.490, 0.980] | 21 / 23 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 44 | 1.442 | [0.975, 2.063] | 13 / 31 |
| zinnia_noir | halo2_lit | verify_s | 44 | 1.328 | [1.053, 1.632] | 11 / 33 |
| zinnia_noir | noir_lit | prove_core_s | 44 | 0.832 | [0.684, 0.971] | 24 / 20 |
| zinnia_noir | noir_lit | end_to_end_s | 44 | 0.818 | [0.662, 0.968] | 23 / 21 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 44 | 0.846 | [0.713, 0.969] | 30 / 14 |
| zinnia_noir | noir_lit | verify_s | 44 | 1.031 | [0.976, 1.088] | 21 / 23 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 44 | 0.922 | [0.847, 0.986] | 26 / 18 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 44 | 0.931 | [0.855, 0.994] | 25 / 19 |
| zinnia_halo2 | halo2_hw | verify_s | 44 | 0.972 | [0.918, 1.030] | 26 / 18 |
| zinnia_noir | noir_hw | prove_core_s | 44 | 0.994 | [0.961, 1.028] | 23 / 21 |
| zinnia_noir | noir_hw | end_to_end_s | 44 | 0.993 | [0.962, 1.026] | 23 / 21 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 44 | 0.981 | [0.954, 1.010] | 29 / 15 |
| zinnia_noir | noir_hw | verify_s | 44 | 1.013 | [0.955, 1.074] | 20 / 24 |
| zinnia_halo2 | noir_hw | prove_core_s | 44 | 0.954 | [0.801, 1.148] | 24 / 20 |
| zinnia_halo2 | noir_hw | end_to_end_s | 44 | 0.917 | [0.774, 1.096] | 25 / 19 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 44 | 0.444 | [0.353, 0.563] | 36 / 8 |
| zinnia_halo2 | noir_hw | verify_s | 44 | 0.595 | [0.519, 0.681] | 38 / 6 |
| zinnia_halo2 | halo2_lit | prove_core_s | 44 | 0.646 | [0.499, 0.801] | 33 / 11 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 44 | 0.652 | [0.504, 0.808] | 32 / 12 |
| zinnia_halo2 | halo2_lit | verify_s | 44 | 0.780 | [0.669, 0.887] | 35 / 9 |
| zinnia_halo2 | noir_lit | prove_core_s | 44 | 0.799 | [0.655, 0.976] | 27 / 17 |
| zinnia_halo2 | noir_lit | end_to_end_s | 44 | 0.756 | [0.618, 0.923] | 29 / 15 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 44 | 0.382 | [0.304, 0.485] | 37 / 7 |
| zinnia_halo2 | noir_lit | verify_s | 44 | 0.605 | [0.529, 0.690] | 38 / 6 |
| zinnia_noir | halo2_hw | prove_core_s | 44 | 0.960 | [0.768, 1.187] | 20 / 24 |
| zinnia_noir | halo2_hw | end_to_end_s | 44 | 1.008 | [0.814, 1.238] | 18 / 26 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 44 | 2.059 | [1.613, 2.611] | 9 / 35 |
| zinnia_noir | halo2_hw | verify_s | 44 | 1.656 | [1.434, 1.906] | 4 / 40 |
| zinnia_noir | halo2_lit | prove_core_s | 44 | 0.673 | [0.465, 0.937] | 21 / 23 |
| zinnia_noir | halo2_lit | end_to_end_s | 44 | 0.706 | [0.490, 0.980] | 21 / 23 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 44 | 1.442 | [0.975, 2.063] | 13 / 31 |
| zinnia_noir | halo2_lit | verify_s | 44 | 1.328 | [1.053, 1.632] | 11 / 33 |
| zinnia_noir | noir_lit | prove_core_s | 44 | 0.832 | [0.684, 0.971] | 24 / 20 |
| zinnia_noir | noir_lit | end_to_end_s | 44 | 0.818 | [0.662, 0.968] | 23 / 21 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 44 | 0.846 | [0.713, 0.969] | 30 / 14 |
| zinnia_noir | noir_lit | verify_s | 44 | 1.031 | [0.976, 1.088] | 21 / 23 |

## Excluded cases

(none)

## Cases without a program

No program for the case on that system (e.g. no handwritten baseline yet); not a failure.

(none)

## Failures by system

