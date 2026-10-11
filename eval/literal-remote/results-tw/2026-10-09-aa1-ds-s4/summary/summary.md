# Epoch 2026-10-09-aa1-ds-s4

Ratios are Zinnia / baseline of per-case medians; below 1 means Zinnia is faster.
Only cases with status `ok` (proved, verified, outputs equal to NumPy, tampered proof rejected) enter a comparison.

`zinnia_noir` `end_to_end_s` witness step: noir-execute (7 cases); `end_to_end_nargo_s` uses `nargo execute`.
`noir_hw` `end_to_end_s` witness step: noir-execute (7 cases); `end_to_end_nargo_s` uses `nargo execute`.
`noir_lit` `end_to_end_s` witness step: noir-execute (7 cases); `end_to_end_nargo_s` uses `nargo execute`.

## Coverage

| system | cases | ok | other statuses |
|---|---|---|---|
| zinnia_halo2 | 7 | 7 | - |
| zinnia_noir | 7 | 7 | - |
| halo2_hw | 7 | 7 | - |
| noir_hw | 7 | 7 | - |
| halo2_lit | 7 | 7 | - |
| noir_lit | 7 | 7 | - |

## Headline: all-supported subset without copy-only cases

5 cases completed by every system that do more than copy inputs: ds1000_case0419, ds1000_case0430, ds1000_case0451, ds1000_case0456, ds1000_case0457

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 5 | 1.138 | [1.043, 1.270] | 0 / 5 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 5 | 1.145 | [1.047, 1.282] | 0 / 5 |
| zinnia_halo2 | halo2_hw | verify_s | 5 | 0.946 | [0.886, 1.008] | 4 / 1 |
| zinnia_noir | noir_hw | prove_core_s | 5 | 1.060 | [0.964, 1.165] | 1 / 4 |
| zinnia_noir | noir_hw | end_to_end_s | 5 | 1.065 | [0.972, 1.168] | 1 / 4 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 5 | 0.995 | [0.924, 1.075] | 3 / 2 |
| zinnia_noir | noir_hw | verify_s | 5 | 0.903 | [0.849, 0.957] | 5 / 0 |
| zinnia_halo2 | noir_hw | prove_core_s | 5 | 1.017 | [0.611, 1.566] | 2 / 3 |
| zinnia_halo2 | noir_hw | end_to_end_s | 5 | 0.986 | [0.591, 1.510] | 2 / 3 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 5 | 0.428 | [0.200, 0.767] | 5 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 5 | 0.549 | [0.419, 0.731] | 5 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 5 | 0.892 | [0.794, 0.984] | 4 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 5 | 0.897 | [0.801, 0.986] | 4 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 5 | 0.914 | [0.799, 1.029] | 3 / 2 |
| zinnia_halo2 | noir_lit | prove_core_s | 5 | 0.932 | [0.581, 1.305] | 2 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_s | 5 | 0.908 | [0.565, 1.265] | 2 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 5 | 0.417 | [0.198, 0.707] | 5 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 5 | 0.622 | [0.389, 0.903] | 4 / 1 |
| zinnia_noir | halo2_hw | prove_core_s | 5 | 1.185 | [0.898, 1.876] | 3 / 2 |
| zinnia_noir | halo2_hw | end_to_end_s | 5 | 1.237 | [0.935, 1.977] | 3 / 2 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 5 | 2.661 | [1.716, 5.390] | 0 / 5 |
| zinnia_noir | halo2_hw | verify_s | 5 | 1.557 | [1.202, 2.049] | 0 / 5 |
| zinnia_noir | halo2_lit | prove_core_s | 5 | 0.929 | [0.596, 1.597] | 4 / 1 |
| zinnia_noir | halo2_lit | end_to_end_s | 5 | 0.970 | [0.620, 1.682] | 4 / 1 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 5 | 2.086 | [1.139, 4.586] | 1 / 4 |
| zinnia_noir | halo2_lit | verify_s | 5 | 1.503 | [1.059, 2.223] | 1 / 4 |
| zinnia_noir | noir_lit | prove_core_s | 5 | 0.971 | [0.932, 1.005] | 3 / 2 |
| zinnia_noir | noir_lit | end_to_end_s | 5 | 0.982 | [0.934, 1.026] | 3 / 2 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 5 | 0.970 | [0.937, 1.004] | 3 / 2 |
| zinnia_noir | noir_lit | verify_s | 5 | 1.024 | [0.854, 1.204] | 2 / 3 |

## Copy-only cases (reported apart)

2 all-supported cases that only copy inputs: ds1000_case0448, ds1000_case0449

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 2 | 0.920 | [0.902, 0.939] | 2 / 0 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 2 | 0.925 | [0.906, 0.944] | 2 / 0 |
| zinnia_halo2 | halo2_hw | verify_s | 2 | 0.946 | [0.895, 1.000] | 2 / 0 |
| zinnia_noir | noir_hw | prove_core_s | 2 | 1.031 | [0.994, 1.070] | 1 / 1 |
| zinnia_noir | noir_hw | end_to_end_s | 2 | 1.054 | [1.021, 1.089] | 0 / 2 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 2 | 0.965 | [0.940, 0.991] | 2 / 0 |
| zinnia_noir | noir_hw | verify_s | 2 | 1.058 | [0.881, 1.272] | 1 / 1 |
| zinnia_halo2 | noir_hw | prove_core_s | 2 | 0.641 | [0.583, 0.704] | 2 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_s | 2 | 0.633 | [0.578, 0.694] | 2 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 2 | 0.277 | [0.265, 0.290] | 2 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 2 | 0.461 | [0.427, 0.498] | 2 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 2 | 0.912 | [0.909, 0.915] | 2 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 2 | 0.917 | [0.914, 0.920] | 2 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 2 | 1.036 | [0.865, 1.240] | 1 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 2 | 0.641 | [0.567, 0.725] | 2 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_s | 2 | 0.628 | [0.555, 0.710] | 2 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 2 | 0.281 | [0.259, 0.304] | 2 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 2 | 0.371 | [0.346, 0.398] | 2 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 2 | 1.482 | [1.273, 1.725] | 0 / 2 |
| zinnia_noir | halo2_hw | end_to_end_s | 2 | 1.540 | [1.333, 1.779] | 0 / 2 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 2 | 3.224 | [2.939, 3.537] | 0 / 2 |
| zinnia_noir | halo2_hw | verify_s | 2 | 2.172 | [1.846, 2.555] | 0 / 2 |
| zinnia_noir | halo2_lit | prove_core_s | 2 | 1.469 | [1.292, 1.670] | 0 / 2 |
| zinnia_noir | halo2_lit | end_to_end_s | 2 | 1.527 | [1.353, 1.723] | 0 / 2 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 2 | 3.196 | [2.983, 3.425] | 0 / 2 |
| zinnia_noir | halo2_lit | verify_s | 2 | 2.378 | [1.784, 3.170] | 0 / 2 |
| zinnia_noir | noir_lit | prove_core_s | 2 | 1.032 | [1.023, 1.041] | 0 / 2 |
| zinnia_noir | noir_lit | end_to_end_s | 2 | 1.045 | [1.045, 1.045] | 0 / 2 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 2 | 0.978 | [0.970, 0.986] | 2 / 0 |
| zinnia_noir | noir_lit | verify_s | 2 | 0.853 | [0.821, 0.885] | 2 / 0 |

## All-supported subset

7 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): ds1000_case0419, ds1000_case0430, ds1000_case0448, ds1000_case0449, ds1000_case0451, ds1000_case0456, ds1000_case0457

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 7 | 1.071 | [0.974, 1.194] | 2 / 5 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 7 | 1.077 | [0.978, 1.203] | 2 / 5 |
| zinnia_halo2 | halo2_hw | verify_s | 7 | 0.946 | [0.899, 0.996] | 6 / 1 |
| zinnia_noir | noir_hw | prove_core_s | 7 | 1.052 | [0.979, 1.129] | 2 / 5 |
| zinnia_noir | noir_hw | end_to_end_s | 7 | 1.062 | [0.990, 1.138] | 1 / 6 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 7 | 0.986 | [0.934, 1.048] | 5 / 2 |
| zinnia_noir | noir_hw | verify_s | 7 | 0.945 | [0.867, 1.054] | 6 / 1 |
| zinnia_halo2 | noir_hw | prove_core_s | 7 | 0.891 | [0.602, 1.304] | 4 / 3 |
| zinnia_halo2 | noir_hw | end_to_end_s | 7 | 0.869 | [0.588, 1.264] | 4 / 3 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 7 | 0.378 | [0.220, 0.618] | 7 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 7 | 0.522 | [0.430, 0.651] | 7 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 7 | 0.898 | [0.827, 0.962] | 6 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 7 | 0.903 | [0.833, 0.966] | 6 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 7 | 0.947 | [0.835, 1.073] | 4 / 3 |
| zinnia_halo2 | noir_lit | prove_core_s | 7 | 0.838 | [0.593, 1.143] | 4 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_s | 7 | 0.817 | [0.579, 1.112] | 4 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 7 | 0.373 | [0.219, 0.586] | 7 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 7 | 0.537 | [0.378, 0.763] | 6 / 1 |
| zinnia_noir | halo2_hw | prove_core_s | 7 | 1.263 | [0.967, 1.768] | 3 / 4 |
| zinnia_noir | halo2_hw | end_to_end_s | 7 | 1.317 | [1.004, 1.846] | 3 / 4 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 7 | 2.811 | [1.916, 4.643] | 0 / 7 |
| zinnia_noir | halo2_hw | verify_s | 7 | 1.712 | [1.354, 2.140] | 0 / 7 |
| zinnia_noir | halo2_lit | prove_core_s | 7 | 1.059 | [0.713, 1.603] | 4 / 3 |
| zinnia_noir | halo2_lit | end_to_end_s | 7 | 1.104 | [0.744, 1.678] | 4 / 3 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 7 | 2.356 | [1.415, 4.180] | 1 / 6 |
| zinnia_noir | halo2_lit | verify_s | 7 | 1.714 | [1.229, 2.417] | 1 / 6 |
| zinnia_noir | noir_lit | prove_core_s | 7 | 0.988 | [0.952, 1.019] | 3 / 4 |
| zinnia_noir | noir_lit | end_to_end_s | 7 | 0.999 | [0.957, 1.035] | 3 / 4 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 7 | 0.972 | [0.948, 0.996] | 5 / 2 |
| zinnia_noir | noir_lit | verify_s | 7 | 0.972 | [0.850, 1.119] | 4 / 3 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 7 | 1.071 | [0.974, 1.194] | 2 / 5 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 7 | 1.077 | [0.978, 1.203] | 2 / 5 |
| zinnia_halo2 | halo2_hw | verify_s | 7 | 0.946 | [0.899, 0.996] | 6 / 1 |
| zinnia_noir | noir_hw | prove_core_s | 7 | 1.052 | [0.979, 1.129] | 2 / 5 |
| zinnia_noir | noir_hw | end_to_end_s | 7 | 1.062 | [0.990, 1.138] | 1 / 6 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 7 | 0.986 | [0.934, 1.048] | 5 / 2 |
| zinnia_noir | noir_hw | verify_s | 7 | 0.945 | [0.867, 1.054] | 6 / 1 |
| zinnia_halo2 | noir_hw | prove_core_s | 7 | 0.891 | [0.602, 1.304] | 4 / 3 |
| zinnia_halo2 | noir_hw | end_to_end_s | 7 | 0.869 | [0.588, 1.264] | 4 / 3 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 7 | 0.378 | [0.220, 0.618] | 7 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 7 | 0.522 | [0.430, 0.651] | 7 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 7 | 0.898 | [0.827, 0.962] | 6 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 7 | 0.903 | [0.833, 0.966] | 6 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 7 | 0.947 | [0.835, 1.073] | 4 / 3 |
| zinnia_halo2 | noir_lit | prove_core_s | 7 | 0.838 | [0.593, 1.143] | 4 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_s | 7 | 0.817 | [0.579, 1.112] | 4 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 7 | 0.373 | [0.219, 0.586] | 7 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 7 | 0.537 | [0.378, 0.763] | 6 / 1 |
| zinnia_noir | halo2_hw | prove_core_s | 7 | 1.263 | [0.967, 1.768] | 3 / 4 |
| zinnia_noir | halo2_hw | end_to_end_s | 7 | 1.317 | [1.004, 1.846] | 3 / 4 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 7 | 2.811 | [1.916, 4.643] | 0 / 7 |
| zinnia_noir | halo2_hw | verify_s | 7 | 1.712 | [1.354, 2.140] | 0 / 7 |
| zinnia_noir | halo2_lit | prove_core_s | 7 | 1.059 | [0.713, 1.603] | 4 / 3 |
| zinnia_noir | halo2_lit | end_to_end_s | 7 | 1.104 | [0.744, 1.678] | 4 / 3 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 7 | 2.356 | [1.415, 4.180] | 1 / 6 |
| zinnia_noir | halo2_lit | verify_s | 7 | 1.714 | [1.229, 2.417] | 1 / 6 |
| zinnia_noir | noir_lit | prove_core_s | 7 | 0.988 | [0.952, 1.019] | 3 / 4 |
| zinnia_noir | noir_lit | end_to_end_s | 7 | 0.999 | [0.957, 1.035] | 3 / 4 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 7 | 0.972 | [0.948, 0.996] | 5 / 2 |
| zinnia_noir | noir_lit | verify_s | 7 | 0.972 | [0.850, 1.119] | 4 / 3 |

## Excluded cases

(none)

## Cases without a program

No program for the case on that system (e.g. no handwritten baseline yet); not a failure.

(none)

## Failures by system

