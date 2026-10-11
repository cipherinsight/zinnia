# Epoch 2026-10-09-aa1-ds-s0

Ratios are Zinnia / baseline of per-case medians; below 1 means Zinnia is faster.
Only cases with status `ok` (proved, verified, outputs equal to NumPy, tampered proof rejected) enter a comparison.

`zinnia_noir` `end_to_end_s` witness step: noir-execute (7 cases); `end_to_end_nargo_s` uses `nargo execute`.
`noir_hw` `end_to_end_s` witness step: noir-execute (6 cases); `end_to_end_nargo_s` uses `nargo execute`.
`noir_lit` `end_to_end_s` witness step: noir-execute (6 cases); `end_to_end_nargo_s` uses `nargo execute`.

## Coverage

| system | cases | ok | other statuses |
|---|---|---|---|
| zinnia_halo2 | 7 | 7 | - |
| zinnia_noir | 7 | 7 | - |
| halo2_hw | 7 | 7 | - |
| noir_hw | 7 | 6 | prove_failed: 1 |
| halo2_lit | 7 | 7 | - |
| noir_lit | 7 | 6 | prove_failed: 1 |

## Headline: all-supported subset without copy-only cases

2 cases completed by every system that do more than copy inputs: ds1000_case0313, ds1000_case0330

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 2 | 1.116 | [1.080, 1.154] | 0 / 2 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 2 | 1.125 | [1.085, 1.167] | 0 / 2 |
| zinnia_halo2 | halo2_hw | verify_s | 2 | 0.968 | [0.842, 1.112] | 1 / 1 |
| zinnia_noir | noir_hw | prove_core_s | 2 | 0.971 | [0.893, 1.055] | 1 / 1 |
| zinnia_noir | noir_hw | end_to_end_s | 2 | 0.991 | [0.922, 1.066] | 1 / 1 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 2 | 1.002 | [0.985, 1.020] | 1 / 1 |
| zinnia_noir | noir_hw | verify_s | 2 | 1.085 | [0.982, 1.199] | 1 / 1 |
| zinnia_halo2 | noir_hw | prove_core_s | 2 | 1.448 | [1.326, 1.581] | 0 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_s | 2 | 1.421 | [1.308, 1.544] | 0 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 2 | 0.750 | [0.677, 0.831] | 2 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 2 | 0.915 | [0.829, 1.011] | 1 / 1 |
| zinnia_halo2 | halo2_lit | prove_core_s | 2 | 0.910 | [0.820, 1.010] | 1 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 2 | 0.918 | [0.828, 1.017] | 1 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 2 | 0.940 | [0.844, 1.046] | 1 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 2 | 1.371 | [1.279, 1.470] | 0 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_s | 2 | 1.345 | [1.253, 1.445] | 0 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 2 | 0.741 | [0.662, 0.830] | 2 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 2 | 0.933 | [0.838, 1.038] | 1 / 1 |
| zinnia_noir | halo2_hw | prove_core_s | 2 | 0.748 | [0.721, 0.777] | 2 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 2 | 0.785 | [0.749, 0.822] | 2 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 2 | 1.504 | [1.333, 1.698] | 0 / 2 |
| zinnia_noir | halo2_hw | verify_s | 2 | 1.147 | [1.080, 1.218] | 0 / 2 |
| zinnia_noir | halo2_lit | prove_core_s | 2 | 0.610 | [0.552, 0.674] | 2 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 2 | 0.640 | [0.584, 0.702] | 2 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 2 | 1.227 | [1.206, 1.248] | 0 / 2 |
| zinnia_noir | halo2_lit | verify_s | 2 | 1.114 | [1.016, 1.221] | 0 / 2 |
| zinnia_noir | noir_lit | prove_core_s | 2 | 0.919 | [0.861, 0.981] | 2 / 0 |
| zinnia_noir | noir_lit | end_to_end_s | 2 | 0.938 | [0.883, 0.997] | 2 / 0 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 2 | 0.991 | [0.964, 1.019] | 1 / 1 |
| zinnia_noir | noir_lit | verify_s | 2 | 1.105 | [0.814, 1.502] | 1 / 1 |

## Copy-only cases (reported apart)

4 all-supported cases that only copy inputs: ds1000_case0302, ds1000_case0303, ds1000_case0304, ds1000_case0337

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 4 | 1.098 | [1.001, 1.204] | 1 / 3 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 4 | 1.102 | [1.005, 1.208] | 1 / 3 |
| zinnia_halo2 | halo2_hw | verify_s | 4 | 1.041 | [1.015, 1.071] | 0 / 4 |
| zinnia_noir | noir_hw | prove_core_s | 4 | 0.931 | [0.830, 1.069] | 3 / 1 |
| zinnia_noir | noir_hw | end_to_end_s | 4 | 0.929 | [0.838, 1.056] | 3 / 1 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 4 | 0.947 | [0.892, 1.049] | 3 / 1 |
| zinnia_noir | noir_hw | verify_s | 4 | 1.094 | [0.917, 1.305] | 2 / 2 |
| zinnia_halo2 | noir_hw | prove_core_s | 4 | 0.553 | [0.501, 0.624] | 4 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_s | 4 | 0.542 | [0.489, 0.612] | 4 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 4 | 0.256 | [0.229, 0.279] | 4 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 4 | 0.442 | [0.417, 0.468] | 4 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 4 | 1.037 | [0.954, 1.126] | 2 / 2 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 4 | 1.040 | [0.958, 1.128] | 2 / 2 |
| zinnia_halo2 | halo2_lit | verify_s | 4 | 0.932 | [0.896, 0.966] | 4 / 0 |
| zinnia_halo2 | noir_lit | prove_core_s | 4 | 0.619 | [0.544, 0.673] | 4 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_s | 4 | 0.605 | [0.534, 0.661] | 4 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 4 | 0.261 | [0.221, 0.292] | 4 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 4 | 0.454 | [0.407, 0.489] | 4 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 4 | 1.846 | [1.491, 2.400] | 0 / 4 |
| zinnia_noir | halo2_hw | end_to_end_s | 4 | 1.888 | [1.522, 2.440] | 0 / 4 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 4 | 4.076 | [3.320, 5.232] | 0 / 4 |
| zinnia_noir | halo2_hw | verify_s | 4 | 2.575 | [2.245, 3.047] | 0 / 4 |
| zinnia_noir | halo2_lit | prove_core_s | 4 | 1.743 | [1.421, 2.325] | 0 / 4 |
| zinnia_noir | halo2_lit | end_to_end_s | 4 | 1.782 | [1.452, 2.360] | 0 / 4 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 4 | 3.847 | [3.166, 5.060] | 0 / 4 |
| zinnia_noir | halo2_lit | verify_s | 4 | 2.306 | [1.982, 2.682] | 0 / 4 |
| zinnia_noir | noir_lit | prove_core_s | 4 | 1.040 | [0.974, 1.152] | 2 / 2 |
| zinnia_noir | noir_lit | end_to_end_s | 4 | 1.036 | [0.965, 1.140] | 2 / 2 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 4 | 0.966 | [0.918, 1.017] | 3 / 1 |
| zinnia_noir | noir_lit | verify_s | 4 | 1.122 | [1.001, 1.291] | 1 / 3 |

## All-supported subset

6 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): ds1000_case0302, ds1000_case0303, ds1000_case0304, ds1000_case0313, ds1000_case0330, ds1000_case0337

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 6 | 1.104 | [1.035, 1.175] | 1 / 5 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 6 | 1.110 | [1.039, 1.180] | 1 / 5 |
| zinnia_halo2 | halo2_hw | verify_s | 6 | 1.016 | [0.939, 1.079] | 1 / 5 |
| zinnia_noir | noir_hw | prove_core_s | 6 | 0.944 | [0.861, 1.046] | 4 / 2 |
| zinnia_noir | noir_hw | end_to_end_s | 6 | 0.949 | [0.868, 1.042] | 4 / 2 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 6 | 0.965 | [0.911, 1.030] | 4 / 2 |
| zinnia_noir | noir_hw | verify_s | 6 | 1.091 | [0.959, 1.241] | 3 / 3 |
| zinnia_halo2 | noir_hw | prove_core_s | 6 | 0.763 | [0.539, 1.113] | 4 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_s | 6 | 0.747 | [0.529, 1.092] | 4 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 6 | 0.366 | [0.250, 0.558] | 6 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 6 | 0.564 | [0.435, 0.757] | 5 / 1 |
| zinnia_halo2 | halo2_lit | prove_core_s | 6 | 0.993 | [0.907, 1.079] | 3 / 3 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 6 | 0.997 | [0.913, 1.082] | 3 / 3 |
| zinnia_halo2 | halo2_lit | verify_s | 6 | 0.935 | [0.885, 0.990] | 5 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 6 | 0.806 | [0.606, 1.107] | 4 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_s | 6 | 0.789 | [0.593, 1.087] | 4 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 6 | 0.370 | [0.253, 0.564] | 6 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 6 | 0.577 | [0.446, 0.776] | 5 / 1 |
| zinnia_noir | halo2_hw | prove_core_s | 6 | 1.366 | [0.938, 1.983] | 2 / 4 |
| zinnia_noir | halo2_hw | end_to_end_s | 6 | 1.409 | [0.979, 2.028] | 2 / 4 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 6 | 2.924 | [1.934, 4.364] | 0 / 6 |
| zinnia_noir | halo2_hw | verify_s | 6 | 1.967 | [1.414, 2.681] | 0 / 6 |
| zinnia_noir | halo2_lit | prove_core_s | 6 | 1.228 | [0.789, 1.866] | 2 / 4 |
| zinnia_noir | halo2_lit | end_to_end_s | 6 | 1.266 | [0.823, 1.907] | 2 / 4 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 6 | 2.628 | [1.671, 4.104] | 0 / 6 |
| zinnia_noir | halo2_lit | verify_s | 6 | 1.809 | [1.326, 2.405] | 0 / 6 |
| zinnia_noir | noir_lit | prove_core_s | 6 | 0.998 | [0.924, 1.092] | 4 / 2 |
| zinnia_noir | noir_lit | end_to_end_s | 6 | 1.002 | [0.936, 1.088] | 4 / 2 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 6 | 0.974 | [0.934, 1.012] | 4 / 2 |
| zinnia_noir | noir_lit | verify_s | 6 | 1.116 | [0.948, 1.319] | 2 / 4 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 7 | 1.086 | [1.020, 1.156] | 2 / 5 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 7 | 1.091 | [1.025, 1.161] | 2 / 5 |
| zinnia_halo2 | halo2_hw | verify_s | 7 | 1.042 | [0.960, 1.120] | 1 / 6 |
| zinnia_noir | noir_hw | prove_core_s | 6 | 0.944 | [0.861, 1.046] | 4 / 2 |
| zinnia_noir | noir_hw | end_to_end_s | 6 | 0.949 | [0.868, 1.042] | 4 / 2 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 6 | 0.965 | [0.911, 1.030] | 4 / 2 |
| zinnia_noir | noir_hw | verify_s | 6 | 1.091 | [0.959, 1.241] | 3 / 3 |
| zinnia_halo2 | noir_hw | prove_core_s | 6 | 0.763 | [0.539, 1.113] | 4 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_s | 6 | 0.747 | [0.529, 1.092] | 4 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 6 | 0.366 | [0.250, 0.558] | 6 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 6 | 0.564 | [0.435, 0.757] | 5 / 1 |
| zinnia_halo2 | halo2_lit | prove_core_s | 7 | 0.996 | [0.921, 1.070] | 3 / 4 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 7 | 1.001 | [0.928, 1.074] | 3 / 4 |
| zinnia_halo2 | halo2_lit | verify_s | 7 | 0.942 | [0.896, 0.990] | 6 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 6 | 0.806 | [0.606, 1.107] | 4 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_s | 6 | 0.789 | [0.593, 1.087] | 4 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 6 | 0.370 | [0.253, 0.564] | 6 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 6 | 0.577 | [0.446, 0.776] | 5 / 1 |
| zinnia_noir | halo2_hw | prove_core_s | 7 | 1.495 | [1.027, 2.122] | 2 / 5 |
| zinnia_noir | halo2_hw | end_to_end_s | 7 | 1.554 | [1.067, 2.209] | 2 / 5 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 7 | 3.496 | [2.181, 5.678] | 0 / 7 |
| zinnia_noir | halo2_hw | verify_s | 7 | 2.080 | [1.527, 2.738] | 0 / 7 |
| zinnia_noir | halo2_lit | prove_core_s | 7 | 1.372 | [0.886, 2.068] | 2 / 5 |
| zinnia_noir | halo2_lit | end_to_end_s | 7 | 1.426 | [0.925, 2.151] | 2 / 5 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 7 | 3.209 | [1.886, 5.517] | 0 / 7 |
| zinnia_noir | halo2_lit | verify_s | 7 | 1.880 | [1.429, 2.410] | 0 / 7 |
| zinnia_noir | noir_lit | prove_core_s | 6 | 0.998 | [0.924, 1.092] | 4 / 2 |
| zinnia_noir | noir_lit | end_to_end_s | 6 | 1.002 | [0.936, 1.088] | 4 / 2 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 6 | 0.974 | [0.934, 1.012] | 4 / 2 |
| zinnia_noir | noir_lit | verify_s | 6 | 1.116 | [0.948, 1.319] | 2 / 4 |

## Excluded cases

(none)

## Cases without a program

No program for the case on that system (e.g. no handwritten baseline yet); not a failure.

(none)

## Failures by system

- noir_hw ds1000_case0291: prove_failed at warmup: /home/zhantong/.bb/bb prove exited 1: 
Scheme is: ultra_honk, num threads: 8 (mem: 32.02 MiB)
Assertion failed: (witness.size() == constraints.max_witness_index
- noir_lit ds1000_case0291: prove_failed at warmup: /home/zhantong/.bb/bb prove exited 1: 
Scheme is: ultra_honk, num threads: 8 (mem: 32.13 MiB)
Assertion failed: (witness.size() == constraints.max_witness_index
