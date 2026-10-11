# Epoch 2026-10-09-aa1-ds-s5

Ratios are Zinnia / baseline of per-case medians; below 1 means Zinnia is faster.
Only cases with status `ok` (proved, verified, outputs equal to NumPy, tampered proof rejected) enter a comparison.

`zinnia_noir` `end_to_end_s` witness step: noir-execute (4 cases); `end_to_end_nargo_s` uses `nargo execute`.
`noir_hw` `end_to_end_s` witness step: noir-execute (6 cases); `end_to_end_nargo_s` uses `nargo execute`.
`noir_lit` `end_to_end_s` witness step: noir-execute (6 cases); `end_to_end_nargo_s` uses `nargo execute`.

## Coverage

| system | cases | ok | other statuses |
|---|---|---|---|
| zinnia_halo2 | 6 | 4 | wrong_output: 2 |
| zinnia_noir | 6 | 4 | wrong_output: 2 |
| halo2_hw | 6 | 6 | - |
| noir_hw | 6 | 6 | - |
| halo2_lit | 6 | 6 | - |
| noir_lit | 6 | 6 | - |

## Headline: all-supported subset without copy-only cases

3 cases completed by every system that do more than copy inputs: ds1000_case0458, ds1000_case0463, ds1000_case0502

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 3 | 0.917 | [0.735, 1.174] | 2 / 1 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 3 | 0.929 | [0.747, 1.193] | 2 / 1 |
| zinnia_halo2 | halo2_hw | verify_s | 3 | 0.983 | [0.921, 1.018] | 1 / 2 |
| zinnia_noir | noir_hw | prove_core_s | 3 | 1.112 | [0.885, 1.333] | 1 / 2 |
| zinnia_noir | noir_hw | end_to_end_s | 3 | 1.112 | [0.866, 1.356] | 1 / 2 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 3 | 1.033 | [0.876, 1.232] | 1 / 2 |
| zinnia_noir | noir_hw | verify_s | 3 | 1.120 | [0.974, 1.448] | 2 / 1 |
| zinnia_halo2 | noir_hw | prove_core_s | 3 | 1.467 | [0.892, 2.049] | 1 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_s | 3 | 1.417 | [0.867, 1.939] | 1 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 3 | 0.678 | [0.367, 1.005] | 2 / 1 |
| zinnia_halo2 | noir_hw | verify_s | 3 | 0.913 | [0.538, 1.286] | 1 / 2 |
| zinnia_halo2 | halo2_lit | prove_core_s | 3 | 0.577 | [0.462, 0.722] | 3 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 3 | 0.585 | [0.466, 0.733] | 3 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 3 | 0.905 | [0.810, 1.002] | 2 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 3 | 1.091 | [0.573, 1.928] | 1 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_s | 3 | 1.049 | [0.540, 1.872] | 1 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 3 | 0.598 | [0.309, 1.016] | 2 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 3 | 0.916 | [0.624, 1.260] | 2 / 1 |
| zinnia_noir | halo2_hw | prove_core_s | 3 | 0.694 | [0.317, 1.165] | 2 / 1 |
| zinnia_noir | halo2_hw | end_to_end_s | 3 | 0.729 | [0.334, 1.215] | 2 / 1 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 3 | 1.416 | [0.651, 2.509] | 1 / 2 |
| zinnia_noir | halo2_hw | verify_s | 3 | 1.207 | [0.920, 1.845] | 1 / 2 |
| zinnia_noir | halo2_lit | prove_core_s | 3 | 0.437 | [0.312, 0.603] | 3 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 3 | 0.459 | [0.327, 0.628] | 3 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 3 | 0.892 | [0.639, 1.297] | 2 / 1 |
| zinnia_noir | halo2_lit | verify_s | 3 | 1.111 | [0.828, 1.468] | 1 / 2 |
| zinnia_noir | noir_lit | prove_core_s | 3 | 0.827 | [0.748, 0.907] | 3 / 0 |
| zinnia_noir | noir_lit | end_to_end_s | 3 | 0.823 | [0.728, 0.915] | 3 / 0 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 3 | 0.910 | [0.861, 0.987] | 3 / 0 |
| zinnia_noir | noir_lit | verify_s | 3 | 1.125 | [0.886, 1.419] | 1 / 2 |

## Copy-only cases (reported apart)

1 all-supported cases that only copy inputs: ds1000_case0477

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 1 | 1.162 | [1.162, 1.162] | 0 / 1 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 1 | 1.166 | [1.166, 1.166] | 0 / 1 |
| zinnia_halo2 | halo2_hw | verify_s | 1 | 1.316 | [1.316, 1.316] | 0 / 1 |
| zinnia_noir | noir_hw | prove_core_s | 1 | 0.947 | [0.947, 0.947] | 1 / 0 |
| zinnia_noir | noir_hw | end_to_end_s | 1 | 0.953 | [0.953, 0.953] | 1 / 0 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 1 | 0.886 | [0.886, 0.886] | 1 / 0 |
| zinnia_noir | noir_hw | verify_s | 1 | 1.264 | [1.264, 1.264] | 0 / 1 |
| zinnia_halo2 | noir_hw | prove_core_s | 1 | 0.507 | [0.507, 0.507] | 1 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_s | 1 | 0.495 | [0.495, 0.495] | 1 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 1 | 0.216 | [0.216, 0.216] | 1 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 1 | 0.472 | [0.472, 0.472] | 1 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 1 | 1.192 | [1.192, 1.192] | 0 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 1 | 1.197 | [1.197, 1.197] | 0 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 1 | 0.972 | [0.972, 0.972] | 1 / 0 |
| zinnia_halo2 | noir_lit | prove_core_s | 1 | 0.495 | [0.495, 0.495] | 1 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_s | 1 | 0.482 | [0.482, 0.482] | 1 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 1 | 0.215 | [0.215, 0.215] | 1 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 1 | 0.379 | [0.379, 0.379] | 1 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 1 | 2.170 | [2.170, 2.170] | 0 / 1 |
| zinnia_noir | halo2_hw | end_to_end_s | 1 | 2.246 | [2.246, 2.246] | 0 / 1 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 1 | 4.784 | [4.784, 4.784] | 0 / 1 |
| zinnia_noir | halo2_hw | verify_s | 1 | 3.522 | [3.522, 3.522] | 0 / 1 |
| zinnia_noir | halo2_lit | prove_core_s | 1 | 2.227 | [2.227, 2.227] | 0 / 1 |
| zinnia_noir | halo2_lit | end_to_end_s | 1 | 2.305 | [2.305, 2.305] | 0 / 1 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 1 | 4.908 | [4.908, 4.908] | 0 / 1 |
| zinnia_noir | halo2_lit | verify_s | 1 | 2.600 | [2.600, 2.600] | 0 / 1 |
| zinnia_noir | noir_lit | prove_core_s | 1 | 0.925 | [0.925, 0.925] | 1 / 0 |
| zinnia_noir | noir_lit | end_to_end_s | 1 | 0.929 | [0.929, 0.929] | 1 / 0 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 1 | 0.883 | [0.883, 0.883] | 1 / 0 |
| zinnia_noir | noir_lit | verify_s | 1 | 1.014 | [1.014, 1.014] | 0 / 1 |

## All-supported subset

4 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): ds1000_case0458, ds1000_case0463, ds1000_case0477, ds1000_case0502

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 4 | 0.973 | [0.810, 1.168] | 2 / 2 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 4 | 0.984 | [0.820, 1.180] | 2 / 2 |
| zinnia_halo2 | halo2_hw | verify_s | 4 | 1.058 | [0.944, 1.233] | 1 / 3 |
| zinnia_noir | noir_hw | prove_core_s | 4 | 1.068 | [0.916, 1.246] | 2 / 2 |
| zinnia_noir | noir_hw | end_to_end_s | 4 | 1.070 | [0.909, 1.259] | 2 / 2 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 4 | 0.994 | [0.881, 1.134] | 2 / 2 |
| zinnia_noir | noir_hw | verify_s | 4 | 1.155 | [0.986, 1.353] | 2 / 2 |
| zinnia_halo2 | noir_hw | prove_core_s | 4 | 1.125 | [0.673, 1.882] | 2 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_s | 4 | 1.089 | [0.655, 1.811] | 2 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 4 | 0.510 | [0.282, 0.922] | 3 / 1 |
| zinnia_halo2 | noir_hw | verify_s | 4 | 0.774 | [0.504, 1.189] | 2 / 2 |
| zinnia_halo2 | halo2_lit | prove_core_s | 4 | 0.692 | [0.516, 0.995] | 3 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 4 | 0.700 | [0.522, 1.002] | 3 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 4 | 0.921 | [0.848, 0.987] | 3 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 4 | 0.896 | [0.533, 1.506] | 2 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_s | 4 | 0.864 | [0.510, 1.462] | 2 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 4 | 0.463 | [0.258, 0.831] | 3 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 4 | 0.735 | [0.480, 1.110] | 3 / 1 |
| zinnia_noir | halo2_hw | prove_core_s | 4 | 0.923 | [0.439, 1.744] | 2 / 2 |
| zinnia_noir | halo2_hw | end_to_end_s | 4 | 0.966 | [0.461, 1.815] | 2 / 2 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 4 | 1.919 | [0.913, 3.713] | 1 / 3 |
| zinnia_noir | halo2_hw | verify_s | 4 | 1.578 | [0.977, 2.594] | 1 / 3 |
| zinnia_noir | halo2_lit | prove_core_s | 4 | 0.657 | [0.367, 1.490] | 3 / 1 |
| zinnia_noir | halo2_lit | end_to_end_s | 4 | 0.687 | [0.385, 1.550] | 3 / 1 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 4 | 1.366 | [0.739, 3.171] | 2 / 2 |
| zinnia_noir | halo2_lit | verify_s | 4 | 1.374 | [0.955, 2.110] | 1 / 3 |
| zinnia_noir | noir_lit | prove_core_s | 4 | 0.850 | [0.785, 0.916] | 4 / 0 |
| zinnia_noir | noir_lit | end_to_end_s | 4 | 0.848 | [0.771, 0.922] | 4 / 0 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 4 | 0.903 | [0.868, 0.960] | 4 / 0 |
| zinnia_noir | noir_lit | verify_s | 4 | 1.096 | [0.942, 1.305] | 1 / 3 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 4 | 0.973 | [0.810, 1.168] | 2 / 2 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 4 | 0.984 | [0.820, 1.180] | 2 / 2 |
| zinnia_halo2 | halo2_hw | verify_s | 4 | 1.058 | [0.944, 1.233] | 1 / 3 |
| zinnia_noir | noir_hw | prove_core_s | 4 | 1.068 | [0.916, 1.246] | 2 / 2 |
| zinnia_noir | noir_hw | end_to_end_s | 4 | 1.070 | [0.909, 1.259] | 2 / 2 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 4 | 0.994 | [0.881, 1.134] | 2 / 2 |
| zinnia_noir | noir_hw | verify_s | 4 | 1.155 | [0.986, 1.353] | 2 / 2 |
| zinnia_halo2 | noir_hw | prove_core_s | 4 | 1.125 | [0.673, 1.882] | 2 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_s | 4 | 1.089 | [0.655, 1.811] | 2 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 4 | 0.510 | [0.282, 0.922] | 3 / 1 |
| zinnia_halo2 | noir_hw | verify_s | 4 | 0.774 | [0.504, 1.189] | 2 / 2 |
| zinnia_halo2 | halo2_lit | prove_core_s | 4 | 0.692 | [0.516, 0.995] | 3 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 4 | 0.700 | [0.522, 1.002] | 3 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 4 | 0.921 | [0.848, 0.987] | 3 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 4 | 0.896 | [0.533, 1.506] | 2 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_s | 4 | 0.864 | [0.510, 1.462] | 2 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 4 | 0.463 | [0.258, 0.831] | 3 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 4 | 0.735 | [0.480, 1.110] | 3 / 1 |
| zinnia_noir | halo2_hw | prove_core_s | 4 | 0.923 | [0.439, 1.744] | 2 / 2 |
| zinnia_noir | halo2_hw | end_to_end_s | 4 | 0.966 | [0.461, 1.815] | 2 / 2 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 4 | 1.919 | [0.913, 3.713] | 1 / 3 |
| zinnia_noir | halo2_hw | verify_s | 4 | 1.578 | [0.977, 2.594] | 1 / 3 |
| zinnia_noir | halo2_lit | prove_core_s | 4 | 0.657 | [0.367, 1.490] | 3 / 1 |
| zinnia_noir | halo2_lit | end_to_end_s | 4 | 0.687 | [0.385, 1.550] | 3 / 1 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 4 | 1.366 | [0.739, 3.171] | 2 / 2 |
| zinnia_noir | halo2_lit | verify_s | 4 | 1.374 | [0.955, 2.110] | 1 / 3 |
| zinnia_noir | noir_lit | prove_core_s | 4 | 0.850 | [0.785, 0.916] | 4 / 0 |
| zinnia_noir | noir_lit | end_to_end_s | 4 | 0.848 | [0.771, 0.922] | 4 / 0 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 4 | 0.903 | [0.868, 0.960] | 4 / 0 |
| zinnia_noir | noir_lit | verify_s | 4 | 1.096 | [0.942, 1.305] | 1 / 3 |

## Excluded cases

(none)

## Cases without a program

No program for the case on that system (e.g. no handwritten baseline yet); not a failure.

(none)

## Failures by system

- zinnia_halo2 ds1000_case0467: wrong_output at warmup: output shapes [[2], [2], [3], [3], [3], [3], [2], [2], [2], [2], [2], [3], [3], [3], [3], [3], [3], [2], [2], [2], [2], [2], [2], [3], [3], [3], [3], [3], [3], 
- zinnia_halo2 ds1000_case0468: wrong_output at warmup: output shapes [[2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], 
- zinnia_noir ds1000_case0467: wrong_output at warmup: output shapes [[2], [2], [3], [3], [3], [3], [2], [2], [2], [2], [2], [3], [3], [3], [3], [3], [3], [2], [2], [2], [2], [2], [2], [3], [3], [3], [3], [3], [3], 
- zinnia_noir ds1000_case0468: wrong_output at warmup: output shapes [[2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], 
