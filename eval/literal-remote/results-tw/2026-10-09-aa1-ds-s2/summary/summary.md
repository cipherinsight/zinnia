# Epoch 2026-10-09-aa1-ds-s2

Ratios are Zinnia / baseline of per-case medians; below 1 means Zinnia is faster.
Only cases with status `ok` (proved, verified, outputs equal to NumPy, tampered proof rejected) enter a comparison.

`zinnia_noir` `end_to_end_s` witness step: noir-execute (6 cases); `end_to_end_nargo_s` uses `nargo execute`.
`noir_hw` `end_to_end_s` witness step: noir-execute (6 cases); `end_to_end_nargo_s` uses `nargo execute`.
`noir_lit` `end_to_end_s` witness step: noir-execute (6 cases); `end_to_end_nargo_s` uses `nargo execute`.

## Coverage

| system | cases | ok | other statuses |
|---|---|---|---|
| zinnia_halo2 | 6 | 6 | - |
| zinnia_noir | 6 | 6 | - |
| halo2_hw | 6 | 6 | - |
| noir_hw | 6 | 6 | - |
| halo2_lit | 6 | 6 | - |
| noir_lit | 6 | 6 | - |

## Headline: all-supported subset without copy-only cases

4 cases completed by every system that do more than copy inputs: ds1000_case0365, ds1000_case0370, ds1000_case0373, ds1000_case0382

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 4 | 0.479 | [0.220, 1.044] | 3 / 1 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 4 | 0.484 | [0.222, 1.056] | 3 / 1 |
| zinnia_halo2 | halo2_hw | verify_s | 4 | 0.661 | [0.414, 1.055] | 3 / 1 |
| zinnia_noir | noir_hw | prove_core_s | 4 | 0.778 | [0.533, 0.999] | 3 / 1 |
| zinnia_noir | noir_hw | end_to_end_s | 4 | 0.783 | [0.542, 0.993] | 4 / 0 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 4 | 0.896 | [0.770, 1.021] | 2 / 2 |
| zinnia_noir | noir_hw | verify_s | 4 | 0.949 | [0.803, 1.131] | 2 / 2 |
| zinnia_halo2 | noir_hw | prove_core_s | 4 | 0.701 | [0.287, 1.396] | 2 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_s | 4 | 0.680 | [0.279, 1.339] | 2 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 4 | 0.340 | [0.136, 0.716] | 4 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 4 | 0.546 | [0.315, 0.908] | 3 / 1 |
| zinnia_halo2 | halo2_lit | prove_core_s | 4 | 0.118 | [0.024, 0.586] | 4 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 4 | 0.119 | [0.024, 0.593] | 4 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 4 | 0.379 | [0.173, 0.833] | 4 / 0 |
| zinnia_halo2 | noir_lit | prove_core_s | 4 | 0.328 | [0.111, 0.970] | 4 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_s | 4 | 0.298 | [0.094, 0.941] | 4 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 4 | 0.178 | [0.061, 0.429] | 4 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 4 | 0.526 | [0.340, 0.813] | 4 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 4 | 0.531 | [0.236, 1.197] | 3 / 1 |
| zinnia_noir | halo2_hw | end_to_end_s | 4 | 0.558 | [0.249, 1.240] | 3 / 1 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 4 | 1.276 | [0.494, 2.718] | 1 / 3 |
| zinnia_noir | halo2_hw | verify_s | 4 | 1.149 | [0.647, 1.778] | 1 / 3 |
| zinnia_noir | halo2_lit | prove_core_s | 4 | 0.131 | [0.030, 0.570] | 4 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 4 | 0.137 | [0.031, 0.598] | 4 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 4 | 0.313 | [0.078, 1.258] | 3 / 1 |
| zinnia_noir | halo2_lit | verify_s | 4 | 0.659 | [0.321, 1.289] | 3 / 1 |
| zinnia_noir | noir_lit | prove_core_s | 4 | 0.364 | [0.140, 0.943] | 3 / 1 |
| zinnia_noir | noir_lit | end_to_end_s | 4 | 0.343 | [0.124, 0.950] | 3 / 1 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 4 | 0.469 | [0.241, 0.911] | 4 / 0 |
| zinnia_noir | noir_lit | verify_s | 4 | 0.915 | [0.859, 1.007] | 3 / 1 |

## Copy-only cases (reported apart)

2 all-supported cases that only copy inputs: ds1000_case0386, ds1000_case0387

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 2 | 0.958 | [0.952, 0.965] | 2 / 0 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 2 | 1.002 | [0.981, 1.025] | 1 / 1 |
| zinnia_halo2 | halo2_hw | verify_s | 2 | 0.911 | [0.908, 0.915] | 2 / 0 |
| zinnia_noir | noir_hw | prove_core_s | 2 | 1.056 | [1.021, 1.091] | 0 / 2 |
| zinnia_noir | noir_hw | end_to_end_s | 2 | 1.072 | [1.030, 1.115] | 0 / 2 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 2 | 1.516 | [1.474, 1.560] | 0 / 2 |
| zinnia_noir | noir_hw | verify_s | 2 | 1.175 | [1.039, 1.330] | 0 / 2 |
| zinnia_halo2 | noir_hw | prove_core_s | 2 | 5.173 | [4.800, 5.575] | 0 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_s | 2 | 4.278 | [4.036, 4.535] | 0 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 2 | 3.816 | [3.551, 4.102] | 0 / 2 |
| zinnia_halo2 | noir_hw | verify_s | 2 | 1.124 | [1.058, 1.194] | 0 / 2 |
| zinnia_halo2 | halo2_lit | prove_core_s | 2 | 0.981 | [0.974, 0.988] | 2 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 2 | 1.025 | [1.016, 1.035] | 0 / 2 |
| zinnia_halo2 | halo2_lit | verify_s | 2 | 0.936 | [0.913, 0.960] | 2 / 0 |
| zinnia_halo2 | noir_lit | prove_core_s | 2 | 4.943 | [4.908, 4.978] | 0 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_s | 2 | 4.098 | [4.057, 4.140] | 0 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 2 | 3.591 | [3.579, 3.603] | 0 / 2 |
| zinnia_halo2 | noir_lit | verify_s | 2 | 1.091 | [1.001, 1.191] | 0 / 2 |
| zinnia_noir | halo2_hw | prove_core_s | 2 | 0.196 | [0.186, 0.205] | 2 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 2 | 0.251 | [0.241, 0.262] | 2 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 2 | 0.398 | [0.352, 0.450] | 2 / 0 |
| zinnia_noir | halo2_hw | verify_s | 2 | 0.953 | [0.796, 1.141] | 1 / 1 |
| zinnia_noir | halo2_lit | prove_core_s | 2 | 0.200 | [0.193, 0.207] | 2 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 2 | 0.257 | [0.250, 0.264] | 2 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 2 | 0.407 | [0.365, 0.455] | 2 / 0 |
| zinnia_noir | halo2_lit | verify_s | 2 | 0.979 | [0.836, 1.148] | 1 / 1 |
| zinnia_noir | noir_lit | prove_core_s | 2 | 1.009 | [0.975, 1.044] | 1 / 1 |
| zinnia_noir | noir_lit | end_to_end_s | 2 | 1.027 | [0.998, 1.057] | 1 / 1 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 2 | 1.427 | [1.295, 1.573] | 0 / 2 |
| zinnia_noir | noir_lit | verify_s | 2 | 1.142 | [1.036, 1.257] | 0 / 2 |

## All-supported subset

6 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): ds1000_case0365, ds1000_case0370, ds1000_case0373, ds1000_case0382, ds1000_case0386, ds1000_case0387

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 6 | 0.604 | [0.335, 1.021] | 5 / 1 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 6 | 0.617 | [0.339, 1.046] | 4 / 2 |
| zinnia_halo2 | halo2_hw | verify_s | 6 | 0.736 | [0.518, 1.008] | 5 / 1 |
| zinnia_noir | noir_hw | prove_core_s | 6 | 0.861 | [0.649, 1.037] | 3 / 3 |
| zinnia_noir | noir_hw | end_to_end_s | 6 | 0.870 | [0.660, 1.043] | 4 / 2 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 6 | 1.068 | [0.858, 1.329] | 2 / 4 |
| zinnia_noir | noir_hw | verify_s | 6 | 1.019 | [0.870, 1.193] | 2 / 4 |
| zinnia_halo2 | noir_hw | prove_core_s | 6 | 1.365 | [0.537, 3.301] | 2 / 4 |
| zinnia_halo2 | noir_hw | end_to_end_s | 6 | 1.255 | [0.507, 2.865] | 2 / 4 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 6 | 0.761 | [0.259, 2.137] | 4 / 2 |
| zinnia_halo2 | noir_hw | verify_s | 6 | 0.694 | [0.423, 1.057] | 3 / 3 |
| zinnia_halo2 | halo2_lit | prove_core_s | 6 | 0.239 | [0.057, 0.816] | 6 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 6 | 0.244 | [0.057, 0.845] | 4 / 2 |
| zinnia_halo2 | halo2_lit | verify_s | 6 | 0.513 | [0.271, 0.900] | 6 / 0 |
| zinnia_halo2 | noir_lit | prove_core_s | 6 | 0.810 | [0.228, 2.872] | 4 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_s | 6 | 0.713 | [0.203, 2.502] | 4 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 6 | 0.484 | [0.132, 1.769] | 4 / 2 |
| zinnia_halo2 | noir_lit | verify_s | 6 | 0.671 | [0.447, 0.983] | 4 / 2 |
| zinnia_noir | halo2_hw | prove_core_s | 6 | 0.381 | [0.213, 0.758] | 5 / 1 |
| zinnia_noir | halo2_hw | end_to_end_s | 6 | 0.427 | [0.245, 0.810] | 5 / 1 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 6 | 0.866 | [0.453, 1.756] | 3 / 3 |
| zinnia_noir | halo2_hw | verify_s | 6 | 1.080 | [0.747, 1.479] | 2 / 4 |
| zinnia_noir | halo2_lit | prove_core_s | 6 | 0.151 | [0.056, 0.398] | 6 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 6 | 0.169 | [0.063, 0.447] | 6 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 6 | 0.342 | [0.126, 0.833] | 5 / 1 |
| zinnia_noir | halo2_lit | verify_s | 6 | 0.752 | [0.420, 1.211] | 4 / 2 |
| zinnia_noir | noir_lit | prove_core_s | 6 | 0.511 | [0.249, 1.042] | 4 / 2 |
| zinnia_noir | noir_lit | end_to_end_s | 6 | 0.494 | [0.232, 1.053] | 4 / 2 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 6 | 0.679 | [0.359, 1.202] | 4 / 2 |
| zinnia_noir | noir_lit | verify_s | 6 | 0.985 | [0.891, 1.109] | 3 / 3 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 6 | 0.604 | [0.335, 1.021] | 5 / 1 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 6 | 0.617 | [0.339, 1.046] | 4 / 2 |
| zinnia_halo2 | halo2_hw | verify_s | 6 | 0.736 | [0.518, 1.008] | 5 / 1 |
| zinnia_noir | noir_hw | prove_core_s | 6 | 0.861 | [0.649, 1.037] | 3 / 3 |
| zinnia_noir | noir_hw | end_to_end_s | 6 | 0.870 | [0.660, 1.043] | 4 / 2 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 6 | 1.068 | [0.858, 1.329] | 2 / 4 |
| zinnia_noir | noir_hw | verify_s | 6 | 1.019 | [0.870, 1.193] | 2 / 4 |
| zinnia_halo2 | noir_hw | prove_core_s | 6 | 1.365 | [0.537, 3.301] | 2 / 4 |
| zinnia_halo2 | noir_hw | end_to_end_s | 6 | 1.255 | [0.507, 2.865] | 2 / 4 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 6 | 0.761 | [0.259, 2.137] | 4 / 2 |
| zinnia_halo2 | noir_hw | verify_s | 6 | 0.694 | [0.423, 1.057] | 3 / 3 |
| zinnia_halo2 | halo2_lit | prove_core_s | 6 | 0.239 | [0.057, 0.816] | 6 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 6 | 0.244 | [0.057, 0.845] | 4 / 2 |
| zinnia_halo2 | halo2_lit | verify_s | 6 | 0.513 | [0.271, 0.900] | 6 / 0 |
| zinnia_halo2 | noir_lit | prove_core_s | 6 | 0.810 | [0.228, 2.872] | 4 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_s | 6 | 0.713 | [0.203, 2.502] | 4 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 6 | 0.484 | [0.132, 1.769] | 4 / 2 |
| zinnia_halo2 | noir_lit | verify_s | 6 | 0.671 | [0.447, 0.983] | 4 / 2 |
| zinnia_noir | halo2_hw | prove_core_s | 6 | 0.381 | [0.213, 0.758] | 5 / 1 |
| zinnia_noir | halo2_hw | end_to_end_s | 6 | 0.427 | [0.245, 0.810] | 5 / 1 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 6 | 0.866 | [0.453, 1.756] | 3 / 3 |
| zinnia_noir | halo2_hw | verify_s | 6 | 1.080 | [0.747, 1.479] | 2 / 4 |
| zinnia_noir | halo2_lit | prove_core_s | 6 | 0.151 | [0.056, 0.398] | 6 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 6 | 0.169 | [0.063, 0.447] | 6 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 6 | 0.342 | [0.126, 0.833] | 5 / 1 |
| zinnia_noir | halo2_lit | verify_s | 6 | 0.752 | [0.420, 1.211] | 4 / 2 |
| zinnia_noir | noir_lit | prove_core_s | 6 | 0.511 | [0.249, 1.042] | 4 / 2 |
| zinnia_noir | noir_lit | end_to_end_s | 6 | 0.494 | [0.232, 1.053] | 4 / 2 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 6 | 0.679 | [0.359, 1.202] | 4 / 2 |
| zinnia_noir | noir_lit | verify_s | 6 | 0.985 | [0.891, 1.109] | 3 / 3 |

## Excluded cases

(none)

## Cases without a program

No program for the case on that system (e.g. no handwritten baseline yet); not a failure.

(none)

## Failures by system

