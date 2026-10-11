# Epoch 2026-10-09-aa1-ds-s3

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

5 cases completed by every system that do more than copy inputs: ds1000_case0400, ds1000_case0401, ds1000_case0413, ds1000_case0417, ds1000_case0418

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 5 | 0.961 | [0.789, 1.121] | 2 / 3 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 5 | 0.966 | [0.794, 1.125] | 2 / 3 |
| zinnia_halo2 | halo2_hw | verify_s | 5 | 0.977 | [0.760, 1.193] | 2 / 3 |
| zinnia_noir | noir_hw | prove_core_s | 5 | 0.959 | [0.832, 1.104] | 3 / 2 |
| zinnia_noir | noir_hw | end_to_end_s | 5 | 0.955 | [0.828, 1.101] | 3 / 2 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 5 | 0.970 | [0.909, 1.042] | 3 / 2 |
| zinnia_noir | noir_hw | verify_s | 5 | 1.047 | [0.892, 1.229] | 2 / 3 |
| zinnia_halo2 | noir_hw | prove_core_s | 5 | 0.602 | [0.391, 0.929] | 4 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_s | 5 | 0.582 | [0.373, 0.909] | 4 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 5 | 0.218 | [0.120, 0.396] | 5 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 5 | 0.392 | [0.317, 0.508] | 5 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 5 | 0.992 | [0.852, 1.160] | 2 / 3 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 5 | 0.996 | [0.855, 1.165] | 2 / 3 |
| zinnia_halo2 | halo2_lit | verify_s | 5 | 0.882 | [0.743, 1.040] | 3 / 2 |
| zinnia_halo2 | noir_lit | prove_core_s | 5 | 0.624 | [0.462, 0.848] | 5 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_s | 5 | 0.604 | [0.443, 0.833] | 5 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 5 | 0.220 | [0.130, 0.375] | 5 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 5 | 0.371 | [0.299, 0.471] | 5 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 5 | 1.529 | [1.017, 2.301] | 2 / 3 |
| zinnia_noir | halo2_hw | end_to_end_s | 5 | 1.585 | [1.049, 2.395] | 2 / 3 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 5 | 4.296 | [2.275, 8.082] | 0 / 5 |
| zinnia_noir | halo2_hw | verify_s | 5 | 2.608 | [1.808, 3.884] | 0 / 5 |
| zinnia_noir | halo2_lit | prove_core_s | 5 | 1.579 | [1.026, 2.430] | 1 / 4 |
| zinnia_noir | halo2_lit | end_to_end_s | 5 | 1.636 | [1.059, 2.527] | 1 / 4 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 5 | 4.433 | [2.296, 8.559] | 0 / 5 |
| zinnia_noir | halo2_lit | verify_s | 5 | 2.355 | [1.730, 3.411] | 0 / 5 |
| zinnia_noir | noir_lit | prove_core_s | 5 | 0.993 | [0.889, 1.083] | 2 / 3 |
| zinnia_noir | noir_lit | end_to_end_s | 5 | 0.992 | [0.896, 1.078] | 2 / 3 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 5 | 0.980 | [0.873, 1.075] | 2 / 3 |
| zinnia_noir | noir_lit | verify_s | 5 | 0.990 | [0.901, 1.088] | 3 / 2 |

## Copy-only cases (reported apart)

2 all-supported cases that only copy inputs: ds1000_case0392, ds1000_case0393

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 2 | 1.131 | [1.117, 1.146] | 0 / 2 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 2 | 1.137 | [1.123, 1.152] | 0 / 2 |
| zinnia_halo2 | halo2_hw | verify_s | 2 | 0.923 | [0.887, 0.960] | 2 / 0 |
| zinnia_noir | noir_hw | prove_core_s | 2 | 1.065 | [1.027, 1.104] | 0 / 2 |
| zinnia_noir | noir_hw | end_to_end_s | 2 | 1.058 | [1.021, 1.096] | 0 / 2 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 2 | 0.963 | [0.957, 0.970] | 2 / 0 |
| zinnia_noir | noir_hw | verify_s | 2 | 1.163 | [1.107, 1.222] | 0 / 2 |
| zinnia_halo2 | noir_hw | prove_core_s | 2 | 0.970 | [0.853, 1.102] | 1 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_s | 2 | 0.941 | [0.829, 1.067] | 1 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 2 | 0.418 | [0.365, 0.480] | 2 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 2 | 0.719 | [0.645, 0.802] | 2 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 2 | 1.019 | [1.014, 1.024] | 0 / 2 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 2 | 1.025 | [1.019, 1.031] | 0 / 2 |
| zinnia_halo2 | halo2_lit | verify_s | 2 | 0.856 | [0.805, 0.911] | 2 / 0 |
| zinnia_halo2 | noir_lit | prove_core_s | 2 | 0.988 | [0.891, 1.095] | 1 / 1 |
| zinnia_halo2 | noir_lit | end_to_end_s | 2 | 0.959 | [0.871, 1.055] | 1 / 1 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 2 | 0.425 | [0.365, 0.494] | 2 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 2 | 0.590 | [0.523, 0.666] | 2 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 2 | 1.242 | [1.147, 1.345] | 0 / 2 |
| zinnia_noir | halo2_hw | end_to_end_s | 2 | 1.279 | [1.183, 1.383] | 0 / 2 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 2 | 2.617 | [2.295, 2.986] | 0 / 2 |
| zinnia_noir | halo2_hw | verify_s | 2 | 1.492 | [1.351, 1.648] | 0 / 2 |
| zinnia_noir | halo2_lit | prove_core_s | 2 | 1.119 | [1.016, 1.234] | 0 / 2 |
| zinnia_noir | halo2_lit | end_to_end_s | 2 | 1.153 | [1.047, 1.269] | 0 / 2 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 2 | 2.359 | [2.031, 2.740] | 0 / 2 |
| zinnia_noir | halo2_lit | verify_s | 2 | 1.385 | [1.382, 1.387] | 0 / 2 |
| zinnia_noir | noir_lit | prove_core_s | 2 | 1.085 | [1.073, 1.096] | 0 / 2 |
| zinnia_noir | noir_lit | end_to_end_s | 2 | 1.078 | [1.073, 1.084] | 0 / 2 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 2 | 0.977 | [0.971, 0.984] | 2 / 0 |
| zinnia_noir | noir_lit | verify_s | 2 | 0.955 | [0.797, 1.143] | 1 / 1 |

## All-supported subset

7 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): ds1000_case0392, ds1000_case0393, ds1000_case0400, ds1000_case0401, ds1000_case0413, ds1000_case0417, ds1000_case0418

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 7 | 1.007 | [0.860, 1.130] | 2 / 5 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 7 | 1.012 | [0.864, 1.135] | 2 / 5 |
| zinnia_halo2 | halo2_hw | verify_s | 7 | 0.961 | [0.816, 1.127] | 4 / 3 |
| zinnia_noir | noir_hw | prove_core_s | 7 | 0.988 | [0.881, 1.092] | 3 / 4 |
| zinnia_noir | noir_hw | end_to_end_s | 7 | 0.984 | [0.880, 1.088] | 3 / 4 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 7 | 0.968 | [0.923, 1.019] | 5 / 2 |
| zinnia_noir | noir_hw | verify_s | 7 | 1.079 | [0.952, 1.216] | 2 / 5 |
| zinnia_halo2 | noir_hw | prove_core_s | 7 | 0.690 | [0.475, 0.966] | 5 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_s | 7 | 0.668 | [0.457, 0.937] | 5 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 7 | 0.263 | [0.157, 0.418] | 7 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 7 | 0.466 | [0.358, 0.609] | 7 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 7 | 1.000 | [0.897, 1.117] | 2 / 5 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 7 | 1.005 | [0.902, 1.122] | 2 / 5 |
| zinnia_halo2 | halo2_lit | verify_s | 7 | 0.875 | [0.769, 0.987] | 5 / 2 |
| zinnia_halo2 | noir_lit | prove_core_s | 7 | 0.712 | [0.533, 0.912] | 6 / 1 |
| zinnia_halo2 | noir_lit | end_to_end_s | 7 | 0.689 | [0.511, 0.891] | 6 / 1 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 7 | 0.266 | [0.167, 0.399] | 7 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 7 | 0.424 | [0.338, 0.532] | 7 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 7 | 1.441 | [1.100, 1.988] | 2 / 5 |
| zinnia_noir | halo2_hw | end_to_end_s | 7 | 1.491 | [1.134, 2.068] | 2 / 5 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 7 | 3.729 | [2.392, 6.220] | 0 / 7 |
| zinnia_noir | halo2_hw | verify_s | 7 | 2.224 | [1.634, 3.173] | 0 / 7 |
| zinnia_noir | halo2_lit | prove_core_s | 7 | 1.431 | [1.049, 2.051] | 1 / 6 |
| zinnia_noir | halo2_lit | end_to_end_s | 7 | 1.480 | [1.082, 2.125] | 1 / 6 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 7 | 3.702 | [2.290, 6.427] | 0 / 7 |
| zinnia_noir | halo2_lit | verify_s | 7 | 2.024 | [1.543, 2.863] | 0 / 7 |
| zinnia_noir | noir_lit | prove_core_s | 7 | 1.019 | [0.935, 1.086] | 2 / 5 |
| zinnia_noir | noir_lit | end_to_end_s | 7 | 1.016 | [0.939, 1.080] | 2 / 5 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 7 | 0.979 | [0.907, 1.049] | 4 / 3 |
| zinnia_noir | noir_lit | verify_s | 7 | 0.980 | [0.887, 1.080] | 4 / 3 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 7 | 1.007 | [0.860, 1.130] | 2 / 5 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 7 | 1.012 | [0.864, 1.135] | 2 / 5 |
| zinnia_halo2 | halo2_hw | verify_s | 7 | 0.961 | [0.816, 1.127] | 4 / 3 |
| zinnia_noir | noir_hw | prove_core_s | 7 | 0.988 | [0.881, 1.092] | 3 / 4 |
| zinnia_noir | noir_hw | end_to_end_s | 7 | 0.984 | [0.880, 1.088] | 3 / 4 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 7 | 0.968 | [0.923, 1.019] | 5 / 2 |
| zinnia_noir | noir_hw | verify_s | 7 | 1.079 | [0.952, 1.216] | 2 / 5 |
| zinnia_halo2 | noir_hw | prove_core_s | 7 | 0.690 | [0.475, 0.966] | 5 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_s | 7 | 0.668 | [0.457, 0.937] | 5 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 7 | 0.263 | [0.157, 0.418] | 7 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 7 | 0.466 | [0.358, 0.609] | 7 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 7 | 1.000 | [0.897, 1.117] | 2 / 5 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 7 | 1.005 | [0.902, 1.122] | 2 / 5 |
| zinnia_halo2 | halo2_lit | verify_s | 7 | 0.875 | [0.769, 0.987] | 5 / 2 |
| zinnia_halo2 | noir_lit | prove_core_s | 7 | 0.712 | [0.533, 0.912] | 6 / 1 |
| zinnia_halo2 | noir_lit | end_to_end_s | 7 | 0.689 | [0.511, 0.891] | 6 / 1 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 7 | 0.266 | [0.167, 0.399] | 7 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 7 | 0.424 | [0.338, 0.532] | 7 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 7 | 1.441 | [1.100, 1.988] | 2 / 5 |
| zinnia_noir | halo2_hw | end_to_end_s | 7 | 1.491 | [1.134, 2.068] | 2 / 5 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 7 | 3.729 | [2.392, 6.220] | 0 / 7 |
| zinnia_noir | halo2_hw | verify_s | 7 | 2.224 | [1.634, 3.173] | 0 / 7 |
| zinnia_noir | halo2_lit | prove_core_s | 7 | 1.431 | [1.049, 2.051] | 1 / 6 |
| zinnia_noir | halo2_lit | end_to_end_s | 7 | 1.480 | [1.082, 2.125] | 1 / 6 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 7 | 3.702 | [2.290, 6.427] | 0 / 7 |
| zinnia_noir | halo2_lit | verify_s | 7 | 2.024 | [1.543, 2.863] | 0 / 7 |
| zinnia_noir | noir_lit | prove_core_s | 7 | 1.019 | [0.935, 1.086] | 2 / 5 |
| zinnia_noir | noir_lit | end_to_end_s | 7 | 1.016 | [0.939, 1.080] | 2 / 5 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 7 | 0.979 | [0.907, 1.049] | 4 / 3 |
| zinnia_noir | noir_lit | verify_s | 7 | 0.980 | [0.887, 1.080] | 4 / 3 |

## Excluded cases

(none)

## Cases without a program

No program for the case on that system (e.g. no handwritten baseline yet); not a failure.

(none)

## Failures by system

