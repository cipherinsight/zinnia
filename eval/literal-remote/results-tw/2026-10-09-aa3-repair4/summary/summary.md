# Epoch 2026-10-09-aa3-repair4

Ratios are Zinnia / baseline of per-case medians; below 1 means Zinnia is faster.
Only cases with status `ok` (proved, verified, outputs equal to NumPy, tampered proof rejected) enter a comparison.

`zinnia_noir` `end_to_end_s` witness step: noir-execute (4 cases); `end_to_end_nargo_s` uses `nargo execute`.
`noir_hw` `end_to_end_s` witness step: noir-execute (4 cases); `end_to_end_nargo_s` uses `nargo execute`.
`noir_lit` `end_to_end_s` witness step: noir-execute (4 cases); `end_to_end_nargo_s` uses `nargo execute`.

## Coverage

| system | cases | ok | other statuses |
|---|---|---|---|
| zinnia_halo2 | 4 | 4 | - |
| zinnia_noir | 4 | 4 | - |
| halo2_hw | 4 | 4 | - |
| noir_hw | 4 | 4 | - |
| halo2_lit | 4 | 4 | - |
| noir_lit | 4 | 4 | - |

## Headline: all-supported subset without copy-only cases

2 cases completed by every system that do more than copy inputs: ds1000_case0291, ds1000_case0349

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 2 | 1.101 | [1.013, 1.198] | 0 / 2 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 2 | 1.106 | [1.017, 1.202] | 0 / 2 |
| zinnia_halo2 | halo2_hw | verify_s | 2 | 1.067 | [1.019, 1.116] | 0 / 2 |
| zinnia_noir | noir_hw | prove_core_s | 2 | 0.952 | [0.894, 1.015] | 1 / 1 |
| zinnia_noir | noir_hw | end_to_end_s | 2 | 0.951 | [0.871, 1.040] | 1 / 1 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 2 | 0.924 | [0.922, 0.926] | 2 / 0 |
| zinnia_noir | noir_hw | verify_s | 2 | 0.935 | [0.886, 0.986] | 2 / 0 |
| zinnia_halo2 | noir_hw | prove_core_s | 2 | 0.376 | [0.321, 0.441] | 2 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_s | 2 | 0.358 | [0.301, 0.426] | 2 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 2 | 0.094 | [0.083, 0.107] | 2 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 2 | 0.254 | [0.235, 0.275] | 2 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 2 | 0.900 | [0.738, 1.097] | 1 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 2 | 0.904 | [0.742, 1.101] | 1 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 2 | 1.020 | [1.016, 1.024] | 0 / 2 |
| zinnia_halo2 | noir_lit | prove_core_s | 2 | 0.357 | [0.318, 0.402] | 2 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_s | 2 | 0.340 | [0.306, 0.379] | 2 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 2 | 0.094 | [0.085, 0.103] | 2 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 2 | 0.247 | [0.239, 0.255] | 2 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 2 | 2.788 | [2.753, 2.823] | 0 / 2 |
| zinnia_noir | halo2_hw | end_to_end_s | 2 | 2.937 | [2.931, 2.942] | 0 / 2 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 2 | 10.827 | [10.362, 11.313] | 0 / 2 |
| zinnia_noir | halo2_hw | verify_s | 2 | 3.928 | [3.851, 4.006] | 0 / 2 |
| zinnia_noir | halo2_lit | prove_core_s | 2 | 2.279 | [2.058, 2.523] | 0 / 2 |
| zinnia_noir | halo2_lit | end_to_end_s | 2 | 2.400 | [2.144, 2.687] | 0 / 2 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 2 | 8.849 | [8.245, 9.497] | 0 / 2 |
| zinnia_noir | halo2_lit | verify_s | 2 | 3.756 | [3.674, 3.840] | 0 / 2 |
| zinnia_noir | noir_lit | prove_core_s | 2 | 0.904 | [0.885, 0.924] | 2 / 0 |
| zinnia_noir | noir_lit | end_to_end_s | 2 | 0.904 | [0.885, 0.925] | 2 / 0 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 2 | 0.921 | [0.892, 0.950] | 2 / 0 |
| zinnia_noir | noir_lit | verify_s | 2 | 0.911 | [0.860, 0.964] | 2 / 0 |

## Copy-only cases (reported apart)

2 all-supported cases that only copy inputs: ds1000_case0467, ds1000_case0468

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 2 | 0.934 | [0.886, 0.984] | 2 / 0 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 2 | 0.937 | [0.890, 0.988] | 2 / 0 |
| zinnia_halo2 | halo2_hw | verify_s | 2 | 0.950 | [0.894, 1.010] | 1 / 1 |
| zinnia_noir | noir_hw | prove_core_s | 2 | 1.002 | [0.925, 1.084] | 1 / 1 |
| zinnia_noir | noir_hw | end_to_end_s | 2 | 0.999 | [0.922, 1.082] | 1 / 1 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 2 | 0.974 | [0.914, 1.037] | 1 / 1 |
| zinnia_noir | noir_hw | verify_s | 2 | 1.042 | [0.976, 1.112] | 1 / 1 |
| zinnia_halo2 | noir_hw | prove_core_s | 2 | 0.557 | [0.519, 0.597] | 2 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_s | 2 | 0.543 | [0.506, 0.582] | 2 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 2 | 0.248 | [0.232, 0.265] | 2 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 2 | 0.412 | [0.382, 0.445] | 2 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 2 | 1.009 | [0.892, 1.143] | 1 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 2 | 1.013 | [0.895, 1.147] | 1 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 2 | 0.943 | [0.910, 0.977] | 2 / 0 |
| zinnia_halo2 | noir_lit | prove_core_s | 2 | 0.531 | [0.498, 0.565] | 2 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_s | 2 | 0.515 | [0.477, 0.555] | 2 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 2 | 0.243 | [0.232, 0.254] | 2 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 2 | 0.436 | [0.356, 0.534] | 2 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 2 | 1.680 | [1.579, 1.786] | 0 / 2 |
| zinnia_noir | halo2_hw | end_to_end_s | 2 | 1.725 | [1.620, 1.836] | 0 / 2 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 2 | 3.682 | [3.500, 3.873] | 0 / 2 |
| zinnia_noir | halo2_hw | verify_s | 2 | 2.401 | [2.286, 2.522] | 0 / 2 |
| zinnia_noir | halo2_lit | prove_core_s | 2 | 1.816 | [1.590, 2.075] | 0 / 2 |
| zinnia_noir | halo2_lit | end_to_end_s | 2 | 1.865 | [1.631, 2.132] | 0 / 2 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 2 | 3.980 | [3.523, 4.497] | 0 / 2 |
| zinnia_noir | halo2_lit | verify_s | 2 | 2.382 | [2.272, 2.498] | 0 / 2 |
| zinnia_noir | noir_lit | prove_core_s | 2 | 0.955 | [0.889, 1.026] | 1 / 1 |
| zinnia_noir | noir_lit | end_to_end_s | 2 | 0.947 | [0.869, 1.032] | 1 / 1 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 2 | 0.954 | [0.913, 0.998] | 2 / 0 |
| zinnia_noir | noir_lit | verify_s | 2 | 1.101 | [0.910, 1.333] | 1 / 1 |

## All-supported subset

4 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): ds1000_case0291, ds1000_case0349, ds1000_case0467, ds1000_case0468

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 4 | 1.014 | [0.916, 1.140] | 2 / 2 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 4 | 1.018 | [0.920, 1.144] | 2 / 2 |
| zinnia_halo2 | halo2_hw | verify_s | 4 | 1.007 | [0.924, 1.089] | 1 / 3 |
| zinnia_noir | noir_hw | prove_core_s | 4 | 0.977 | [0.910, 1.049] | 2 / 2 |
| zinnia_noir | noir_hw | end_to_end_s | 4 | 0.975 | [0.896, 1.061] | 2 / 2 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 4 | 0.948 | [0.917, 1.007] | 3 / 1 |
| zinnia_noir | noir_hw | verify_s | 4 | 0.987 | [0.910, 1.076] | 3 / 1 |
| zinnia_halo2 | noir_hw | prove_core_s | 4 | 0.458 | [0.362, 0.557] | 4 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_s | 4 | 0.441 | [0.343, 0.543] | 4 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 4 | 0.153 | [0.094, 0.248] | 4 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 4 | 0.323 | [0.254, 0.412] | 4 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 4 | 0.953 | [0.811, 1.120] | 2 / 2 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 4 | 0.957 | [0.815, 1.124] | 2 / 2 |
| zinnia_halo2 | halo2_lit | verify_s | 4 | 0.981 | [0.936, 1.020] | 2 / 2 |
| zinnia_halo2 | noir_lit | prove_core_s | 4 | 0.435 | [0.355, 0.531] | 4 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_s | 4 | 0.419 | [0.340, 0.515] | 4 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 4 | 0.151 | [0.094, 0.243] | 4 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 4 | 0.328 | [0.247, 0.444] | 4 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 4 | 2.164 | [1.680, 2.788] | 0 / 4 |
| zinnia_noir | halo2_hw | end_to_end_s | 4 | 2.251 | [1.725, 2.937] | 0 / 4 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 4 | 6.314 | [3.682, 10.827] | 0 / 4 |
| zinnia_noir | halo2_hw | verify_s | 4 | 3.071 | [2.401, 3.928] | 0 / 4 |
| zinnia_noir | halo2_lit | prove_core_s | 4 | 2.034 | [1.699, 2.398] | 0 / 4 |
| zinnia_noir | halo2_lit | end_to_end_s | 4 | 2.116 | [1.746, 2.536] | 0 / 4 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 4 | 5.935 | [3.980, 8.849] | 0 / 4 |
| zinnia_noir | halo2_lit | verify_s | 4 | 2.991 | [2.382, 3.756] | 0 / 4 |
| zinnia_noir | noir_lit | prove_core_s | 4 | 0.929 | [0.887, 0.990] | 3 / 1 |
| zinnia_noir | noir_lit | end_to_end_s | 4 | 0.925 | [0.877, 0.993] | 3 / 1 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 4 | 0.937 | [0.902, 0.976] | 4 / 0 |
| zinnia_noir | noir_lit | verify_s | 4 | 1.001 | [0.884, 1.212] | 3 / 1 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 4 | 1.014 | [0.916, 1.140] | 2 / 2 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 4 | 1.018 | [0.920, 1.144] | 2 / 2 |
| zinnia_halo2 | halo2_hw | verify_s | 4 | 1.007 | [0.924, 1.089] | 1 / 3 |
| zinnia_noir | noir_hw | prove_core_s | 4 | 0.977 | [0.910, 1.049] | 2 / 2 |
| zinnia_noir | noir_hw | end_to_end_s | 4 | 0.975 | [0.896, 1.061] | 2 / 2 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 4 | 0.948 | [0.917, 1.007] | 3 / 1 |
| zinnia_noir | noir_hw | verify_s | 4 | 0.987 | [0.910, 1.076] | 3 / 1 |
| zinnia_halo2 | noir_hw | prove_core_s | 4 | 0.458 | [0.362, 0.557] | 4 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_s | 4 | 0.441 | [0.343, 0.543] | 4 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 4 | 0.153 | [0.094, 0.248] | 4 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 4 | 0.323 | [0.254, 0.412] | 4 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 4 | 0.953 | [0.811, 1.120] | 2 / 2 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 4 | 0.957 | [0.815, 1.124] | 2 / 2 |
| zinnia_halo2 | halo2_lit | verify_s | 4 | 0.981 | [0.936, 1.020] | 2 / 2 |
| zinnia_halo2 | noir_lit | prove_core_s | 4 | 0.435 | [0.355, 0.531] | 4 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_s | 4 | 0.419 | [0.340, 0.515] | 4 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 4 | 0.151 | [0.094, 0.243] | 4 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 4 | 0.328 | [0.247, 0.444] | 4 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 4 | 2.164 | [1.680, 2.788] | 0 / 4 |
| zinnia_noir | halo2_hw | end_to_end_s | 4 | 2.251 | [1.725, 2.937] | 0 / 4 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 4 | 6.314 | [3.682, 10.827] | 0 / 4 |
| zinnia_noir | halo2_hw | verify_s | 4 | 3.071 | [2.401, 3.928] | 0 / 4 |
| zinnia_noir | halo2_lit | prove_core_s | 4 | 2.034 | [1.699, 2.398] | 0 / 4 |
| zinnia_noir | halo2_lit | end_to_end_s | 4 | 2.116 | [1.746, 2.536] | 0 / 4 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 4 | 5.935 | [3.980, 8.849] | 0 / 4 |
| zinnia_noir | halo2_lit | verify_s | 4 | 2.991 | [2.382, 3.756] | 0 / 4 |
| zinnia_noir | noir_lit | prove_core_s | 4 | 0.929 | [0.887, 0.990] | 3 / 1 |
| zinnia_noir | noir_lit | end_to_end_s | 4 | 0.925 | [0.877, 0.993] | 3 / 1 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 4 | 0.937 | [0.902, 0.976] | 4 / 0 |
| zinnia_noir | noir_lit | verify_s | 4 | 1.001 | [0.884, 1.212] | 3 / 1 |

## Excluded cases

(none)

## Cases without a program

No program for the case on that system (e.g. no handwritten baseline yet); not a failure.

(none)

## Failures by system

