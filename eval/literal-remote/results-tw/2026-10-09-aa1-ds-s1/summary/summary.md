# Epoch 2026-10-09-aa1-ds-s1

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

2 cases completed by every system that do more than copy inputs: ds1000_case0356, ds1000_case0357

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 2 | 1.025 | [0.988, 1.064] | 1 / 1 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 2 | 1.031 | [0.996, 1.068] | 1 / 1 |
| zinnia_halo2 | halo2_hw | verify_s | 2 | 0.975 | [0.900, 1.057] | 1 / 1 |
| zinnia_noir | noir_hw | prove_core_s | 2 | 0.875 | [0.724, 1.058] | 1 / 1 |
| zinnia_noir | noir_hw | end_to_end_s | 2 | 0.884 | [0.737, 1.060] | 1 / 1 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 2 | 0.898 | [0.841, 0.958] | 2 / 0 |
| zinnia_noir | noir_hw | verify_s | 2 | 1.347 | [1.253, 1.448] | 0 / 2 |
| zinnia_halo2 | noir_hw | prove_core_s | 2 | 1.078 | [0.932, 1.246] | 1 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_s | 2 | 1.063 | [0.921, 1.228] | 1 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 2 | 0.529 | [0.486, 0.575] | 2 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 2 | 0.725 | [0.669, 0.787] | 2 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 2 | 0.816 | [0.767, 0.869] | 2 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 2 | 0.821 | [0.772, 0.873] | 2 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 2 | 0.981 | [0.893, 1.077] | 1 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 2 | 1.243 | [1.166, 1.325] | 0 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_s | 2 | 1.209 | [1.147, 1.274] | 0 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 2 | 0.574 | [0.523, 0.630] | 2 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 2 | 0.518 | [0.516, 0.521] | 2 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 2 | 0.833 | [0.768, 0.903] | 2 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 2 | 0.857 | [0.797, 0.922] | 2 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 2 | 1.752 | [1.725, 1.779] | 0 / 2 |
| zinnia_noir | halo2_hw | verify_s | 2 | 1.811 | [1.686, 1.945] | 0 / 2 |
| zinnia_noir | halo2_lit | prove_core_s | 2 | 0.663 | [0.596, 0.738] | 2 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 2 | 0.682 | [0.618, 0.754] | 2 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 2 | 1.394 | [1.337, 1.454] | 0 / 2 |
| zinnia_noir | halo2_lit | verify_s | 2 | 1.821 | [1.673, 1.982] | 0 / 2 |
| zinnia_noir | noir_lit | prove_core_s | 2 | 1.009 | [0.906, 1.125] | 1 / 1 |
| zinnia_noir | noir_lit | end_to_end_s | 2 | 1.005 | [0.918, 1.100] | 1 / 1 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 2 | 0.975 | [0.906, 1.049] | 1 / 1 |
| zinnia_noir | noir_lit | verify_s | 2 | 0.962 | [0.959, 0.966] | 2 / 0 |

## Copy-only cases (reported apart)

4 all-supported cases that only copy inputs: ds1000_case0339, ds1000_case0341, ds1000_case0342, ds1000_case0343

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 4 | 0.985 | [0.904, 1.073] | 2 / 2 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 4 | 0.989 | [0.908, 1.077] | 2 / 2 |
| zinnia_halo2 | halo2_hw | verify_s | 4 | 0.881 | [0.826, 0.981] | 3 / 1 |
| zinnia_noir | noir_hw | prove_core_s | 4 | 0.928 | [0.889, 0.968] | 4 / 0 |
| zinnia_noir | noir_hw | end_to_end_s | 4 | 0.924 | [0.884, 0.965] | 4 / 0 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 4 | 0.895 | [0.873, 0.918] | 4 / 0 |
| zinnia_noir | noir_hw | verify_s | 4 | 0.978 | [0.936, 1.031] | 3 / 1 |
| zinnia_halo2 | noir_hw | prove_core_s | 4 | 0.713 | [0.658, 0.772] | 4 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_s | 4 | 0.696 | [0.640, 0.754] | 4 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 4 | 0.326 | [0.308, 0.351] | 4 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 4 | 0.376 | [0.371, 0.385] | 4 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 4 | 1.041 | [0.912, 1.187] | 2 / 2 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 4 | 1.045 | [0.916, 1.192] | 2 / 2 |
| zinnia_halo2 | halo2_lit | verify_s | 4 | 0.972 | [0.867, 1.102] | 3 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 4 | 0.765 | [0.733, 0.786] | 4 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_s | 4 | 0.746 | [0.712, 0.767] | 4 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 4 | 0.338 | [0.313, 0.356] | 4 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 4 | 0.475 | [0.345, 0.595] | 4 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 4 | 1.281 | [1.193, 1.345] | 0 / 4 |
| zinnia_noir | halo2_hw | end_to_end_s | 4 | 1.313 | [1.232, 1.378] | 0 / 4 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 4 | 2.718 | [2.595, 2.847] | 0 / 4 |
| zinnia_noir | halo2_hw | verify_s | 4 | 2.294 | [2.101, 2.492] | 0 / 4 |
| zinnia_noir | halo2_lit | prove_core_s | 4 | 1.354 | [1.205, 1.540] | 0 / 4 |
| zinnia_noir | halo2_lit | end_to_end_s | 4 | 1.387 | [1.239, 1.584] | 0 / 4 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 4 | 2.871 | [2.400, 3.530] | 0 / 4 |
| zinnia_noir | halo2_lit | verify_s | 4 | 2.532 | [2.308, 2.769] | 0 / 4 |
| zinnia_noir | noir_lit | prove_core_s | 4 | 0.996 | [0.932, 1.087] | 3 / 1 |
| zinnia_noir | noir_lit | end_to_end_s | 4 | 0.991 | [0.929, 1.067] | 3 / 1 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 4 | 0.929 | [0.887, 0.967] | 4 / 0 |
| zinnia_noir | noir_lit | verify_s | 4 | 1.237 | [0.963, 1.525] | 1 / 3 |

## All-supported subset

6 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): ds1000_case0339, ds1000_case0341, ds1000_case0342, ds1000_case0343, ds1000_case0356, ds1000_case0357

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 6 | 0.998 | [0.942, 1.057] | 3 / 3 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 6 | 1.003 | [0.947, 1.062] | 3 / 3 |
| zinnia_halo2 | halo2_hw | verify_s | 6 | 0.911 | [0.842, 0.989] | 4 / 2 |
| zinnia_noir | noir_hw | prove_core_s | 6 | 0.910 | [0.821, 0.990] | 5 / 1 |
| zinnia_noir | noir_hw | end_to_end_s | 6 | 0.910 | [0.829, 0.987] | 5 / 1 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 6 | 0.896 | [0.866, 0.927] | 6 / 0 |
| zinnia_noir | noir_hw | verify_s | 6 | 1.088 | [0.965, 1.252] | 3 / 3 |
| zinnia_halo2 | noir_hw | prove_core_s | 6 | 0.818 | [0.694, 0.992] | 5 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_s | 6 | 0.802 | [0.678, 0.977] | 5 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 6 | 0.383 | [0.321, 0.467] | 6 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 6 | 0.468 | [0.374, 0.601] | 6 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 6 | 0.960 | [0.848, 1.104] | 4 / 2 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 6 | 0.964 | [0.852, 1.109] | 4 / 2 |
| zinnia_halo2 | halo2_lit | verify_s | 6 | 0.975 | [0.894, 1.070] | 4 / 2 |
| zinnia_halo2 | noir_lit | prove_core_s | 6 | 0.900 | [0.760, 1.089] | 4 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_s | 6 | 0.877 | [0.740, 1.056] | 4 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 6 | 0.403 | [0.333, 0.502] | 6 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 6 | 0.489 | [0.391, 0.571] | 6 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 6 | 1.110 | [0.925, 1.302] | 2 / 4 |
| zinnia_noir | halo2_hw | end_to_end_s | 6 | 1.139 | [0.956, 1.334] | 2 / 4 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 6 | 2.348 | [1.993, 2.744] | 0 / 6 |
| zinnia_noir | halo2_hw | verify_s | 6 | 2.120 | [1.898, 2.359] | 0 / 6 |
| zinnia_noir | halo2_lit | prove_core_s | 6 | 1.067 | [0.793, 1.393] | 2 / 4 |
| zinnia_noir | halo2_lit | end_to_end_s | 6 | 1.095 | [0.817, 1.429] | 2 / 4 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 6 | 2.256 | [1.670, 3.047] | 0 / 6 |
| zinnia_noir | halo2_lit | verify_s | 6 | 2.268 | [1.950, 2.609] | 0 / 6 |
| zinnia_noir | noir_lit | prove_core_s | 6 | 1.000 | [0.933, 1.075] | 4 / 2 |
| zinnia_noir | noir_lit | end_to_end_s | 6 | 0.995 | [0.937, 1.058] | 4 / 2 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 6 | 0.944 | [0.901, 0.992] | 5 / 1 |
| zinnia_noir | noir_lit | verify_s | 6 | 1.137 | [0.951, 1.371] | 3 / 3 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 7 | 1.001 | [0.948, 1.052] | 3 / 4 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 7 | 1.006 | [0.953, 1.057] | 3 / 4 |
| zinnia_halo2 | halo2_hw | verify_s | 7 | 0.927 | [0.861, 1.002] | 4 / 3 |
| zinnia_noir | noir_hw | prove_core_s | 6 | 0.910 | [0.821, 0.990] | 5 / 1 |
| zinnia_noir | noir_hw | end_to_end_s | 6 | 0.910 | [0.829, 0.987] | 5 / 1 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 6 | 0.896 | [0.866, 0.927] | 6 / 0 |
| zinnia_noir | noir_hw | verify_s | 6 | 1.088 | [0.965, 1.252] | 3 / 3 |
| zinnia_halo2 | noir_hw | prove_core_s | 6 | 0.818 | [0.694, 0.992] | 5 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_s | 6 | 0.802 | [0.678, 0.977] | 5 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 6 | 0.383 | [0.321, 0.467] | 6 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 6 | 0.468 | [0.374, 0.601] | 6 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 7 | 0.972 | [0.868, 1.100] | 4 / 3 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 7 | 0.976 | [0.872, 1.105] | 4 / 3 |
| zinnia_halo2 | halo2_lit | verify_s | 7 | 0.993 | [0.914, 1.078] | 4 / 3 |
| zinnia_halo2 | noir_lit | prove_core_s | 6 | 0.900 | [0.760, 1.089] | 4 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_s | 6 | 0.877 | [0.740, 1.056] | 4 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 6 | 0.403 | [0.333, 0.502] | 6 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 6 | 0.489 | [0.391, 0.571] | 6 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 7 | 1.281 | [0.979, 1.760] | 2 / 5 |
| zinnia_noir | halo2_hw | end_to_end_s | 7 | 1.317 | [1.007, 1.816] | 2 / 5 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 7 | 2.945 | [2.093, 4.763] | 0 / 7 |
| zinnia_noir | halo2_hw | verify_s | 7 | 2.216 | [1.956, 2.502] | 0 / 7 |
| zinnia_noir | halo2_lit | prove_core_s | 7 | 1.244 | [0.867, 1.806] | 2 / 5 |
| zinnia_noir | halo2_lit | end_to_end_s | 7 | 1.278 | [0.890, 1.864] | 2 / 5 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 7 | 2.858 | [1.816, 4.866] | 0 / 7 |
| zinnia_noir | halo2_lit | verify_s | 7 | 2.372 | [2.030, 2.743] | 0 / 7 |
| zinnia_noir | noir_lit | prove_core_s | 6 | 1.000 | [0.933, 1.075] | 4 / 2 |
| zinnia_noir | noir_lit | end_to_end_s | 6 | 0.995 | [0.937, 1.058] | 4 / 2 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 6 | 0.944 | [0.901, 0.992] | 5 / 1 |
| zinnia_noir | noir_lit | verify_s | 6 | 1.137 | [0.951, 1.371] | 3 / 3 |

## Excluded cases

(none)

## Cases without a program

No program for the case on that system (e.g. no handwritten baseline yet); not a failure.

(none)

## Failures by system

- noir_hw ds1000_case0349: prove_failed at warmup: /home/zhantong/.bb/bb prove exited 1: 
Scheme is: ultra_honk, num threads: 8 (mem: 32.09 MiB)
Assertion failed: (witness.size() == constraints.max_witness_index
- noir_lit ds1000_case0349: prove_failed at warmup: /home/zhantong/.bb/bb prove exited 1: 
Scheme is: ultra_honk, num threads: 8 (mem: 32.06 MiB)
Assertion failed: (witness.size() == constraints.max_witness_index
