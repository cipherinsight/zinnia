# Epoch 2026-10-11-fz-ds-s2

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

34 cases completed by every system that do more than copy inputs: ds1000_case0300, ds1000_case0310, ds1000_case0313, ds1000_case0318, ds1000_case0321, ds1000_case0324, ds1000_case0329, ds1000_case0335, ds1000_case0345, ds1000_case0349, ds1000_case0356, ds1000_case0363, ds1000_case0368, ds1000_case0373, ds1000_case0381, ds1000_case0400, ds1000_case0403, ds1000_case0413, ds1000_case0416, ds1000_case0419, ds1000_case0428, ds1000_case0431, ds1000_case0438, ds1000_case0441, ds1000_case0444, ds1000_case0450, ds1000_case0453, ds1000_case0457, ds1000_case0463, ds1000_case0478, ds1000_case0485, ds1000_case0503, ds1000_case0507, ds1000_case0510

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 34 | 0.862 | [0.749, 0.960] | 23 / 11 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 34 | 0.870 | [0.756, 0.969] | 23 / 11 |
| zinnia_halo2 | halo2_hw | verify_s | 34 | 1.004 | [0.910, 1.096] | 16 / 18 |
| zinnia_noir | noir_hw | prove_core_s | 34 | 0.947 | [0.854, 1.021] | 23 / 11 |
| zinnia_noir | noir_hw | end_to_end_s | 34 | 0.953 | [0.861, 1.026] | 24 / 10 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 34 | 0.959 | [0.911, 0.999] | 22 / 12 |
| zinnia_noir | noir_hw | verify_s | 34 | 1.036 | [0.958, 1.118] | 17 / 17 |
| zinnia_halo2 | noir_hw | prove_core_s | 34 | 0.865 | [0.695, 1.056] | 16 / 18 |
| zinnia_halo2 | noir_hw | end_to_end_s | 34 | 0.816 | [0.643, 1.011] | 17 / 17 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 34 | 0.387 | [0.294, 0.499] | 32 / 2 |
| zinnia_halo2 | noir_hw | verify_s | 34 | 0.642 | [0.542, 0.753] | 27 / 7 |
| zinnia_halo2 | halo2_lit | prove_core_s | 34 | 0.531 | [0.369, 0.705] | 28 / 6 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 34 | 0.535 | [0.372, 0.711] | 28 / 6 |
| zinnia_halo2 | halo2_lit | verify_s | 34 | 0.743 | [0.623, 0.856] | 28 / 6 |
| zinnia_halo2 | noir_lit | prove_core_s | 34 | 0.758 | [0.583, 0.945] | 23 / 11 |
| zinnia_halo2 | noir_lit | end_to_end_s | 34 | 0.706 | [0.531, 0.897] | 24 / 10 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 34 | 0.351 | [0.265, 0.453] | 32 / 2 |
| zinnia_halo2 | noir_lit | verify_s | 34 | 0.611 | [0.521, 0.708] | 31 / 3 |
| zinnia_noir | halo2_hw | prove_core_s | 34 | 0.943 | [0.766, 1.156] | 18 / 16 |
| zinnia_noir | halo2_hw | end_to_end_s | 34 | 1.015 | [0.812, 1.283] | 17 / 17 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 34 | 2.154 | [1.686, 2.796] | 3 / 31 |
| zinnia_noir | halo2_hw | verify_s | 34 | 1.622 | [1.420, 1.863] | 5 / 29 |
| zinnia_noir | halo2_lit | prove_core_s | 34 | 0.581 | [0.391, 0.825] | 25 / 9 |
| zinnia_noir | halo2_lit | end_to_end_s | 34 | 0.625 | [0.415, 0.912] | 25 / 9 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 34 | 1.326 | [0.889, 1.958] | 11 / 23 |
| zinnia_noir | halo2_lit | verify_s | 34 | 1.200 | [0.988, 1.457] | 11 / 23 |
| zinnia_noir | noir_lit | prove_core_s | 34 | 0.830 | [0.670, 0.975] | 19 / 15 |
| zinnia_noir | noir_lit | end_to_end_s | 34 | 0.824 | [0.660, 0.973] | 20 / 14 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 34 | 0.869 | [0.758, 0.961] | 21 / 13 |
| zinnia_noir | noir_lit | verify_s | 34 | 0.986 | [0.926, 1.046] | 17 / 17 |

## Copy-only cases (reported apart)

10 all-supported cases that only copy inputs: ds1000_case0303, ds1000_case0338, ds1000_case0341, ds1000_case0359, ds1000_case0385, ds1000_case0389, ds1000_case0393, ds1000_case0435, ds1000_case0471, ds1000_case0500

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 10 | 0.978 | [0.911, 1.052] | 6 / 4 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 10 | 0.991 | [0.921, 1.066] | 5 / 5 |
| zinnia_halo2 | halo2_hw | verify_s | 10 | 1.067 | [0.962, 1.201] | 5 / 5 |
| zinnia_noir | noir_hw | prove_core_s | 10 | 0.994 | [0.947, 1.041] | 4 / 6 |
| zinnia_noir | noir_hw | end_to_end_s | 10 | 0.989 | [0.936, 1.041] | 5 / 5 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 10 | 1.010 | [0.932, 1.128] | 7 / 3 |
| zinnia_noir | noir_hw | verify_s | 10 | 0.984 | [0.850, 1.129] | 4 / 6 |
| zinnia_halo2 | noir_hw | prove_core_s | 10 | 0.946 | [0.612, 1.563] | 7 / 3 |
| zinnia_halo2 | noir_hw | end_to_end_s | 10 | 0.901 | [0.597, 1.442] | 7 / 3 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 10 | 0.442 | [0.259, 0.827] | 8 / 2 |
| zinnia_halo2 | noir_hw | verify_s | 10 | 0.670 | [0.518, 0.864] | 8 / 2 |
| zinnia_halo2 | halo2_lit | prove_core_s | 10 | 1.012 | [0.954, 1.067] | 4 / 6 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 10 | 1.025 | [0.966, 1.080] | 3 / 7 |
| zinnia_halo2 | halo2_lit | verify_s | 10 | 1.008 | [0.902, 1.152] | 5 / 5 |
| zinnia_halo2 | noir_lit | prove_core_s | 10 | 1.000 | [0.649, 1.613] | 5 / 5 |
| zinnia_halo2 | noir_lit | end_to_end_s | 10 | 0.953 | [0.631, 1.496] | 6 / 4 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 10 | 0.447 | [0.262, 0.838] | 8 / 2 |
| zinnia_halo2 | noir_lit | verify_s | 10 | 0.670 | [0.543, 0.840] | 8 / 2 |
| zinnia_noir | halo2_hw | prove_core_s | 10 | 1.027 | [0.638, 1.574] | 4 / 6 |
| zinnia_noir | halo2_hw | end_to_end_s | 10 | 1.088 | [0.704, 1.621] | 4 / 6 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 10 | 2.263 | [1.365, 3.525] | 2 / 8 |
| zinnia_noir | halo2_hw | verify_s | 10 | 1.567 | [1.118, 2.097] | 2 / 8 |
| zinnia_noir | halo2_lit | prove_core_s | 10 | 1.063 | [0.648, 1.652] | 4 / 6 |
| zinnia_noir | halo2_lit | end_to_end_s | 10 | 1.125 | [0.717, 1.700] | 3 / 7 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 10 | 2.342 | [1.385, 3.723] | 2 / 8 |
| zinnia_noir | halo2_lit | verify_s | 10 | 1.479 | [1.056, 1.995] | 2 / 8 |
| zinnia_noir | noir_lit | prove_core_s | 10 | 1.050 | [0.990, 1.115] | 3 / 7 |
| zinnia_noir | noir_lit | end_to_end_s | 10 | 1.046 | [0.991, 1.104] | 3 / 7 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 10 | 1.021 | [0.944, 1.145] | 8 / 2 |
| zinnia_noir | noir_lit | verify_s | 10 | 0.983 | [0.897, 1.065] | 4 / 6 |

## All-supported subset

44 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): ds1000_case0300, ds1000_case0303, ds1000_case0310, ds1000_case0313, ds1000_case0318, ds1000_case0321, ds1000_case0324, ds1000_case0329, ds1000_case0335, ds1000_case0338, ds1000_case0341, ds1000_case0345, ds1000_case0349, ds1000_case0356, ds1000_case0359, ds1000_case0363, ds1000_case0368, ds1000_case0373, ds1000_case0381, ds1000_case0385, ds1000_case0389, ds1000_case0393, ds1000_case0400, ds1000_case0403, ds1000_case0413, ds1000_case0416, ds1000_case0419, ds1000_case0428, ds1000_case0431, ds1000_case0435, ds1000_case0438, ds1000_case0441, ds1000_case0444, ds1000_case0450, ds1000_case0453, ds1000_case0457, ds1000_case0463, ds1000_case0471, ds1000_case0478, ds1000_case0485, ds1000_case0500, ds1000_case0503, ds1000_case0507, ds1000_case0510

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 44 | 0.887 | [0.792, 0.967] | 29 / 15 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 44 | 0.896 | [0.799, 0.977] | 28 / 16 |
| zinnia_halo2 | halo2_hw | verify_s | 44 | 1.018 | [0.937, 1.095] | 21 / 23 |
| zinnia_noir | noir_hw | prove_core_s | 44 | 0.957 | [0.881, 1.016] | 27 / 17 |
| zinnia_noir | noir_hw | end_to_end_s | 44 | 0.961 | [0.885, 1.019] | 29 / 15 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 44 | 0.970 | [0.926, 1.011] | 29 / 15 |
| zinnia_noir | noir_hw | verify_s | 44 | 1.024 | [0.956, 1.096] | 21 / 23 |
| zinnia_halo2 | noir_hw | prove_core_s | 44 | 0.883 | [0.726, 1.070] | 23 / 21 |
| zinnia_halo2 | noir_hw | end_to_end_s | 44 | 0.834 | [0.682, 1.018] | 24 / 20 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 44 | 0.399 | [0.314, 0.508] | 40 / 4 |
| zinnia_halo2 | noir_hw | verify_s | 44 | 0.648 | [0.562, 0.743] | 35 / 9 |
| zinnia_halo2 | halo2_lit | prove_core_s | 44 | 0.614 | [0.455, 0.774] | 32 / 12 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 44 | 0.620 | [0.459, 0.783] | 31 / 13 |
| zinnia_halo2 | halo2_lit | verify_s | 44 | 0.797 | [0.689, 0.896] | 33 / 11 |
| zinnia_halo2 | noir_lit | prove_core_s | 44 | 0.807 | [0.640, 0.994] | 28 / 16 |
| zinnia_halo2 | noir_lit | end_to_end_s | 44 | 0.756 | [0.591, 0.940] | 30 / 14 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 44 | 0.371 | [0.288, 0.475] | 40 / 4 |
| zinnia_halo2 | noir_lit | verify_s | 44 | 0.624 | [0.547, 0.706] | 39 / 5 |
| zinnia_noir | halo2_hw | prove_core_s | 44 | 0.962 | [0.794, 1.156] | 22 / 22 |
| zinnia_noir | halo2_hw | end_to_end_s | 44 | 1.031 | [0.844, 1.257] | 21 / 23 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 44 | 2.179 | [1.741, 2.724] | 5 / 39 |
| zinnia_noir | halo2_hw | verify_s | 44 | 1.609 | [1.417, 1.821] | 7 / 37 |
| zinnia_noir | halo2_lit | prove_core_s | 44 | 0.666 | [0.478, 0.900] | 29 / 15 |
| zinnia_noir | halo2_lit | end_to_end_s | 44 | 0.714 | [0.509, 0.977] | 28 / 16 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 44 | 1.509 | [1.078, 2.083] | 13 / 31 |
| zinnia_noir | halo2_lit | verify_s | 44 | 1.259 | [1.063, 1.486] | 13 / 31 |
| zinnia_noir | noir_lit | prove_core_s | 44 | 0.875 | [0.737, 0.994] | 22 / 22 |
| zinnia_noir | noir_lit | end_to_end_s | 44 | 0.870 | [0.728, 0.992] | 23 / 21 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 44 | 0.901 | [0.806, 0.980] | 29 / 15 |
| zinnia_noir | noir_lit | verify_s | 44 | 0.986 | [0.934, 1.036] | 21 / 23 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 44 | 0.887 | [0.792, 0.967] | 29 / 15 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 44 | 0.896 | [0.799, 0.977] | 28 / 16 |
| zinnia_halo2 | halo2_hw | verify_s | 44 | 1.018 | [0.937, 1.095] | 21 / 23 |
| zinnia_noir | noir_hw | prove_core_s | 44 | 0.957 | [0.881, 1.016] | 27 / 17 |
| zinnia_noir | noir_hw | end_to_end_s | 44 | 0.961 | [0.885, 1.019] | 29 / 15 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 44 | 0.970 | [0.926, 1.011] | 29 / 15 |
| zinnia_noir | noir_hw | verify_s | 44 | 1.024 | [0.956, 1.096] | 21 / 23 |
| zinnia_halo2 | noir_hw | prove_core_s | 44 | 0.883 | [0.726, 1.070] | 23 / 21 |
| zinnia_halo2 | noir_hw | end_to_end_s | 44 | 0.834 | [0.682, 1.018] | 24 / 20 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 44 | 0.399 | [0.314, 0.508] | 40 / 4 |
| zinnia_halo2 | noir_hw | verify_s | 44 | 0.648 | [0.562, 0.743] | 35 / 9 |
| zinnia_halo2 | halo2_lit | prove_core_s | 44 | 0.614 | [0.455, 0.774] | 32 / 12 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 44 | 0.620 | [0.459, 0.783] | 31 / 13 |
| zinnia_halo2 | halo2_lit | verify_s | 44 | 0.797 | [0.689, 0.896] | 33 / 11 |
| zinnia_halo2 | noir_lit | prove_core_s | 44 | 0.807 | [0.640, 0.994] | 28 / 16 |
| zinnia_halo2 | noir_lit | end_to_end_s | 44 | 0.756 | [0.591, 0.940] | 30 / 14 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 44 | 0.371 | [0.288, 0.475] | 40 / 4 |
| zinnia_halo2 | noir_lit | verify_s | 44 | 0.624 | [0.547, 0.706] | 39 / 5 |
| zinnia_noir | halo2_hw | prove_core_s | 44 | 0.962 | [0.794, 1.156] | 22 / 22 |
| zinnia_noir | halo2_hw | end_to_end_s | 44 | 1.031 | [0.844, 1.257] | 21 / 23 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 44 | 2.179 | [1.741, 2.724] | 5 / 39 |
| zinnia_noir | halo2_hw | verify_s | 44 | 1.609 | [1.417, 1.821] | 7 / 37 |
| zinnia_noir | halo2_lit | prove_core_s | 44 | 0.666 | [0.478, 0.900] | 29 / 15 |
| zinnia_noir | halo2_lit | end_to_end_s | 44 | 0.714 | [0.509, 0.977] | 28 / 16 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 44 | 1.509 | [1.078, 2.083] | 13 / 31 |
| zinnia_noir | halo2_lit | verify_s | 44 | 1.259 | [1.063, 1.486] | 13 / 31 |
| zinnia_noir | noir_lit | prove_core_s | 44 | 0.875 | [0.737, 0.994] | 22 / 22 |
| zinnia_noir | noir_lit | end_to_end_s | 44 | 0.870 | [0.728, 0.992] | 23 / 21 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 44 | 0.901 | [0.806, 0.980] | 29 / 15 |
| zinnia_noir | noir_lit | verify_s | 44 | 0.986 | [0.934, 1.036] | 21 / 23 |

## Excluded cases

(none)

## Cases without a program

No program for the case on that system (e.g. no handwritten baseline yet); not a failure.

(none)

## Failures by system

