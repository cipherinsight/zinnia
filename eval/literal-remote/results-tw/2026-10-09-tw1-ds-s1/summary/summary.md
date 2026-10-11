# Epoch 2026-10-09-tw1-ds-s1

Ratios are Zinnia / baseline of per-case medians; below 1 means Zinnia is faster.
Only cases with status `ok` (proved, verified, outputs equal to NumPy, tampered proof rejected) enter a comparison.

`zinnia_noir` `end_to_end_s` witness step: noir-execute (13 cases); `end_to_end_nargo_s` uses `nargo execute`.
`noir_hw` `end_to_end_s` witness step: noir-execute (13 cases); `end_to_end_nargo_s` uses `nargo execute`.
`noir_lit` `end_to_end_s` witness step: noir-execute (13 cases); `end_to_end_nargo_s` uses `nargo execute`.

## Coverage

| system | cases | ok | other statuses |
|---|---|---|---|
| zinnia_halo2 | 13 | 13 | - |
| zinnia_noir | 13 | 13 | - |
| halo2_hw | 13 | 13 | - |
| noir_hw | 13 | 13 | - |
| halo2_lit | 13 | 13 | - |
| noir_lit | 13 | 13 | - |

## Headline: all-supported subset without copy-only cases

8 cases completed by every system that do more than copy inputs: ds1000_case0300, ds1000_case0318, ds1000_case0326, ds1000_case0375, ds1000_case0402, ds1000_case0422, ds1000_case0443, ds1000_case0508

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 8 | 0.996 | [0.911, 1.096] | 4 / 4 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 8 | 1.014 | [0.921, 1.127] | 4 / 4 |
| zinnia_halo2 | halo2_hw | verify_s | 8 | 0.993 | [0.894, 1.105] | 3 / 5 |
| zinnia_noir | noir_hw | prove_core_s | 8 | 1.044 | [0.973, 1.135] | 3 / 5 |
| zinnia_noir | noir_hw | end_to_end_s | 8 | 1.022 | [0.972, 1.067] | 3 / 5 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 8 | 1.012 | [0.971, 1.047] | 2 / 6 |
| zinnia_noir | noir_hw | verify_s | 8 | 0.967 | [0.835, 1.114] | 4 / 4 |
| zinnia_halo2 | noir_hw | prove_core_s | 8 | 0.978 | [0.788, 1.194] | 4 / 4 |
| zinnia_halo2 | noir_hw | end_to_end_s | 8 | 0.826 | [0.507, 1.164] | 4 / 4 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 8 | 0.425 | [0.272, 0.595] | 8 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 8 | 0.591 | [0.480, 0.712] | 8 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 8 | 0.327 | [0.159, 0.617] | 8 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 8 | 0.332 | [0.161, 0.626] | 8 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 8 | 0.621 | [0.408, 0.884] | 6 / 2 |
| zinnia_halo2 | noir_lit | prove_core_s | 8 | 0.619 | [0.457, 0.816] | 7 / 1 |
| zinnia_halo2 | noir_lit | end_to_end_s | 8 | 0.511 | [0.324, 0.762] | 7 / 1 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 8 | 0.304 | [0.204, 0.429] | 8 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 8 | 0.582 | [0.448, 0.749] | 7 / 1 |
| zinnia_noir | halo2_hw | prove_core_s | 8 | 1.063 | [0.828, 1.491] | 4 / 4 |
| zinnia_noir | halo2_hw | end_to_end_s | 8 | 1.254 | [0.865, 2.254] | 4 / 4 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 8 | 2.411 | [1.655, 4.174] | 0 / 8 |
| zinnia_noir | halo2_hw | verify_s | 8 | 1.624 | [1.267, 2.116] | 1 / 7 |
| zinnia_noir | halo2_lit | prove_core_s | 8 | 0.349 | [0.145, 0.788] | 6 / 2 |
| zinnia_noir | halo2_lit | end_to_end_s | 8 | 0.411 | [0.154, 1.104] | 6 / 2 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 8 | 0.791 | [0.298, 2.093] | 4 / 4 |
| zinnia_noir | halo2_lit | verify_s | 8 | 1.016 | [0.631, 1.704] | 5 / 3 |
| zinnia_noir | noir_lit | prove_core_s | 8 | 0.661 | [0.441, 0.927] | 7 / 1 |
| zinnia_noir | noir_lit | end_to_end_s | 8 | 0.632 | [0.410, 0.906] | 7 / 1 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 8 | 0.724 | [0.517, 0.952] | 7 / 1 |
| zinnia_noir | noir_lit | verify_s | 8 | 0.953 | [0.854, 1.042] | 4 / 4 |

## Copy-only cases (reported apart)

5 all-supported cases that only copy inputs: ds1000_case0340, ds1000_case0360, ds1000_case0435, ds1000_case0472, ds1000_case0496

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 5 | 0.989 | [0.946, 1.042] | 4 / 1 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 5 | 1.001 | [0.954, 1.053] | 3 / 2 |
| zinnia_halo2 | halo2_hw | verify_s | 5 | 0.943 | [0.878, 1.012] | 3 / 2 |
| zinnia_noir | noir_hw | prove_core_s | 5 | 1.018 | [0.990, 1.053] | 2 / 3 |
| zinnia_noir | noir_hw | end_to_end_s | 5 | 1.014 | [0.987, 1.044] | 3 / 2 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 5 | 0.980 | [0.937, 1.026] | 3 / 2 |
| zinnia_noir | noir_hw | verify_s | 5 | 1.147 | [1.048, 1.322] | 0 / 5 |
| zinnia_halo2 | noir_hw | prove_core_s | 5 | 0.971 | [0.531, 1.798] | 3 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_s | 5 | 0.920 | [0.521, 1.642] | 3 / 2 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 5 | 0.481 | [0.255, 0.981] | 4 / 1 |
| zinnia_halo2 | noir_hw | verify_s | 5 | 0.652 | [0.452, 0.938] | 3 / 2 |
| zinnia_halo2 | halo2_lit | prove_core_s | 5 | 1.007 | [0.955, 1.075] | 3 / 2 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 5 | 1.019 | [0.965, 1.087] | 2 / 3 |
| zinnia_halo2 | halo2_lit | verify_s | 5 | 0.939 | [0.886, 1.006] | 4 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 5 | 0.939 | [0.517, 1.706] | 3 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_s | 5 | 0.900 | [0.509, 1.587] | 3 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 5 | 0.486 | [0.252, 0.982] | 4 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 5 | 0.654 | [0.538, 0.796] | 5 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 5 | 1.037 | [0.534, 1.851] | 2 / 3 |
| zinnia_noir | halo2_hw | end_to_end_s | 5 | 1.103 | [0.624, 1.902] | 2 / 3 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 5 | 2.039 | [1.037, 3.685] | 1 / 4 |
| zinnia_noir | halo2_hw | verify_s | 5 | 1.657 | [1.218, 2.222] | 1 / 4 |
| zinnia_noir | halo2_lit | prove_core_s | 5 | 1.055 | [0.551, 1.944] | 2 / 3 |
| zinnia_noir | halo2_lit | end_to_end_s | 5 | 1.123 | [0.631, 1.998] | 2 / 3 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 5 | 2.076 | [1.056, 3.871] | 1 / 4 |
| zinnia_noir | halo2_lit | verify_s | 5 | 1.651 | [1.173, 2.342] | 1 / 4 |
| zinnia_noir | noir_lit | prove_core_s | 5 | 0.985 | [0.926, 1.034] | 3 / 2 |
| zinnia_noir | noir_lit | end_to_end_s | 5 | 0.992 | [0.938, 1.040] | 3 / 2 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 5 | 0.990 | [0.947, 1.035] | 3 / 2 |
| zinnia_noir | noir_lit | verify_s | 5 | 1.151 | [1.028, 1.298] | 0 / 5 |

## All-supported subset

13 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): ds1000_case0300, ds1000_case0318, ds1000_case0326, ds1000_case0340, ds1000_case0360, ds1000_case0375, ds1000_case0402, ds1000_case0422, ds1000_case0435, ds1000_case0443, ds1000_case0472, ds1000_case0496, ds1000_case0508

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 13 | 0.993 | [0.937, 1.055] | 8 / 5 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 13 | 1.009 | [0.948, 1.078] | 7 / 6 |
| zinnia_halo2 | halo2_hw | verify_s | 13 | 0.973 | [0.905, 1.047] | 6 / 7 |
| zinnia_noir | noir_hw | prove_core_s | 13 | 1.034 | [0.990, 1.089] | 5 / 8 |
| zinnia_noir | noir_hw | end_to_end_s | 13 | 1.019 | [0.986, 1.050] | 6 / 7 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 13 | 0.999 | [0.968, 1.030] | 5 / 8 |
| zinnia_noir | noir_hw | verify_s | 13 | 1.033 | [0.924, 1.151] | 4 / 9 |
| zinnia_halo2 | noir_hw | prove_core_s | 13 | 0.975 | [0.742, 1.299] | 7 / 6 |
| zinnia_halo2 | noir_hw | end_to_end_s | 13 | 0.861 | [0.595, 1.208] | 7 / 6 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 13 | 0.446 | [0.306, 0.645] | 12 / 1 |
| zinnia_halo2 | noir_hw | verify_s | 13 | 0.614 | [0.507, 0.745] | 11 / 2 |
| zinnia_halo2 | halo2_lit | prove_core_s | 13 | 0.504 | [0.288, 0.800] | 11 / 2 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 13 | 0.511 | [0.292, 0.810] | 10 / 3 |
| zinnia_halo2 | halo2_lit | verify_s | 13 | 0.728 | [0.543, 0.916] | 10 / 3 |
| zinnia_halo2 | noir_lit | prove_core_s | 13 | 0.727 | [0.530, 1.010] | 10 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_s | 13 | 0.635 | [0.431, 0.929] | 10 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 13 | 0.364 | [0.251, 0.536] | 12 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 13 | 0.609 | [0.504, 0.732] | 12 / 1 |
| zinnia_noir | halo2_hw | prove_core_s | 13 | 1.053 | [0.768, 1.439] | 6 / 7 |
| zinnia_noir | halo2_hw | end_to_end_s | 13 | 1.194 | [0.828, 1.825] | 6 / 7 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 13 | 2.261 | [1.540, 3.406] | 1 / 12 |
| zinnia_noir | halo2_hw | verify_s | 13 | 1.637 | [1.334, 2.007] | 2 / 11 |
| zinnia_noir | halo2_lit | prove_core_s | 13 | 0.534 | [0.270, 0.994] | 8 / 5 |
| zinnia_noir | halo2_lit | end_to_end_s | 13 | 0.605 | [0.293, 1.198] | 8 / 5 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 13 | 1.146 | [0.553, 2.255] | 5 / 8 |
| zinnia_noir | halo2_lit | verify_s | 13 | 1.224 | [0.849, 1.745] | 6 / 7 |
| zinnia_noir | noir_lit | prove_core_s | 13 | 0.770 | [0.579, 0.962] | 10 / 3 |
| zinnia_noir | noir_lit | end_to_end_s | 13 | 0.752 | [0.552, 0.953] | 10 / 3 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 13 | 0.817 | [0.645, 0.976] | 10 / 3 |
| zinnia_noir | noir_lit | verify_s | 13 | 1.024 | [0.934, 1.118] | 4 / 9 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 13 | 0.993 | [0.937, 1.055] | 8 / 5 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 13 | 1.009 | [0.948, 1.078] | 7 / 6 |
| zinnia_halo2 | halo2_hw | verify_s | 13 | 0.973 | [0.905, 1.047] | 6 / 7 |
| zinnia_noir | noir_hw | prove_core_s | 13 | 1.034 | [0.990, 1.089] | 5 / 8 |
| zinnia_noir | noir_hw | end_to_end_s | 13 | 1.019 | [0.986, 1.050] | 6 / 7 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 13 | 0.999 | [0.968, 1.030] | 5 / 8 |
| zinnia_noir | noir_hw | verify_s | 13 | 1.033 | [0.924, 1.151] | 4 / 9 |
| zinnia_halo2 | noir_hw | prove_core_s | 13 | 0.975 | [0.742, 1.299] | 7 / 6 |
| zinnia_halo2 | noir_hw | end_to_end_s | 13 | 0.861 | [0.595, 1.208] | 7 / 6 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 13 | 0.446 | [0.306, 0.645] | 12 / 1 |
| zinnia_halo2 | noir_hw | verify_s | 13 | 0.614 | [0.507, 0.745] | 11 / 2 |
| zinnia_halo2 | halo2_lit | prove_core_s | 13 | 0.504 | [0.288, 0.800] | 11 / 2 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 13 | 0.511 | [0.292, 0.810] | 10 / 3 |
| zinnia_halo2 | halo2_lit | verify_s | 13 | 0.728 | [0.543, 0.916] | 10 / 3 |
| zinnia_halo2 | noir_lit | prove_core_s | 13 | 0.727 | [0.530, 1.010] | 10 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_s | 13 | 0.635 | [0.431, 0.929] | 10 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 13 | 0.364 | [0.251, 0.536] | 12 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 13 | 0.609 | [0.504, 0.732] | 12 / 1 |
| zinnia_noir | halo2_hw | prove_core_s | 13 | 1.053 | [0.768, 1.439] | 6 / 7 |
| zinnia_noir | halo2_hw | end_to_end_s | 13 | 1.194 | [0.828, 1.825] | 6 / 7 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 13 | 2.261 | [1.540, 3.406] | 1 / 12 |
| zinnia_noir | halo2_hw | verify_s | 13 | 1.637 | [1.334, 2.007] | 2 / 11 |
| zinnia_noir | halo2_lit | prove_core_s | 13 | 0.534 | [0.270, 0.994] | 8 / 5 |
| zinnia_noir | halo2_lit | end_to_end_s | 13 | 0.605 | [0.293, 1.198] | 8 / 5 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 13 | 1.146 | [0.553, 2.255] | 5 / 8 |
| zinnia_noir | halo2_lit | verify_s | 13 | 1.224 | [0.849, 1.745] | 6 / 7 |
| zinnia_noir | noir_lit | prove_core_s | 13 | 0.770 | [0.579, 0.962] | 10 / 3 |
| zinnia_noir | noir_lit | end_to_end_s | 13 | 0.752 | [0.552, 0.953] | 10 / 3 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 13 | 0.817 | [0.645, 0.976] | 10 / 3 |
| zinnia_noir | noir_lit | verify_s | 13 | 1.024 | [0.934, 1.118] | 4 / 9 |

## Excluded cases

- ds1000_case0291: returns a.shape; the annotation fixes it to (2, 5, 5, 2), so `a` is never read
- ds1000_case0295: ZinniaException: IR generation failed: shape element at axis 1 must be a compile-time constant int
- ds1000_case0296: same kernel as 0295 after renaming local variables
- ds1000_case0297: ZinniaException: IR generation failed: shape element at axis 1 must be a compile-time constant int
- ds1000_case0299: ZinniaException: IR generation failed: shape element at axis 1 must be a compile-time constant int
- ds1000_case0302: same outputs up to a fixed reordering or reshaping, on the same inputs, like 0301
- ds1000_case0303: copies 20 input cells to outputs with no arithmetic (only which cells, their order and unread input cells differ), like 0301
- ds1000_case0304: copies 20 input cells to outputs with no arithmetic (only which cells, their order and unread input cells differ), like 0301
- ds1000_case0313: same kernel as 0309 after renaming local variables
- ds1000_case0330: computes `a ** power` like 0329; 0329 assigns the result back to `a`, 0330 to `result`
- ds1000_case0331: ZinniaException: IR generation failed: np.gcd is not implemented
- ds1000_case0332: same kernel as 0331 after renaming local variables
- ds1000_case0337: copies 8 input cells to outputs with no arithmetic (only which cells, their order and unread input cells differ), like 0336
- ds1000_case0339: copies 16 input cells to outputs with no arithmetic (only which cells, their order and unread input cells differ), like 0338
- ds1000_case0341: same kernel as 0340 after renaming local variables
- ds1000_case0342: same kernel as 0340 after renaming local variables
- ds1000_case0343: same outputs up to a fixed reordering or reshaping, on the same inputs, like 0340
- ds1000_case0344: non_numeric_data: test input mystr: test input is a str
- ds1000_case0349: returns a.shape[1]; the annotation fixes it to 6, so `a` is never read
- ds1000_case0354: ZinniaException: IR generation failed: np.append is not implemented
- ds1000_case0356: same kernel as 0355 after renaming local variables
- ds1000_case0357: same outputs up to a fixed reordering or reshaping, on the same inputs, like 0355
- ds1000_case0365: same kernel as 0363 after renaming local variables
- ds1000_case0366: ZinniaException: IR generation failed: np.insert: this bounded slice supports one scalar obj
- ds1000_case0367: imports: import copy; non_numeric_data: test input array_of_arrays: object array (e.g. arrays of different shapes)
- ds1000_case0370: same kernel as 0368 after renaming local variables
- ds1000_case0373: returns np.arange(1, n+1)/n for n = len(grades) = 50, a constant; the sort is dead code
- ds1000_case0382: same outputs up to a fixed reordering or reshaping, on the same inputs, like 0381
- ds1000_case0387: same outputs up to a fixed reordering or reshaping, on the same inputs, like 0385
- ds1000_case0388: same outputs up to a fixed reordering or reshaping, on the same inputs, like 0385
- ds1000_case0390: same outputs up to a fixed reordering or reshaping, on the same inputs, like 0385
- ds1000_case0392: copies 30 input cells to outputs with no arithmetic (only which cells, their order and unread input cells differ), like 0340
- ds1000_case0393: same outputs up to a fixed reordering or reshaping, on the same inputs, like 0391
- ds1000_case0400: no inputs; returns the constant np.array([])
- ds1000_case0401: no inputs; returns the constant np.array([[], [], []])
- ds1000_case0405: ZinniaException: IR generation failed: np.bincount: x entry value is not statically bounded — its runtime value could be anything, and the output length is `max(x)+1`. The previous compiler revision silently capped the output at 256 with a proof-time witness assert; that has been removed. To compile this program, either pass a literal int array, or annotate the input with `@requires(lambda x: forall(x, lambda v: v <= K))` so the compiler can derive a tight, sound upper bound.
- ds1000_case0406: ZinniaException: IR generation failed: shape element at axis 0 must be a compile-time constant int
- ds1000_case0407: ZinniaException: IR generation failed: np.bincount: x entry value is not statically bounded — its runtime value could be anything, and the output length is `max(x)+1`. The previous compiler revision silently capped the output at 256 with a proof-time witness assert; that has been removed. To compile this program, either pass a literal int array, or annotate the input with `@requires(lambda x: forall(x, lambda v: v <= K))` so the compiler can derive a tight, sound upper bound.
- ds1000_case0408: ZinniaException: IR generation failed: shape element at axis 0 must be a compile-time constant int
- ds1000_case0411: ZinniaException: IR generation failed: Unknown function `slice` in @zk_circuit. If `slice` is a helper you defined, decorate it with @zk_chip (or @zk_external for off-circuit witness functions). registered chips: fill_crop
- ds1000_case0413: compiles, but Zinnia refuses to build a witness for either fixture input: ZinniaException: Cannot build witness for `x`: `Complex` values are not supported as circuit inputs.
- ds1000_case0417: same outputs up to a fixed reordering or reshaping, on the same inputs, like 0414
- ds1000_case0418: same outputs up to a fixed reordering or reshaping, on the same inputs, like 0416
- ds1000_case0419: same outputs up to a fixed reordering or reshaping, on the same inputs, like 0416
- ds1000_case0427: compiles, but no backend synthesizes the circuit: halo2-base: RuntimeError: Circuit synthesis error: halo2-base backend does not yet support bit_xor_i; noir: RuntimeError: Unsupported IR operation in circuit: noir-emit:bit_xor_i
- ds1000_case0430: same kernel as 0428 after renaming local variables
- ds1000_case0448: copies 24 input cells to outputs with no arithmetic (only which cells, their order and unread input cells differ), like 0361
- ds1000_case0449: same kernel as 0448 after renaming local variables; 0448 itself duplicates 0361: copies 24 input cells to outputs with no arithmetic (only which cells, their order and unread input cells differ), like 0361
- ds1000_case0451: no inputs; returns the constant np.zeros((20, 10, 10, 2))
- ds1000_case0459: non_numeric_data: test input A: array of strings
- ds1000_case0460: non_numeric_data: test input A: array of strings
- ds1000_case0463: same outputs up to a fixed reordering or reshaping, on the same inputs, like 0462
- ds1000_case0467: ZinniaException: IR generation failed: IRBuilder::ensure_ptr: cannot materialize a composite (List/Tuple) as a scalar IR pointer; got List(CompositeData { elements_type: [Integer], values: [Integer(ScalarValue { static_val: Some(-1), stmt_id: Some(40), value_id: ValueId(55) })], value_id: ValueId(58) }). A composite reached a scalar IR op; the right fix is at the call site (e.g. dispatch a per-element loop or convert to a StaticArray), not silently taking the first element.
- ds1000_case0468: same outputs up to a fixed reordering or reshaping, on the same inputs, like 0467
- ds1000_case0469: non_finite_values: test input a: test input contains NaN or infinity
- ds1000_case0470: non_finite_values: test input a: test input contains NaN or infinity
- ds1000_case0473: non_numeric_data: test input CNTS: ragged sequence (setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (4,) + inhomogeneous part.)
- ds1000_case0489: killed by signal 9 after 14 s: exceeded the memory cap (MEMRUN_MAX=8G)
- ds1000_case0492: non_finite_values: test input s1: test input contains NaN or infinity
- ds1000_case0498: same kernel as 0495 after renaming local variables
- ds1000_case0499: same outputs up to a fixed reordering or reshaping, on the same inputs, like 0496
- ds1000_case0502: same kernel as 0501 after renaming local variables

## Cases without a program

No program for the case on that system (e.g. no handwritten baseline yet); not a failure.

(none)

## Failures by system

