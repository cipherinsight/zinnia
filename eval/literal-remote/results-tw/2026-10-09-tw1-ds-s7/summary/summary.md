# Epoch 2026-10-09-tw1-ds-s7

Ratios are Zinnia / baseline of per-case medians; below 1 means Zinnia is faster.
Only cases with status `ok` (proved, verified, outputs equal to NumPy, tampered proof rejected) enter a comparison.

`zinnia_noir` `end_to_end_s` witness step: noir-execute (12 cases); `end_to_end_nargo_s` uses `nargo execute`.
`noir_hw` `end_to_end_s` witness step: noir-execute (12 cases); `end_to_end_nargo_s` uses `nargo execute`.
`noir_lit` `end_to_end_s` witness step: noir-execute (12 cases); `end_to_end_nargo_s` uses `nargo execute`.

## Coverage

| system | cases | ok | other statuses |
|---|---|---|---|
| zinnia_halo2 | 12 | 12 | - |
| zinnia_noir | 12 | 12 | - |
| halo2_hw | 12 | 12 | - |
| noir_hw | 12 | 12 | - |
| halo2_lit | 12 | 12 | - |
| noir_lit | 12 | 12 | - |

## Headline: all-supported subset without copy-only cases

11 cases completed by every system that do more than copy inputs: ds1000_case0314, ds1000_case0324, ds1000_case0358, ds1000_case0369, ds1000_case0398, ds1000_case0416, ds1000_case0433, ds1000_case0441, ds1000_case0462, ds1000_case0491, ds1000_case0505

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 11 | 0.927 | [0.840, 1.011] | 7 / 4 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 11 | 0.933 | [0.848, 1.016] | 7 / 4 |
| zinnia_halo2 | halo2_hw | verify_s | 11 | 0.956 | [0.883, 1.035] | 7 / 4 |
| zinnia_noir | noir_hw | prove_core_s | 11 | 0.953 | [0.903, 1.005] | 7 / 4 |
| zinnia_noir | noir_hw | end_to_end_s | 11 | 0.962 | [0.911, 1.013] | 7 / 4 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 11 | 0.966 | [0.942, 0.991] | 8 / 3 |
| zinnia_noir | noir_hw | verify_s | 11 | 1.160 | [1.044, 1.276] | 3 / 8 |
| zinnia_halo2 | noir_hw | prove_core_s | 11 | 0.811 | [0.631, 1.087] | 8 / 3 |
| zinnia_halo2 | noir_hw | end_to_end_s | 11 | 0.793 | [0.619, 1.052] | 8 / 3 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 11 | 0.409 | [0.305, 0.579] | 10 / 1 |
| zinnia_halo2 | noir_hw | verify_s | 11 | 0.639 | [0.532, 0.768] | 10 / 1 |
| zinnia_halo2 | halo2_lit | prove_core_s | 11 | 0.801 | [0.697, 0.899] | 9 / 2 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 11 | 0.806 | [0.703, 0.904] | 9 / 2 |
| zinnia_halo2 | halo2_lit | verify_s | 11 | 0.939 | [0.839, 1.051] | 6 / 5 |
| zinnia_halo2 | noir_lit | prove_core_s | 11 | 0.783 | [0.577, 1.089] | 8 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_s | 11 | 0.760 | [0.563, 1.055] | 8 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 11 | 0.402 | [0.291, 0.582] | 10 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 11 | 0.610 | [0.504, 0.745] | 10 / 1 |
| zinnia_noir | halo2_hw | prove_core_s | 11 | 1.089 | [0.733, 1.488] | 3 / 8 |
| zinnia_noir | halo2_hw | end_to_end_s | 11 | 1.132 | [0.772, 1.533] | 2 / 9 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 11 | 2.201 | [1.473, 3.015] | 1 / 10 |
| zinnia_noir | halo2_hw | verify_s | 11 | 1.736 | [1.361, 2.192] | 1 / 10 |
| zinnia_noir | halo2_lit | prove_core_s | 11 | 0.941 | [0.643, 1.332] | 5 / 6 |
| zinnia_noir | halo2_lit | end_to_end_s | 11 | 0.978 | [0.677, 1.374] | 4 / 7 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 11 | 1.902 | [1.297, 2.696] | 2 / 9 |
| zinnia_noir | halo2_lit | verify_s | 11 | 1.706 | [1.322, 2.182] | 1 / 10 |
| zinnia_noir | noir_lit | prove_core_s | 11 | 0.919 | [0.855, 0.989] | 8 / 3 |
| zinnia_noir | noir_lit | end_to_end_s | 11 | 0.923 | [0.856, 0.993] | 8 / 3 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 11 | 0.949 | [0.901, 1.001] | 9 / 2 |
| zinnia_noir | noir_lit | verify_s | 11 | 1.108 | [1.011, 1.222] | 4 / 7 |

## Copy-only cases (reported apart)

1 all-supported cases that only copy inputs: ds1000_case0336

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 1 | 0.835 | [0.835, 0.835] | 1 / 0 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 1 | 0.838 | [0.838, 0.838] | 1 / 0 |
| zinnia_halo2 | halo2_hw | verify_s | 1 | 1.147 | [1.147, 1.147] | 0 / 1 |
| zinnia_noir | noir_hw | prove_core_s | 1 | 1.041 | [1.041, 1.041] | 0 / 1 |
| zinnia_noir | noir_hw | end_to_end_s | 1 | 1.045 | [1.045, 1.045] | 0 / 1 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 1 | 0.997 | [0.997, 0.997] | 1 / 0 |
| zinnia_noir | noir_hw | verify_s | 1 | 1.074 | [1.074, 1.074] | 0 / 1 |
| zinnia_halo2 | noir_hw | prove_core_s | 1 | 0.364 | [0.364, 0.364] | 1 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_s | 1 | 0.356 | [0.356, 0.356] | 1 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 1 | 0.169 | [0.169, 0.169] | 1 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 1 | 0.348 | [0.348, 0.348] | 1 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 1 | 0.912 | [0.912, 0.912] | 1 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 1 | 0.915 | [0.915, 0.915] | 1 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 1 | 1.067 | [1.067, 1.067] | 0 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 1 | 0.355 | [0.355, 0.355] | 1 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_s | 1 | 0.350 | [0.350, 0.350] | 1 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 1 | 0.162 | [0.162, 0.162] | 1 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 1 | 0.328 | [0.328, 0.328] | 1 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 1 | 2.389 | [2.389, 2.389] | 0 / 1 |
| zinnia_noir | halo2_hw | end_to_end_s | 1 | 2.459 | [2.459, 2.459] | 0 / 1 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 1 | 4.940 | [4.940, 4.940] | 0 / 1 |
| zinnia_noir | halo2_hw | verify_s | 1 | 3.541 | [3.541, 3.541] | 0 / 1 |
| zinnia_noir | halo2_lit | prove_core_s | 1 | 2.610 | [2.610, 2.610] | 0 / 1 |
| zinnia_noir | halo2_lit | end_to_end_s | 1 | 2.686 | [2.686, 2.686] | 0 / 1 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 1 | 5.396 | [5.396, 5.396] | 0 / 1 |
| zinnia_noir | halo2_lit | verify_s | 1 | 3.294 | [3.294, 3.294] | 0 / 1 |
| zinnia_noir | noir_lit | prove_core_s | 1 | 1.016 | [1.016, 1.016] | 0 / 1 |
| zinnia_noir | noir_lit | end_to_end_s | 1 | 1.028 | [1.028, 1.028] | 0 / 1 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 1 | 0.953 | [0.953, 0.953] | 1 / 0 |
| zinnia_noir | noir_lit | verify_s | 1 | 1.012 | [1.012, 1.012] | 0 / 1 |

## All-supported subset

12 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): ds1000_case0314, ds1000_case0324, ds1000_case0336, ds1000_case0358, ds1000_case0369, ds1000_case0398, ds1000_case0416, ds1000_case0433, ds1000_case0441, ds1000_case0462, ds1000_case0491, ds1000_case0505

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 12 | 0.919 | [0.841, 0.997] | 8 / 4 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 12 | 0.924 | [0.848, 1.002] | 8 / 4 |
| zinnia_halo2 | halo2_hw | verify_s | 12 | 0.971 | [0.897, 1.050] | 7 / 5 |
| zinnia_noir | noir_hw | prove_core_s | 12 | 0.960 | [0.911, 1.010] | 7 / 5 |
| zinnia_noir | noir_hw | end_to_end_s | 12 | 0.969 | [0.919, 1.018] | 7 / 5 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 12 | 0.969 | [0.945, 0.992] | 9 / 3 |
| zinnia_noir | noir_hw | verify_s | 12 | 1.152 | [1.050, 1.260] | 3 / 9 |
| zinnia_halo2 | noir_hw | prove_core_s | 12 | 0.759 | [0.585, 1.016] | 9 / 3 |
| zinnia_halo2 | noir_hw | end_to_end_s | 12 | 0.741 | [0.573, 0.984] | 9 / 3 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 12 | 0.380 | [0.282, 0.535] | 11 / 1 |
| zinnia_halo2 | noir_hw | verify_s | 12 | 0.607 | [0.502, 0.737] | 11 / 1 |
| zinnia_halo2 | halo2_lit | prove_core_s | 12 | 0.809 | [0.711, 0.901] | 10 / 2 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 12 | 0.814 | [0.716, 0.905] | 10 / 2 |
| zinnia_halo2 | halo2_lit | verify_s | 12 | 0.949 | [0.854, 1.051] | 6 / 6 |
| zinnia_halo2 | noir_lit | prove_core_s | 12 | 0.733 | [0.544, 1.012] | 9 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_s | 12 | 0.713 | [0.530, 0.981] | 9 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 12 | 0.373 | [0.270, 0.536] | 11 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 12 | 0.579 | [0.472, 0.711] | 11 / 1 |
| zinnia_noir | halo2_hw | prove_core_s | 12 | 1.162 | [0.793, 1.592] | 3 / 9 |
| zinnia_noir | halo2_hw | end_to_end_s | 12 | 1.208 | [0.833, 1.642] | 2 / 10 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 12 | 2.355 | [1.600, 3.232] | 1 / 11 |
| zinnia_noir | halo2_hw | verify_s | 12 | 1.843 | [1.448, 2.348] | 1 / 11 |
| zinnia_noir | halo2_lit | prove_core_s | 12 | 1.024 | [0.700, 1.461] | 5 / 7 |
| zinnia_noir | halo2_lit | end_to_end_s | 12 | 1.064 | [0.733, 1.508] | 4 / 8 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 12 | 2.074 | [1.406, 2.966] | 2 / 10 |
| zinnia_noir | halo2_lit | verify_s | 12 | 1.802 | [1.404, 2.300] | 1 / 11 |
| zinnia_noir | noir_lit | prove_core_s | 12 | 0.927 | [0.864, 0.994] | 8 / 4 |
| zinnia_noir | noir_lit | end_to_end_s | 12 | 0.931 | [0.866, 0.998] | 8 / 4 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 12 | 0.949 | [0.906, 0.997] | 10 / 2 |
| zinnia_noir | noir_lit | verify_s | 12 | 1.099 | [1.010, 1.203] | 4 / 8 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 12 | 0.919 | [0.841, 0.997] | 8 / 4 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 12 | 0.924 | [0.848, 1.002] | 8 / 4 |
| zinnia_halo2 | halo2_hw | verify_s | 12 | 0.971 | [0.897, 1.050] | 7 / 5 |
| zinnia_noir | noir_hw | prove_core_s | 12 | 0.960 | [0.911, 1.010] | 7 / 5 |
| zinnia_noir | noir_hw | end_to_end_s | 12 | 0.969 | [0.919, 1.018] | 7 / 5 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 12 | 0.969 | [0.945, 0.992] | 9 / 3 |
| zinnia_noir | noir_hw | verify_s | 12 | 1.152 | [1.050, 1.260] | 3 / 9 |
| zinnia_halo2 | noir_hw | prove_core_s | 12 | 0.759 | [0.585, 1.016] | 9 / 3 |
| zinnia_halo2 | noir_hw | end_to_end_s | 12 | 0.741 | [0.573, 0.984] | 9 / 3 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 12 | 0.380 | [0.282, 0.535] | 11 / 1 |
| zinnia_halo2 | noir_hw | verify_s | 12 | 0.607 | [0.502, 0.737] | 11 / 1 |
| zinnia_halo2 | halo2_lit | prove_core_s | 12 | 0.809 | [0.711, 0.901] | 10 / 2 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 12 | 0.814 | [0.716, 0.905] | 10 / 2 |
| zinnia_halo2 | halo2_lit | verify_s | 12 | 0.949 | [0.854, 1.051] | 6 / 6 |
| zinnia_halo2 | noir_lit | prove_core_s | 12 | 0.733 | [0.544, 1.012] | 9 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_s | 12 | 0.713 | [0.530, 0.981] | 9 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 12 | 0.373 | [0.270, 0.536] | 11 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 12 | 0.579 | [0.472, 0.711] | 11 / 1 |
| zinnia_noir | halo2_hw | prove_core_s | 12 | 1.162 | [0.793, 1.592] | 3 / 9 |
| zinnia_noir | halo2_hw | end_to_end_s | 12 | 1.208 | [0.833, 1.642] | 2 / 10 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 12 | 2.355 | [1.600, 3.232] | 1 / 11 |
| zinnia_noir | halo2_hw | verify_s | 12 | 1.843 | [1.448, 2.348] | 1 / 11 |
| zinnia_noir | halo2_lit | prove_core_s | 12 | 1.024 | [0.700, 1.461] | 5 / 7 |
| zinnia_noir | halo2_lit | end_to_end_s | 12 | 1.064 | [0.733, 1.508] | 4 / 8 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 12 | 2.074 | [1.406, 2.966] | 2 / 10 |
| zinnia_noir | halo2_lit | verify_s | 12 | 1.802 | [1.404, 2.300] | 1 / 11 |
| zinnia_noir | noir_lit | prove_core_s | 12 | 0.927 | [0.864, 0.994] | 8 / 4 |
| zinnia_noir | noir_lit | end_to_end_s | 12 | 0.931 | [0.866, 0.998] | 8 / 4 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 12 | 0.949 | [0.906, 0.997] | 10 / 2 |
| zinnia_noir | noir_lit | verify_s | 12 | 1.099 | [1.010, 1.203] | 4 / 8 |

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

