# Epoch 2026-10-09-tw1-ds-s5

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

11 cases completed by every system that do more than copy inputs: ds1000_case0311, ds1000_case0322, ds1000_case0334, ds1000_case0353, ds1000_case0364, ds1000_case0414, ds1000_case0429, ds1000_case0439, ds1000_case0453, ds1000_case0485, ds1000_case0503

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 11 | 0.962 | [0.868, 1.056] | 6 / 5 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 11 | 0.971 | [0.877, 1.068] | 6 / 5 |
| zinnia_halo2 | halo2_hw | verify_s | 11 | 0.964 | [0.849, 1.087] | 6 / 5 |
| zinnia_noir | noir_hw | prove_core_s | 11 | 1.026 | [0.950, 1.114] | 4 / 7 |
| zinnia_noir | noir_hw | end_to_end_s | 11 | 1.029 | [0.952, 1.115] | 4 / 7 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 11 | 1.029 | [0.968, 1.126] | 6 / 5 |
| zinnia_noir | noir_hw | verify_s | 11 | 0.909 | [0.823, 1.011] | 7 / 4 |
| zinnia_halo2 | noir_hw | prove_core_s | 11 | 1.224 | [0.835, 1.800] | 5 / 6 |
| zinnia_halo2 | noir_hw | end_to_end_s | 11 | 1.181 | [0.814, 1.718] | 5 / 6 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 11 | 0.639 | [0.411, 0.997] | 8 / 3 |
| zinnia_halo2 | noir_hw | verify_s | 11 | 0.669 | [0.514, 0.875] | 8 / 3 |
| zinnia_halo2 | halo2_lit | prove_core_s | 11 | 0.777 | [0.694, 0.870] | 10 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 11 | 0.785 | [0.700, 0.879] | 10 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 11 | 0.863 | [0.763, 0.968] | 10 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 11 | 1.190 | [0.843, 1.661] | 4 / 7 |
| zinnia_halo2 | noir_lit | end_to_end_s | 11 | 1.148 | [0.821, 1.588] | 4 / 7 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 11 | 0.627 | [0.414, 0.946] | 8 / 3 |
| zinnia_halo2 | noir_lit | verify_s | 11 | 0.676 | [0.535, 0.845] | 9 / 2 |
| zinnia_noir | halo2_hw | prove_core_s | 11 | 0.806 | [0.561, 1.166] | 7 / 4 |
| zinnia_noir | halo2_hw | end_to_end_s | 11 | 0.846 | [0.596, 1.210] | 7 / 4 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 11 | 1.564 | [1.060, 2.333] | 3 / 8 |
| zinnia_noir | halo2_hw | verify_s | 11 | 1.312 | [0.974, 1.765] | 2 / 9 |
| zinnia_noir | halo2_lit | prove_core_s | 11 | 0.652 | [0.449, 0.946] | 8 / 3 |
| zinnia_noir | halo2_lit | end_to_end_s | 11 | 0.684 | [0.478, 0.983] | 8 / 3 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 11 | 1.264 | [0.847, 1.898] | 3 / 8 |
| zinnia_noir | halo2_lit | verify_s | 11 | 1.173 | [0.892, 1.529] | 4 / 7 |
| zinnia_noir | noir_lit | prove_core_s | 11 | 0.998 | [0.924, 1.072] | 4 / 7 |
| zinnia_noir | noir_lit | end_to_end_s | 11 | 1.001 | [0.928, 1.072] | 4 / 7 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 11 | 1.010 | [0.951, 1.092] | 6 / 5 |
| zinnia_noir | noir_lit | verify_s | 11 | 0.919 | [0.819, 1.041] | 7 / 4 |

## Copy-only cases (reported apart)

1 all-supported cases that only copy inputs: ds1000_case0389

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 1 | 1.052 | [1.052, 1.052] | 0 / 1 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 1 | 1.059 | [1.059, 1.059] | 0 / 1 |
| zinnia_halo2 | halo2_hw | verify_s | 1 | 0.847 | [0.847, 0.847] | 1 / 0 |
| zinnia_noir | noir_hw | prove_core_s | 1 | 0.830 | [0.830, 0.830] | 1 / 0 |
| zinnia_noir | noir_hw | end_to_end_s | 1 | 0.846 | [0.846, 0.846] | 1 / 0 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 1 | 0.862 | [0.862, 0.862] | 1 / 0 |
| zinnia_noir | noir_hw | verify_s | 1 | 0.977 | [0.977, 0.977] | 1 / 0 |
| zinnia_halo2 | noir_hw | prove_core_s | 1 | 0.803 | [0.803, 0.803] | 1 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_s | 1 | 0.795 | [0.795, 0.795] | 1 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 1 | 0.427 | [0.427, 0.427] | 1 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 1 | 0.546 | [0.546, 0.546] | 1 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 1 | 1.030 | [1.030, 1.030] | 0 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 1 | 1.037 | [1.037, 1.037] | 0 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 1 | 0.961 | [0.961, 0.961] | 1 / 0 |
| zinnia_halo2 | noir_lit | prove_core_s | 1 | 0.941 | [0.941, 0.941] | 1 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_s | 1 | 0.918 | [0.918, 0.918] | 1 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 1 | 0.465 | [0.465, 0.465] | 1 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 1 | 0.600 | [0.600, 0.600] | 1 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 1 | 1.088 | [1.088, 1.088] | 0 / 1 |
| zinnia_noir | halo2_hw | end_to_end_s | 1 | 1.128 | [1.128, 1.128] | 0 / 1 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 1 | 2.139 | [2.139, 2.139] | 0 / 1 |
| zinnia_noir | halo2_hw | verify_s | 1 | 1.517 | [1.517, 1.517] | 0 / 1 |
| zinnia_noir | halo2_lit | prove_core_s | 1 | 1.065 | [1.065, 1.065] | 0 / 1 |
| zinnia_noir | halo2_lit | end_to_end_s | 1 | 1.104 | [1.104, 1.104] | 0 / 1 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 1 | 2.094 | [2.094, 2.094] | 0 / 1 |
| zinnia_noir | halo2_lit | verify_s | 1 | 1.721 | [1.721, 1.721] | 0 / 1 |
| zinnia_noir | noir_lit | prove_core_s | 1 | 0.974 | [0.974, 0.974] | 1 / 0 |
| zinnia_noir | noir_lit | end_to_end_s | 1 | 0.977 | [0.977, 0.977] | 1 / 0 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 1 | 0.939 | [0.939, 0.939] | 1 / 0 |
| zinnia_noir | noir_lit | verify_s | 1 | 1.075 | [1.075, 1.075] | 0 / 1 |

## All-supported subset

12 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): ds1000_case0311, ds1000_case0322, ds1000_case0334, ds1000_case0353, ds1000_case0364, ds1000_case0389, ds1000_case0414, ds1000_case0429, ds1000_case0439, ds1000_case0453, ds1000_case0485, ds1000_case0503

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 12 | 0.969 | [0.881, 1.057] | 6 / 6 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 12 | 0.978 | [0.889, 1.069] | 6 / 6 |
| zinnia_halo2 | halo2_hw | verify_s | 12 | 0.954 | [0.850, 1.071] | 7 / 5 |
| zinnia_noir | noir_hw | prove_core_s | 12 | 1.008 | [0.933, 1.092] | 5 / 7 |
| zinnia_noir | noir_hw | end_to_end_s | 12 | 1.012 | [0.938, 1.095] | 5 / 7 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 12 | 1.014 | [0.950, 1.106] | 7 / 5 |
| zinnia_noir | noir_hw | verify_s | 12 | 0.915 | [0.833, 1.010] | 8 / 4 |
| zinnia_halo2 | noir_hw | prove_core_s | 12 | 1.182 | [0.831, 1.702] | 6 / 6 |
| zinnia_halo2 | noir_hw | end_to_end_s | 12 | 1.142 | [0.810, 1.625] | 6 / 6 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 12 | 0.618 | [0.414, 0.938] | 9 / 3 |
| zinnia_halo2 | noir_hw | verify_s | 12 | 0.657 | [0.518, 0.843] | 9 / 3 |
| zinnia_halo2 | halo2_lit | prove_core_s | 12 | 0.796 | [0.710, 0.889] | 10 / 2 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 12 | 0.803 | [0.716, 0.897] | 10 / 2 |
| zinnia_halo2 | halo2_lit | verify_s | 12 | 0.870 | [0.775, 0.970] | 11 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 12 | 1.167 | [0.856, 1.591] | 5 / 7 |
| zinnia_halo2 | noir_lit | end_to_end_s | 12 | 1.127 | [0.833, 1.520] | 5 / 7 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 12 | 0.612 | [0.419, 0.898] | 9 / 3 |
| zinnia_halo2 | noir_lit | verify_s | 12 | 0.669 | [0.541, 0.820] | 10 / 2 |
| zinnia_noir | halo2_hw | prove_core_s | 12 | 0.827 | [0.587, 1.153] | 7 / 5 |
| zinnia_noir | halo2_hw | end_to_end_s | 12 | 0.867 | [0.622, 1.196] | 7 / 5 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 12 | 1.605 | [1.113, 2.300] | 3 / 9 |
| zinnia_noir | halo2_hw | verify_s | 12 | 1.328 | [1.014, 1.745] | 2 / 10 |
| zinnia_noir | halo2_lit | prove_core_s | 12 | 0.679 | [0.477, 0.962] | 8 / 4 |
| zinnia_noir | halo2_lit | end_to_end_s | 12 | 0.712 | [0.506, 0.998] | 8 / 4 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 12 | 1.318 | [0.900, 1.925] | 3 / 9 |
| zinnia_noir | halo2_lit | verify_s | 12 | 1.211 | [0.936, 1.551] | 4 / 8 |
| zinnia_noir | noir_lit | prove_core_s | 12 | 0.996 | [0.929, 1.064] | 5 / 7 |
| zinnia_noir | noir_lit | end_to_end_s | 12 | 0.999 | [0.934, 1.064] | 5 / 7 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 12 | 1.004 | [0.950, 1.081] | 7 / 5 |
| zinnia_noir | noir_lit | verify_s | 12 | 0.931 | [0.834, 1.047] | 7 / 5 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 12 | 0.969 | [0.881, 1.057] | 6 / 6 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 12 | 0.978 | [0.889, 1.069] | 6 / 6 |
| zinnia_halo2 | halo2_hw | verify_s | 12 | 0.954 | [0.850, 1.071] | 7 / 5 |
| zinnia_noir | noir_hw | prove_core_s | 12 | 1.008 | [0.933, 1.092] | 5 / 7 |
| zinnia_noir | noir_hw | end_to_end_s | 12 | 1.012 | [0.938, 1.095] | 5 / 7 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 12 | 1.014 | [0.950, 1.106] | 7 / 5 |
| zinnia_noir | noir_hw | verify_s | 12 | 0.915 | [0.833, 1.010] | 8 / 4 |
| zinnia_halo2 | noir_hw | prove_core_s | 12 | 1.182 | [0.831, 1.702] | 6 / 6 |
| zinnia_halo2 | noir_hw | end_to_end_s | 12 | 1.142 | [0.810, 1.625] | 6 / 6 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 12 | 0.618 | [0.414, 0.938] | 9 / 3 |
| zinnia_halo2 | noir_hw | verify_s | 12 | 0.657 | [0.518, 0.843] | 9 / 3 |
| zinnia_halo2 | halo2_lit | prove_core_s | 12 | 0.796 | [0.710, 0.889] | 10 / 2 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 12 | 0.803 | [0.716, 0.897] | 10 / 2 |
| zinnia_halo2 | halo2_lit | verify_s | 12 | 0.870 | [0.775, 0.970] | 11 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 12 | 1.167 | [0.856, 1.591] | 5 / 7 |
| zinnia_halo2 | noir_lit | end_to_end_s | 12 | 1.127 | [0.833, 1.520] | 5 / 7 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 12 | 0.612 | [0.419, 0.898] | 9 / 3 |
| zinnia_halo2 | noir_lit | verify_s | 12 | 0.669 | [0.541, 0.820] | 10 / 2 |
| zinnia_noir | halo2_hw | prove_core_s | 12 | 0.827 | [0.587, 1.153] | 7 / 5 |
| zinnia_noir | halo2_hw | end_to_end_s | 12 | 0.867 | [0.622, 1.196] | 7 / 5 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 12 | 1.605 | [1.113, 2.300] | 3 / 9 |
| zinnia_noir | halo2_hw | verify_s | 12 | 1.328 | [1.014, 1.745] | 2 / 10 |
| zinnia_noir | halo2_lit | prove_core_s | 12 | 0.679 | [0.477, 0.962] | 8 / 4 |
| zinnia_noir | halo2_lit | end_to_end_s | 12 | 0.712 | [0.506, 0.998] | 8 / 4 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 12 | 1.318 | [0.900, 1.925] | 3 / 9 |
| zinnia_noir | halo2_lit | verify_s | 12 | 1.211 | [0.936, 1.551] | 4 / 8 |
| zinnia_noir | noir_lit | prove_core_s | 12 | 0.996 | [0.929, 1.064] | 5 / 7 |
| zinnia_noir | noir_lit | end_to_end_s | 12 | 0.999 | [0.934, 1.064] | 5 / 7 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 12 | 1.004 | [0.950, 1.081] | 7 / 5 |
| zinnia_noir | noir_lit | verify_s | 12 | 0.931 | [0.834, 1.047] | 7 / 5 |

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

