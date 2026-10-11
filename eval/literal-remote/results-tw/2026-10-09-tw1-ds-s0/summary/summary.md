# Epoch 2026-10-09-tw1-ds-s0

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

8 cases completed by every system that do more than copy inputs: ds1000_case0298, ds1000_case0325, ds1000_case0374, ds1000_case0399, ds1000_case0420, ds1000_case0434, ds1000_case0442, ds1000_case0507

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 8 | 0.938 | [0.864, 1.005] | 5 / 3 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 8 | 0.945 | [0.870, 1.012] | 5 / 3 |
| zinnia_halo2 | halo2_hw | verify_s | 8 | 0.929 | [0.827, 1.034] | 5 / 3 |
| zinnia_noir | noir_hw | prove_core_s | 8 | 1.043 | [0.974, 1.116] | 3 / 5 |
| zinnia_noir | noir_hw | end_to_end_s | 8 | 1.045 | [0.975, 1.119] | 3 / 5 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 8 | 0.999 | [0.954, 1.045] | 3 / 5 |
| zinnia_noir | noir_hw | verify_s | 8 | 0.985 | [0.843, 1.160] | 5 / 3 |
| zinnia_halo2 | noir_hw | prove_core_s | 8 | 0.824 | [0.678, 1.052] | 7 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_s | 8 | 0.808 | [0.664, 1.031] | 7 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 8 | 0.389 | [0.315, 0.510] | 8 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 8 | 0.534 | [0.453, 0.641] | 8 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 8 | 0.519 | [0.305, 0.816] | 7 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 8 | 0.523 | [0.308, 0.820] | 7 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 8 | 0.707 | [0.510, 0.905] | 7 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 8 | 0.624 | [0.559, 0.697] | 8 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_s | 8 | 0.604 | [0.532, 0.679] | 8 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 8 | 0.323 | [0.294, 0.359] | 8 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 8 | 0.526 | [0.474, 0.587] | 8 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 8 | 1.187 | [0.931, 1.445] | 1 / 7 |
| zinnia_noir | halo2_hw | end_to_end_s | 8 | 1.223 | [0.967, 1.478] | 1 / 7 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 8 | 2.430 | [1.917, 2.930] | 0 / 8 |
| zinnia_noir | halo2_hw | verify_s | 8 | 1.712 | [1.417, 2.083] | 0 / 8 |
| zinnia_noir | halo2_lit | prove_core_s | 8 | 0.657 | [0.305, 1.236] | 4 / 4 |
| zinnia_noir | halo2_lit | end_to_end_s | 8 | 0.677 | [0.316, 1.267] | 4 / 4 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 8 | 1.344 | [0.622, 2.502] | 2 / 6 |
| zinnia_noir | halo2_lit | verify_s | 8 | 1.302 | [0.863, 1.845] | 2 / 6 |
| zinnia_noir | noir_lit | prove_core_s | 8 | 0.790 | [0.582, 0.962] | 6 / 2 |
| zinnia_noir | noir_lit | end_to_end_s | 8 | 0.782 | [0.564, 0.964] | 6 / 2 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 8 | 0.831 | [0.670, 0.952] | 8 / 0 |
| zinnia_noir | noir_lit | verify_s | 8 | 0.968 | [0.922, 1.008] | 5 / 3 |

## Copy-only cases (reported apart)

5 all-supported cases that only copy inputs: ds1000_case0317, ds1000_case0338, ds1000_case0359, ds1000_case0471, ds1000_case0495

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 5 | 0.988 | [0.898, 1.092] | 3 / 2 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 5 | 1.007 | [0.925, 1.106] | 3 / 2 |
| zinnia_halo2 | halo2_hw | verify_s | 5 | 1.059 | [0.986, 1.188] | 2 / 3 |
| zinnia_noir | noir_hw | prove_core_s | 5 | 0.997 | [0.956, 1.033] | 3 / 2 |
| zinnia_noir | noir_hw | end_to_end_s | 5 | 1.000 | [0.947, 1.047] | 2 / 3 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 5 | 1.017 | [0.951, 1.093] | 2 / 3 |
| zinnia_noir | noir_hw | verify_s | 5 | 1.170 | [1.011, 1.366] | 1 / 4 |
| zinnia_halo2 | noir_hw | prove_core_s | 5 | 1.251 | [0.613, 2.357] | 2 / 3 |
| zinnia_halo2 | noir_hw | end_to_end_s | 5 | 1.181 | [0.594, 2.170] | 2 / 3 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 5 | 0.656 | [0.295, 1.366] | 2 / 3 |
| zinnia_halo2 | noir_hw | verify_s | 5 | 0.830 | [0.577, 1.196] | 3 / 2 |
| zinnia_halo2 | halo2_lit | prove_core_s | 5 | 0.970 | [0.884, 1.063] | 3 / 2 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 5 | 0.988 | [0.905, 1.080] | 3 / 2 |
| zinnia_halo2 | halo2_lit | verify_s | 5 | 1.053 | [0.972, 1.142] | 2 / 3 |
| zinnia_halo2 | noir_lit | prove_core_s | 5 | 1.218 | [0.655, 2.248] | 2 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_s | 5 | 1.151 | [0.622, 2.051] | 2 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 5 | 0.648 | [0.303, 1.312] | 2 / 3 |
| zinnia_halo2 | noir_lit | verify_s | 5 | 0.775 | [0.472, 1.156] | 3 / 2 |
| zinnia_noir | halo2_hw | prove_core_s | 5 | 0.788 | [0.391, 1.585] | 3 / 2 |
| zinnia_noir | halo2_hw | end_to_end_s | 5 | 0.853 | [0.445, 1.635] | 3 / 2 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 5 | 1.561 | [0.745, 3.291] | 3 / 2 |
| zinnia_noir | halo2_hw | verify_s | 5 | 1.492 | [0.946, 2.409] | 2 / 3 |
| zinnia_noir | halo2_lit | prove_core_s | 5 | 0.773 | [0.401, 1.488] | 3 / 2 |
| zinnia_noir | halo2_lit | end_to_end_s | 5 | 0.837 | [0.457, 1.536] | 3 / 2 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 5 | 1.532 | [0.765, 3.071] | 3 / 2 |
| zinnia_noir | halo2_lit | verify_s | 5 | 1.484 | [0.968, 2.353] | 1 / 4 |
| zinnia_noir | noir_lit | prove_core_s | 5 | 0.970 | [0.918, 1.021] | 3 / 2 |
| zinnia_noir | noir_lit | end_to_end_s | 5 | 0.975 | [0.930, 1.017] | 3 / 2 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 5 | 1.004 | [0.950, 1.087] | 3 / 2 |
| zinnia_noir | noir_lit | verify_s | 5 | 1.092 | [0.854, 1.369] | 1 / 4 |

## All-supported subset

13 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): ds1000_case0298, ds1000_case0317, ds1000_case0325, ds1000_case0338, ds1000_case0359, ds1000_case0374, ds1000_case0399, ds1000_case0420, ds1000_case0434, ds1000_case0442, ds1000_case0471, ds1000_case0495, ds1000_case0507

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 13 | 0.957 | [0.897, 1.018] | 8 / 5 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 13 | 0.969 | [0.909, 1.028] | 8 / 5 |
| zinnia_halo2 | halo2_hw | verify_s | 13 | 0.977 | [0.896, 1.064] | 7 / 6 |
| zinnia_noir | noir_hw | prove_core_s | 13 | 1.025 | [0.980, 1.074] | 6 / 7 |
| zinnia_noir | noir_hw | end_to_end_s | 13 | 1.027 | [0.980, 1.078] | 5 / 8 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 13 | 1.006 | [0.968, 1.048] | 5 / 8 |
| zinnia_noir | noir_hw | verify_s | 13 | 1.052 | [0.930, 1.194] | 6 / 7 |
| zinnia_halo2 | noir_hw | prove_core_s | 13 | 0.968 | [0.720, 1.346] | 9 / 4 |
| zinnia_halo2 | noir_hw | end_to_end_s | 13 | 0.935 | [0.702, 1.283] | 9 / 4 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 13 | 0.475 | [0.339, 0.696] | 10 / 3 |
| zinnia_halo2 | noir_hw | verify_s | 13 | 0.633 | [0.516, 0.794] | 11 / 2 |
| zinnia_halo2 | halo2_lit | prove_core_s | 13 | 0.660 | [0.449, 0.901] | 10 / 3 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 13 | 0.668 | [0.454, 0.911] | 10 / 3 |
| zinnia_halo2 | halo2_lit | verify_s | 13 | 0.824 | [0.648, 0.985] | 9 / 4 |
| zinnia_halo2 | noir_lit | prove_core_s | 13 | 0.807 | [0.611, 1.136] | 10 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_s | 13 | 0.774 | [0.592, 1.068] | 10 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 13 | 0.422 | [0.309, 0.614] | 10 / 3 |
| zinnia_halo2 | noir_lit | verify_s | 13 | 0.610 | [0.499, 0.762] | 11 / 2 |
| zinnia_noir | halo2_hw | prove_core_s | 13 | 1.013 | [0.717, 1.378] | 4 / 9 |
| zinnia_noir | halo2_hw | end_to_end_s | 13 | 1.065 | [0.773, 1.424] | 4 / 9 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 13 | 2.050 | [1.430, 2.840] | 3 / 10 |
| zinnia_noir | halo2_hw | verify_s | 13 | 1.624 | [1.301, 2.014] | 2 / 11 |
| zinnia_noir | halo2_lit | prove_core_s | 13 | 0.699 | [0.409, 1.126] | 7 / 6 |
| zinnia_noir | halo2_lit | end_to_end_s | 13 | 0.734 | [0.435, 1.167] | 7 / 6 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 13 | 1.414 | [0.828, 2.298] | 5 / 8 |
| zinnia_noir | halo2_lit | verify_s | 13 | 1.369 | [1.013, 1.812] | 3 / 10 |
| zinnia_noir | noir_lit | prove_core_s | 13 | 0.855 | [0.698, 0.974] | 9 / 4 |
| zinnia_noir | noir_lit | end_to_end_s | 13 | 0.851 | [0.686, 0.975] | 9 / 4 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 13 | 0.894 | [0.770, 0.990] | 11 / 2 |
| zinnia_noir | noir_lit | verify_s | 13 | 1.014 | [0.914, 1.130] | 6 / 7 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 13 | 0.957 | [0.897, 1.018] | 8 / 5 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 13 | 0.969 | [0.909, 1.028] | 8 / 5 |
| zinnia_halo2 | halo2_hw | verify_s | 13 | 0.977 | [0.896, 1.064] | 7 / 6 |
| zinnia_noir | noir_hw | prove_core_s | 13 | 1.025 | [0.980, 1.074] | 6 / 7 |
| zinnia_noir | noir_hw | end_to_end_s | 13 | 1.027 | [0.980, 1.078] | 5 / 8 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 13 | 1.006 | [0.968, 1.048] | 5 / 8 |
| zinnia_noir | noir_hw | verify_s | 13 | 1.052 | [0.930, 1.194] | 6 / 7 |
| zinnia_halo2 | noir_hw | prove_core_s | 13 | 0.968 | [0.720, 1.346] | 9 / 4 |
| zinnia_halo2 | noir_hw | end_to_end_s | 13 | 0.935 | [0.702, 1.283] | 9 / 4 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 13 | 0.475 | [0.339, 0.696] | 10 / 3 |
| zinnia_halo2 | noir_hw | verify_s | 13 | 0.633 | [0.516, 0.794] | 11 / 2 |
| zinnia_halo2 | halo2_lit | prove_core_s | 13 | 0.660 | [0.449, 0.901] | 10 / 3 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 13 | 0.668 | [0.454, 0.911] | 10 / 3 |
| zinnia_halo2 | halo2_lit | verify_s | 13 | 0.824 | [0.648, 0.985] | 9 / 4 |
| zinnia_halo2 | noir_lit | prove_core_s | 13 | 0.807 | [0.611, 1.136] | 10 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_s | 13 | 0.774 | [0.592, 1.068] | 10 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 13 | 0.422 | [0.309, 0.614] | 10 / 3 |
| zinnia_halo2 | noir_lit | verify_s | 13 | 0.610 | [0.499, 0.762] | 11 / 2 |
| zinnia_noir | halo2_hw | prove_core_s | 13 | 1.013 | [0.717, 1.378] | 4 / 9 |
| zinnia_noir | halo2_hw | end_to_end_s | 13 | 1.065 | [0.773, 1.424] | 4 / 9 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 13 | 2.050 | [1.430, 2.840] | 3 / 10 |
| zinnia_noir | halo2_hw | verify_s | 13 | 1.624 | [1.301, 2.014] | 2 / 11 |
| zinnia_noir | halo2_lit | prove_core_s | 13 | 0.699 | [0.409, 1.126] | 7 / 6 |
| zinnia_noir | halo2_lit | end_to_end_s | 13 | 0.734 | [0.435, 1.167] | 7 / 6 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 13 | 1.414 | [0.828, 2.298] | 5 / 8 |
| zinnia_noir | halo2_lit | verify_s | 13 | 1.369 | [1.013, 1.812] | 3 / 10 |
| zinnia_noir | noir_lit | prove_core_s | 13 | 0.855 | [0.698, 0.974] | 9 / 4 |
| zinnia_noir | noir_lit | end_to_end_s | 13 | 0.851 | [0.686, 0.975] | 9 / 4 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 13 | 0.894 | [0.770, 0.990] | 11 / 2 |
| zinnia_noir | noir_lit | verify_s | 13 | 1.014 | [0.914, 1.130] | 6 / 7 |

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

