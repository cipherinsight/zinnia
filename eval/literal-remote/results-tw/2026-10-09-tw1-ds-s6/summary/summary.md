# Epoch 2026-10-09-tw1-ds-s6

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

11 cases completed by every system that do more than copy inputs: ds1000_case0312, ds1000_case0323, ds1000_case0335, ds1000_case0355, ds1000_case0368, ds1000_case0415, ds1000_case0431, ds1000_case0440, ds1000_case0454, ds1000_case0490, ds1000_case0504

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 11 | 0.877 | [0.825, 0.926] | 11 / 0 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 11 | 0.884 | [0.832, 0.934] | 10 / 1 |
| zinnia_halo2 | halo2_hw | verify_s | 11 | 0.862 | [0.809, 0.916] | 11 / 0 |
| zinnia_noir | noir_hw | prove_core_s | 11 | 0.974 | [0.931, 1.018] | 6 / 5 |
| zinnia_noir | noir_hw | end_to_end_s | 11 | 0.976 | [0.933, 1.021] | 5 / 6 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 11 | 0.991 | [0.958, 1.025] | 7 / 4 |
| zinnia_noir | noir_hw | verify_s | 11 | 1.000 | [0.878, 1.133] | 5 / 6 |
| zinnia_halo2 | noir_hw | prove_core_s | 11 | 1.196 | [0.908, 1.603] | 4 / 7 |
| zinnia_halo2 | noir_hw | end_to_end_s | 11 | 1.147 | [0.873, 1.526] | 5 / 6 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 11 | 0.570 | [0.410, 0.811] | 9 / 2 |
| zinnia_halo2 | noir_hw | verify_s | 11 | 0.587 | [0.480, 0.727] | 10 / 1 |
| zinnia_halo2 | halo2_lit | prove_core_s | 11 | 0.557 | [0.307, 0.779] | 11 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 11 | 0.561 | [0.310, 0.784] | 11 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 11 | 0.636 | [0.413, 0.810] | 11 / 0 |
| zinnia_halo2 | noir_lit | prove_core_s | 11 | 0.902 | [0.559, 1.320] | 6 / 5 |
| zinnia_halo2 | noir_lit | end_to_end_s | 11 | 0.858 | [0.514, 1.276] | 6 / 5 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 11 | 0.441 | [0.289, 0.653] | 10 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 11 | 0.642 | [0.498, 0.829] | 8 / 3 |
| zinnia_noir | halo2_hw | prove_core_s | 11 | 0.714 | [0.524, 0.949] | 7 / 4 |
| zinnia_noir | halo2_hw | end_to_end_s | 11 | 0.753 | [0.557, 0.993] | 7 / 4 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 11 | 1.537 | [1.093, 2.108] | 2 / 9 |
| zinnia_noir | halo2_hw | verify_s | 11 | 1.468 | [1.180, 1.811] | 2 / 9 |
| zinnia_noir | halo2_lit | prove_core_s | 11 | 0.453 | [0.199, 0.780] | 9 / 2 |
| zinnia_noir | halo2_lit | end_to_end_s | 11 | 0.478 | [0.211, 0.816] | 9 / 2 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 11 | 0.975 | [0.417, 1.727] | 3 / 8 |
| zinnia_noir | halo2_lit | verify_s | 11 | 1.084 | [0.623, 1.605] | 4 / 7 |
| zinnia_noir | noir_lit | prove_core_s | 11 | 0.734 | [0.420, 0.988] | 8 / 3 |
| zinnia_noir | noir_lit | end_to_end_s | 11 | 0.730 | [0.405, 1.000] | 9 / 2 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 11 | 0.766 | [0.465, 1.002] | 9 / 2 |
| zinnia_noir | noir_lit | verify_s | 11 | 1.094 | [0.928, 1.283] | 5 / 6 |

## Copy-only cases (reported apart)

1 all-supported cases that only copy inputs: ds1000_case0391

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 1 | 1.061 | [1.061, 1.061] | 0 / 1 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 1 | 1.068 | [1.068, 1.068] | 0 / 1 |
| zinnia_halo2 | halo2_hw | verify_s | 1 | 1.109 | [1.109, 1.109] | 0 / 1 |
| zinnia_noir | noir_hw | prove_core_s | 1 | 1.079 | [1.079, 1.079] | 0 / 1 |
| zinnia_noir | noir_hw | end_to_end_s | 1 | 1.068 | [1.068, 1.068] | 0 / 1 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 1 | 0.984 | [0.984, 0.984] | 1 / 0 |
| zinnia_noir | noir_hw | verify_s | 1 | 1.366 | [1.366, 1.366] | 0 / 1 |
| zinnia_halo2 | noir_hw | prove_core_s | 1 | 0.998 | [0.998, 0.998] | 1 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_s | 1 | 0.959 | [0.959, 0.959] | 1 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 1 | 0.436 | [0.436, 0.436] | 1 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 1 | 0.982 | [0.982, 0.982] | 1 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 1 | 0.976 | [0.976, 0.976] | 1 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 1 | 0.983 | [0.983, 0.983] | 1 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 1 | 1.008 | [1.008, 1.008] | 0 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 1 | 0.954 | [0.954, 0.954] | 1 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_s | 1 | 0.922 | [0.922, 0.922] | 1 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 1 | 0.450 | [0.450, 0.450] | 1 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 1 | 0.671 | [0.671, 0.671] | 1 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 1 | 1.146 | [1.146, 1.146] | 0 / 1 |
| zinnia_noir | halo2_hw | end_to_end_s | 1 | 1.189 | [1.189, 1.189] | 0 / 1 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 1 | 2.410 | [2.410, 2.410] | 0 / 1 |
| zinnia_noir | halo2_hw | verify_s | 1 | 1.543 | [1.543, 1.543] | 0 / 1 |
| zinnia_noir | halo2_lit | prove_core_s | 1 | 1.055 | [1.055, 1.055] | 0 / 1 |
| zinnia_noir | halo2_lit | end_to_end_s | 1 | 1.095 | [1.095, 1.095] | 0 / 1 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 1 | 2.219 | [2.219, 2.219] | 0 / 1 |
| zinnia_noir | halo2_lit | verify_s | 1 | 1.403 | [1.403, 1.403] | 0 / 1 |
| zinnia_noir | noir_lit | prove_core_s | 1 | 1.031 | [1.031, 1.031] | 0 / 1 |
| zinnia_noir | noir_lit | end_to_end_s | 1 | 1.027 | [1.027, 1.027] | 0 / 1 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 1 | 1.015 | [1.015, 1.015] | 0 / 1 |
| zinnia_noir | noir_lit | verify_s | 1 | 0.934 | [0.934, 0.934] | 1 / 0 |

## All-supported subset

12 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): ds1000_case0312, ds1000_case0323, ds1000_case0335, ds1000_case0355, ds1000_case0368, ds1000_case0391, ds1000_case0415, ds1000_case0431, ds1000_case0440, ds1000_case0454, ds1000_case0490, ds1000_case0504

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 12 | 0.891 | [0.837, 0.945] | 11 / 1 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 12 | 0.898 | [0.844, 0.952] | 10 / 2 |
| zinnia_halo2 | halo2_hw | verify_s | 12 | 0.880 | [0.821, 0.944] | 11 / 1 |
| zinnia_noir | noir_hw | prove_core_s | 12 | 0.982 | [0.939, 1.026] | 6 / 6 |
| zinnia_noir | noir_hw | end_to_end_s | 12 | 0.983 | [0.940, 1.027] | 5 / 7 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 12 | 0.990 | [0.961, 1.022] | 8 / 4 |
| zinnia_noir | noir_hw | verify_s | 12 | 1.026 | [0.903, 1.159] | 5 / 7 |
| zinnia_halo2 | noir_hw | prove_core_s | 12 | 1.179 | [0.910, 1.542] | 5 / 7 |
| zinnia_halo2 | noir_hw | end_to_end_s | 12 | 1.130 | [0.874, 1.470] | 6 / 6 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 12 | 0.557 | [0.410, 0.772] | 10 / 2 |
| zinnia_halo2 | noir_hw | verify_s | 12 | 0.613 | [0.502, 0.752] | 11 / 1 |
| zinnia_halo2 | halo2_lit | prove_core_s | 12 | 0.583 | [0.336, 0.805] | 12 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 12 | 0.588 | [0.339, 0.810] | 12 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 12 | 0.661 | [0.440, 0.838] | 11 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 12 | 0.906 | [0.592, 1.280] | 7 / 5 |
| zinnia_halo2 | noir_lit | end_to_end_s | 12 | 0.863 | [0.546, 1.238] | 7 / 5 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 12 | 0.441 | [0.303, 0.631] | 11 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 12 | 0.645 | [0.508, 0.812] | 9 / 3 |
| zinnia_noir | halo2_hw | prove_core_s | 12 | 0.742 | [0.552, 0.976] | 7 / 5 |
| zinnia_noir | halo2_hw | end_to_end_s | 12 | 0.782 | [0.587, 1.021] | 7 / 5 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 12 | 1.595 | [1.159, 2.157] | 2 / 10 |
| zinnia_noir | halo2_hw | verify_s | 12 | 1.474 | [1.206, 1.782] | 2 / 10 |
| zinnia_noir | halo2_lit | prove_core_s | 12 | 0.486 | [0.225, 0.815] | 9 / 3 |
| zinnia_noir | halo2_lit | end_to_end_s | 12 | 0.512 | [0.238, 0.852] | 9 / 3 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 12 | 1.044 | [0.472, 1.800] | 3 / 9 |
| zinnia_noir | halo2_lit | verify_s | 12 | 1.107 | [0.661, 1.591] | 4 / 8 |
| zinnia_noir | noir_lit | prove_core_s | 12 | 0.755 | [0.453, 0.993] | 8 / 4 |
| zinnia_noir | noir_lit | end_to_end_s | 12 | 0.751 | [0.438, 1.002] | 9 / 3 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 12 | 0.784 | [0.497, 1.004] | 9 / 3 |
| zinnia_noir | noir_lit | verify_s | 12 | 1.080 | [0.929, 1.248] | 6 / 6 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 12 | 0.891 | [0.837, 0.945] | 11 / 1 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 12 | 0.898 | [0.844, 0.952] | 10 / 2 |
| zinnia_halo2 | halo2_hw | verify_s | 12 | 0.880 | [0.821, 0.944] | 11 / 1 |
| zinnia_noir | noir_hw | prove_core_s | 12 | 0.982 | [0.939, 1.026] | 6 / 6 |
| zinnia_noir | noir_hw | end_to_end_s | 12 | 0.983 | [0.940, 1.027] | 5 / 7 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 12 | 0.990 | [0.961, 1.022] | 8 / 4 |
| zinnia_noir | noir_hw | verify_s | 12 | 1.026 | [0.903, 1.159] | 5 / 7 |
| zinnia_halo2 | noir_hw | prove_core_s | 12 | 1.179 | [0.910, 1.542] | 5 / 7 |
| zinnia_halo2 | noir_hw | end_to_end_s | 12 | 1.130 | [0.874, 1.470] | 6 / 6 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 12 | 0.557 | [0.410, 0.772] | 10 / 2 |
| zinnia_halo2 | noir_hw | verify_s | 12 | 0.613 | [0.502, 0.752] | 11 / 1 |
| zinnia_halo2 | halo2_lit | prove_core_s | 12 | 0.583 | [0.336, 0.805] | 12 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 12 | 0.588 | [0.339, 0.810] | 12 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 12 | 0.661 | [0.440, 0.838] | 11 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 12 | 0.906 | [0.592, 1.280] | 7 / 5 |
| zinnia_halo2 | noir_lit | end_to_end_s | 12 | 0.863 | [0.546, 1.238] | 7 / 5 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 12 | 0.441 | [0.303, 0.631] | 11 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 12 | 0.645 | [0.508, 0.812] | 9 / 3 |
| zinnia_noir | halo2_hw | prove_core_s | 12 | 0.742 | [0.552, 0.976] | 7 / 5 |
| zinnia_noir | halo2_hw | end_to_end_s | 12 | 0.782 | [0.587, 1.021] | 7 / 5 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 12 | 1.595 | [1.159, 2.157] | 2 / 10 |
| zinnia_noir | halo2_hw | verify_s | 12 | 1.474 | [1.206, 1.782] | 2 / 10 |
| zinnia_noir | halo2_lit | prove_core_s | 12 | 0.486 | [0.225, 0.815] | 9 / 3 |
| zinnia_noir | halo2_lit | end_to_end_s | 12 | 0.512 | [0.238, 0.852] | 9 / 3 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 12 | 1.044 | [0.472, 1.800] | 3 / 9 |
| zinnia_noir | halo2_lit | verify_s | 12 | 1.107 | [0.661, 1.591] | 4 / 8 |
| zinnia_noir | noir_lit | prove_core_s | 12 | 0.755 | [0.453, 0.993] | 8 / 4 |
| zinnia_noir | noir_lit | end_to_end_s | 12 | 0.751 | [0.438, 1.002] | 9 / 3 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 12 | 0.784 | [0.497, 1.004] | 9 / 3 |
| zinnia_noir | noir_lit | verify_s | 12 | 1.080 | [0.929, 1.248] | 6 / 6 |

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

