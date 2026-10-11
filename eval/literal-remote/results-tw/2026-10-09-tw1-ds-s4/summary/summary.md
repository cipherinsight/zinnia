# Epoch 2026-10-09-tw1-ds-s4

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

11 cases completed by every system that do more than copy inputs: ds1000_case0310, ds1000_case0321, ds1000_case0329, ds1000_case0347, ds1000_case0363, ds1000_case0412, ds1000_case0428, ds1000_case0438, ds1000_case0452, ds1000_case0481, ds1000_case0501

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 11 | 0.909 | [0.836, 0.968] | 10 / 1 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 11 | 0.916 | [0.843, 0.975] | 10 / 1 |
| zinnia_halo2 | halo2_hw | verify_s | 11 | 0.921 | [0.825, 1.026] | 8 / 3 |
| zinnia_noir | noir_hw | prove_core_s | 11 | 0.986 | [0.942, 1.033] | 6 / 5 |
| zinnia_noir | noir_hw | end_to_end_s | 11 | 0.988 | [0.945, 1.034] | 7 / 4 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 11 | 0.971 | [0.942, 1.005] | 9 / 2 |
| zinnia_noir | noir_hw | verify_s | 11 | 0.982 | [0.858, 1.121] | 6 / 5 |
| zinnia_halo2 | noir_hw | prove_core_s | 11 | 0.959 | [0.776, 1.183] | 5 / 6 |
| zinnia_halo2 | noir_hw | end_to_end_s | 11 | 0.935 | [0.760, 1.148] | 6 / 5 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 11 | 0.473 | [0.377, 0.597] | 11 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 11 | 0.635 | [0.526, 0.761] | 10 / 1 |
| zinnia_halo2 | halo2_lit | prove_core_s | 11 | 0.700 | [0.611, 0.794] | 11 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 11 | 0.705 | [0.616, 0.799] | 11 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 11 | 0.832 | [0.761, 0.911] | 10 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 11 | 0.915 | [0.733, 1.150] | 8 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_s | 11 | 0.890 | [0.714, 1.114] | 8 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 11 | 0.463 | [0.366, 0.590] | 11 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 11 | 0.646 | [0.548, 0.770] | 10 / 1 |
| zinnia_noir | halo2_hw | prove_core_s | 11 | 0.935 | [0.715, 1.183] | 6 / 5 |
| zinnia_noir | halo2_hw | end_to_end_s | 11 | 0.967 | [0.742, 1.220] | 6 / 5 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 11 | 1.881 | [1.416, 2.416] | 1 / 10 |
| zinnia_noir | halo2_hw | verify_s | 11 | 1.424 | [1.190, 1.707] | 1 / 10 |
| zinnia_noir | halo2_lit | prove_core_s | 11 | 0.720 | [0.577, 0.883] | 8 / 3 |
| zinnia_noir | halo2_lit | end_to_end_s | 11 | 0.745 | [0.599, 0.909] | 8 / 3 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 11 | 1.449 | [1.146, 1.795] | 2 / 9 |
| zinnia_noir | halo2_lit | verify_s | 11 | 1.287 | [1.132, 1.478] | 1 / 10 |
| zinnia_noir | noir_lit | prove_core_s | 11 | 0.941 | [0.902, 0.981] | 8 / 3 |
| zinnia_noir | noir_lit | end_to_end_s | 11 | 0.940 | [0.903, 0.980] | 9 / 2 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 11 | 0.951 | [0.933, 0.971] | 10 / 1 |
| zinnia_noir | noir_lit | verify_s | 11 | 1.000 | [0.896, 1.132] | 6 / 5 |

## Copy-only cases (reported apart)

1 all-supported cases that only copy inputs: ds1000_case0385

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 1 | 0.978 | [0.978, 0.978] | 1 / 0 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 1 | 1.028 | [1.028, 1.028] | 0 / 1 |
| zinnia_halo2 | halo2_hw | verify_s | 1 | 0.898 | [0.898, 0.898] | 1 / 0 |
| zinnia_noir | noir_hw | prove_core_s | 1 | 1.094 | [1.094, 1.094] | 0 / 1 |
| zinnia_noir | noir_hw | end_to_end_s | 1 | 1.088 | [1.088, 1.088] | 0 / 1 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 1 | 1.603 | [1.603, 1.603] | 0 / 1 |
| zinnia_noir | noir_hw | verify_s | 1 | 0.989 | [0.989, 0.989] | 1 / 0 |
| zinnia_halo2 | noir_hw | prove_core_s | 1 | 5.364 | [5.364, 5.364] | 0 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_s | 1 | 4.554 | [4.554, 4.554] | 0 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 1 | 4.080 | [4.080, 4.080] | 0 / 1 |
| zinnia_halo2 | noir_hw | verify_s | 1 | 1.097 | [1.097, 1.097] | 0 / 1 |
| zinnia_halo2 | halo2_lit | prove_core_s | 1 | 1.004 | [1.004, 1.004] | 0 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 1 | 1.055 | [1.055, 1.055] | 0 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 1 | 1.061 | [1.061, 1.061] | 0 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 1 | 5.439 | [5.439, 5.439] | 0 / 1 |
| zinnia_halo2 | noir_lit | end_to_end_s | 1 | 4.601 | [4.601, 4.601] | 0 / 1 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 1 | 4.056 | [4.056, 4.056] | 0 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 1 | 1.157 | [1.157, 1.157] | 0 / 1 |
| zinnia_noir | halo2_hw | prove_core_s | 1 | 0.199 | [0.199, 0.199] | 1 / 0 |
| zinnia_noir | halo2_hw | end_to_end_s | 1 | 0.246 | [0.246, 0.246] | 1 / 0 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 1 | 0.404 | [0.404, 0.404] | 1 / 0 |
| zinnia_noir | halo2_hw | verify_s | 1 | 0.809 | [0.809, 0.809] | 1 / 0 |
| zinnia_noir | halo2_lit | prove_core_s | 1 | 0.205 | [0.205, 0.205] | 1 / 0 |
| zinnia_noir | halo2_lit | end_to_end_s | 1 | 0.252 | [0.252, 0.252] | 1 / 0 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 1 | 0.414 | [0.414, 0.414] | 1 / 0 |
| zinnia_noir | halo2_lit | verify_s | 1 | 0.956 | [0.956, 0.956] | 1 / 0 |
| zinnia_noir | noir_lit | prove_core_s | 1 | 1.110 | [1.110, 1.110] | 0 / 1 |
| zinnia_noir | noir_lit | end_to_end_s | 1 | 1.099 | [1.099, 1.099] | 0 / 1 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 1 | 1.593 | [1.593, 1.593] | 0 / 1 |
| zinnia_noir | noir_lit | verify_s | 1 | 1.044 | [1.044, 1.044] | 0 / 1 |

## All-supported subset

12 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): ds1000_case0310, ds1000_case0321, ds1000_case0329, ds1000_case0347, ds1000_case0363, ds1000_case0385, ds1000_case0412, ds1000_case0428, ds1000_case0438, ds1000_case0452, ds1000_case0481, ds1000_case0501

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 12 | 0.914 | [0.846, 0.969] | 11 / 1 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 12 | 0.925 | [0.855, 0.981] | 10 / 2 |
| zinnia_halo2 | halo2_hw | verify_s | 12 | 0.919 | [0.831, 1.014] | 9 / 3 |
| zinnia_noir | noir_hw | prove_core_s | 12 | 0.995 | [0.950, 1.042] | 6 / 6 |
| zinnia_noir | noir_hw | end_to_end_s | 12 | 0.996 | [0.953, 1.042] | 7 / 5 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 12 | 1.013 | [0.950, 1.113] | 9 / 3 |
| zinnia_noir | noir_hw | verify_s | 12 | 0.983 | [0.869, 1.110] | 7 / 5 |
| zinnia_halo2 | noir_hw | prove_core_s | 12 | 1.107 | [0.828, 1.584] | 5 / 7 |
| zinnia_halo2 | noir_hw | end_to_end_s | 12 | 1.067 | [0.810, 1.488] | 6 / 6 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 12 | 0.566 | [0.403, 0.876] | 11 / 1 |
| zinnia_halo2 | noir_hw | verify_s | 12 | 0.665 | [0.548, 0.797] | 10 / 2 |
| zinnia_halo2 | halo2_lit | prove_core_s | 12 | 0.722 | [0.627, 0.822] | 11 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 12 | 0.729 | [0.633, 0.834] | 11 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 12 | 0.849 | [0.774, 0.929] | 10 / 2 |
| zinnia_halo2 | noir_lit | prove_core_s | 12 | 1.061 | [0.779, 1.550] | 8 / 4 |
| zinnia_halo2 | noir_lit | end_to_end_s | 12 | 1.021 | [0.760, 1.456] | 8 / 4 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 12 | 0.555 | [0.392, 0.866] | 11 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 12 | 0.678 | [0.568, 0.813] | 10 / 2 |
| zinnia_noir | halo2_hw | prove_core_s | 12 | 0.822 | [0.576, 1.112] | 7 / 5 |
| zinnia_noir | halo2_hw | end_to_end_s | 12 | 0.863 | [0.622, 1.149] | 7 / 5 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 12 | 1.655 | [1.153, 2.262] | 2 / 10 |
| zinnia_noir | halo2_hw | verify_s | 12 | 1.359 | [1.135, 1.638] | 2 / 10 |
| zinnia_noir | halo2_lit | prove_core_s | 12 | 0.648 | [0.485, 0.839] | 9 / 3 |
| zinnia_noir | halo2_lit | end_to_end_s | 12 | 0.681 | [0.522, 0.867] | 9 / 3 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 12 | 1.305 | [0.971, 1.705] | 3 / 9 |
| zinnia_noir | halo2_lit | verify_s | 12 | 1.256 | [1.111, 1.438] | 2 / 10 |
| zinnia_noir | noir_lit | prove_core_s | 12 | 0.954 | [0.912, 1.000] | 8 / 4 |
| zinnia_noir | noir_lit | end_to_end_s | 12 | 0.953 | [0.912, 0.996] | 9 / 3 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 12 | 0.993 | [0.937, 1.090] | 10 / 2 |
| zinnia_noir | noir_lit | verify_s | 12 | 1.003 | [0.907, 1.124] | 6 / 6 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 12 | 0.914 | [0.846, 0.969] | 11 / 1 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 12 | 0.925 | [0.855, 0.981] | 10 / 2 |
| zinnia_halo2 | halo2_hw | verify_s | 12 | 0.919 | [0.831, 1.014] | 9 / 3 |
| zinnia_noir | noir_hw | prove_core_s | 12 | 0.995 | [0.950, 1.042] | 6 / 6 |
| zinnia_noir | noir_hw | end_to_end_s | 12 | 0.996 | [0.953, 1.042] | 7 / 5 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 12 | 1.013 | [0.950, 1.113] | 9 / 3 |
| zinnia_noir | noir_hw | verify_s | 12 | 0.983 | [0.869, 1.110] | 7 / 5 |
| zinnia_halo2 | noir_hw | prove_core_s | 12 | 1.107 | [0.828, 1.584] | 5 / 7 |
| zinnia_halo2 | noir_hw | end_to_end_s | 12 | 1.067 | [0.810, 1.488] | 6 / 6 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 12 | 0.566 | [0.403, 0.876] | 11 / 1 |
| zinnia_halo2 | noir_hw | verify_s | 12 | 0.665 | [0.548, 0.797] | 10 / 2 |
| zinnia_halo2 | halo2_lit | prove_core_s | 12 | 0.722 | [0.627, 0.822] | 11 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 12 | 0.729 | [0.633, 0.834] | 11 / 1 |
| zinnia_halo2 | halo2_lit | verify_s | 12 | 0.849 | [0.774, 0.929] | 10 / 2 |
| zinnia_halo2 | noir_lit | prove_core_s | 12 | 1.061 | [0.779, 1.550] | 8 / 4 |
| zinnia_halo2 | noir_lit | end_to_end_s | 12 | 1.021 | [0.760, 1.456] | 8 / 4 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 12 | 0.555 | [0.392, 0.866] | 11 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 12 | 0.678 | [0.568, 0.813] | 10 / 2 |
| zinnia_noir | halo2_hw | prove_core_s | 12 | 0.822 | [0.576, 1.112] | 7 / 5 |
| zinnia_noir | halo2_hw | end_to_end_s | 12 | 0.863 | [0.622, 1.149] | 7 / 5 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 12 | 1.655 | [1.153, 2.262] | 2 / 10 |
| zinnia_noir | halo2_hw | verify_s | 12 | 1.359 | [1.135, 1.638] | 2 / 10 |
| zinnia_noir | halo2_lit | prove_core_s | 12 | 0.648 | [0.485, 0.839] | 9 / 3 |
| zinnia_noir | halo2_lit | end_to_end_s | 12 | 0.681 | [0.522, 0.867] | 9 / 3 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 12 | 1.305 | [0.971, 1.705] | 3 / 9 |
| zinnia_noir | halo2_lit | verify_s | 12 | 1.256 | [1.111, 1.438] | 2 / 10 |
| zinnia_noir | noir_lit | prove_core_s | 12 | 0.954 | [0.912, 1.000] | 8 / 4 |
| zinnia_noir | noir_lit | end_to_end_s | 12 | 0.953 | [0.912, 0.996] | 9 / 3 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 12 | 0.993 | [0.937, 1.090] | 10 / 2 |
| zinnia_noir | noir_lit | verify_s | 12 | 1.003 | [0.907, 1.124] | 6 / 6 |

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

