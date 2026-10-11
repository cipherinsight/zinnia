# Epoch 2026-10-09-tw1-ds-s3

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

10 cases completed by every system that do more than copy inputs: ds1000_case0309, ds1000_case0320, ds1000_case0346, ds1000_case0383, ds1000_case0409, ds1000_case0426, ds1000_case0437, ds1000_case0450, ds1000_case0480, ds1000_case0510

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 10 | 0.999 | [0.949, 1.052] | 5 / 5 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 10 | 1.010 | [0.959, 1.063] | 5 / 5 |
| zinnia_halo2 | halo2_hw | verify_s | 10 | 1.153 | [1.040, 1.275] | 2 / 8 |
| zinnia_noir | noir_hw | prove_core_s | 10 | 0.987 | [0.948, 1.031] | 6 / 4 |
| zinnia_noir | noir_hw | end_to_end_s | 10 | 0.993 | [0.958, 1.033] | 6 / 4 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 10 | 0.981 | [0.950, 1.016] | 6 / 4 |
| zinnia_noir | noir_hw | verify_s | 10 | 1.037 | [0.970, 1.117] | 3 / 7 |
| zinnia_halo2 | noir_hw | prove_core_s | 10 | 1.121 | [0.905, 1.412] | 3 / 7 |
| zinnia_halo2 | noir_hw | end_to_end_s | 10 | 1.090 | [0.884, 1.367] | 5 / 5 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 10 | 0.547 | [0.427, 0.710] | 9 / 1 |
| zinnia_halo2 | noir_hw | verify_s | 10 | 0.756 | [0.658, 0.856] | 10 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 10 | 0.387 | [0.183, 0.712] | 10 / 0 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 10 | 0.391 | [0.185, 0.718] | 10 / 0 |
| zinnia_halo2 | halo2_lit | verify_s | 10 | 0.722 | [0.464, 1.029] | 6 / 4 |
| zinnia_halo2 | noir_lit | prove_core_s | 10 | 0.843 | [0.565, 1.183] | 6 / 4 |
| zinnia_halo2 | noir_lit | end_to_end_s | 10 | 0.805 | [0.526, 1.142] | 6 / 4 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 10 | 0.444 | [0.318, 0.613] | 9 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 10 | 0.748 | [0.635, 0.866] | 10 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 10 | 0.880 | [0.719, 1.097] | 6 / 4 |
| zinnia_noir | halo2_hw | end_to_end_s | 10 | 0.920 | [0.753, 1.142] | 6 / 4 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 10 | 1.812 | [1.453, 2.271] | 1 / 9 |
| zinnia_noir | halo2_hw | verify_s | 10 | 1.583 | [1.408, 1.781] | 0 / 10 |
| zinnia_noir | halo2_lit | prove_core_s | 10 | 0.341 | [0.149, 0.673] | 9 / 1 |
| zinnia_noir | halo2_lit | end_to_end_s | 10 | 0.356 | [0.156, 0.703] | 9 / 1 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 10 | 0.701 | [0.300, 1.409] | 4 / 6 |
| zinnia_noir | halo2_lit | verify_s | 10 | 0.991 | [0.620, 1.431] | 4 / 6 |
| zinnia_noir | noir_lit | prove_core_s | 10 | 0.743 | [0.495, 0.952] | 8 / 2 |
| zinnia_noir | noir_lit | end_to_end_s | 10 | 0.734 | [0.475, 0.956] | 8 / 2 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 10 | 0.798 | [0.569, 0.969] | 9 / 1 |
| zinnia_noir | noir_lit | verify_s | 10 | 1.026 | [0.931, 1.139] | 5 / 5 |

## Copy-only cases (reported apart)

3 all-supported cases that only copy inputs: ds1000_case0328, ds1000_case0362, ds1000_case0500

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 3 | 1.008 | [0.927, 1.069] | 1 / 2 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 3 | 1.013 | [0.930, 1.076] | 1 / 2 |
| zinnia_halo2 | halo2_hw | verify_s | 3 | 0.902 | [0.614, 1.251] | 2 / 1 |
| zinnia_noir | noir_hw | prove_core_s | 3 | 0.967 | [0.916, 1.055] | 2 / 1 |
| zinnia_noir | noir_hw | end_to_end_s | 3 | 0.967 | [0.920, 1.056] | 2 / 1 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 3 | 0.947 | [0.886, 0.997] | 3 / 0 |
| zinnia_noir | noir_hw | verify_s | 3 | 0.938 | [0.877, 1.010] | 2 / 1 |
| zinnia_halo2 | noir_hw | prove_core_s | 3 | 0.639 | [0.308, 0.964] | 3 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_s | 3 | 0.624 | [0.302, 0.943] | 3 / 0 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 3 | 0.297 | [0.146, 0.441] | 3 / 0 |
| zinnia_halo2 | noir_hw | verify_s | 3 | 0.455 | [0.403, 0.510] | 3 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 3 | 0.986 | [0.963, 1.000] | 2 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 3 | 0.991 | [0.965, 1.007] | 1 / 2 |
| zinnia_halo2 | halo2_lit | verify_s | 3 | 0.860 | [0.703, 1.052] | 2 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 3 | 0.614 | [0.272, 1.001] | 2 / 1 |
| zinnia_halo2 | noir_lit | end_to_end_s | 3 | 0.599 | [0.267, 0.964] | 3 / 0 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 3 | 0.283 | [0.134, 0.414] | 3 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 3 | 0.409 | [0.294, 0.503] | 3 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 3 | 1.525 | [1.077, 2.816] | 0 / 3 |
| zinnia_noir | halo2_hw | end_to_end_s | 3 | 1.571 | [1.124, 2.864] | 0 / 3 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 3 | 3.231 | [2.431, 5.659] | 0 / 3 |
| zinnia_noir | halo2_hw | verify_s | 3 | 1.860 | [1.354, 2.287] | 0 / 3 |
| zinnia_noir | halo2_lit | prove_core_s | 3 | 1.492 | [1.038, 2.923] | 0 / 3 |
| zinnia_noir | halo2_lit | end_to_end_s | 3 | 1.537 | [1.084, 2.970] | 0 / 3 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 3 | 3.160 | [2.275, 5.869] | 0 / 3 |
| zinnia_noir | halo2_lit | verify_s | 3 | 1.774 | [1.551, 1.923] | 0 / 3 |
| zinnia_noir | noir_lit | prove_core_s | 3 | 0.928 | [0.826, 1.043] | 2 / 1 |
| zinnia_noir | noir_lit | end_to_end_s | 3 | 0.930 | [0.822, 1.044] | 2 / 1 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 3 | 0.903 | [0.816, 0.976] | 3 / 0 |
| zinnia_noir | noir_lit | verify_s | 3 | 0.843 | [0.639, 1.018] | 2 / 1 |

## All-supported subset

13 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): ds1000_case0309, ds1000_case0320, ds1000_case0328, ds1000_case0346, ds1000_case0362, ds1000_case0383, ds1000_case0409, ds1000_case0426, ds1000_case0437, ds1000_case0450, ds1000_case0480, ds1000_case0500, ds1000_case0510

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 13 | 1.001 | [0.959, 1.045] | 6 / 7 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 13 | 1.010 | [0.967, 1.055] | 6 / 7 |
| zinnia_halo2 | halo2_hw | verify_s | 13 | 1.090 | [0.956, 1.220] | 4 / 9 |
| zinnia_noir | noir_hw | prove_core_s | 13 | 0.983 | [0.949, 1.020] | 8 / 5 |
| zinnia_noir | noir_hw | end_to_end_s | 13 | 0.987 | [0.955, 1.022] | 8 / 5 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 13 | 0.973 | [0.945, 1.003] | 9 / 4 |
| zinnia_noir | noir_hw | verify_s | 13 | 1.013 | [0.958, 1.083] | 5 / 8 |
| zinnia_halo2 | noir_hw | prove_core_s | 13 | 0.985 | [0.754, 1.252] | 6 / 7 |
| zinnia_halo2 | noir_hw | end_to_end_s | 13 | 0.958 | [0.736, 1.212] | 8 / 5 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 13 | 0.475 | [0.357, 0.617] | 12 / 1 |
| zinnia_halo2 | noir_hw | verify_s | 13 | 0.672 | [0.572, 0.782] | 13 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 13 | 0.480 | [0.262, 0.794] | 12 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 13 | 0.484 | [0.264, 0.800] | 11 / 2 |
| zinnia_halo2 | halo2_lit | verify_s | 13 | 0.752 | [0.537, 0.990] | 8 / 5 |
| zinnia_halo2 | noir_lit | prove_core_s | 13 | 0.784 | [0.557, 1.058] | 8 / 5 |
| zinnia_halo2 | noir_lit | end_to_end_s | 13 | 0.752 | [0.529, 1.022] | 9 / 4 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 13 | 0.401 | [0.293, 0.534] | 12 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 13 | 0.650 | [0.532, 0.779] | 13 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 13 | 0.999 | [0.801, 1.284] | 6 / 7 |
| zinnia_noir | halo2_hw | end_to_end_s | 13 | 1.041 | [0.838, 1.330] | 6 / 7 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 13 | 2.070 | [1.640, 2.658] | 1 / 12 |
| zinnia_noir | halo2_hw | verify_s | 13 | 1.643 | [1.469, 1.842] | 0 / 13 |
| zinnia_noir | halo2_lit | prove_core_s | 13 | 0.479 | [0.235, 0.903] | 9 / 4 |
| zinnia_noir | halo2_lit | end_to_end_s | 13 | 0.499 | [0.245, 0.938] | 9 / 4 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 13 | 0.993 | [0.478, 1.897] | 4 / 9 |
| zinnia_noir | halo2_lit | verify_s | 13 | 1.133 | [0.774, 1.539] | 4 / 9 |
| zinnia_noir | noir_lit | prove_core_s | 13 | 0.782 | [0.570, 0.949] | 10 / 3 |
| zinnia_noir | noir_lit | end_to_end_s | 13 | 0.775 | [0.553, 0.952] | 10 / 3 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 13 | 0.821 | [0.633, 0.956] | 12 / 1 |
| zinnia_noir | noir_lit | verify_s | 13 | 0.980 | [0.881, 1.084] | 7 / 6 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 13 | 1.001 | [0.959, 1.045] | 6 / 7 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 13 | 1.010 | [0.967, 1.055] | 6 / 7 |
| zinnia_halo2 | halo2_hw | verify_s | 13 | 1.090 | [0.956, 1.220] | 4 / 9 |
| zinnia_noir | noir_hw | prove_core_s | 13 | 0.983 | [0.949, 1.020] | 8 / 5 |
| zinnia_noir | noir_hw | end_to_end_s | 13 | 0.987 | [0.955, 1.022] | 8 / 5 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 13 | 0.973 | [0.945, 1.003] | 9 / 4 |
| zinnia_noir | noir_hw | verify_s | 13 | 1.013 | [0.958, 1.083] | 5 / 8 |
| zinnia_halo2 | noir_hw | prove_core_s | 13 | 0.985 | [0.754, 1.252] | 6 / 7 |
| zinnia_halo2 | noir_hw | end_to_end_s | 13 | 0.958 | [0.736, 1.212] | 8 / 5 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 13 | 0.475 | [0.357, 0.617] | 12 / 1 |
| zinnia_halo2 | noir_hw | verify_s | 13 | 0.672 | [0.572, 0.782] | 13 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 13 | 0.480 | [0.262, 0.794] | 12 / 1 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 13 | 0.484 | [0.264, 0.800] | 11 / 2 |
| zinnia_halo2 | halo2_lit | verify_s | 13 | 0.752 | [0.537, 0.990] | 8 / 5 |
| zinnia_halo2 | noir_lit | prove_core_s | 13 | 0.784 | [0.557, 1.058] | 8 / 5 |
| zinnia_halo2 | noir_lit | end_to_end_s | 13 | 0.752 | [0.529, 1.022] | 9 / 4 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 13 | 0.401 | [0.293, 0.534] | 12 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 13 | 0.650 | [0.532, 0.779] | 13 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 13 | 0.999 | [0.801, 1.284] | 6 / 7 |
| zinnia_noir | halo2_hw | end_to_end_s | 13 | 1.041 | [0.838, 1.330] | 6 / 7 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 13 | 2.070 | [1.640, 2.658] | 1 / 12 |
| zinnia_noir | halo2_hw | verify_s | 13 | 1.643 | [1.469, 1.842] | 0 / 13 |
| zinnia_noir | halo2_lit | prove_core_s | 13 | 0.479 | [0.235, 0.903] | 9 / 4 |
| zinnia_noir | halo2_lit | end_to_end_s | 13 | 0.499 | [0.245, 0.938] | 9 / 4 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 13 | 0.993 | [0.478, 1.897] | 4 / 9 |
| zinnia_noir | halo2_lit | verify_s | 13 | 1.133 | [0.774, 1.539] | 4 / 9 |
| zinnia_noir | noir_lit | prove_core_s | 13 | 0.782 | [0.570, 0.949] | 10 / 3 |
| zinnia_noir | noir_lit | end_to_end_s | 13 | 0.775 | [0.553, 0.952] | 10 / 3 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 13 | 0.821 | [0.633, 0.956] | 12 / 1 |
| zinnia_noir | noir_lit | verify_s | 13 | 0.980 | [0.881, 1.084] | 7 / 6 |

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

