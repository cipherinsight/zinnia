# Epoch 2026-10-09-tw1-ds-s2

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

9 cases completed by every system that do more than copy inputs: ds1000_case0319, ds1000_case0345, ds1000_case0381, ds1000_case0403, ds1000_case0425, ds1000_case0436, ds1000_case0444, ds1000_case0478, ds1000_case0509

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 9 | 0.858 | [0.591, 1.067] | 3 / 6 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 9 | 0.872 | [0.600, 1.089] | 3 / 6 |
| zinnia_halo2 | halo2_hw | verify_s | 9 | 0.832 | [0.661, 0.975] | 7 / 2 |
| zinnia_noir | noir_hw | prove_core_s | 9 | 1.019 | [0.997, 1.044] | 5 / 4 |
| zinnia_noir | noir_hw | end_to_end_s | 9 | 1.015 | [0.991, 1.041] | 4 / 5 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 9 | 1.001 | [0.972, 1.030] | 5 / 4 |
| zinnia_noir | noir_hw | verify_s | 9 | 1.045 | [0.902, 1.208] | 4 / 5 |
| zinnia_halo2 | noir_hw | prove_core_s | 9 | 1.013 | [0.724, 1.360] | 4 / 5 |
| zinnia_halo2 | noir_hw | end_to_end_s | 9 | 0.863 | [0.507, 1.300] | 4 / 5 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 9 | 0.452 | [0.267, 0.693] | 8 / 1 |
| zinnia_halo2 | noir_hw | verify_s | 9 | 0.590 | [0.458, 0.746] | 8 / 1 |
| zinnia_halo2 | halo2_lit | prove_core_s | 9 | 0.334 | [0.152, 0.679] | 7 / 2 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 9 | 0.340 | [0.155, 0.691] | 7 / 2 |
| zinnia_halo2 | halo2_lit | verify_s | 9 | 0.556 | [0.372, 0.795] | 8 / 1 |
| zinnia_halo2 | noir_lit | prove_core_s | 9 | 0.706 | [0.469, 1.032] | 7 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_s | 9 | 0.597 | [0.350, 0.962] | 7 / 2 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 9 | 0.342 | [0.212, 0.527] | 9 / 0 |
| zinnia_halo2 | noir_lit | verify_s | 9 | 0.571 | [0.440, 0.715] | 9 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 9 | 0.863 | [0.489, 1.430] | 5 / 4 |
| zinnia_noir | halo2_hw | end_to_end_s | 9 | 1.025 | [0.532, 2.023] | 5 / 4 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 9 | 1.931 | [0.985, 3.781] | 2 / 7 |
| zinnia_noir | halo2_hw | verify_s | 9 | 1.473 | [1.039, 2.048] | 2 / 7 |
| zinnia_noir | halo2_lit | prove_core_s | 9 | 0.336 | [0.129, 0.894] | 7 / 2 |
| zinnia_noir | halo2_lit | end_to_end_s | 9 | 0.399 | [0.138, 1.240] | 7 / 2 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 9 | 0.752 | [0.258, 2.316] | 5 / 4 |
| zinnia_noir | halo2_lit | verify_s | 9 | 0.985 | [0.570, 1.700] | 4 / 5 |
| zinnia_noir | noir_lit | prove_core_s | 9 | 0.710 | [0.463, 0.937] | 7 / 2 |
| zinnia_noir | noir_lit | end_to_end_s | 9 | 0.701 | [0.444, 0.944] | 7 / 2 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 9 | 0.758 | [0.525, 0.938] | 9 / 0 |
| zinnia_noir | noir_lit | verify_s | 9 | 1.010 | [0.913, 1.112] | 4 / 5 |

## Copy-only cases (reported apart)

4 all-supported cases that only copy inputs: ds1000_case0301, ds1000_case0327, ds1000_case0361, ds1000_case0497

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 4 | 1.005 | [0.947, 1.051] | 1 / 3 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 4 | 1.012 | [0.951, 1.061] | 1 / 3 |
| zinnia_halo2 | halo2_hw | verify_s | 4 | 0.928 | [0.785, 1.080] | 3 / 1 |
| zinnia_noir | noir_hw | prove_core_s | 4 | 0.994 | [0.893, 1.107] | 2 / 2 |
| zinnia_noir | noir_hw | end_to_end_s | 4 | 0.992 | [0.888, 1.107] | 2 / 2 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 4 | 0.971 | [0.935, 1.009] | 3 / 1 |
| zinnia_noir | noir_hw | verify_s | 4 | 0.996 | [0.900, 1.089] | 2 / 2 |
| zinnia_halo2 | noir_hw | prove_core_s | 4 | 0.797 | [0.548, 1.463] | 3 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_s | 4 | 0.774 | [0.534, 1.407] | 3 / 1 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 4 | 0.396 | [0.264, 0.808] | 3 / 1 |
| zinnia_halo2 | noir_hw | verify_s | 4 | 0.535 | [0.377, 0.757] | 4 / 0 |
| zinnia_halo2 | halo2_lit | prove_core_s | 4 | 1.012 | [0.991, 1.040] | 2 / 2 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 4 | 1.019 | [0.995, 1.043] | 2 / 2 |
| zinnia_halo2 | halo2_lit | verify_s | 4 | 1.012 | [0.825, 1.188] | 1 / 3 |
| zinnia_halo2 | noir_lit | prove_core_s | 4 | 0.807 | [0.557, 1.577] | 3 / 1 |
| zinnia_halo2 | noir_lit | end_to_end_s | 4 | 0.784 | [0.546, 1.510] | 3 / 1 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 4 | 0.401 | [0.273, 0.828] | 3 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 4 | 0.547 | [0.404, 0.740] | 4 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 4 | 1.254 | [0.663, 1.766] | 1 / 3 |
| zinnia_noir | halo2_hw | end_to_end_s | 4 | 1.296 | [0.699, 1.800] | 1 / 3 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 4 | 2.485 | [1.264, 3.614] | 1 / 3 |
| zinnia_noir | halo2_hw | verify_s | 4 | 1.729 | [1.156, 2.585] | 1 / 3 |
| zinnia_noir | halo2_lit | prove_core_s | 4 | 1.262 | [0.643, 1.807] | 1 / 3 |
| zinnia_noir | halo2_lit | end_to_end_s | 4 | 1.305 | [0.680, 1.853] | 1 / 3 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 4 | 2.502 | [1.216, 3.714] | 1 / 3 |
| zinnia_noir | halo2_lit | verify_s | 4 | 1.885 | [1.226, 2.898] | 1 / 3 |
| zinnia_noir | noir_lit | prove_core_s | 4 | 1.007 | [0.951, 1.078] | 3 / 1 |
| zinnia_noir | noir_lit | end_to_end_s | 4 | 1.004 | [0.949, 1.078] | 3 / 1 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 4 | 0.985 | [0.960, 1.016] | 3 / 1 |
| zinnia_noir | noir_lit | verify_s | 4 | 1.019 | [0.910, 1.141] | 1 / 3 |

## All-supported subset

13 cases completed by every system of the epoch (zinnia_halo2, zinnia_noir, halo2_hw, noir_hw, halo2_lit, noir_lit): ds1000_case0301, ds1000_case0319, ds1000_case0327, ds1000_case0345, ds1000_case0361, ds1000_case0381, ds1000_case0403, ds1000_case0425, ds1000_case0436, ds1000_case0444, ds1000_case0478, ds1000_case0497, ds1000_case0509

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 13 | 0.901 | [0.692, 1.051] | 4 / 9 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 13 | 0.913 | [0.701, 1.068] | 4 / 9 |
| zinnia_halo2 | halo2_hw | verify_s | 13 | 0.860 | [0.726, 0.977] | 10 / 3 |
| zinnia_noir | noir_hw | prove_core_s | 13 | 1.011 | [0.973, 1.051] | 7 / 6 |
| zinnia_noir | noir_hw | end_to_end_s | 13 | 1.007 | [0.968, 1.049] | 6 / 7 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 13 | 0.992 | [0.967, 1.016] | 8 / 5 |
| zinnia_noir | noir_hw | verify_s | 13 | 1.030 | [0.925, 1.147] | 6 / 7 |
| zinnia_halo2 | noir_hw | prove_core_s | 13 | 0.941 | [0.711, 1.232] | 7 / 6 |
| zinnia_halo2 | noir_hw | end_to_end_s | 13 | 0.835 | [0.568, 1.169] | 7 / 6 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 13 | 0.434 | [0.293, 0.621] | 11 / 2 |
| zinnia_halo2 | noir_hw | verify_s | 13 | 0.573 | [0.466, 0.700] | 12 / 1 |
| zinnia_halo2 | halo2_lit | prove_core_s | 13 | 0.470 | [0.248, 0.804] | 9 / 4 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 13 | 0.476 | [0.252, 0.815] | 9 / 4 |
| zinnia_halo2 | halo2_lit | verify_s | 13 | 0.669 | [0.479, 0.889] | 9 / 4 |
| zinnia_halo2 | noir_lit | prove_core_s | 13 | 0.735 | [0.533, 1.014] | 10 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_s | 13 | 0.649 | [0.430, 0.945] | 10 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 13 | 0.359 | [0.248, 0.516] | 12 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 13 | 0.563 | [0.462, 0.678] | 13 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 13 | 0.968 | [0.617, 1.433] | 6 / 7 |
| zinnia_noir | halo2_hw | end_to_end_s | 13 | 1.102 | [0.665, 1.790] | 6 / 7 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 13 | 2.086 | [1.233, 3.375] | 3 / 10 |
| zinnia_noir | halo2_hw | verify_s | 13 | 1.547 | [1.174, 2.006] | 3 / 10 |
| zinnia_noir | halo2_lit | prove_core_s | 13 | 0.505 | [0.229, 1.054] | 8 / 5 |
| zinnia_noir | halo2_lit | end_to_end_s | 13 | 0.575 | [0.248, 1.297] | 8 / 5 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 13 | 1.089 | [0.464, 2.446] | 6 / 7 |
| zinnia_noir | halo2_lit | verify_s | 13 | 1.203 | [0.772, 1.819] | 5 / 8 |
| zinnia_noir | noir_lit | prove_core_s | 13 | 0.790 | [0.575, 0.968] | 10 / 3 |
| zinnia_noir | noir_lit | end_to_end_s | 13 | 0.783 | [0.557, 0.971] | 10 / 3 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 13 | 0.822 | [0.630, 0.959] | 12 / 1 |
| zinnia_noir | noir_lit | verify_s | 13 | 1.013 | [0.938, 1.092] | 5 / 8 |

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|
| zinnia_halo2 | halo2_hw | prove_core_s | 13 | 0.901 | [0.692, 1.051] | 4 / 9 |
| zinnia_halo2 | halo2_hw | end_to_end_s | 13 | 0.913 | [0.701, 1.068] | 4 / 9 |
| zinnia_halo2 | halo2_hw | verify_s | 13 | 0.860 | [0.726, 0.977] | 10 / 3 |
| zinnia_noir | noir_hw | prove_core_s | 13 | 1.011 | [0.973, 1.051] | 7 / 6 |
| zinnia_noir | noir_hw | end_to_end_s | 13 | 1.007 | [0.968, 1.049] | 6 / 7 |
| zinnia_noir | noir_hw | end_to_end_nargo_s | 13 | 0.992 | [0.967, 1.016] | 8 / 5 |
| zinnia_noir | noir_hw | verify_s | 13 | 1.030 | [0.925, 1.147] | 6 / 7 |
| zinnia_halo2 | noir_hw | prove_core_s | 13 | 0.941 | [0.711, 1.232] | 7 / 6 |
| zinnia_halo2 | noir_hw | end_to_end_s | 13 | 0.835 | [0.568, 1.169] | 7 / 6 |
| zinnia_halo2 | noir_hw | end_to_end_nargo_s | 13 | 0.434 | [0.293, 0.621] | 11 / 2 |
| zinnia_halo2 | noir_hw | verify_s | 13 | 0.573 | [0.466, 0.700] | 12 / 1 |
| zinnia_halo2 | halo2_lit | prove_core_s | 13 | 0.470 | [0.248, 0.804] | 9 / 4 |
| zinnia_halo2 | halo2_lit | end_to_end_s | 13 | 0.476 | [0.252, 0.815] | 9 / 4 |
| zinnia_halo2 | halo2_lit | verify_s | 13 | 0.669 | [0.479, 0.889] | 9 / 4 |
| zinnia_halo2 | noir_lit | prove_core_s | 13 | 0.735 | [0.533, 1.014] | 10 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_s | 13 | 0.649 | [0.430, 0.945] | 10 / 3 |
| zinnia_halo2 | noir_lit | end_to_end_nargo_s | 13 | 0.359 | [0.248, 0.516] | 12 / 1 |
| zinnia_halo2 | noir_lit | verify_s | 13 | 0.563 | [0.462, 0.678] | 13 / 0 |
| zinnia_noir | halo2_hw | prove_core_s | 13 | 0.968 | [0.617, 1.433] | 6 / 7 |
| zinnia_noir | halo2_hw | end_to_end_s | 13 | 1.102 | [0.665, 1.790] | 6 / 7 |
| zinnia_noir | halo2_hw | end_to_end_nargo_s | 13 | 2.086 | [1.233, 3.375] | 3 / 10 |
| zinnia_noir | halo2_hw | verify_s | 13 | 1.547 | [1.174, 2.006] | 3 / 10 |
| zinnia_noir | halo2_lit | prove_core_s | 13 | 0.505 | [0.229, 1.054] | 8 / 5 |
| zinnia_noir | halo2_lit | end_to_end_s | 13 | 0.575 | [0.248, 1.297] | 8 / 5 |
| zinnia_noir | halo2_lit | end_to_end_nargo_s | 13 | 1.089 | [0.464, 2.446] | 6 / 7 |
| zinnia_noir | halo2_lit | verify_s | 13 | 1.203 | [0.772, 1.819] | 5 / 8 |
| zinnia_noir | noir_lit | prove_core_s | 13 | 0.790 | [0.575, 0.968] | 10 / 3 |
| zinnia_noir | noir_lit | end_to_end_s | 13 | 0.783 | [0.557, 0.971] | 10 / 3 |
| zinnia_noir | noir_lit | end_to_end_nargo_s | 13 | 0.822 | [0.630, 0.959] | 12 / 1 |
| zinnia_noir | noir_lit | verify_s | 13 | 1.013 | [0.938, 1.092] | 5 / 8 |

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

