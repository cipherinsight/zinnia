# Batch 4 handwritten baseline validation

Validated 2026-10-09 using NumPy 2.2.2, halo2 dev-fast, nargo
1.0.0-beta.20 and bb 5.0.0-nightly.20260324. All 19 entries pass the full
`hw_check.check` protocol for both targets: fixture A/B, keys from A and
proof from B, NumPy outputs, tampered public rejection, out-of-domain input
rejection, and zero redundant halo2 input checks. These are correctness
checks and circuit sizes, not proving-time benchmark measurements.
The compact machine-readable record is `batch4-validation.json`.

The completed goodExpoMeasure baseline rounds the Q32 product to hundredths
with ties to even, truncates and wraps uint8. It no longer rejects all
negative-sigma results above one. Ising now rounds beta to float32 before
its acceptance predicate; both targets pass the added domain regression
with all +1 spins and beta = 186065279 / 2^32. Before this correction the
unrounded beta flipped spins incorrectly. Halo2 tests reject false rounding
splits/parities and false/nonintegral float32 exponents; Noir tests check
rounding/wrapping and reject false/nonintegral exponents. Lint: zero errors.

## Remaining semantic limitation

**goodExpoMeasure remains provisional for the full input domain.** Its A/B
fixtures pass exactly, but an approximate exp followed by discrete rounding
can disagree with float64 at a threshold. Moreover, exact rational
`floor(255*n/100)` differs from NumPy float64 `(n/100)*255` for some n:
`n=12820` gives NumPy integer 32690, exact rational 32691 (bytes 178 vs 179).
This is reproducible with NumPy 2.2.2:

```python
import numpy as np
n = np.array([12820], dtype=np.float64)
assert int((n / 100 * 255).astype(np.uint8)[0]) == 178
assert (255 * 12820 // 100) % 256 == 179
```

Fixing this requires an agreed discrete numerical semantics or faithful
float64 exp/round/cast behavior; passing A/B is not a full-domain claim.
The existing shared exp gadget also documents its Q32 overflow boundary
limitation. No Zinnia-emitted code, IR, or circuit sizes were inspected.

## Circuit sizes

| Entry | Halo2 advice | Halo2 lookup | Noir gates |
|---|---:|---:|---:|
| `pythran_goodexpomeasure__goodExpoMeasure` | 390,479 | 78,353 | 67,794 |
| `pythran_ising__ising_step` | 1,973 | 371 | 3,095 |
| `npbench_crc16_crc16__crc16` | 27,193 | 442 | 11,137 |
| `npbench_weather_stencils_vadv_vadv__vadv` | 1,674,072 | 404,288 | 295,114 |
| `npbench_deep_learning_conv2d_bias_conv2d__conv2d_bias` | 554,040 | 103,360 | 127,033 |
| `npbench_polybench_durbin_durbin__kernel` | 10,753 | 2,599 | 5,651 |
| `npbench_polybench_syr2k_syr2k__kernel` | 102,312 | 26,796 | 19,170 |
| `npbench_polybench_syrk_syrk__kernel` | 69,361 | 18,244 | 14,032 |
| `npbench_weather_stencils_hdiff_hdiff__hdiff` | 416,220 | 88,760 | 76,089 |
| `pythran_fbcorr__fbcorr` | 146,934 | 17,424 | 41,504 |
| `pythran_tax__tax` | 3,938 | 671 | 3,302 |
| `npbench_polybench_gemm_gemm__kernel` | 56,878 | 13,136 | 13,774 |
| `npbench_polybench_mvt_mvt__kernel` | 11,616 | 2,944 | 4,951 |
| `pythran_euclidean_distance_square__euclidean_distance_square` | 1,563 | 425 | 3,009 |
| `pythran_lap__laplacian` | 4,239 | 1,105 | 3,505 |
| `pythran_multitype__check` | 104 | 28 | 2,831 |
| `pythran_multitype__check2` | 59 | 14 | 2,825 |
| `pythran_multitype__times` | 48 | 12 | 388 |
| `pythran_periodic_bc__periodic_bc` | 720 | 216 | 2,947 |
