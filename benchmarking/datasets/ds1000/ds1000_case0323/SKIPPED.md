# Baseline notes for ds1000_case0323

`kernel(degree: Integer) -> np.cos(np.deg2rad(degree))`

All 5 baselines compute the same 4-term Taylor expansion of cos:

```
rad   = degree * π/180
x2    = rad * rad
x4    = x2 * x2
x6    = x4 * x2
term2 = x2 / 2
term4 = x4 / 24
term6 = x6 / 720
cos   = 1 - term2 + term4 - term6
assert |cos - claimed| ≤ 0.005
```

- **Halo2**: FixedPointChip 2^63, primitive `qmul`/`qdiv`/`qsub`/`qadd` (no `qcos`).
- **SP1 / Risc0**: native f64.
- **Cairo / Noir**: ×10^9 integer quantisation (the higher scale is needed to keep precision through the three sequential multiplications).

Tolerance is set to 0.005 — comfortably above the worst-case 4-term Taylor truncation error for |x| ≤ π and the FixedPointChip rounding accumulated across the Taylor chain.
