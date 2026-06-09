# Baseline notes for ds1000_case0329

`kernel(a: Float[Array, "8 6"], power: Integer) -> a ** power`

| Framework | Status                                                                                              |
|-----------|-----------------------------------------------------------------------------------------------------|
| Halo2     | written; two explicit `qmul` ops per cell (no `qpow` shortcut). FixedPointChip 2^63 quantisation.   |
| SP1       | written; two explicit `f64` multiplications per cell (no `.powi`).                                  |
| Risc0     | written; two explicit `f64` multiplications per cell (no `.powi`).                                  |
| Cairo     | written; ×1000 quantisation, two explicit u128 multiplications. Result fully scaled by 10^9.       |
| Noir      | written; ×1000 quantisation, two explicit Field multiplications. Result fully scaled by 10^9.       |

**power=3 is baked compile-time** in every baseline so the algorithm and control flow are identical: two multiplications + one equality check per cell.
