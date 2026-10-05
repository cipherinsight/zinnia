//! DS-1000 case 0426, halo2-base baseline.
//! Kernel: `((a[:, None] & (1 << np.arange(m))[::-1]) > 0).astype(int)` with
//! `m = 15` on `a: Integer[20]`: the low 15 bits of each value, most significant
//! first (`Integer[20, 15]`).
//! `a & 2^k > 0` is bit `k` of `a`'s two's complement, i.e. bit `k` of
//! `a mod 2^15`, also for negative `a`. `q.floor_div_const` splits
//! `a = h * 2^15 + l` (`l` in `[0, 2^15)`, `h` in the window, so the split is
//! unique) and halo2-base `num_to_bits` decomposes `l` into its 15 bits.

use crate::prelude::*;

const M: usize = 15;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.ints("a", &[20])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_ints(ctx, &a);
    let mut result = Vec::with_capacity(a.len() * M);
    for x in a {
        let (_, low) = q.floor_div_const(ctx, x, 1 << M);
        let bits = q.gate.num_to_bits(ctx, low, M); // least significant first
        result.extend(bits.iter().rev());
    }
    out.ints(&result);
    Ok(())
}
