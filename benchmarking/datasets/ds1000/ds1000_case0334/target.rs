//! DS-1000 case 0334, halo2-base baseline.
//! Kernel: `np.mean([a, b, c], axis=0)` on three `Float[50]`.
//! The Q32 mean `floor((a + b + c) * 2^32 / (3 * 2^32))` equals
//! `floor((a + b + c) / 3)`, a division by the constant 3.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let (a, b, c) = (inputs.floats("a", &[50])?, inputs.floats("b", &[50])?, inputs.floats("c", &[50])?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let (a, b, c) = (q.load_floats(ctx, &a), q.load_floats(ctx, &b), q.load_floats(ctx, &c));
    let mean: Vec<Cell> = (0..50)
        .map(|i| {
            let s = q.sum(ctx, [a[i], b[i], c[i]]);
            q.floor_div_const(ctx, s, 3).0.cell
        })
        .collect();
    out.floats(&mean);
    Ok(())
}
