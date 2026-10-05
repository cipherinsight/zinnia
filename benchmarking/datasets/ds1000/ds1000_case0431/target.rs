//! DS-1000 case 0431, halo2-base baseline.
//! Kernel: `~((a > mean - 2*std) & (a < mean + 2*std))` on `Float[8]`.
//! `a.mean()` and `a.std()` appear twice in the kernel and are computed once.
//! The variance sums the squared deviations with one rescale (`q.dot`); `/ 8`
//! is a division by a constant; the square root is the exact floor square root.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[8])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let total = q.sum(ctx, a.iter().copied());
    let mean = q.floor_div_const(ctx, total, 8).0.cell;
    let dev: Vec<Cell> = a.iter().map(|x| q.sub(ctx, *x, mean)).collect();
    let sq_sum = q.dot(ctx, &dev, &dev);
    let var = q.floor_div_const(ctx, sq_sum.cell, 8).0;
    let std = q.sqrt(ctx, var.cell);
    let spread = q.gate.mul(ctx, std.cell, Constant(fe(2)));
    let low = q.sub(ctx, mean, spread);
    let low = q.bound(ctx, low);
    let high = q.add(ctx, mean, spread);
    let high = q.bound(ctx, high);
    let mut outlier = Vec::with_capacity(8);
    for x in &a {
        let x = q.bound(ctx, *x);
        let above_low = q.lt(ctx, low, x);
        let below_high = q.lt(ctx, x, high);
        let inside = q.gate.and(ctx, above_low, below_high);
        outlier.push(q.gate.not(ctx, inside));
    }
    out.bools(&outlier);
    Ok(())
}
