//! DS-1000 case 0429, halo2-base baseline.
//! Kernel: `(a.mean() - 2*a.std(), a.mean() + 2*a.std())` on `Float[30]`.
//! `a.std()` appears twice in the kernel and is computed once. The variance
//! sums the squared deviations with one rescale (`q.dot`); `/ 30` is a division
//! by a constant; the square root is the exact floor square root.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[30])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let total = q.sum(ctx, a.iter().copied());
    let mean = q.floor_div_const(ctx, total, 30).0.cell;
    let dev: Vec<Cell> = a.iter().map(|x| q.sub(ctx, *x, mean)).collect();
    let sq_sum = q.dot(ctx, &dev, &dev);
    let var = q.floor_div_const(ctx, sq_sum.cell, 30).0;
    let std = q.sqrt(ctx, var.cell);
    let spread = q.gate.mul(ctx, std.cell, Constant(fe(2)));
    let low = q.sub(ctx, mean, spread);
    let high = q.add(ctx, mean, spread);
    out.floats(&[low, high]);
    Ok(())
}
