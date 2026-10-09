// npbench_polybench_durbin_durbin__kernel: literal translation of kernel.py (LITERAL_BASELINES.md).
// Each dot rescales once; the slice product is fully materialized before updating y. No first-iteration shortcut.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = r_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let r = inputs.floats("r", &[N])?;
    let q = Q32::new(builder); let ctx = builder.main(0);
    let r = q.load_floats(ctx, &r);
    let zero = q.float_const(ctx, 0.0);
    let mut y = vec![zero; N];
    let mut alpha = q.neg(ctx, r[0]);
    let mut beta = q.float_const(ctx, 1.0);
    y[0] = q.neg(ctx, r[0]);
    for k in 1..N {
        let square = q.mul(ctx, alpha, alpha).cell;
        let factor = q.sub(ctx, Constant(fe(SCALE)), square);
        beta = q.mul(ctx, beta, factor).cell;
        let flipped: Vec<Cell> = r[..k].iter().rev().copied().collect();
        let dot = q.dot(ctx, &flipped, &y[..k]).cell;
        let sum = q.add(ctx, r[k], dot);
        let numerator = q.neg(ctx, sum);
        let divisor = q.bound(ctx, beta);
        alpha = q.div(ctx, numerator, divisor).cell;
        let products: Vec<Cell> = (0..k).map(|j| q.mul(ctx, alpha, y[k-1-j]).cell).collect();
        for j in 0..k { y[j] = q.add(ctx, y[j], products[j]); }
        y[k] = alpha;
    }
    out.floats(&y); Ok(())
}
