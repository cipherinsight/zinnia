//! pythran_wdist__slow_wdist (Pythran wdist), halo2-base baseline.
//! Kernel: `D[ii, jj] = sqrt(sum_k ((A[k, ii] - B[k, jj]) / W[k, ii]) ** 2)`
//! on A: Float[K, MM], B: Float[K, NN], W: Float[K, MM]. Each quotient is a
//! float division (NumPy materializes `wdiff`); each square rounds before
//! the sum, followed by
//! the exact floor square root. A divisor `W[k, ii]` serves every `jj`, so its
//! sign test runs once (`q.abs`, then `div_by_abs`: `q.div` with the
//! magnitude hoisted out of the loop over `jj`).

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const K: usize = 2; // = A_0
const MM: usize = 3; // = A_1
const NN: usize = 3; // = B_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("A", &[K, MM])?;
    let b = inputs.floats("B", &[K, NN])?;
    let w = inputs.floats("W", &[K, MM])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let b = q.load_floats(ctx, &b);
    let w = q.load_floats_bounded(ctx, &w);
    let w_abs: Vec<Cell> = w.iter().map(|wv| q.abs(ctx, *wv)).collect();
    let mut d = Vec::with_capacity(MM * NN);
    for ii in 0..MM {
        for jj in 0..NN {
            let wdiff: Vec<Cell> = (0..K)
                .map(|k| {
                    let diff = q.sub(ctx, a[k * MM + ii], b[k * NN + jj]);
                    q.div_by_abs(ctx, diff, w[k * MM + ii].cell, w_abs[k * MM + ii]).cell
                })
                .collect();
            let squares: Vec<Cell> = wdiff.iter().map(|x| q.mul(ctx, *x, *x).cell).collect();
            let total = q.sum(ctx, squares);
            // Frozen Halo2 sqrt/guard requires a nonnegative signed64 radicand.
            q.range.range_check(ctx, total, 63);
            d.push(q.sqrt(ctx, total).cell);
        }
    }
    out.floats(&d);
    Ok(())
}
