// pythran_wdist__slow_wdist: literal translation of kernel.py (LITERAL_BASELINES.md).
// Each ii,jj builds difference, quotient and square arrays in separate passes, then sum and sqrt.
use crate::prelude::*;
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
    let w = q.load_floats(ctx, &w);
    let zero = q.float_const(ctx, 0.0);
    let mut d = vec![zero; MM * NN];
    for ii in 0..MM {
        for jj in 0..NN {
            let diff: Vec<Cell> = (0..K)
                .map(|k| q.sub(ctx, a[k * MM + ii], b[k * NN + jj]))
                .collect();
            let wdiff: Vec<Cell> = (0..K)
                .map(|k| {
                    let den = q.bound(ctx, w[k * MM + ii]);
                    q.div(ctx, diff[k], den).cell
                })
                .collect();
            let squares: Vec<Cell> = wdiff.iter().map(|x| q.pow_const(ctx, *x, 2)).collect();
            let mut total = zero;
            for x in squares {
                total = q.add(ctx, total, x);
            }
            d[ii * NN + jj] = q.sqrt(ctx, total).cell;
        }
    }
    out.floats(&d);
    Ok(())
}
