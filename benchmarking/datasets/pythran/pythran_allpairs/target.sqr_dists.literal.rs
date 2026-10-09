// pythran_allpairs__sqr_dists: literal translation of kernel.py (LITERAL_BASELINES.md).
// Difference and squared arrays are separate full passes; running sum in index order, one rounding per square.
use crate::prelude::*;
const NX: usize = 16; // = X_0
const NY: usize = 16; // = Y_0
const K: usize = 16; // = X_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.floats("X", &[NX, K])?;
    let y = inputs.floats("Y", &[NY, K])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let x = q.load_floats(ctx, &x);
    let y = q.load_floats(ctx, &y);
    let zero = q.float_const(ctx, 0.0);
    let mut result = vec![zero; NY * NX];
    for i in 0..NY {
        for j in 0..NX {
            let diff: Vec<Cell> = (0..K)
                .map(|k| q.sub(ctx, x[j * K + k], y[i * K + k]))
                .collect();
            let squares: Vec<Cell> = diff.iter().map(|v| q.pow_const(ctx, *v, 2)).collect();
            let mut sum = zero;
            for v in squares {
                sum = q.add(ctx, sum, v);
            }
            result[i * NX + j] = sum;
        }
    }
    out.floats(&result);
    Ok(())
}
