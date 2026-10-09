//! DS-1000 case 0454: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Full elementwise intermediate, then index-order row reduction, then broadcast division; integer division directly yields Q32.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let raw = inputs.ints("X", &[5, 4])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let x = q.load_ints(ctx, &raw);
    // np.abs(X), computed in full before the reduction.
    let magnitudes: Vec<Cell> = x.iter().map(|v| q.abs_int64(ctx, *v)).collect();
    let mut norms = Vec::new();
    for i in 0..5 {
        let mut acc = magnitudes[i * 4];
        for j in 1..4 {
            let lhs = q.bound(ctx, acc);
            let rhs = q.bound(ctx, magnitudes[i * 4 + j]);
            acc = q.max(ctx, lhs, rhs).cell;
        }
        norms.push(acc);
    }
    // result = X / norms.reshape(-1, 1)
    let mut result = Vec::new();
    for i in 0..5 {
        for j in 0..4 {
            let numerator = x[i * 4 + j];
            let divisor = q.bound(ctx, norms[i]);
            result.push(q.div(ctx, numerator, divisor).cell);
        }
    }
    out.floats(&result);
    Ok(())
}
