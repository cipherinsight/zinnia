//! DS-1000 case 0453: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Full elementwise intermediate, then index-order row reduction, then broadcast division; integer division directly yields Q32.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let raw = inputs.ints("X", &[5, 4])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let x = q.load_ints(ctx, &raw);
    // X*X: exact integer products, materialized before sum.
    let squares: Vec<Cell> = x.iter().map(|v| q.gate.mul(ctx, *v, *v)).collect();
    let mut norms = Vec::new();
    for i in 0..5 {
        let mut acc = ctx.load_zero();
        for j in 0..4 {
            acc = q.add(ctx, acc, squares[i * 4 + j]);
        }
        norms.push(acc);
    }
    // sqrt after the complete sum array.
    let norms: Vec<Cell> = norms
        .iter()
        .map(|v| {
            let f = q.int_to_float(ctx, *v);
            q.sqrt(ctx, f).cell
        })
        .collect();
    // result = X / norms.reshape(-1, 1)
    let mut result = Vec::new();
    for i in 0..5 {
        for j in 0..4 {
            let numerator = q.int_to_float(ctx, x[i * 4 + j]);
            let divisor = q.bound(ctx, norms[i]);
            result.push(q.div(ctx, numerator, divisor).cell);
        }
    }
    out.floats(&result);
    Ok(())
}
