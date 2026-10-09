//! DS-1000 case 0439: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Transpose and newaxis are wire views; every broadcast product is computed, including both symmetric halves.
use crate::prelude::*;


pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let xv = inputs.floats("X", &[10, 20])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let x = q.load_floats(ctx, &xv);
    // Each transpose/newaxis expression is a wire view; full broadcast shape.
    let mut result = Vec::new();
    for feature in 0..20 {
        for row in 0..10 {
            for col in 0..10 {
                result.push(q.mul(ctx, x[row * 20 + feature], x[col * 20 + feature]).cell);
            }
        }
    }
    out.floats(&result);
    Ok(())
}
