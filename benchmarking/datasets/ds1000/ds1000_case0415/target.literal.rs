//! DS-1000 case 0415: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Reshape is wiring; max uses an index-order pairwise scan with fresh bounds.
use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let data = inputs.floats("data", &[8])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let data = q.load_floats(ctx, &data);
    // Slice and reshape into two consecutive bins of four.
    let mut result = Vec::new();
    for b in 0..2 {
        let mut best = data[b * 4];
        for k in 1..4 {
            let left = q.bound(ctx, best);
            let right = q.bound(ctx, data[b * 4 + k]);
            best = q.max(ctx, left, right).cell;
        }
        result.push(best);
    }
    out.floats(&result);
    Ok(())
}
