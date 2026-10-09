//! DS-1000 case 0335: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Stack the three arrays, then pairwise max in axis index order with fresh bounds.
use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[50])?;
    let b = inputs.floats("b", &[50])?;
    let c = inputs.floats("c", &[50])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let arrays = [q.load_floats(ctx, &a), q.load_floats(ctx, &b), q.load_floats(ctx, &c)];
    let mut result = Vec::new();
    for i in 0..50 {
        let mut best = arrays[0][i];
        for j in 1..3 {
            let b = q.bound(ctx, best);
            let x = q.bound(ctx, arrays[j][i]);
            best = q.max(ctx, b, x).cell;
        }
        result.push(best);
    }
    out.floats(&result);
    Ok(())
}
