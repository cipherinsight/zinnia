//! DS-1000 case 0436: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Compute all row extrema by pairwise scan, then broadcast equality; fresh bounds at each comparison.
use crate::prelude::*;


pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let av = inputs.floats("a", &[10, 5])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_floats(ctx, &av);
    let mut extrema = Vec::new();
    for row in 0..10 {
        let mut best = a[row * 5];
        for col in 1..5 {
            let left = q.bound(ctx, best);
            let right = q.bound(ctx, a[row * 5 + col]);
            best = q.max(ctx, left, right).cell;
        }
        extrema.push(best);
    }
    let mut mask = Vec::new();
    for row in 0..10 {
        for col in 0..5 { mask.push(q.eq(ctx, extrema[row], a[row * 5 + col])); }
    }
    out.bools(&mask);
    Ok(())
}
