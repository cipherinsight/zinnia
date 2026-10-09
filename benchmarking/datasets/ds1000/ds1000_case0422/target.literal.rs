//! DS-1000 case 0422: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Valid correlation when the second input is longer uses lag 49-k; each defining sum rescales once.
use crate::prelude::*;


pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let av = inputs.floats("a", &[50])?;
    let bv = inputs.floats("b", &[50])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_floats(ctx, &av);
    let b = q.load_floats(ctx, &bv);
    // hstack((b[1:], b)) in full.
    let mut stacked = b[1..].to_vec();
    stacked.extend_from_slice(&b);
    let mut result = Vec::new();
    for k in 0..50 {
        let mut acc = q.int_const(ctx, 0);
        for n in 0..50 {
            let product = q.gate.mul(ctx, a[n], stacked[n + 49 - k]);
            acc = q.add(ctx, acc, product);
        }
        result.push(q.rescale(ctx, acc).cell);
    }
    out.floats(&result);
    Ok(())
}
