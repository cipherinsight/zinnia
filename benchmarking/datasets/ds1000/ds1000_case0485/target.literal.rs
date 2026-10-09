//! DS-1000 case 0485: literal translation of kernel.py (LITERAL_BASELINES.md).
//! tensordot uses its defining sum in index order with one final Q32 rescale per output (L5).
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let ra = inputs.floats("A", &[5, 6, 3])?;
    let rb = inputs.floats("B", &[3, 3])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_floats(ctx, &ra);
    let b = q.load_floats(ctx, &rb);
    let mut result = Vec::new();
    for i in 0..5 {
        for j in 0..6 {
            for k in 0..3 {
                let mut raw = ctx.load_zero();
                for p in 0..3 {
                    let term = q.gate.mul(ctx, a[(i * 6 + j) * 3 + p], b[p * 3 + k]);
                    raw = q.add(ctx, raw, term);
                }
                result.push(q.rescale(ctx, raw).cell);
            }
        }
    }
    out.floats(&result);
    Ok(())
}
