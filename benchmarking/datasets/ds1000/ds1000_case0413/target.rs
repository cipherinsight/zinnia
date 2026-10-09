//! DS-1000 case 0413: optimized baseline.
//! Complex values are interleaved (real, imaginary) Q32 components, each checked.
//! Exact nonzero imaginary test; length counts complex items, padded unit is two.
//! Shared compaction carries both components in stable input order.
use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.complexes("x", &[10])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let x = q.load_floats(ctx, &x);
    let mut keep = Vec::with_capacity(10);
    let mut weights = Vec::with_capacity(10);
    for i in 0..10 {
        let zero = q.eq(ctx, x[2 * i + 1], Constant(Fr::ZERO));
        let k = q.gate.not(ctx, zero);
        keep.push(k);
        // im is zero exactly when k is zero, so k * im is already im.
        weights.push(vec![Some(q.gate.mul(ctx, k, x[2 * i])), Some(x[2 * i + 1])]);
    }
    let count = q.sum(ctx, keep.iter().copied());
    let result = q.compact(ctx, &keep, &weights);
    out.dynamic(count, &result, Kind::Float, 2);
    Ok(())
}
