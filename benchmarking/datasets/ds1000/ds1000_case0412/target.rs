//! DS-1000 case 0412, halo2-base baseline.
//! Kernel: `x[x >= 0]` on `Float[10]` (data-dependent length).
//! Output: the count of kept values, then the kept values in order, zero-padded
//! to 10. Compaction: `q.compact` (a triangular shift network, no equality
//! tests).

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.floats("x", &[10])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x = q.load_floats_bounded(ctx, &x);
    let mut keep = Vec::with_capacity(10);
    let mut weights = Vec::with_capacity(10);
    for xi in &x {
        let neg = q.is_neg(ctx, *xi);
        let k = q.gate.not(ctx, neg);
        weights.push(vec![Some(q.gate.mul(ctx, k, xi.cell))]);
        keep.push(k);
    }
    let count = q.sum(ctx, keep.iter().copied());
    let kept = q.compact(ctx, &keep, &weights);
    out.dynamic(count, &kept, Kind::Float, 1);
    Ok(())
}
