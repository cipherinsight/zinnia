//! DS-1000 case 0457, literal translation of kernel.py (LITERAL_BASELINES.md).
//! dim = 6 is the input-width constant; the source body never reads it.
//! Materialize all broadcast differences, then every norm in row-major order.
//! Norm is a running sum of squared terms, rescaled once, then shared sqrt.
//! This matches optimized rounding and its squared-norm window; no symmetry.
//! Diagonal zeros are omitted from public instances only (L7).
use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let raw = inputs.floats("a", &[8, 6])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_floats(ctx, &raw);
    // a - a[:, None], broadcast to (8, 8, 6).
    let mut difference = Vec::with_capacity(8 * 8 * 6);
    for i in 0..8 {
        for j in 0..8 {
            for k in 0..6 {
                difference.push(q.sub(ctx, a[j * 6 + k], a[i * 6 + k]));
            }
        }
    }
    // np.linalg.norm(..., axis=-1): defining reduction at each position.
    let mut result = Vec::with_capacity(64);
    for row in difference.chunks(6) {
        let mut sum = q.float_const(ctx, 0.0);
        for &d in row {
            let term = q.gate.mul(ctx, d, d);
            sum = q.add(ctx, sum, term);
        }
        let squared = q.rescale(ctx, sum);
        result.push(q.sqrt(ctx, squared.cell).cell);
    }
    for i in 0..8 {
        for j in 0..8 {
            if i == j { out.const_floats(&[0.0]); }
            else { out.float(result[i * 8 + j]); }
        }
    }
    Ok(())
}
