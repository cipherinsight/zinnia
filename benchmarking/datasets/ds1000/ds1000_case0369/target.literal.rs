// DS-1000 case 0369, literal translation of kernel.py (LITERAL_BASELINES.md).
// Full isclose broadcast, including self comparisons; tolerance recomputed per element, then running all reduction.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let raw = inputs.floats("a", &[3, 3])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_floats(ctx, &raw);
    let mut close = Vec::new();
    // np.isclose: abs(x-y) <= atol + rtol*abs(y), for the full broadcast array.
    for i in 0..3 {
        for j in 0..3 {
            let y = a[i * 3];
            let diff = q.sub(ctx, a[i * 3 + j], y);
            let gap = q.abs_int64(ctx, diff);
            let magnitude = q.abs_int64(ctx, y);
            let relative = q.mul(ctx, Constant(fe(quantize(1e-5))), magnitude).cell;
            let tolerance = q.add(ctx, Constant(Fr::ZERO), relative);
            let left = q.bound(ctx, gap);
            let right = q.bound(ctx, tolerance);
            close.push(q.le(ctx, left, right));
        }
    }
    let mut result = close[0];
    for x in &close[1..] {
        result = q.gate.and(ctx, result, *x);
    }
    out.bool(result);
    Ok(())
}
