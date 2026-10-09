//! DS-1000 case 0373: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Bubble sort: all 1225 adjacent comparisons, bounding at each comparison.
//! Constant ratios use floor(i * 2^32 / 50) in both variants, with no public cells.
use crate::prelude::*;

fn ecdf_result(q: &Q32, ctx: &mut Ctx, x: &[Cell]) -> Vec<f64> {
    // xs = np.sort(x)
    let mut xs = x.to_vec();
    for end in (1..xs.len()).rev() {
        for j in 0..end {
            let a = q.bound(ctx, xs[j]);
            let b = q.bound(ctx, xs[j + 1]);
            let swap = q.gt(ctx, a, b);
            let lo = q.gate.select(ctx, xs[j + 1], xs[j], swap);
            let hi = q.gate.select(ctx, xs[j], xs[j + 1], swap);
            xs[j] = lo;
            xs[j + 1] = hi;
        }
    }
    // ys = np.arange(1, len(xs)+1)/float(len(xs))
    // All operands are shape constants. Fold division with Q32 floor rounding.
    let ys = (1..=xs.len()).map(|i| ((i as i128 * SCALE) / xs.len() as i128) as f64 / SCALE as f64).collect();
    ys
}

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let grades = inputs.floats("grades", &[50])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let grades = q.load_floats(ctx, &grades);
    let result = ecdf_result(&q, ctx, &grades);
    out.const_floats(&result);
    Ok(())
}
