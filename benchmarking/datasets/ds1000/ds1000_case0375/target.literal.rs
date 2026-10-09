// DS-1000 case 0375, literal translation of kernel.py (LITERAL_BASELINES.md).
// ecdf_result remains a helper: full stable bubble sort and constant-folded ys.
// Argmax scans (best value, first best index); branches compute both results then select.
use crate::prelude::*;
fn ecdf_result(q: &Q32, ctx: &mut Ctx, x: &[Cell]) -> (Vec<Cell>, Vec<Cell>) {
    // xs = np.sort(x): stable bubble sort over all 50 values.
    let mut xs = x.to_vec();
    for pass in 0..49 {
        for j in 0..49 - pass {
            let a = q.bound(ctx, xs[j]);
            let b = q.bound(ctx, xs[j + 1]);
            let swap = q.gt(ctx, a, b);
            let (left, right) = (xs[j], xs[j + 1]);
            xs[j] = q.gate.select(ctx, right, left, swap);
            xs[j + 1] = q.gate.select(ctx, left, right, swap);
        }
    }
    // ys = arange(1, len(xs)+1)/float(len(xs)): a constant array (L3).
    let ys = (1..51)
        .map(|i| q.float_const(ctx, i as f64 / 50.0))
        .collect();
    (xs, ys)
}
fn argmax(q: &Q32, ctx: &mut Ctx, values: &[Cell]) -> Cell {
    let mut best = values[0];
    let mut index = q.int_const(ctx, 0);
    for i in 1..values.len() {
        let a = q.bound(ctx, values[i]);
        let b = q.bound(ctx, best);
        let greater = q.gt(ctx, a, b);
        best = q.gate.select(ctx, values[i], best, greater);
        index = q.gate.select(ctx, Constant(fe(i as i128)), index, greater);
    }
    index
}
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let raw = inputs.floats("grades", &[50])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let grades = q.load_floats(ctx, &raw);
    let threshold = q.load_float(ctx, inputs.float("threshold")?);
    let (resultx, resulty) = ecdf_result(&q, ctx, &grades);
    let mut mask = Vec::new();
    for y in &resulty {
        let a = q.bound(ctx, *y);
        let b = q.bound(ctx, threshold);
        mask.push(q.gt(ctx, a, b));
    }
    let t = argmax(&q, ctx, &mask);
    let low = resultx[0];
    let high = q.index(ctx, &resultx, t);
    out.floats(&[low, high]);
    Ok(())
}
