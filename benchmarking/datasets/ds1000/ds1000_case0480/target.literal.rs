//! DS-1000 case 0480: literal translation of kernel.py (LITERAL_BASELINES.md).
//! argmax scans left to right with strict improvement; both indexed rechecks are computed.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let rx = inputs.ints("x", &[8])?; let ry = inputs.ints("y", &[8])?;
    let ra = inputs.int("a")?; let rb = inputs.int("b")?;
    let q = Q32::new(builder); let ctx = builder.main(0);
    let x = q.load_ints(ctx, &rx); let y = q.load_ints(ctx, &ry);
    let a = q.load_int(ctx, ra); let b = q.load_int(ctx, rb);
    let xm: Vec<Cell> = x.iter().map(|v| q.eq(ctx, *v, a)).collect();
    let ym: Vec<Cell> = y.iter().map(|v| q.eq(ctx, *v, b)).collect();
    let hits: Vec<Cell> = (0..8).map(|i| q.gate.and(ctx, xm[i], ym[i])).collect();
    let mut best = hits[0]; let mut result = ctx.load_zero();
    for i in 1..8 {
        let lhs = q.bound(ctx, hits[i]); let rhs = q.bound(ctx, best);
        let improve = q.gt(ctx, lhs, rhs);
        best = q.gate.select(ctx, hits[i], best, improve);
        result = q.gate.select(ctx, Constant(fe(i as i128)), result, improve);
    }
    let xr = q.index(ctx, &x, result);
    let eqx = q.eq(ctx, xr, a); let nx = q.gate.not(ctx, eqx);
    let yr = q.index(ctx, &y, result);
    let eqy = q.eq(ctx, yr, b); let ny = q.gate.not(ctx, eqy);
    let condition = q.gate.or(ctx, nx, ny);
    result = q.gate.select(ctx, Constant(fe(-1)), result, condition);
    out.int(result);
    Ok(())
}
