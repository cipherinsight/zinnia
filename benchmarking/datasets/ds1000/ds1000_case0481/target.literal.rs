//! DS-1000 case 0481: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Compaction uses prefix counts and an equality test for every input/output pair, with zero padding.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let rx = inputs.ints("x", &[9])?;
    let ry = inputs.ints("y", &[9])?;
    let ra = inputs.int("a")?;
    let rb = inputs.int("b")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let x = q.load_ints(ctx, &rx);
    let y = q.load_ints(ctx, &ry);
    let a = q.load_int(ctx, ra);
    let b = q.load_int(ctx, rb);
    let xm: Vec<Cell> = x.iter().map(|v| q.eq(ctx, *v, a)).collect();
    let ym: Vec<Cell> = y.iter().map(|v| q.eq(ctx, *v, b)).collect();
    let keep: Vec<Cell> = (0..9).map(|i| q.gate.and(ctx, xm[i], ym[i])).collect();
    // Count retained elements, then equality-test each input/output pair.
    let mut count = ctx.load_zero();
    let mut before = Vec::new();
    for i in 0..9 {
        before.push(count);
        count = q.add(ctx, count, keep[i]);
    }
    let mut result = Vec::new();
    for j in 0..9 {
        let mut value = ctx.load_zero();
        for i in 0..9 {
            let eq = q.eq(ctx, before[i], Constant(fe(j as i128)));
            let hit = q.gate.and(ctx, keep[i], eq);
            let term = q.gate.mul(ctx, hit, Constant(fe(i as i128)));
            value = q.add(ctx, value, term);
        }
        result.push(value);
    }
    out.dynamic(count, &result, Kind::Int, 1);
    Ok(())
}
