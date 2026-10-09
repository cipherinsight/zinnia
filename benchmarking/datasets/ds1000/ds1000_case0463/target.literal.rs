//! Input is a 20x1 column; Boolean indexing then reshape(-1, 1) retains
//! the same row-major values. The public length counts rows of one value.
//! DS-1000 case 0463: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Compaction uses prefix counts and an equality test for every input/output pair, with zero padding.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let raw = inputs.ints("a", &[20, 1])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_ints(ctx, &raw);
    let mut keep = vec![q.int_const(ctx, 1); 20];
    for i in 1..20 {
        let eq = q.eq(ctx, a[i], a[i - 1]);
        keep[i] = q.gate.not(ctx, eq);
    }
    let nz: Vec<Cell> = a
        .iter()
        .map(|v| {
            let eq = q.eq(ctx, *v, Constant(Fr::ZERO));
            q.gate.not(ctx, eq)
        })
        .collect();
    for i in 0..20 {
        keep[i] = q.gate.and(ctx, keep[i], nz[i]);
    }
    // Count retained elements, then equality-test each input/output pair.
    let mut count = ctx.load_zero();
    let mut before = Vec::new();
    for i in 0..20 {
        before.push(count);
        count = q.add(ctx, count, keep[i]);
    }
    let mut result = Vec::new();
    for j in 0..20 {
        let mut value = ctx.load_zero();
        for i in 0..20 {
            let eq = q.eq(ctx, before[i], Constant(fe(j as i128)));
            let hit = q.gate.and(ctx, keep[i], eq);
            let term = q.gate.mul(ctx, hit, a[i]);
            value = q.add(ctx, value, term);
        }
        result.push(value);
    }
    out.dynamic(count, &result, Kind::Int, 1);
    Ok(())
}
