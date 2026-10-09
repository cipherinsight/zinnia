//! DS-1000 case 0444: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Compute each comparison array independently in expression order. Compaction uses prefix counts and an equality test for every input/output pair.
use crate::prelude::*;


pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let av = inputs.ints("A", &[20])?;
    let bv = inputs.ints("B", &[3])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_ints(ctx, &av);
    let b = q.load_ints(ctx, &bv);
    let above0: Vec<Cell> = a.iter().map(|x| {
        let xb = q.bound(ctx, *x);
        let bb = q.bound(ctx, b[0]);
        q.gt(ctx, xb, bb)
    }).collect();
    let below1: Vec<Cell> = a.iter().map(|x| {
        let xb = q.bound(ctx, *x);
        let bb = q.bound(ctx, b[1]);
        q.lt(ctx, xb, bb)
    }).collect();
    let first: Vec<Cell> = (0..20).map(|i| q.gate.and(ctx, above0[i], below1[i])).collect();
    let above1: Vec<Cell> = a.iter().map(|x| {
        let xb = q.bound(ctx, *x);
        let bb = q.bound(ctx, b[1]);
        q.gt(ctx, xb, bb)
    }).collect();
    let below2: Vec<Cell> = a.iter().map(|x| {
        let xb = q.bound(ctx, *x);
        let bb = q.bound(ctx, b[2]);
        q.lt(ctx, xb, bb)
    }).collect();
    let second: Vec<Cell> = (0..20).map(|i| q.gate.and(ctx, above1[i], below2[i])).collect();
    let keep: Vec<Cell> = (0..20).map(|i| q.gate.or(ctx, first[i], second[i])).collect();
    let mut prefix = vec![q.int_const(ctx, 0)];
    for i in 0..20 { prefix.push(q.add(ctx, prefix[i], keep[i])); }
    let mut result = Vec::new();
    for j in 0..20 {
        let mut sum = q.int_const(ctx, 0);
        for i in 0..20 {
            let slot = q.eq(ctx, prefix[i], Constant(fe(j as i128)));
            let active = q.gate.and(ctx, keep[i], slot);
            let term = q.gate.mul(ctx, active, a[i]);
            sum = q.add(ctx, sum, term);
        }
        result.push(sum);
    }
    out.dynamic(prefix[20], &result, Kind::Int, 1);
    Ok(())
}
