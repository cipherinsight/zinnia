//! DS-1000 case 0442: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Membership is an equality/OR scan. Compaction uses prefix counts and an equality test for every input/output pair.
use crate::prelude::*;


pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let av = inputs.ints("A", &[20])?;
    let bv = inputs.ints("B", &[3])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_ints(ctx, &av);
    let b = q.load_ints(ctx, &bv);
    // np.in1d: equality against every B element and running OR.
    let mut membership = Vec::new();
    for x in &a {
        let mut found = q.int_const(ctx, 0);
        for y in &b {
            let same = q.eq(ctx, *x, *y);
            found = q.gate.or(ctx, found, same);
        }
        membership.push(found);
    }
    let keep: Vec<Cell> = membership.iter().map(|v| q.gate.not(ctx, *v)).collect();
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
