//! DS-1000 case 0322: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Minimum by pairwise scan; argwhere by prefix count and equality-test compaction.
use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.ints("a", &[2, 2])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_ints(ctx, &a);
    let mut low = a[0];
    for i in 1..4 {
        let l = q.bound(ctx, low);
        let x = q.bound(ctx, a[i]);
        low = q.min(ctx, l, x).cell;
    }
    let mask: Vec<Cell> = a.iter().map(|x| q.eq(ctx, *x, low)).collect();
    let mut count = q.int_const(ctx, 0);
    let mut positions = Vec::new();
    for i in 0..4 {
        positions.push(count);
        count = q.add(ctx, count, mask[i]);
    }
    let mut coords = Vec::new();
    for j in 0..4 {
        for axis in 0..2 {
            let mut acc = q.int_const(ctx, 0);
            for i in 0..4 {
                let at = q.eq(ctx, positions[i], Constant(fe(j as i128)));
                let keep = q.gate.and(ctx, mask[i], at);
                let coord = if axis == 0 { i / 2 } else { i % 2 };
                let term = q.gate.mul(ctx, keep, Constant(fe(coord as i128)));
                acc = q.add(ctx, acc, term);
            }
            coords.push(acc);
        }
    }
    out.dynamic(count, &coords, Kind::Int, 2);
    Ok(())
}
