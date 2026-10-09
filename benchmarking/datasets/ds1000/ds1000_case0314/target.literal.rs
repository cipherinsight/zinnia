//! DS-1000 case 0314: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Separate argmax, unravel, minimum, scatter, argmax and unravel; strict scans preserve first ties.
use crate::prelude::*;

fn argmax(q: &Q32, ctx: &mut Ctx, a: &[Cell]) -> Cell {
    let mut best = a[0];
    let mut idx = q.int_const(ctx, 0);
    for i in 1..a.len() {
        let x = q.bound(ctx, a[i]);
        let b = q.bound(ctx, best);
        let better = q.gt(ctx, x, b);
        best = q.gate.select(ctx, a[i], best, better);
        idx = q.gate.select(ctx, Constant(fe(i as i128)), idx, better);
    }
    idx
}
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let input = inputs.floats("a", &[8, 6])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let mut a = q.load_floats(ctx, &input);
    // idx = np.unravel_index(a.argmax(), a.shape)
    let flat = argmax(&q, ctx, &a);
    let (row, col) = q.divmod_small(ctx, flat, 6, 6);
    // a[idx] = a.min(): complete the RHS before any write.
    let mut low = a[0];
    for i in 1..48 {
        let l = q.bound(ctx, low);
        let x = q.bound(ctx, a[i]);
        low = q.min(ctx, l, x).cell;
    }
    for i in 0..8 {
        for j in 0..6 {
            let r = q.eq(ctx, row, Constant(fe(i as i128)));
            let c = q.eq(ctx, col, Constant(fe(j as i128)));
            let hit = q.gate.and(ctx, r, c);
            a[i * 6 + j] = q.gate.select(ctx, low, a[i * 6 + j], hit);
        }
    }
    // result = np.unravel_index(a.argmax(), a.shape)
    let flat = argmax(&q, ctx, &a);
    let (row, col) = q.divmod_small(ctx, flat, 6, 6);
    out.ints(&[row, col]);
    Ok(())
}
