//! DS-1000 case 0314, halo2-base baseline.
//! Kernel: on `Float[8, 6]`, `idx = unravel(a.argmax())`, `a[idx] = a.min()`,
//! then `unravel(a.argmax())` again (C order).
//! The first pass finds the first maximum's flat index and the minimum; the
//! write `a[idx] = min` is a select per cell against that index.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[8, 6])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats_bounded(ctx, &a);
    let (mut best, mut low) = (a[0], a[0]);
    let mut best_idx = q.int_const(ctx, 0);
    for (i, x) in a.iter().enumerate().skip(1) {
        let better = q.gt(ctx, *x, best);
        best = q.select_i64(ctx, *x, best, better);
        best_idx = q.gate.select(ctx, Constant(fe(i as i128)), best_idx, better);
        low = q.min(ctx, low, *x);
    }

    let mut best2 = None;
    let mut best2_idx = q.int_const(ctx, 0);
    for (i, x) in a.iter().enumerate() {
        let hit = q.eq(ctx, best_idx, Constant(fe(i as i128)));
        let v = q.select_i64(ctx, low, *x, hit);
        best2 = Some(match best2 {
            None => v,
            Some(b) => {
                let better = q.gt(ctx, v, b);
                best2_idx = q.gate.select(ctx, Constant(fe(i as i128)), best2_idx, better);
                q.select_i64(ctx, v, b, better)
            }
        });
    }
    let (row, col) = q.divmod_small(ctx, best2_idx, 6, 6);
    out.ints(&[row, col]);
    Ok(())
}
