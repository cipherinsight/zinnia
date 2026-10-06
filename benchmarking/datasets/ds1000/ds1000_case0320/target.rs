//! DS-1000 case 0320, halo2-base baseline.
//! Kernel: `np.unravel_index(a.argmin(), a.shape)` on `Float[5, 6]`.
//! The flat index `k` of the first minimum is below 30 by construction (a select
//! of loop constants), so the C-order unravel `(k // 6, k % 6)` uses halo2-base's
//! small-operand `div_mod`.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[5, 6])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats_bounded(ctx, &a);
    let mut best = a[0];
    let mut best_idx = q.int_const(ctx, 0);
    for (i, x) in a.iter().enumerate().skip(1) {
        let better = q.lt(ctx, *x, best);
        best = q.select_i64(ctx, *x, best, better);
        best_idx = q.gate.select(ctx, Constant(fe(i as i128)), best_idx, better);
    }
    let (row, col) = q.divmod_small(ctx, best_idx, 6, 5);
    out.ints(&[row, col]);
    Ok(())
}
