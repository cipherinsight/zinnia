//! DS-1000 case 0311, halo2-base baseline.
//! Kernel: `np.unravel_index(a.argmax(), a.shape, order='F')` on `Float[8, 6]`.
//! The flat argmax index `k` (C order) is below 48 by construction (a select of
//! loop constants), so the Fortran-order unravel `(k % 8, k // 8)` uses
//! halo2-base's small-operand `div_mod`.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[8, 6])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let a = q.bound_all(ctx, &a);
    let mut best = a[0];
    let mut best_idx = q.int_const(ctx, 0);
    for (i, x) in a.iter().enumerate().skip(1) {
        let better = q.gt(ctx, *x, best);
        best = q.select_i64(ctx, *x, best, better);
        best_idx = q.gate.select(ctx, Constant(fe(i as i128)), best_idx, better);
    }
    let (col, row) = q.divmod_small(ctx, best_idx, 8, 6);
    out.ints(&[row, col]);
    Ok(())
}
