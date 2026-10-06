//! DS-1000 case 0309, halo2-base baseline.
//! Kernel: `a.argmax()` on `Float[8, 6]`: the flat index of the first maximum.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[8, 6])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats_bounded(ctx, &a);
    let mut best = a[0];
    let mut best_idx = q.int_const(ctx, 0);
    for (i, x) in a.iter().enumerate().skip(1) {
        // Strictly greater, so ties keep the first index (NumPy's rule).
        let better = q.gt(ctx, *x, best);
        best = q.select_i64(ctx, *x, best, better);
        best_idx = q.gate.select(ctx, Constant(fe(i as i128)), best_idx, better);
    }
    out.int(best_idx);
    Ok(())
}
