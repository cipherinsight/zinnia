//! pythran_l2norm__l2_norm (Pythran l2norm), halo2-base baseline.
//! Kernel: `np.sqrt(np.sum(np.abs(x) ** 2, 1))` on `x: Float[N, COLS]`.
//! `np.abs(x) ** 2` is `x ** 2`, so the absolute value is skipped; each row's
//! squares are rounded individually before the sum and square root.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = x_0
const COLS: usize = 16; // = x_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.floats("x", &[N, COLS])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x = q.load_floats(ctx, &x);
    let norms: Vec<Cell> = x
        .chunks(COLS)
        .map(|row| {
            let squares: Vec<Cell> = row.iter().map(|x| q.mul(ctx, *x, *x).cell).collect();
            let total = q.sum(ctx, squares);
            // Frozen Halo2 sqrt/guard requires a nonnegative signed64 radicand.
            q.range.range_check(ctx, total, 63);
            q.sqrt(ctx, total).cell
        })
        .collect();
    out.floats(&norms);
    Ok(())
}
