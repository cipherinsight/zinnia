//! pythran_l2norm__l2_norm (Pythran l2norm), halo2-base baseline.
//! Kernel: `np.sqrt(np.sum(np.abs(x) ** 2, 1))` on `x: Float[N, COLS]`.
//! `np.abs(x) ** 2` is `x ** 2`, so the absolute value is skipped; each row's
//! sum of squares is one reduction, rescaled once (`q.dot`, NPB_BASELINES.md
//! section 5, rule 3), and its square root is the exact floor square root.

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
            let squares = q.dot(ctx, row, row);
            q.sqrt(ctx, squares.cell).cell
        })
        .collect();
    out.floats(&norms);
    Ok(())
}
