//! pythran_euclidean_distance_square__euclidean_distance_square (Pythran
//! euclidean_distance_square), halo2-base baseline.
//! Kernel: `-2 * np.dot(x1, x2.T) + np.sum(np.square(x1), axis=1)[:, None]
//! + np.sum(np.square(x2), axis=1)` on `x1: Float[R1, D]`, `x2: Float[R2, D]`.
//! Each dot product and each row sum of squares is one reduction, rescaled
//! once (`q.dot`; NPB_BASELINES.md section 5.3); `-2 *` is exact.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const R1: usize = 1; // = x1_0
const R2: usize = 5; // = x2_0
const D: usize = 3; // = x1_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x1 = inputs.floats("x1", &[R1, D])?;
    let x2 = inputs.floats("x2", &[R2, D])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x1 = q.load_floats(ctx, &x1);
    let x2 = q.load_floats(ctx, &x2);
    let squares1: Vec<Cell> = x1.chunks(D).map(|row| q.dot(ctx, row, row).cell).collect();
    let squares2: Vec<Cell> = x2.chunks(D).map(|row| q.dot(ctx, row, row).cell).collect();
    let mut result: Vec<Cell> = Vec::with_capacity(R1 * R2);
    for (row1, square1) in x1.chunks(D).zip(&squares1) {
        for (row2, square2) in x2.chunks(D).zip(&squares2) {
            let gram = q.dot(ctx, row1, row2);
            let squares = q.add(ctx, *square1, *square2);
            result.push(q.gate.mul_add(ctx, gram.cell, Constant(-Fr::from(2)), squares));
        }
    }
    out.floats(&result);
    Ok(())
}
