//! pythran_allpairs__sqr_dists (Pythran allpairs), halo2-base baseline.
//! Kernel: `np.array([[np.sum((x - y) ** 2) for x in X] for y in Y])` on
//! X: Float[M, N], Y: Float[P, N]; the output is `P x M`, row `j` holding the
//! squared distances from `Y[j]` to every row of `X`.
//! Preserve (x - y) ** 2 followed by sum: each square is rounded before
//! reduction. Compute the source's pairwise differences directly.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.sqr_dists.json.
const M: usize = 16; // = X_0
const N: usize = 16; // = X_1
const P: usize = 16; // = Y_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.floats("X", &[M, N])?;
    let y = inputs.floats("Y", &[P, N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x = q.load_floats(ctx, &x);
    let y = q.load_floats(ctx, &y);
    let x_rows: Vec<&[Cell]> = x.chunks(N).collect();
    let y_rows: Vec<&[Cell]> = y.chunks(N).collect();
    let mut result = Vec::with_capacity(P * M);
    for j in 0..P {
        for i in 0..M {
            let squares: Vec<Cell> = x_rows[i].iter().zip(y_rows[j]).map(|(a, b)| {
                let diff = q.sub(ctx, *a, *b);
                q.mul(ctx, diff, diff).cell
            }).collect();
            result.push(q.sum(ctx, squares));
        }
    }
    out.floats(&result);
    Ok(())
}
