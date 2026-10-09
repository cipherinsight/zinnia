//! pythran_matmul__matrix_multiply (Pythran matmul), halo2-base baseline.
//! Kernel: the triple loop `r += m0[i][k] * m1[k][j]` for m0: Float[N, K],
//! m1: Float[K, P], returning the `N x P` products as nested lists.
//! (`zero(n, m)` builds `m` rows of `n`, so the kernel runs only for
//! `N == P`, as at the fixture's shape.)
//! Each output accumulates its products into one variable, so it is one
//! reduction rescaled once (`q.dot`, NPB_BASELINES.md section 5.3).

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = m0_0
const K: usize = 16; // = m0_1
const P: usize = 16; // = m1_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let m0 = inputs.floats("m0", &[N, K])?;
    let m1 = inputs.floats("m1", &[K, P])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let m0 = q.load_floats(ctx, &m0);
    let m1 = q.load_floats(ctx, &m1);
    let columns: Vec<Vec<Cell>> = (0..P).map(|j| (0..K).map(|k| m1[k * P + j]).collect()).collect();
    let result: Vec<Cell> = m0
        .chunks(K)
        .flat_map(|row| columns.iter().map(|column| q.dot(ctx, row, column).cell).collect::<Vec<_>>())
        .collect();
    out.floats(&result);
    Ok(())
}
