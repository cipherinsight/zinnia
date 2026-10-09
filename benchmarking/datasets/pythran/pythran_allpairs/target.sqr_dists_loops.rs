//! pythran_allpairs__sqr_dists_loops (Pythran allpairs), halo2-base baseline.
//! Kernel: `D[i, j] = np.sum((X[i] - Y[j]) ** 2)` for every row `i` of
//! X: Float[M, N] and row `j` of Y: Float[P, N]; the output is `D: M x P`.
//! Each sum of squares is one reduction, rescaled once (NPB_BASELINES.md 5.3).
//! Its raw value is computed as `|x|^2 + |y|^2 - 2 x . y`, which is the same
//! integer as `sum((x - y)^2)`: the squared norms are computed once per row,
//! so each pair costs one inner product.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.sqr_dists_loops.json.
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
    let raw_dot = |ctx: &mut Ctx, a: &[Cell], b: &[Cell]| {
        q.gate.inner_product(ctx, a.iter().map(|v| Existing(*v)), b.iter().map(|v| Existing(*v)))
    };
    let x_norms: Vec<Cell> = x_rows.iter().map(|r| raw_dot(ctx, r, r)).collect();
    let y_norms: Vec<Cell> = y_rows.iter().map(|r| raw_dot(ctx, r, r)).collect();
    let mut result = Vec::with_capacity(M * P);
    for i in 0..M {
        for j in 0..P {
            let cross = raw_dot(ctx, x_rows[i], y_rows[j]);
            let norms = q.add(ctx, x_norms[i], y_norms[j]);
            let raw = q.gate.mul_add(ctx, cross, Constant(-Fr::from(2)), norms);
            result.push(q.rescale(ctx, raw).cell);
        }
    }
    out.floats(&result);
    Ok(())
}
