//! npbench_polybench_symm_symm__kernel (NPBench polybench/symm), halo2-base baseline.
//! Kernel: on alpha, beta: Float, C: Float[M, N], A: Float[M, M],
//! B: Float[M, N], in place (the output is `C` after the call):
//!
//! ```text
//! C *= beta
//! for i: for j:
//!     C[:i, j] += alpha * B[i, j] * A[i, :i]
//!     temp2[j] = B[:i, j] @ A[i, :i]
//! C[i, :] += alpha * B[i, :] * A[i, i] + alpha * temp2   (inside the loop over i)
//! ```
//!
//! In NumPy's evaluation order, every product is rounded: `alpha * B[i, j]`
//! (the same product as row `i` of `alpha * B[i, :]`, computed once), then
//! its products with `A[i, :i]` and `A[i, i]`, and `alpha * temp2[j]`;
//! `temp2[j]` is one reduction, rescaled once (`q.dot`); for `i = 0` it is the
//! empty 0.0, and `alpha * 0.0` is 0. The additions are exact. Only the lower
//! triangle of `A` (with the diagonal) is read, and only it is loaded.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const M: usize = 10; // = C_0
const N: usize = 13; // = C_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let alpha = inputs.float("alpha")?;
    let beta = inputs.float("beta")?;
    let c = inputs.floats("C", &[M, N])?;
    let a = inputs.floats("A", &[M, M])?;
    let b = inputs.floats("B", &[M, N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let alpha = q.load_float(ctx, alpha);
    let beta = q.load_float(ctx, beta);
    let c = q.load_floats(ctx, &c);
    // a_low[i] = A[i, :i + 1]
    let a_low: Vec<Vec<Cell>> = (0..M).map(|i| (0..=i).map(|k| q.load_float(ctx, a[i * M + k])).collect()).collect();
    let b = q.load_floats(ctx, &b);

    let mut c: Vec<Cell> = c.iter().map(|x| q.mul(ctx, *x, beta).cell).collect();
    for i in 0..M {
        for j in 0..N {
            let alpha_b = q.mul(ctx, alpha, b[i * N + j]).cell;
            for k in 0..i {
                let term = q.mul(ctx, alpha_b, a_low[i][k]);
                c[k * N + j] = q.add(ctx, c[k * N + j], term.cell);
            }
            let diagonal = q.mul(ctx, alpha_b, a_low[i][i]);
            let mut update = diagonal.cell;
            if i > 0 {
                let column: Vec<Cell> = (0..i).map(|k| b[k * N + j]).collect();
                let temp2 = q.dot(ctx, &column, &a_low[i][..i]);
                let scaled = q.mul(ctx, alpha, temp2.cell);
                update = q.add(ctx, update, scaled.cell);
            }
            c[i * N + j] = q.add(ctx, c[i * N + j], update);
        }
    }
    out.floats(&c);
    Ok(())
}
