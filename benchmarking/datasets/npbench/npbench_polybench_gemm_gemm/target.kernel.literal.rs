//! npbench_polybench_gemm_gemm__kernel, halo2-base literal baseline.
//! Literal translation of kernel.py (LITERAL_BASELINES.md).
//! Source: `C[:] = alpha * A @ B + beta * C` on C: Float[NI, NJ],
//! A: Float[NI, NK], B: Float[NK, NJ], scalars alpha, beta.
//! Translation choices:
//! - NumPy evaluates this in its usual left-to-right, broadcast-then-reduce
//!   order: `alpha * A` is a fresh array, one rounded product per element
//!   (L5 elementwise row); `(alpha * A) @ B` is one reduction per output
//!   element, rescaled once (L5 single-call-reduction row, `q.dot`);
//!   `beta * C` is a fresh array, one rounded product per element; the final
//!   `+` is exact elementwise addition. No fusing of the two `*` with the
//!   `@` or the `+`.
//! - The output is `C` after the call (`mutated_outputs`).

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const NI: usize = 14; // = C_0
const NJ: usize = 15; // = C_1
const NK: usize = 16; // = A_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let alpha = inputs.float("alpha")?;
    let beta = inputs.float("beta")?;
    let c = inputs.floats("C", &[NI, NJ])?;
    let a = inputs.floats("A", &[NI, NK])?;
    let b = inputs.floats("B", &[NK, NJ])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let alpha = q.load_float(ctx, alpha);
    let beta = q.load_float(ctx, beta);
    let c = q.load_floats(ctx, &c);
    let a = q.load_floats(ctx, &a);
    let b = q.load_floats(ctx, &b);

    // alpha * A (fresh array, one rounded product per element)
    let alpha_a: Vec<Cell> = a.iter().map(|x| q.mul(ctx, alpha, *x).cell).collect();
    // (alpha * A) @ B: one reduction per output element, rescaled once.
    let columns: Vec<Vec<Cell>> = (0..NJ)
        .map(|j| (0..NK).map(|k| b[k * NJ + j]).collect())
        .collect();
    let mut result: Vec<Cell> = Vec::with_capacity(NI * NJ);
    for i in 0..NI {
        let row = &alpha_a[i * NK..(i + 1) * NK];
        for (j, column) in columns.iter().enumerate() {
            let product = q.dot(ctx, row, column);
            result.push(product.cell);
        }
    }
    // beta * C (fresh array, one rounded product per element)
    let beta_c: Vec<Cell> = c.iter().map(|x| q.mul(ctx, beta, *x).cell).collect();
    let result: Vec<Cell> = result
        .iter()
        .zip(&beta_c)
        .map(|(x, y)| q.add(ctx, *x, *y))
        .collect();
    out.floats(&result);
    Ok(())
}
