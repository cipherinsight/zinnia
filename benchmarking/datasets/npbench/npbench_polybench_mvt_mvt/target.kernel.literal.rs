//! npbench_polybench_mvt_mvt__kernel, halo2-base literal baseline.
//! Literal translation of kernel.py (LITERAL_BASELINES.md).
//! Source:
//!   x1 += A @ y_1
//!   x2 += y_2 @ A
//! on A: Float[N0, N1], x1: Float[N0], x2: Float[N1], y_1: Float[N1],
//! y_2: Float[N0].
//! Translation choices:
//! - Each `@` is one library call, so L5 gives it one rescale per output
//!   element (`q.dot`); `+=` is exact addition onto the existing value,
//!   computed as its own statement after the matching `@`.
//! - The two statements are independent (`x1`'s update never reads `x2`,
//!   and vice versa) and are computed in source order; the outputs are `x1`
//!   and `x2` after the call (`mutated_outputs`).

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N0: usize = 16; // = A_0
const N1: usize = 16; // = A_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x1 = inputs.floats("x1", &[N0])?;
    let x2 = inputs.floats("x2", &[N1])?;
    let y_1 = inputs.floats("y_1", &[N1])?;
    let y_2 = inputs.floats("y_2", &[N0])?;
    let a = inputs.floats("A", &[N0, N1])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x1 = q.load_floats(ctx, &x1);
    let x2 = q.load_floats(ctx, &x2);
    let y_1 = q.load_floats(ctx, &y_1);
    let y_2 = q.load_floats(ctx, &y_2);
    let a = q.load_floats(ctx, &a);

    // x1 += A @ y_1
    let x1: Vec<Cell> = (0..N0)
        .map(|i| {
            let product = q.dot(ctx, &a[i * N1..(i + 1) * N1], &y_1);
            q.add(ctx, x1[i], product.cell)
        })
        .collect();
    // x2 += y_2 @ A
    let x2: Vec<Cell> = (0..N1)
        .map(|j| {
            let column: Vec<Cell> = (0..N0).map(|i| a[i * N1 + j]).collect();
            let product = q.dot(ctx, &y_2, &column);
            q.add(ctx, x2[j], product.cell)
        })
        .collect();
    out.floats(&x1);
    out.floats(&x2);
    Ok(())
}
