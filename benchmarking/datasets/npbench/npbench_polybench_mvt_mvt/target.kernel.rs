//! npbench_polybench_mvt_mvt__kernel (NPBench polybench/mvt), halo2-base baseline.
//! Kernel: `x1 += A @ y_1; x2 += y_2 @ A` on `A: Float[N0, N1]` and vectors
//! `x1: Float[N0]`, `x2: Float[N1]`, `y_1: Float[N1]`, `y_2: Float[N0]`, in
//! place; the outputs are `x1` and `x2` after the call (`mutated_outputs`).
//! Each element of either product is one reduction, rescaled once (`q.dot`).

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
    let x1: Vec<Cell> = (0..N0)
        .map(|i| {
            let product = q.dot(ctx, &a[i * N1..(i + 1) * N1], &y_1);
            q.add(ctx, x1[i], product.cell)
        })
        .collect();
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
