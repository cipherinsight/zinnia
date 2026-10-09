//! npbench_polybench_gesummv_gesummv__kernel (NPBench polybench/gesummv), halo2-base baseline.
//! Kernel: `alpha * A @ x + beta * B @ x` on A, B: Float[M, N], x: Float[N].
//! `*` and `@` group left to right, so each term is `(alpha * A) @ x`: every
//! element of `alpha * A` is a rounded product, then each row's dot product
//! with `x` is one reduction, rescaled once (`q.dot`).

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const M: usize = 16; // = A_0
const N: usize = 16; // = A_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let alpha = inputs.float("alpha")?;
    let beta = inputs.float("beta")?;
    let a = inputs.floats("A", &[M, N])?;
    let b = inputs.floats("B", &[M, N])?;
    let x = inputs.floats("x", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let alpha = q.load_float(ctx, alpha);
    let beta = q.load_float(ctx, beta);
    let a = q.load_floats(ctx, &a);
    let b = q.load_floats(ctx, &b);
    let x = q.load_floats(ctx, &x);
    let scaled_dot = |ctx: &mut Ctx, factor: Cell, row: &[Cell]| -> Cell {
        let scaled: Vec<Cell> = row.iter().map(|v| q.mul(ctx, factor, *v).cell).collect();
        q.dot(ctx, &scaled, &x).cell
    };
    let result: Vec<Cell> = (0..M)
        .map(|i| {
            let first = scaled_dot(ctx, alpha, &a[i * N..(i + 1) * N]);
            let second = scaled_dot(ctx, beta, &b[i * N..(i + 1) * N]);
            q.add(ctx, first, second)
        })
        .collect();
    out.floats(&result);
    Ok(())
}
