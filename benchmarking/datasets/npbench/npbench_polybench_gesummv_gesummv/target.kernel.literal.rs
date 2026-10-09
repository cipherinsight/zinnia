//! npbench_polybench_gesummv_gesummv__kernel, halo2-base baseline.
//! Literal translation of kernel.py (LITERAL_BASELINES.md).
//! `return alpha * A @ x + beta * B @ x` on A, B: Float[M, N], x: Float[N].
//! Python's `*` and `@` share one precedence level and group left to right,
//! so the source expression is `((alpha * A) @ x) + ((beta * B) @ x)`:
//! `alpha * A` is an elementwise array (every element a rounded product),
//! then each row's dot product with `x` is one NumPy reduction, rescaled
//! once (L5); the two terms are added elementwise.

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

    // Complete each array expression before evaluating the next one.
    let scaled_a: Vec<Cell> = a.iter().map(|v| q.mul(ctx, alpha, *v).cell).collect();
    let first: Vec<Cell> = scaled_a.chunks(N).map(|row| q.dot(ctx, row, &x).cell).collect();
    let scaled_b: Vec<Cell> = b.iter().map(|v| q.mul(ctx, beta, *v).cell).collect();
    let second: Vec<Cell> = scaled_b.chunks(N).map(|row| q.dot(ctx, row, &x).cell).collect();
    let result: Vec<Cell> = (0..M).map(|i| q.add(ctx, first[i], second[i])).collect();
    out.floats(&result);
    Ok(())
}
