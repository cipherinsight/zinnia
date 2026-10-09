//! npbench_polybench_atax_atax__kernel, halo2-base baseline.
//! Literal translation of kernel.py (LITERAL_BASELINES.md).
//! `return A @ x @ A` on A: Float[M, N], x: Float[N]. Python evaluates `@`
//! left to right: `y = A @ x` (M dot products of a row of A with x), then
//! `y @ A` (N dot products of y with a column of A). Each `@` is one NumPy
//! reduction, so each dot product is a single term-by-term sum rescaled once
//! (L5: "np.dot, @ ... the single rescale ... applies only to these
//! one-call reductions").

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const M: usize = 13; // = A_0
const N: usize = 16; // = A_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("A", &[M, N])?;
    let x = inputs.floats("x", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let x = q.load_floats(ctx, &x);

    // y = A @ x
    let y: Vec<Cell> = a.chunks(N).map(|row| q.dot(ctx, row, &x).cell).collect();

    // return y @ A
    let result: Vec<Cell> = (0..N)
        .map(|j| {
            let column: Vec<Cell> = (0..M).map(|i| a[i * N + j]).collect();
            q.dot(ctx, &y, &column).cell
        })
        .collect();
    out.floats(&result);
    Ok(())
}
