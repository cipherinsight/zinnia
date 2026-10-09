//! npbench_polybench_atax_atax__kernel (NPBench polybench/atax), halo2-base baseline.
//! Kernel: `A @ x @ A` on A: Float[M, N], x: Float[N], evaluated left to
//! right: `y = A @ x`, then `y @ A`.
//! Each element of either product is one reduction, rescaled once (`q.dot`).

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
    let y: Vec<Cell> = a.chunks(N).map(|row| q.dot(ctx, row, &x).cell).collect();
    let result: Vec<Cell> = (0..N)
        .map(|j| {
            let column: Vec<Cell> = (0..M).map(|i| a[i * N + j]).collect();
            q.dot(ctx, &y, &column).cell
        })
        .collect();
    out.floats(&result);
    Ok(())
}
