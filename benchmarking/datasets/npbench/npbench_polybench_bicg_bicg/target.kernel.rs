//! npbench_polybench_bicg_bicg__kernel (NPBench polybench/bicg), halo2-base baseline.
//! Kernel: `(r @ A, A @ p)` on A: Float[M, N], p: Float[N], r: Float[M].
//! Each element of either product is one reduction, rescaled once (`q.dot`).

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const M: usize = 16; // = A_0
const N: usize = 13; // = A_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("A", &[M, N])?;
    let p = inputs.floats("p", &[N])?;
    let r = inputs.floats("r", &[M])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let p = q.load_floats(ctx, &p);
    let r = q.load_floats(ctx, &r);
    let s: Vec<Cell> = (0..N)
        .map(|j| {
            let column: Vec<Cell> = (0..M).map(|i| a[i * N + j]).collect();
            q.dot(ctx, &r, &column).cell
        })
        .collect();
    let t: Vec<Cell> = a.chunks(N).map(|row| q.dot(ctx, row, &p).cell).collect();
    out.floats(&s);
    out.floats(&t);
    Ok(())
}
