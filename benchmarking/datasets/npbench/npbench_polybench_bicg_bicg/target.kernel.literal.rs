//! npbench_polybench_bicg_bicg__kernel, halo2-base baseline: literal
//! translation of kernel.py (LITERAL_BASELINES.md).
//! Kernel: `return (r @ A, A @ p)` on A: Float[M, N], p: Float[N], r: Float[M].
//! Each `@` is one NumPy reduction call, so L5's dot-product row applies: the
//! defining sum over the shared axis, with the single allowed rescale
//! (`q.dot`), one call per output element, in the tuple's source order.

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

    // r @ A: element j is the dot of r with A's column j.
    let s: Vec<Cell> = (0..N)
        .map(|j| {
            let column: Vec<Cell> = (0..M).map(|i| a[i * N + j]).collect();
            q.dot(ctx, &r, &column).cell
        })
        .collect();
    // A @ p: element i is the dot of A's row i with p.
    let t: Vec<Cell> = a.chunks(N).map(|row| q.dot(ctx, row, &p).cell).collect();

    out.floats(&s);
    out.floats(&t);
    Ok(())
}
