//! npbench_polybench_trisolv_trisolv__kernel (NPBench polybench/trisolv), halo2-base baseline.
//! Kernel: forward substitution `x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]`
//! for `i` in order, on L: Float[N, N], x, b: Float[N], in place; the output
//! is `x` after the call (`mutated_outputs`).
//! Each `L[i, :i] @ x[:i]` is one reduction, rescaled once (`q.dot`; the
//! empty one for `i = 0` is 0). The input `x` is overwritten before it is
//! read and the strict upper triangle of `L` is never read, so neither is
//! loaded. Each division takes the diagonal element's magnitude from
//! `q.abs_int64` (`q.div_by_abs`: 7,171 advice cells, against 7,379 for
//! `q.div`'s sign test on the input's window proof).

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = L_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let l = inputs.floats("L", &[N, N])?;
    let b = inputs.floats("b", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let b = q.load_floats(ctx, &b);
    let mut x: Vec<Cell> = Vec::with_capacity(N);
    for i in 0..N {
        let row = q.load_floats(ctx, &l[i * N..i * N + i]);
        let diagonal = q.load_float(ctx, l[i * N + i]);
        let magnitude = q.abs_int64(ctx, diagonal);
        let rest = if i == 0 {
            b[0]
        } else {
            let done = q.dot(ctx, &row, &x);
            q.sub(ctx, b[i], done.cell)
        };
        x.push(q.div_by_abs(ctx, rest, diagonal, magnitude).cell);
    }
    out.floats(&x);
    Ok(())
}
