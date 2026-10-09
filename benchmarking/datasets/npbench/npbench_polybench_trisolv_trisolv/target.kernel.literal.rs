// npbench_polybench_trisolv_trisolv__kernel: literal translation of kernel.py (LITERAL_BASELINES.md).
// Empty dot is constant zero; each nonempty dot has one rescale; division uses a fresh bound.
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
        let diagonal = q.bound(ctx, diagonal);
        let dot = if i == 0 {
            q.float_const(ctx, 0.0)
        } else {
            let dot = q.dot(ctx, &row, &x);
            dot.cell
        };
        let rest = q.sub(ctx, b[i], dot);
        x.push(q.div(ctx, rest, diagonal).cell);
    }
    out.floats(&x);
    Ok(())
}
