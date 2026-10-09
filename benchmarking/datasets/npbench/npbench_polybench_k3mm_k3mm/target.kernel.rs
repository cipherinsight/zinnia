//! npbench_polybench_k3mm_k3mm__kernel (NPBench polybench/k3mm), halo2-base baseline.
//! Kernel: `A @ B @ C @ D` on A: Float[NI, NJ], B: Float[NJ, NK],
//! C: Float[NK, NL], D: Float[NL, NM], evaluated left to right. Each element
//! of each product is one reduction, rescaled once (`q.dot`).

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const NI: usize = 13; // = A_0
const NJ: usize = 15; // = A_1
const NK: usize = 14; // = B_1
const NL: usize = 16; // = C_1
const NM: usize = 16; // = D_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("A", &[NI, NJ])?;
    let b = inputs.floats("B", &[NJ, NK])?;
    let c = inputs.floats("C", &[NK, NL])?;
    let d = inputs.floats("D", &[NL, NM])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let b = q.load_floats(ctx, &b);
    let c = q.load_floats(ctx, &c);
    let d = q.load_floats(ctx, &d);
    let ab = matmul(&q, ctx, &a, &b, NI, NJ, NK);
    let abc = matmul(&q, ctx, &ab, &c, NI, NK, NL);
    let abcd = matmul(&q, ctx, &abc, &d, NI, NL, NM);
    out.floats(&abcd);
    Ok(())
}

/// `x @ y` for `x: [rows, inner]` and `y: [inner, cols]`, row-major.
fn matmul(q: &Q32, ctx: &mut Ctx, x: &[Cell], y: &[Cell], rows: usize, inner: usize, cols: usize) -> Vec<Cell> {
    let columns: Vec<Vec<Cell>> = (0..cols).map(|j| (0..inner).map(|k| y[k * cols + j]).collect()).collect();
    (0..rows)
        .flat_map(|i| {
            let row = &x[i * inner..(i + 1) * inner];
            columns.iter().map(|column| q.dot(ctx, row, column).cell).collect::<Vec<_>>()
        })
        .collect()
}
