//! npbench_polybench_k2mm_k2mm__kernel (NPBench polybench/k2mm), halo2-base baseline.
//! Kernel: `D[:] = alpha * A @ B @ C + beta * D` on A: Float[NI, NK],
//! B: Float[NK, NJ], C: Float[NJ, NL], D: Float[NI, NL]; the output is `D`
//! after the call (`mutated_outputs`).
//! NumPy evaluates `((alpha * A) @ B) @ C`: every element of `alpha * A` and
//! of `beta * D` is a rounded product, every element of either matrix
//! product is one reduction, rescaled once (`q.dot`).

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const NI: usize = 14; // = A_0
const NK: usize = 15; // = A_1
const NJ: usize = 15; // = B_1
const NL: usize = 16; // = C_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let alpha = inputs.float("alpha")?;
    let beta = inputs.float("beta")?;
    let a = inputs.floats("A", &[NI, NK])?;
    let b = inputs.floats("B", &[NK, NJ])?;
    let c = inputs.floats("C", &[NJ, NL])?;
    let d = inputs.floats("D", &[NI, NL])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let alpha = q.load_float(ctx, alpha);
    let beta = q.load_float(ctx, beta);
    let a = q.load_floats(ctx, &a);
    let b = q.load_floats(ctx, &b);
    let c = q.load_floats(ctx, &c);
    let d = q.load_floats(ctx, &d);

    let alpha_a: Vec<Cell> = a.iter().map(|x| q.mul(ctx, alpha, *x).cell).collect();
    let ab = matmul(&q, ctx, &alpha_a, &b, NI, NK, NJ);
    let abc = matmul(&q, ctx, &ab, &c, NI, NJ, NL);
    let result: Vec<Cell> = abc
        .iter()
        .zip(&d)
        .map(|(x, y)| {
            let beta_d = q.mul(ctx, beta, *y);
            q.add(ctx, *x, beta_d)
        })
        .collect();
    out.floats(&result);
    Ok(())
}

/// `x @ y` for x: [rows, inner] and y: [inner, cols], row-major.
fn matmul(q: &Q32, ctx: &mut Ctx, x: &[Cell], y: &[Cell], rows: usize, inner: usize, cols: usize) -> Vec<Cell> {
    let columns: Vec<Vec<Cell>> = (0..cols).map(|j| (0..inner).map(|k| y[k * cols + j]).collect()).collect();
    x.chunks(inner)
        .take(rows)
        .flat_map(|row| columns.iter().map(|column| q.dot(ctx, row, column).cell).collect::<Vec<_>>())
        .collect()
}
