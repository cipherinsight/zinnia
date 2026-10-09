//! npbench_polybench_gemm_gemm__kernel (NPBench polybench/gemm), halo2-base baseline.
//! Kernel: `C[:] = alpha * A @ B + beta * C` on `C: Float[NI, NJ]`,
//! `A: Float[NI, NK]`, `B: Float[NK, NJ]` and scalars `alpha`, `beta`, in
//! place; the output is `C` after the call (`mutated_outputs`).
//! NumPy's order: `alpha * A` is materialized (each element rounded), then
//! `@ B` (each element one reduction, rescaled once), `beta * C` is rounded
//! elementwise, and the sum is exact.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const NI: usize = 14; // = C_0
const NJ: usize = 15; // = C_1
const NK: usize = 16; // = A_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let alpha = inputs.float("alpha")?;
    let beta = inputs.float("beta")?;
    let c = inputs.floats("C", &[NI, NJ])?;
    let a = inputs.floats("A", &[NI, NK])?;
    let b = inputs.floats("B", &[NK, NJ])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let alpha = q.load_float(ctx, alpha);
    let beta = q.load_float(ctx, beta);
    let c = q.load_floats(ctx, &c);
    let a = q.load_floats(ctx, &a);
    let b = q.load_floats(ctx, &b);
    let alpha_a: Vec<Cell> = a.iter().map(|x| q.mul(ctx, alpha, *x).cell).collect();
    let columns: Vec<Vec<Cell>> = (0..NJ).map(|j| (0..NK).map(|k| b[k * NJ + j]).collect()).collect();
    let mut result: Vec<Cell> = Vec::with_capacity(NI * NJ);
    for i in 0..NI {
        for (j, column) in columns.iter().enumerate() {
            let product = q.dot(ctx, &alpha_a[i * NK..(i + 1) * NK], column);
            let scaled = q.mul(ctx, beta, c[i * NJ + j]);
            result.push(q.add(ctx, product.cell, scaled.cell));
        }
    }
    out.floats(&result);
    Ok(())
}
