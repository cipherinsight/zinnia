//! npbench_polybench_syrk_syrk__kernel (NPBench polybench/syrk), halo2-base baseline.
//! Kernel: for each row `i` of `C: Float[N, N]`, `C[i, :i + 1] *= beta`, then
//! for each `k < M`, `C[i, :i + 1] += alpha * A[i, k] * A[:i + 1, k]`, with
//! `A: Float[N, M]`, in place; the output is `C` after the call
//! (`mutated_outputs`; its upper triangle is the input's).
//! NumPy's order (NPB_BASELINES.md section 5.3): `alpha * A[i, k]` is a
//! rounded scalar, computed once per `(i, k)`; each element's product with it
//! is rounded; the updates add exactly, so each element sums its terms once.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 14; // = A_0
const M: usize = 10; // = A_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let alpha = inputs.float("alpha")?;
    let beta = inputs.float("beta")?;
    let c = inputs.floats("C", &[N, N])?;
    let a = inputs.floats("A", &[N, M])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let alpha = q.load_float(ctx, alpha);
    let beta = q.load_float(ctx, beta);
    let mut c = q.load_floats(ctx, &c);
    let a = q.load_floats(ctx, &a);
    for i in 0..N {
        let mut terms: Vec<Vec<Cell>> = (0..=i).map(|j| vec![q.mul(ctx, c[i * N + j], beta).cell]).collect();
        for k in 0..M {
            let scale = q.mul(ctx, alpha, a[i * M + k]);
            for (j, t) in terms.iter_mut().enumerate() {
                t.push(q.mul(ctx, scale, a[j * M + k]).cell);
            }
        }
        for (j, t) in terms.into_iter().enumerate() {
            c[i * N + j] = q.sum(ctx, t);
        }
    }
    out.floats(&c);
    Ok(())
}
