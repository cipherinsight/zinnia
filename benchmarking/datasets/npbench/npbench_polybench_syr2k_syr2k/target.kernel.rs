//! npbench_polybench_syr2k_syr2k__kernel (NPBench polybench/syr2k), halo2-base baseline.
//! Kernel: for each row `i` of `C: Float[N, N]`, `C[i, :i + 1] *= beta`, then
//! for each `k < M`, `C[i, :i + 1] += A[:i + 1, k] * alpha * B[i, k] +
//! B[:i + 1, k] * alpha * A[i, k]`, with `A, B: Float[N, M]`, in place; the
//! output is `C` after the call (`mutated_outputs`; its upper triangle is the
//! input's).
//! NumPy's order (NPB_BASELINES.md section 5.3), left to right: every
//! `A[j, k] * alpha` and `B[j, k] * alpha` is rounded (the same value for
//! every row `i`, so computed once), then each product with `B[i, k]` or
//! `A[i, k]` is rounded; the updates add exactly, so each element sums its
//! terms once.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 13; // = A_0
const M: usize = 9; // = A_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let alpha = inputs.float("alpha")?;
    let beta = inputs.float("beta")?;
    let c = inputs.floats("C", &[N, N])?;
    let a = inputs.floats("A", &[N, M])?;
    let b = inputs.floats("B", &[N, M])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let alpha = q.load_float(ctx, alpha);
    let beta = q.load_float(ctx, beta);
    let mut c = q.load_floats(ctx, &c);
    let a = q.load_floats(ctx, &a);
    let b = q.load_floats(ctx, &b);
    let a_alpha: Vec<Cell> = a.iter().map(|x| q.mul(ctx, *x, alpha).cell).collect();
    let b_alpha: Vec<Cell> = b.iter().map(|x| q.mul(ctx, *x, alpha).cell).collect();
    for i in 0..N {
        let mut terms: Vec<Vec<Cell>> = (0..=i).map(|j| vec![q.mul(ctx, c[i * N + j], beta).cell]).collect();
        for k in 0..M {
            for (j, t) in terms.iter_mut().enumerate() {
                t.push(q.mul(ctx, a_alpha[j * M + k], b[i * M + k]).cell);
                t.push(q.mul(ctx, b_alpha[j * M + k], a[i * M + k]).cell);
            }
        }
        for (j, t) in terms.into_iter().enumerate() {
            c[i * N + j] = q.sum(ctx, t);
        }
    }
    out.floats(&c);
    Ok(())
}
