//! npbench_polybench_syr2k_syr2k__kernel, halo2-base baseline.
//! Literal translation of kernel.py (LITERAL_BASELINES.md).
//! Source:
//!   for i in range(N):
//!       C[i, :i + 1] *= beta
//!       for k in range(M):
//!           C[i, :i + 1] += A[:i + 1, k] * alpha * B[i, k]
//!                          + B[:i + 1, k] * alpha * A[i, k]
//! `C[i, :i + 1] *= beta` is an elementwise array statement: a loop over
//! `j in 0..=i` multiplying each element by `beta` (L5, "element-wise ops,
//! broadcasting"). The second statement's right-hand side is, left to right,
//! `(A[:i+1,k] * alpha) * B[i,k] + (B[:i+1,k] * alpha) * A[i,k]`: for each
//! `j`, `A[j, k] * alpha` and `B[j, k] * alpha` are rounded, each multiplied
//! again by the opposite row's scalar, the two products added, then added
//! into `C[i, j]` (NPB_FAIRNESS.md item 3: products rounded before
//! accumulation, additions exact). `A[j, k] * alpha` does not depend on `i`,
//! but the source recomputes it fresh every time the statement executes
//! (once per `(i, k)` for each `j <= i`, i.e. up to `N - j` times over the
//! outer loop for a given `(j, k)`); a literal translation computes it the
//! same number of times, not once (L3: no hoisting of loop-invariant work).

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

    for i in 0..N {
        // C[i, :i + 1] *= beta
        for j in 0..=i {
            c[i * N + j] = q.mul(ctx, c[i * N + j], beta).cell;
        }
        // for k in range(M): C[i, :i + 1] += ...
        for k in 0..M {
            let aa: Vec<Cell> = (0..=i)
                .map(|j| q.mul(ctx, a[j * M + k], alpha).cell)
                .collect();
            let first: Vec<Cell> = aa
                .iter()
                .map(|v| q.mul(ctx, *v, b[i * M + k]).cell)
                .collect();
            let ba: Vec<Cell> = (0..=i)
                .map(|j| q.mul(ctx, b[j * M + k], alpha).cell)
                .collect();
            let second: Vec<Cell> = ba
                .iter()
                .map(|v| q.mul(ctx, *v, a[i * M + k]).cell)
                .collect();
            let rhs: Vec<Cell> = (0..=i).map(|j| q.add(ctx, first[j], second[j])).collect();
            for j in 0..=i {
                c[i * N + j] = q.add(ctx, c[i * N + j], rhs[j]);
            }
        }
    }
    out.floats(&c);
    Ok(())
}
