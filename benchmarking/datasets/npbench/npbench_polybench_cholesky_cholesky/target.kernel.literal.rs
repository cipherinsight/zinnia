//! npbench_polybench_cholesky_cholesky__kernel, halo2-base literal baseline.
//! Literal translation of kernel.py (LITERAL_BASELINES.md).
//! Source:
//!   A[0, 0] = np.sqrt(A[0, 0])
//!   for i in range(1, N):
//!       for j in range(i):
//!           A[i, j] -= np.dot(A[i, :j], A[j, :j])
//!           A[i, j] /= A[j, j]
//!       A[i, i] -= np.dot(A[i, :i], A[i, :i])
//!       A[i, i] = np.sqrt(A[i, i])
//! Translation choices:
//! - Each `np.dot` is one library call, so L5 gives it one rescale (`q.dot`);
//!   an empty slice (`j == 0`) is the empty sum 0, matching NumPy.
//! - `/=` by `A[j, j]`: by the time the loop reaches this statement, `A[j, j]`
//!   already holds `sqrt(...)` from its own earlier iteration (the same
//!   source assignment, read back), which is non-negative whenever NumPy's
//!   radicand is (NumPy has no witness otherwise); the quotient gadget for a
//!   non-negative divisor applies (`q.div_pos`). This carries the `I64`
//!   `q.sqrt` already produced for that one array slot into the one division
//!   that reads it, not a proof borrowed from an unrelated operation.
//! - In-place mutation of `A` over the `(i, j)` loop nest is a sequence of
//!   scalar assignments in source order; the output is all of `A` after the
//!   call (`mutated_outputs`), including the untouched upper triangle.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 15; // = A_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("A", &[N, N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let mut a = q.load_floats(ctx, &a);
    let at = |i: usize, j: usize| i * N + j;
    // The I64 each diagonal slot holds once np.sqrt has written it, indexed
    // by row; read back by the `/=` statement of later rows.
    let mut diagonal: Vec<I64> = Vec::with_capacity(N);

    // A[0, 0] = np.sqrt(A[0, 0])
    let root = q.sqrt(ctx, a[at(0, 0)]);
    a[at(0, 0)] = root.cell;
    diagonal.push(root);

    for i in 1..N {
        for j in 0..i {
            // A[i, j] -= np.dot(A[i, :j], A[j, :j])
            let mut value = a[at(i, j)];
            if j > 0 {
                let dot = q.dot(ctx, &a[at(i, 0)..at(i, j)], &a[at(j, 0)..at(j, j)]);
                value = q.sub(ctx, value, dot.cell);
            }
            // A[i, j] /= A[j, j]
            a[at(i, j)] = q.div_pos(ctx, value, diagonal[j]).cell;
        }
        // A[i, i] -= np.dot(A[i, :i], A[i, :i])
        let dot = q.dot(ctx, &a[at(i, 0)..at(i, i)], &a[at(i, 0)..at(i, i)]);
        let radicand = q.sub(ctx, a[at(i, i)], dot.cell);
        // A[i, i] = np.sqrt(A[i, i])
        let root = q.sqrt(ctx, radicand);
        a[at(i, i)] = root.cell;
        diagonal.push(root);
    }
    out.floats(&a);
    Ok(())
}
