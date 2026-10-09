//! npbench_polybench_lu_lu__kernel (NPBench polybench/lu), halo2-base baseline.
//! Kernel: LU decomposition without pivoting of `A: Float[N, N]` in place
//! (Doolittle, row by row); the output is `A` after the call
//! (`mutated_outputs`). Row `i`: for `j < i`,
//! `A[i, j] = (A[i, j] - A[i, :j] @ A[:j, j]) / A[j, j]`; for `j >= i`,
//! `A[i, j] -= A[i, :i] @ A[:i, j]`.
//! Each `@` is one reduction, rescaled once (`q.dot`; an empty one is 0 and
//! costs nothing). The divisor `A[j, j]` can have either sign; its magnitude
//! is proven once (`abs_int64`, a sign bit and a 63-bit check) when row `j`
//! is done, and every quotient by it reuses it (`div_by_abs`). The last
//! diagonal element divides nothing, so it gets no such proof.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 15; // = A_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("A", &[N, N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let mut a = q.load_floats(ctx, &a);
    let at = |i: usize, j: usize| i * N + j;
    let mut diagonal_abs: Vec<Cell> = Vec::with_capacity(N);
    for i in 0..N {
        for j in 0..i {
            let numerator = minus_dot(&q, ctx, &a, i, j, j);
            a[at(i, j)] = q.div_by_abs(ctx, numerator, a[at(j, j)], diagonal_abs[j]).cell;
        }
        for j in i..N {
            a[at(i, j)] = minus_dot(&q, ctx, &a, i, j, i);
        }
        if i + 1 < N {
            diagonal_abs.push(q.abs_int64(ctx, a[at(i, i)]));
        }
    }
    out.floats(&a);
    Ok(())
}

/// `A[i, j] - A[i, :len] @ A[:len, j]`.
fn minus_dot(q: &Q32, ctx: &mut Ctx, a: &[Cell], i: usize, j: usize, len: usize) -> Cell {
    if len == 0 {
        return a[i * N + j];
    }
    let row = &a[i * N..i * N + len];
    let column: Vec<Cell> = (0..len).map(|k| a[k * N + j]).collect();
    let dot = q.dot(ctx, row, &column);
    q.sub(ctx, a[i * N + j], dot.cell)
}
