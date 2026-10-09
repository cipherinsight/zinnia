//! npbench_polybench_lu_lu__kernel, halo2-base baseline.
//! Literal translation of kernel.py (LITERAL_BASELINES.md).
//! Each @ is a dot with one rescale; empty dots are constant zero.
//! Subtraction remains explicit even for an empty dot. Division uses the
//! shared canonical_nonzero domain, as in the optimized statement.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 15; // = A_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("A", &[N, N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let mut a = q.load_floats(ctx, &a);
    let at = |i: usize, j: usize| i * N + j;
    for i in 0..N {
        for j in 0..i {
            // A[i, j] -= A[i, :j] @ A[:j, j]
            let after_sub = minus_dot(&q, ctx, &a, i, j, j);
            // A[i, j] /= A[j, j]
            a[at(i, j)] = q.canonical_div_nonzero(ctx, after_sub, a[at(j, j)]).cell;
        }
        for j in i..N {
            // A[i, j] -= A[i, :i] @ A[:i, j]
            a[at(i, j)] = minus_dot(&q, ctx, &a, i, j, i);
        }
    }
    out.floats(&a);
    Ok(())
}

/// `A[i, j] - A[i, :len] @ A[:len, j]` (`A[i, :0] @ A[:0, j]` is 0).
fn minus_dot(q: &Q32, ctx: &mut Ctx, a: &[Cell], i: usize, j: usize, len: usize) -> Cell {
    if len == 0 {
        return q.sub(ctx, a[i * N + j], Constant(Fr::ZERO));
    }
    let row = &a[i * N..i * N + len];
    let column: Vec<Cell> = (0..len).map(|k| a[k * N + j]).collect();
    let dot = q.dot(ctx, row, &column);
    q.sub(ctx, a[i * N + j], dot.cell)
}
