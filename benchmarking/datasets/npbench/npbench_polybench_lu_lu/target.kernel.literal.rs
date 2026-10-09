//! npbench_polybench_lu_lu__kernel, halo2-base baseline.
//! Literal translation of kernel.py (LITERAL_BASELINES.md).
//! LU decomposition without pivoting of `A: Float[N, N]` in place (Doolittle,
//! row by row); the output is `A` after the call (`mutated_outputs`). Source:
//!   for i in range(N):
//!       for j in range(i):
//!           A[i, j] -= A[i, :j] @ A[:j, j]
//!           A[i, j] /= A[j, j]
//!       for j in range(i, N):
//!           A[i, j] -= A[i, :i] @ A[:i, j]
//! Both `@` calls are single NumPy reductions (one rescale each, L5); an
//! empty slice (`j == 0` or `i == 0`) makes the dot product's defining sum
//! empty, i.e. 0, so the preceding subtraction is skipped rather than
//! computed as "minus zero" (the same value either way, no extra gate). The
//! two assignments to `A[i, j]` in the `j < i` branch are two source
//! statements executed in order: the second reads the value the first just
//! wrote, as the source does; this is not common-subexpression elimination,
//! there is nothing recomputed. `/` uses the division gadget the arithmetic
//! contract shares across baselines (`canonical_div_nonzero`): the dividend
//! after subtraction is not bounded to signed64 by this point in the
//! recurrence, which only this gadget's domain covers.

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
        return a[i * N + j];
    }
    let row = &a[i * N..i * N + len];
    let column: Vec<Cell> = (0..len).map(|k| a[k * N + j]).collect();
    let dot = q.dot(ctx, row, &column);
    q.sub(ctx, a[i * N + j], dot.cell)
}
