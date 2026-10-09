//! npbench_polybench_cholesky_cholesky__kernel (NPBench polybench/cholesky), halo2-base baseline.
//! Kernel: Cholesky factorization of `A: Float[N, N]` in place, row by row:
//! `A[i, j] = (A[i, j] - A[i, :j] @ A[j, :j]) / A[j, j]` for `j < i`, then
//! `A[i, i] = sqrt(A[i, i] - A[i, :i] @ A[i, :i])`; the output is all of `A`
//! after the call (`mutated_outputs`), the untouched upper triangle included.
//! Each `np.dot` is one reduction, rescaled once (`q.dot`; an empty one is 0
//! and costs nothing). The divisors are square roots, non-negative by
//! construction, so the quotients are `q.div_pos` (a zero divisor has no
//! witness, as NumPy's result is not finite; a negative radicand has none
//! either, as NumPy's is NaN).

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 15; // = A_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("A", &[N, N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let mut a = q.load_floats(ctx, &a);
    let at = |i: usize, j: usize| i * N + j;
    let mut diagonal: Vec<I64> = Vec::with_capacity(N);
    for i in 0..N {
        for j in 0..i {
            let numerator = minus_dot(&q, ctx, a[at(i, j)], &a[at(i, 0)..at(i, j)], &a[at(j, 0)..at(j, j)]);
            a[at(i, j)] = q.div_pos(ctx, numerator, diagonal[j]).cell;
        }
        let radicand = minus_dot(&q, ctx, a[at(i, i)], &a[at(i, 0)..at(i, i)], &a[at(i, 0)..at(i, i)]);
        let root = q.sqrt(ctx, radicand);
        a[at(i, i)] = root.cell;
        diagonal.push(root);
    }
    out.floats(&a);
    Ok(())
}

/// `value - x @ y` (`value` itself when they are empty).
fn minus_dot(q: &Q32, ctx: &mut Ctx, value: Cell, x: &[Cell], y: &[Cell]) -> Cell {
    if x.is_empty() {
        return value;
    }
    let dot = q.dot(ctx, x, y);
    q.sub(ctx, value, dot.cell)
}
