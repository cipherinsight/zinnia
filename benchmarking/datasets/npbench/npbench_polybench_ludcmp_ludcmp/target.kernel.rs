//! npbench_polybench_ludcmp_ludcmp__kernel (NPBench polybench/ludcmp), halo2-base baseline.
//! Kernel: LU decomposition without pivoting of `A: Float[N, N]` (as
//! polybench/lu), then `L y = b` by forward substitution
//! (`y[i] = b[i] - A[i, :i] @ y[:i]`) and `U x = y` by back substitution
//! (`x[i] = (y[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]`); returns `(x, y)`.
//! `A` is written but not returned, so it is no output.
//! Each `@` is one reduction, rescaled once (`q.dot`; an empty one is 0 and
//! costs nothing). Every diagonal element divides, with either sign: its
//! magnitude is proven once (`abs_int64`, a sign bit and a 63-bit check)
//! and every quotient by it reuses it (`div_by_abs`).

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 15; // = A_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("A", &[N, N])?;
    let b = inputs.floats("b", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let mut a = q.load_floats(ctx, &a);
    let b = q.load_floats(ctx, &b);
    let at = |i: usize, j: usize| i * N + j;
    let mut diagonal_abs: Vec<Cell> = Vec::with_capacity(N);
    for i in 0..N {
        for j in 0..i {
            let column: Vec<Cell> = (0..j).map(|k| a[at(k, j)]).collect();
            let numerator = minus_dot(&q, ctx, a[at(i, j)], &a[at(i, 0)..at(i, j)], &column);
            a[at(i, j)] = q.div_by_abs(ctx, numerator, a[at(j, j)], diagonal_abs[j]).cell;
        }
        for j in i..N {
            let column: Vec<Cell> = (0..i).map(|k| a[at(k, j)]).collect();
            a[at(i, j)] = minus_dot(&q, ctx, a[at(i, j)], &a[at(i, 0)..at(i, i)], &column);
        }
        diagonal_abs.push(q.abs_int64(ctx, a[at(i, i)]));
    }
    // y[i] = b[i] - A[i, :i] @ y[:i]
    let mut y: Vec<Cell> = Vec::with_capacity(N);
    for i in 0..N {
        let yi = minus_dot(&q, ctx, b[i], &a[at(i, 0)..at(i, i)], &y);
        y.push(yi);
    }
    // x[i] = (y[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i], from the last row up
    let mut x: Vec<Cell> = vec![y[0]; N];
    for i in (0..N).rev() {
        let numerator = minus_dot(&q, ctx, y[i], &a[at(i, i + 1)..at(i + 1, 0)], &x[i + 1..]);
        x[i] = q.div_by_abs(ctx, numerator, a[at(i, i)], diagonal_abs[i]).cell;
    }
    out.floats(&x);
    out.floats(&y);
    Ok(())
}

/// `value - row @ column` (`value` itself when they are empty).
fn minus_dot(q: &Q32, ctx: &mut Ctx, value: Cell, row: &[Cell], column: &[Cell]) -> Cell {
    if row.is_empty() {
        return value;
    }
    let dot = q.dot(ctx, row, column);
    q.sub(ctx, value, dot.cell)
}
