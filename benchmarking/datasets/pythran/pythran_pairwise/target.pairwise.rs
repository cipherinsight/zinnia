//! pythran_pairwise__pairwise (Pythran pairwise), halo2-base baseline.
//! Kernel: `D[i, j] = sqrt(sum_k (X[i, k] - X[j, k])^2)` on `X: Float[M, N]`.
//! Each sum of squares is one reduction, rescaled once (`q.dot`,
//! NPB_BASELINES.md section 5, rule 3), and its root the exact floor square
//! root. `D[i, j]` and `D[j, i]` are the same computation (`(a - b)^2` and
//! `(b - a)^2` are equal), so each pair is computed once and output twice;
//! the diagonal is `sqrt(0) = 0` for every input, a constant output (not
//! public, CONVENTIONS 1.3).

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const M: usize = 16; // = X_0
const N: usize = 8; // = X_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.floats("X", &[M, N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x = q.load_floats(ctx, &x);
    let mut d: Vec<Option<Cell>> = vec![None; M * M];
    for i in 0..M {
        for j in i + 1..M {
            let diffs: Vec<Cell> = (0..N).map(|k| q.sub(ctx, x[i * N + k], x[j * N + k])).collect();
            let squares = q.dot(ctx, &diffs, &diffs);
            let root = q.sqrt(ctx, squares.cell).cell;
            d[i * M + j] = Some(root);
            d[j * M + i] = Some(root);
        }
    }
    for value in d {
        match value {
            Some(cell) => out.float(cell),
            None => out.const_floats(&[0.0]), // the diagonal
        }
    }
    Ok(())
}
