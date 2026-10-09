//! pythran_pairwise__pairwise (Pythran pairwise), halo2-base baseline.
//! Kernel: `D[i, j] = sqrt(sum_k (X[i, k] - X[j, k])^2)` on `X: Float[M, N]`.
//! Match every ordered pair in the source; each squared difference rounds
//! before accumulation and sqrt. Only the identically-zero diagonal folds.

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
        for j in 0..M {
            if i == j { continue; }
            let diffs: Vec<Cell> = (0..N).map(|k| q.sub(ctx, x[i * N + k], x[j * N + k])).collect();
            let squares: Vec<Cell> = diffs.iter().map(|x| q.mul(ctx, *x, *x).cell).collect();
            let squared_distance = q.sum(ctx, squares);
            let root = q.sqrt(ctx, squared_distance).cell;
            d[i * M + j] = Some(root);
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
