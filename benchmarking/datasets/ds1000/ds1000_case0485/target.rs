//! DS-1000 case 0485, halo2-base baseline.
//! Kernel: `np.tensordot(A, B, axes=((2), (0)))` on `A: Float[5, 6, 3]`, `B: Float[3, 3]`:
//! `r[i, j, k] = sum_p A[i, j, p] * B[p, k]`, each a dot product with one rescale.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let (a, b) = (inputs.floats("A", &[5, 6, 3])?, inputs.floats("B", &[3, 3])?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let (a, b) = (q.load_floats(ctx, &a), q.load_floats(ctx, &b));
    let b_cols: Vec<Vec<Cell>> = (0..3).map(|k| (0..3).map(|p| b[p * 3 + k]).collect()).collect();
    let mut r = Vec::with_capacity(90);
    for ij in 0..30 {
        for col in &b_cols {
            r.push(q.dot(ctx, &a[ij * 3..ij * 3 + 3], col).cell);
        }
    }
    out.floats(&r);
    Ok(())
}
