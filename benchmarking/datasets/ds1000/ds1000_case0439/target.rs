//! DS-1000 case 0439, halo2-base baseline.
//! Kernel: `X.T[:, :, None] * X.T[:, None]` on `Float[10, 20]`: the `Float[20, 10, 10]`
//! array `r[k, i, j] = X[i, k] * X[j, k]`, each a rescaled product.
//! The product is symmetric in `(i, j)` (the same rounded value), so each of the
//! 20 * 55 unordered pairs is computed once and exposed at both positions.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.floats("X", &[10, 20])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x = q.load_floats(ctx, &x);
    let mut r: Vec<Option<Cell>> = vec![None; 20 * 10 * 10];
    for k in 0..20 {
        for i in 0..10 {
            for j in i..10 {
                let p = q.mul(ctx, x[i * 20 + k], x[j * 20 + k]).cell;
                r[k * 100 + i * 10 + j] = Some(p);
                r[k * 100 + j * 10 + i] = Some(p);
            }
        }
    }
    let r: Vec<Cell> = r.into_iter().map(|p| p.expect("every position is filled")).collect();
    out.floats(&r);
    Ok(())
}
