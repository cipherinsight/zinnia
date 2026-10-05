//! DS-1000 case 0422, halo2-base baseline.
//! Kernel: `np.correlate(a, np.hstack((b[1:], b)), mode='valid')` on two `Float[50]`.
//! `v = hstack((b[1:], b))` has 99 elements, `v[m] = b[(m + 1) % 50]`. NumPy's
//! 'valid' correlation of the shorter `a` with `v` is
//! `out[k] = sum_n a[n] * v[n + 49 - k] = sum_n a[n] * b[(n - k) mod 50]`,
//! `k < 50`: a circular correlation, each output one dot product with one rescale.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let (a, b) = (inputs.floats("a", &[50])?, inputs.floats("b", &[50])?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let (a, b) = (q.load_floats(ctx, &a), q.load_floats(ctx, &b));
    let corr: Vec<Cell> = (0..50)
        .map(|k| {
            let shifted: Vec<Cell> = (0..50).map(|n| b[(n + 50 - k) % 50]).collect();
            q.dot(ctx, &a, &shifted).cell
        })
        .collect();
    out.floats(&corr);
    Ok(())
}
