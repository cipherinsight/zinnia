//! DS-1000 case 0338, halo2-base baseline.
//! Kernel: `np.vstack((np.diag(a), np.diag(np.fliplr(a))))` on `Float[8, 8]`:
//! row 0 is `a[i, i]`, row 1 is `a[i, 7 - i]`. Only those 16 cells (disjoint for
//! an even size) are loaded; they are the outputs.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[8, 8])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let diag: Vec<f64> = (0..8).map(|i| a[i * 8 + i]).collect();
    let anti_diag: Vec<f64> = (0..8).map(|i| a[i * 8 + 7 - i]).collect();
    let diag = q.load_floats(ctx, &diag);
    let anti_diag = q.load_floats(ctx, &anti_diag);
    out.floats(&[diag, anti_diag].concat());
    Ok(())
}
