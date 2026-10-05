//! DS-1000 case 0336, halo2-base baseline.
//! Kernel: `np.diag(np.fliplr(a))` on `Float[8, 8]`: the anti-diagonal `a[i, 7 - i]`.
//! Only the 8 cells the kernel reads are loaded; they are the outputs.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[8, 8])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let anti_diag: Vec<f64> = (0..8).map(|i| a[i * 8 + 7 - i]).collect();
    let anti_diag = q.load_floats(ctx, &anti_diag);
    out.floats(&anti_diag);
    Ok(())
}
