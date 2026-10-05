//! DS-1000 case 0337, halo2-base baseline.
//! Kernel: `np.diag(np.fliplr(a))` on `Float[8, 10]`: `fliplr(a)[i, j] = a[i, 9 - j]`,
//! so the diagonal is `a[i, 9 - i]` for `i < 8`. Only those 8 cells are loaded;
//! they are the outputs.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[8, 10])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let diag: Vec<f64> = (0..8).map(|i| a[i * 10 + 9 - i]).collect();
    let diag = q.load_floats(ctx, &diag);
    out.floats(&diag);
    Ok(())
}
