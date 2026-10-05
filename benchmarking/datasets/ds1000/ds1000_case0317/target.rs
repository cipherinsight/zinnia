//! DS-1000 case 0317, halo2-base baseline.
//! Kernel: `np.array(a)` on `Float[10, 20, 5]`: a copy; the outputs are the
//! input cells.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[10, 20, 5])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    out.floats(&a);
    Ok(())
}
