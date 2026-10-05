//! DS-1000 case 0500, halo2-base baseline.
//! Kernel: `a.reshape(-1, 3)` on `Float[54]`: the output `Float[18, 3]` is the
//! input in the same row-major order.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[54])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    out.floats(&a);
    Ok(())
}
