//! DS-1000 case 0392, halo2-base baseline.
//! Kernel: `a[4:7, :]` on `Float[20, 10]` (`low = 4`, `high = 7` are
//! constants). The outputs are input cells, so the circuit loads only rows
//! 4..7 and exposes them.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[20, 10])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let kept = q.load_floats(ctx, &a[40..70]);
    out.floats(&kept);
    Ok(())
}
