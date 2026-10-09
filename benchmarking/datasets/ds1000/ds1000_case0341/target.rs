//! DS-1000 case 0341, halo2-base baseline.
//! Kernel: `[value for value in X.flat]` on `Integer[5, 6]`: the 30 elements in
//! row-major order, each a loaded input cell.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.ints("X", &[5, 6])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x = q.load_ints(ctx, &x);
    out.ints(&x);
    Ok(())
}
