//! DS-1000 case 0472, halo2-base baseline.
//! Kernel: `a[-1:, ...]` on `Float[6, 2, 9, 3, 7, 4]`: the last of the six
//! leading slices (1512 cells); the outputs are input cells.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[6, 2, 9, 3, 7, 4])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    const SLICE: usize = 2 * 9 * 3 * 7 * 4;
    let last = q.load_floats(ctx, &a[5 * SLICE..]);
    out.floats(&last);
    Ok(())
}
