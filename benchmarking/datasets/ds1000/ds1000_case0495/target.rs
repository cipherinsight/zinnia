//! DS-1000 case 0495, halo2-base baseline.
//! Kernel: `np.pad(a, ((0, 93 - 41), (0, 13 - 13)), 'constant')` on
//! `Float[41, 13]` (`shape = (93, 13)` is a constant): `a` followed by 52 zero
//! rows. The `a` cells are exposed as they are; the zeros are circuit
//! constants, not public cells.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[41, 13])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    out.floats(&a);
    out.const_floats(&[0.0; 52 * 13]);
    Ok(())
}
