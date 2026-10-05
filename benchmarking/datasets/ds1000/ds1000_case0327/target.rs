//! DS-1000 case 0327, halo2-base baseline.
//! Kernel: `np.pad(A, (0, 17 - 10), 'constant')` on `Float[10]` (`length = 17`
//! is a constant): `A` followed by seven zeros. The `A` cells are exposed as
//! they are; the zeros are circuit constants, not public cells.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("A", &[10])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    out.floats(&a);
    out.const_floats(&[0.0; 7]);
    Ok(())
}
