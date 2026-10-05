//! DS-1000 case 0302, halo2-base baseline.
//! Kernel: `np.reshape(A, (5, -1))` on `Float[20]` (`nrow = 5` is a constant):
//! a (5, 4) view of `A` in the same row-major order. The outputs are the input
//! cells, so the circuit only loads and exposes them.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("A", &[20])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    out.floats(&a);
    Ok(())
}
