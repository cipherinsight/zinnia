//! DS-1000 case 0303, halo2-base baseline.
//! Kernel: `np.reshape(A[:(23 // 5) * 5], (-1, 5))` on `Float[23]` (`ncol = 5`
//! is a constant): the first 20 cells of `A` as a (4, 5) array. The outputs are
//! input cells, so the circuit loads only those 20 and exposes them.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("A", &[23])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let kept = q.load_floats(ctx, &a[..20]);
    out.floats(&kept);
    Ok(())
}
