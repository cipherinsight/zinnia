//! DS-1000 case 0328, halo2-base baseline.
//! Kernel: `result = A.copy(); result[3:] = 0` on `Float[5]` (`length = 3` is a
//! constant and not above `A.shape[0]`, so the `np.pad` branch is dead): the
//! first three cells of `A`, then two zeros. Only those three cells are loaded
//! and exposed; the zeros are circuit constants, not public cells.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("A", &[5])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let kept = q.load_floats(ctx, &a[..3]);
    out.floats(&kept);
    out.const_floats(&[0.0; 2]);
    Ok(())
}
