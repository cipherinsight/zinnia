//! DS-1000 case 0502, halo2-base baseline.
//! Kernel: `np.take_along_axis(a, b[..., None], axis=-1)[..., 0]` on `a: Float[13, 13, 2]`, `b: Integer[13, 13]`.
//! Each `b[i, j]` selects among the 2 values `a[i, j, :]` with NumPy's index
//! rules (negative indices wrap, others out of range have no proof), by a
//! one-hot over the valid index values for these short axes.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let (a, b) = (inputs.floats("a", &[13, 13, 2])?, inputs.ints("b", &[13, 13])?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let b = q.load_ints(ctx, &b);
    let picked: Vec<Cell> = (0..169)
        .map(|ij| {
            q.index(ctx, &a[ij * 2..(ij + 1) * 2], b[ij])
        })
        .collect();
    out.floats(&picked);
    Ok(())
}
