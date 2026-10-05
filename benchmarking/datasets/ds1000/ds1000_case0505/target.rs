//! DS-1000 case 0505, halo2-base baseline.
//! Kernel: `a.sum() - np.take_along_axis(a, b[..., None], axis=-1)[..., 0].sum()` on `a: Float[13, 13, 5]`, `b: Integer[13, 13]`.
//! Each `b[i, j]` selects among the 5 values `a[i, j, :]` with NumPy's index
//! rules (negative indices wrap, others out of range have no proof).

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let (a, b) = (inputs.floats("a", &[13, 13, 5])?, inputs.ints("b", &[13, 13])?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let b = q.load_ints(ctx, &b);
    let picked: Vec<Cell> = (0..169)
        .map(|ij| {
            q.index(ctx, &a[ij * 5..(ij + 1) * 5], b[ij])
        })
        .collect();
    let all = q.sum(ctx, a.iter().copied());
    let taken = q.sum(ctx, picked.iter().copied());
    let rest = q.sub(ctx, all, taken);
    out.float(rest);
    Ok(())
}
