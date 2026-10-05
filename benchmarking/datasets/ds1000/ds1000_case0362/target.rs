//! DS-1000 case 0362, halo2-base baseline.
//! Kernel: `del_col = [12, 14, 1, 9]` (a constant), keep the entries
//! `<= a.shape[1] = 6` and subtract 1, then `np.delete(a, del_col, axis=1)` on
//! `Float[8, 6]`: only column 0 is deleted. The outputs are input cells, so the
//! circuit loads only columns 1..6 and exposes them.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[8, 6])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let kept: Vec<f64> = a.chunks(6).flat_map(|row| row[1..].to_vec()).collect();
    let kept = q.load_floats(ctx, &kept);
    out.floats(&kept);
    Ok(())
}
