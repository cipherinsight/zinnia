//! DS-1000 case 0391, halo2-base baseline.
//! Kernel: `a[:, 4:7]` on `Float[20, 10]` (`low = 4`, `high = 7` are
//! constants). The outputs are input cells, so the circuit loads only columns
//! 4..7 and exposes them.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[20, 10])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let kept: Vec<f64> = a.chunks(10).flat_map(|row| row[4..7].to_vec()).collect();
    let kept = q.load_floats(ctx, &kept);
    out.floats(&kept);
    Ok(())
}
