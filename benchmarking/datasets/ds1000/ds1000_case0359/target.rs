//! DS-1000 case 0359, halo2-base baseline.
//! Kernel: `np.delete(a, 2, axis=1)` on `Float[3, 3]`: the outputs are input
//! cells, so the circuit loads only the kept ones and exposes them.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[3, 3])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let kept: Vec<f64> = (0..3).flat_map(|i| [a[i * 3], a[i * 3 + 1]]).collect();
    let kept = q.load_floats(ctx, &kept);
    out.floats(&kept);
    Ok(())
}
