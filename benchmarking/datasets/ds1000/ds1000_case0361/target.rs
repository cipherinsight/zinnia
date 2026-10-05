//! DS-1000 case 0361, halo2-base baseline.
//! Kernel: `np.delete(a, [0, 2], axis=1)` on `Float[6, 6]`: the outputs are
//! input cells, so the circuit loads only the kept ones and exposes them.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[6, 6])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let kept: Vec<f64> = (0..6).flat_map(|i| [1, 3, 4, 5].map(|j| a[i * 6 + j])).collect();
    let kept = q.load_floats(ctx, &kept);
    out.floats(&kept);
    Ok(())
}
