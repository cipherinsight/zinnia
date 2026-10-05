//! DS-1000 case 0360, halo2-base baseline.
//! Kernel: `np.delete(a, 2, axis=0)` on `Float[4, 4]`: the outputs are input
//! cells, so the circuit loads only the kept ones and exposes them.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[4, 4])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let kept: Vec<f64> = [0, 1, 3].iter().flat_map(|i| a[i * 4..i * 4 + 4].to_vec()).collect();
    let kept = q.load_floats(ctx, &kept);
    out.floats(&kept);
    Ok(())
}
