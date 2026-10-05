//! DS-1000 case 0390, halo2-base baseline.
//! Kernel: crop `a` to multiples of `patch_size = 5` (a constant; 100 x 200
//! already are), then
//! `reshape(20, 5, 40, 5).swapaxes(1, 2).transpose(1, 0, 2, 3).reshape(-1, 5, 5)`
//! on `Float[100, 200]`: the 5x5 blocks of `a` as patches, block (i, j) is
//! output patch j * 20 + i. The outputs are input cells, so the circuit only
//! exposes them.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[100, 200])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let mut patches = Vec::with_capacity(20000);
    for j in 0..40 {
        for i in 0..20 {
            for r in 0..5 {
                for s in 0..5 {
                    patches.push(a[(5 * i + r) * 200 + 5 * j + s]);
                }
            }
        }
    }
    out.floats(&patches);
    Ok(())
}
