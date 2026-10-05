//! DS-1000 case 0387, halo2-base baseline.
//! Kernel: `a.reshape(50, 2, 100, 2).swapaxes(1, 2).reshape(-1, 2, 2)` on `Float[100, 200]`: the 2x2 blocks of `a` as patches;
//! block (i, j) is output patch i * 100 + j. The outputs are input cells, so the circuit only exposes them.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[100, 200])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let mut patches = Vec::with_capacity(20000);
    for i in 0..50 {
        for j in 0..100 {
            for r in 0..2 {
                for s in 0..2 {
                    patches.push(a[(2 * i + r) * 200 + 2 * j + s]);
                }
            }
        }
    }
    out.floats(&patches);
    Ok(())
}
