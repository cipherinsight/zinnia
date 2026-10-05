//! DS-1000 case 0409, halo2-base baseline.
//! Kernel: `np.array(x) + np.array(y)` on two `Float[8, 6]` (the copies are no-ops).

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let (x, y) = (inputs.floats("x", &[8, 6])?, inputs.floats("y", &[8, 6])?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let (x, y) = (q.load_floats(ctx, &x), q.load_floats(ctx, &y));
    let z: Vec<Cell> = x.iter().zip(&y).map(|(a, b)| q.add(ctx, *a, *b)).collect();
    out.floats(&z);
    Ok(())
}
