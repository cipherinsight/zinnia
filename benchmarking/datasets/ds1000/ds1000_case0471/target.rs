//! DS-1000 case 0471, halo2-base baseline.
//! Kernel: `Z[..., -1:]` on `Float[6, 2, 9, 3, 7, 4]`: the last entry of every
//! length-4 innermost row, shape (6, 2, 9, 3, 7, 1). The outputs are input
//! cells, so the circuit loads only those 2268 and exposes them.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let z = inputs.floats("Z", &[6, 2, 9, 3, 7, 4])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let last: Vec<f64> = z.chunks(4).map(|row| row[3]).collect();
    let last = q.load_floats(ctx, &last);
    out.floats(&last);
    Ok(())
}
