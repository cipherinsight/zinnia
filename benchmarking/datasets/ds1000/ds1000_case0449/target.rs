//! DS-1000 case 0449 (same kernel as 0448), halo2-base baseline.
//! Kernel: `np.vstack(([x_dists.T], [y_dists.T])).T` on two `Float[3, 4]`: the
//! (3, 4, 2) result pairs `x_dists[i, j]` with `y_dists[i, j]`; the outputs are
//! input cells, so the circuit only exposes them.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let (x, y) = (inputs.floats("x_dists", &[3, 4])?, inputs.floats("y_dists", &[3, 4])?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let (x, y) = (q.load_floats(ctx, &x), q.load_floats(ctx, &y));
    let pairs: Vec<Cell> = (0..12).flat_map(|k| [x[k], y[k]]).collect();
    out.floats(&pairs);
    Ok(())
}
