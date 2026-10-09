//! DS-1000 case 0448: optimized baseline.
//! Kernel: np.vstack(([x_dists.T], [y_dists.T])).T on two 3x4 arrays.
//! The stacked shape is 2x4x3; final transpose has shape 3x4x2.
//! All transposes and stacking are copies of wires, with no arithmetic.
use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.floats("x_dists", &[3, 4])?;
    let y = inputs.floats("y_dists", &[3, 4])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let x = q.load_floats(ctx, &x);
    let y = q.load_floats(ctx, &y);
    let mut result = Vec::with_capacity(24);
    for i in 0..12 { result.push(x[i]); result.push(y[i]); }
    out.floats(&result);
    Ok(())
}
