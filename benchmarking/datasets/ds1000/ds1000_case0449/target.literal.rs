//! DS-1000 case 0449: literal translation of kernel.py (LITERAL_BASELINES.md).
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
    // [x_dists.T], [y_dists.T]: each shape 1x4x3.
    let mut xt = Vec::with_capacity(12);
    let mut yt = Vec::with_capacity(12);
    for col in 0..4 {
        for row in 0..3 { xt.push(x[row * 4 + col]); }
    }
    for col in 0..4 {
        for row in 0..3 { yt.push(y[row * 4 + col]); }
    }
    // vstack gives 2x4x3; .T reverses all axes to 3x4x2.
    let stacked = [xt, yt].concat();
    let mut result = Vec::with_capacity(24);
    for row in 0..3 {
        for col in 0..4 {
            for axis in 0..2 { result.push(stacked[axis * 12 + col * 3 + row]); }
        }
    }
    out.floats(&result);
    Ok(())
}
