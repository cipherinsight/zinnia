//! DS-1000 case 0386, optimized baseline.
//! Sliding windows have shape (99, 199, 2, 2); reshape keeps row-major order.
//! All 20000 inputs are read, and repeated output occurrences share their wires.
use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let raw = inputs.floats("a", &[100, 200])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_floats(ctx, &raw);
    // np.lib.stride_tricks.sliding_window_view(a, window_shape=(2, 2))
    let mut windows = Vec::with_capacity(99 * 199 * 4);
    for i in 0..99 {
        for j in 0..199 {
            for r in 0..2 {
                for s in 0..2 {
                    windows.push(a[(i + r) * 200 + j + s]);
                }
            }
        }
    }
    // .reshape(-1, 2, 2): only the shape changes.
    let result = windows;
    out.floats(&result);
    Ok(())
}
