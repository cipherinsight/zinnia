//! DS-1000 case 0440, halo2-base baseline.
//! Kernel: `X = np.zeros([5, 10]); X[:, i] += np.sqrt(np.diag(Y[i]))` for each `i`
//! on `Y: Float[10, 5, 5]`: `X[j, i] = sqrt(Y[i, j, j])` (adding to 0.0 is exact).
//! 50 exact floor square roots; a negative diagonal entry has no proof (NumPy
//! returns NaN). Only the 50 diagonal entries are loaded; the other 200 inputs
//! never reach the output.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let y = inputs.floats("Y", &[10, 5, 5])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    // Y[i, j, j] in the row-major order of X[j, i].
    let diag: Vec<f64> = (0..5).flat_map(|j| (0..10).map(move |i| (i, j))).map(|(i, j)| y[i * 25 + j * 6]).collect();
    let diag = q.load_floats(ctx, &diag);
    let x: Vec<Cell> = diag.iter().map(|d| q.sqrt(ctx, *d).cell).collect();
    out.floats(&x);
    Ok(())
}
