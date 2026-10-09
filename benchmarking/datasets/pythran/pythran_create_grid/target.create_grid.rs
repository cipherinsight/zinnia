//! pythran_create_grid__create_grid (Pythran create_grid), halo2-base baseline.
//! Kernel: on `x: Float[N]`, the grid `z[i, j] = (x[i], x[j], 0)` reshaped to
//! `(N * N, 3)`. Data movement only (NPB_BASELINES.md section 5, rule 6):
//! the inputs are loaded with their domain checks and copied to the outputs;
//! the third column is the constant 0, not public.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = x_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.floats("x", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x = q.load_floats(ctx, &x);
    for i in 0..N {
        for j in 0..N {
            out.floats(&[x[i], x[j]]);
            out.const_floats(&[0.0]);
        }
    }
    Ok(())
}
