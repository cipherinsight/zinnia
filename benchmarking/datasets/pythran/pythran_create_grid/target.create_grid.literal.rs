// pythran_create_grid__create_grid: literal translation of kernel.py (LITERAL_BASELINES.md).
// Materialize zeros and each slice assignment in order; omit constant output column from publics.
use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = x_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.floats("x", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x = q.load_floats(ctx, &x);
    let mut z = vec![ctx.load_zero(); N*N*3];
    for i in 0..N { for j in 0..N { z[(i*N+j)*3] = x[i]; } }
    for i in 0..N { for j in 0..N { z[(i*N+j)*3+1] = x[j]; } }
    let fast_grid = z;
    for row in 0..N*N { out.floats(&fast_grid[row*3..row*3+2]); out.const_floats(&[0.0]); }
    Ok(())
}
