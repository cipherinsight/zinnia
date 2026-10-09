//! pythran_reverse_cumsum__reverse_cumsum (Pythran reverse_cumsum), halo2-base baseline.
//! Kernel: `np.cumsum(x[::-1])[::-1]` on `x: Float[N]`: element `i` is
//! `x[i] + ... + x[N - 1]`, the suffix sums, built from the end (exact
//! additions, one per element).

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 10; // = x_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.floats("x", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x = q.load_floats(ctx, &x);
    let mut sums = x.clone();
    for i in (0..N - 1).rev() {
        sums[i] = q.add(ctx, x[i], sums[i + 1]);
    }
    out.floats(&sums);
    Ok(())
}
