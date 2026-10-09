//! pythran_smoothing__smoothing (Pythran smoothing), halo2-base baseline.
//! Kernel: exponential smoothing of x: Float[N] with a scalar `alpha`:
//! `s[0] = x[0]`, `s[i] = alpha * x[i] + (1 - alpha) * s[i - 1]`; returns `s`.
//! Both products of a step are rounded (scalar operations), `1 - alpha` is
//! exact and computed once.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = x_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.floats("x", &[N])?;
    let alpha = inputs.float("alpha")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x = q.load_floats(ctx, &x);
    let alpha = q.load_float(ctx, alpha);
    let keep = q.sub(ctx, Constant(fe(SCALE)), alpha);
    let mut s = Vec::with_capacity(N);
    s.push(x[0]);
    for i in 1..N {
        let fresh = q.mul(ctx, alpha, x[i]);
        let carried = q.mul(ctx, keep, s[i - 1]);
        s.push(q.add(ctx, fresh, carried));
    }
    out.floats(&s);
    Ok(())
}
