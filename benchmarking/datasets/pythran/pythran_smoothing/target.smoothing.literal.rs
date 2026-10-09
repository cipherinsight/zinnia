//! pythran_smoothing__smoothing: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Recompute 1-alpha inside each recurrence, in source evaluation order.

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
    let mut s = x.clone();
    for i in 1..N {
        let fresh = q.mul(ctx, alpha, x[i]).cell;
        let keep = q.sub(ctx, Constant(fe(SCALE)), alpha);
        let carried = q.mul(ctx, keep, s[i-1]).cell;
        s[i] = q.add(ctx, fresh, carried);
    }
    out.floats(&s);
    Ok(())
}
