//! DS-1000 case 0420, halo2-base baseline.
//! Kernel: `np.where(x < x_min, x_min, np.where(x > x_max, x_max, 3*x**2 - 2*x**3))`
//! with `x: Float`, `x_min, x_max: Integer` (a float result).
//! The bounds are compared as floats (`float(x_min)`, `float(x_max)`, which
//! like every float must lie in the Q32 window). The polynomial is evaluated
//! at `x` when `x_min <= x <= x_max` and at 0 otherwise: its value is only
//! selected in the first case, and a clamped `x` far outside the bounds would
//! otherwise overflow `x**3` although NumPy's result (a bound) fits.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.float("x")?;
    let (x_min, x_max) = (inputs.int("x_min")?, inputs.int("x_max")?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let xb = q.load_float_bounded(ctx, x);
    let x = xb.cell;
    let (x_min, x_max) = (q.load_int(ctx, x_min), q.load_int(ctx, x_max));
    let lo = q.int_to_float(ctx, x_min);
    let hi = q.int_to_float(ctx, x_max);
    let (lo_b, hi_b) = (q.bound(ctx, lo), q.bound(ctx, hi));
    let below = q.lt(ctx, xb, lo_b);
    let above = q.gt(ctx, xb, hi_b);

    // t = x if neither bound applies, else 0.
    let t = q.gate.mul_not(ctx, below, x);
    let t = q.gate.mul_not(ctx, above, t);
    let t2 = q.mul(ctx, t, t).cell;
    let t3 = q.mul(ctx, t2, t).cell;
    let smooth = q.gate.inner_product(ctx, [Existing(t2), Existing(t3)], [Constant(fe(3)), Constant(fe(-2))]);

    let inner = q.gate.select(ctx, hi, smooth, above);
    let result = q.gate.select(ctx, lo, inner, below);
    out.float(result);
    Ok(())
}
