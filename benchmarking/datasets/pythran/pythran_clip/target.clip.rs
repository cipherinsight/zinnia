//! pythran_clip__clip (Pythran clip), halo2-base baseline.
//! Kernel: for each `z[i]` of z: Complex[N], `limit1(z[i]) * _max` when
//! `abs(z[i]) > _max`, else `z[i]`, with `limit1(x) = 0` when
//! `abs(x) < 1e-6` and `x / abs(x)` otherwise; returns Complex[N].
//! Canonical Q32 complex arithmetic: component squares round before sqrt.
//! Complex/real division keeps the source lowering's rounded numerator x*d
//! and rounded denominator d*d before division; cancellation changes rounding.
//! Inactive divisions use the neutral real denominator 1.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 2; // = z_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let z = inputs.complexes("z", &[N])?;
    let max = inputs.float("_max")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let z = q.load_floats(ctx, &z); // interleaved real and imaginary parts
    let max = q.load_float_bounded(ctx, max);
    let mut result = Vec::with_capacity(2 * N);
    for x in z.chunks(2) {
        let rr = q.mul(ctx, x[0], x[0]).cell;
        let ii = q.mul(ctx, x[1], x[1]).cell;
        let square = q.add(ctx, rr, ii);
        let magnitude = q.sqrt(ctx, square);
        let over = q.gt(ctx, magnitude, max);
        let tiny = q.lt_const(ctx, magnitude, quantize(1e-6));
        let divisor = q.gate.select(ctx, Constant(fe(SCALE)), magnitude.cell, tiny);
        let denominator = q.mul(ctx, divisor, divisor).cell;
        for part in x {
            let numerator = q.mul(ctx, *part, divisor).cell;
            let unit = q.div_by_abs(ctx, numerator, denominator, denominator);
            let limited = q.mul(ctx, unit, max);
            let limited = q.gate.select(ctx, Constant(Fr::ZERO), limited.cell, tiny);
            result.push(q.gate.select(ctx, limited, *part, over));
        }
    }
    out.floats(&result);
    Ok(())
}
