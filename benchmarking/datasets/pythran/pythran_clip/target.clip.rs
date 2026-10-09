//! pythran_clip__clip (Pythran clip), halo2-base baseline.
//! Kernel: for each `z[i]` of z: Complex[N], `limit1(z[i]) * _max` when
//! `abs(z[i]) > _max`, else `z[i]`, with `limit1(x) = 0` when
//! `abs(x) < 1e-6` and `x / abs(x)` otherwise; returns Complex[N].
//! `abs` of a complex value is the exact floor square root of
//! `re^2 + im^2` (`q.isqrt` of the raw sum, scale 2^64): one rounding, as
//! `np.abs` rounds `hypot` once. Both branches are computed and selected; in
//! the unused one the divisor is moved off zero (`abs + [abs < 1e-6]`), so
//! every input has a witness. `x / abs(x)` and `* _max` are componentwise,
//! as NumPy divides and multiplies a complex value by a real one.

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
        let square = q.gate.inner_product(ctx, [Existing(x[0]), Existing(x[1])], [Existing(x[0]), Existing(x[1])]);
        let magnitude = q.isqrt(ctx, square);
        let over = q.gt(ctx, magnitude, max);
        let tiny = q.lt_const(ctx, magnitude, quantize(1e-6));
        let divisor = q.add(ctx, magnitude.cell, tiny);
        for part in x {
            let unit = q.div_by_abs(ctx, *part, divisor, divisor);
            let limited = q.mul(ctx, unit, max);
            let limited = q.gate.select(ctx, Constant(Fr::ZERO), limited.cell, tiny);
            result.push(q.gate.select(ctx, limited, *part, over));
        }
    }
    out.floats(&result);
    Ok(())
}
