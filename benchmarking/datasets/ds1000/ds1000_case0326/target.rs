//! DS-1000 case 0326, halo2-base baseline.
//! Kernel: `np.degrees(np.arcsin(value))` for a `Float` value.
//! No proof unless -1 <= value <= 1 (NumPy returns NaN outside).
//! arcsin(v) = 2 atan(t) with t = v / (1 + sqrt(1 - v^2)), |t| <= 1: the
//! square root is the exact integer root of 2^64 - v_q^2 (no rounded v^2), t
//! a positive-divisor division, atan(t) = t P(t^2) with P a degree-9
//! Chebyshev fit of atan(t)/t on [0, 1] (fit error 2.1e-9 rad), and degrees =
//! atan(t) * 360/pi. Every step is a pinned function of the input, so the
//! output is unique; measured error against NumPy below 2e-7 degrees on
//! [-1, 1].

use crate::prelude::*;

/// P(u) = atan(t)/t, u = t^2, highest degree first.
const ATAN_U: [f64; 10] = [
    -0.001804437510348226,
    0.010962195309438757,
    -0.0312640164156834,
    0.0580399231381612,
    -0.08407845425531969,
    0.10953436857600397,
    -0.14264386880640922,
    0.19998541917604715,
    -0.33333296696567116,
    0.9999999990510089,
];

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let value = inputs.float("value")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let v = q.load_float(ctx, value);
    // -1 <= value <= 1: value + 1 lies in [0, 2].
    let shifted = q.add(ctx, v, Constant(fe(SCALE)));
    q.range.check_less_than_safe(ctx, shifted, 2 * SCALE as u64 + 1);

    // sqrt(1 - v^2) in Q32: floor(sqrt(2^64 - v_q^2)), exact.
    let v_sq = q.gate.mul(ctx, v, v);
    let rest = q.sub(ctx, Constant(fe(SCALE * SCALE)), v_sq);
    let root = q.isqrt(ctx, rest);
    let denominator = q.add(ctx, root.cell, Constant(fe(SCALE))); // 1 + sqrt(1 - v^2) >= 1
    let denominator = q.bound(ctx, denominator);
    let t = q.div_pos(ctx, v, denominator);
    let u = q.mul(ctx, t.cell, t.cell);
    let p = q.polynomial(ctx, u.cell, &ATAN_U);
    let atan = q.mul(ctx, t.cell, p);
    let degrees = q.mul(ctx, atan.cell, Constant(fe(quantize(360.0 / std::f64::consts::PI))));
    out.float(degrees.cell);
    Ok(())
}
