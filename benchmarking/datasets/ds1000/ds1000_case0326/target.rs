//! DS-1000 case 0326, halo2-base baseline.
//! Kernel: `np.degrees(np.arcsin(value))` for a `Float` value.
//! No proof unless -1 <= value <= 1 (NumPy returns NaN outside).
//! arcsin has no exact gadget. The prover supplies y = round(arcsin(value) *
//! 2^32); the circuit checks -pi/2 <= y <= pi/2 and |S(y) - value| <= EPS,
//! where S(y) = y (1 + u (c3 + u (c5 + u (c7 + u c9)))), u = y^2, is a degree-9
//! odd polynomial fitted to sin on [-pi/2, pi/2] (minimax error 4.6e-9; the
//! library's degree-14 polynomial is meant for np.sin on [0, pi] and costs 14
//! rescales here instead of 6). With its Q32 coefficients S is within 7.3e-9
//! of sin on [-pi/2, pi/2], and flooring its 6 products adds at most 40 ulp;
//! pointwise the two stay below 1.5e-8, so EPS = 2e-8 accepts the honest y.
//! Every accepted y has |sin(y) - value| <= 3.5e-8: it lies within
//! 3.5e-8 / sqrt(1 - (|value| + 3.5e-8)^2) rad of arcsin(value), and never
//! more than sqrt(2 * 3.5e-8) = 2.7e-4 rad (0.015 degrees, reached at
//! |value| = 1). degrees = y * (180 / pi), one rescaled product.

use crate::prelude::*;

/// round(pi / 2 * 2^32): arcsin lies in [-HALF_PI_Q, HALF_PI_Q] in Q32.
const HALF_PI_Q: u64 = 6_746_518_852;
/// round(2e-8 * 2^32), the tolerance on S(y) - value.
const EPS_Q: u64 = 86;
/// c9, c7, c5, c3, 1 (highest first), each a Q32 value k / 2^32 (k =
/// 11167, -850688, 35790037, -715827472), so `q.polynomial` quantizes them exactly.
const SIN9: [f64; 5] = [
    11_167.0 / 4_294_967_296.0,
    -850_688.0 / 4_294_967_296.0,
    35_790_037.0 / 4_294_967_296.0,
    -715_827_472.0 / 4_294_967_296.0,
    1.0,
];

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let value = inputs.float("value")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let v = q.load_float(ctx, value);
    // -1 <= value <= 1: value + 1 lies in [0, 2].
    let shifted = q.add(ctx, v, Constant(fe(SCALE)));
    q.range.check_less_than_safe(ctx, shifted, 2 * SCALE as u64 + 1);

    // Hint y = round(arcsin(value) * 2^32), pinned to the window above by
    // y + pi/2 in [0, pi] and S(y) - value + EPS in [0, 2 EPS].
    let y = ctx.load_witness(fe(quantize((quantize(value) as f64 / SCALE as f64).asin())));
    let y_shifted = q.add(ctx, y, Constant(fe(HALF_PI_Q as i128)));
    q.range.check_less_than_safe(ctx, y_shifted, 2 * HALF_PI_Q + 1);
    let u = q.mul(ctx, y, y);
    let s_over_y = q.polynomial(ctx, u.cell, &SIN9);
    let s = q.mul(ctx, y, s_over_y);
    let gap = q.sub(ctx, s.cell, v);
    let gap = q.add(ctx, gap, Constant(fe(EPS_Q as i128)));
    q.range.check_less_than_safe(ctx, gap, 2 * EPS_Q + 1);

    let degrees = q.mul(ctx, y, Constant(fe(quantize(180.0 / std::f64::consts::PI))));
    out.float(degrees.cell);
    Ok(())
}
