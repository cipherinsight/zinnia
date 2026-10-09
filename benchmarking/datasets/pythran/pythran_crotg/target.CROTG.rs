//! pythran_crotg__CROTG (Pythran crotg), halo2-base baseline.
//! Kernel: BLAS CROTG on complex scalars CA, CB. If `abs(CA) == 0`:
//! `C = 0`, `S = 1`, `CA = CB`. Else `SCALE = |CA| + |CB|`,
//! `NORM = SCALE * sqrt(|CA / SCALE| ** 2 + |CB / SCALE| ** 2)`,
//! `ALPHA = CA / |CA|`, `C = |CA| / NORM`, `S = ALPHA * conj(CB) / NORM`,
//! `CA = ALPHA * NORM`. Returns `(CA, CB, C, S)`. The inputs C and S are
//! overwritten before they are read, so they are not loaded.
//! `abs` of a complex value is the exact floor square root of `re^2 + im^2`
//! (`q.isqrt` of the raw sum: one rounding, as `abs` rounds `hypot` once).
//! A complex value divided or multiplied by a real one is componentwise;
//! `ALPHA * conj(CB)` is a complex product, each part one rescale. Both
//! branches are computed and selected; the `else` branch runs on
//! `CA + [CA == 0]` (that is, 1 when CA is 0), so its divisors are positive
//! for every input. `CA == 0` is `re^2 + im^2 == 0`, the same test as
//! `abs(CA) == 0`. The divisors `|CA|`, `SCALE`, `NORM` are positive and below
//! `2^64`, their own magnitudes, so the divisions are
//! `q.div_by_abs(v, d, d)` (no sign test). Where the rounded `NORM` is 0
//! (both magnitudes a few units of 2^-32) there is no proof.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let ca = inputs.complex("CA")?;
    let cb = inputs.complex("CB")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let ca = q.load_floats(ctx, &ca);
    let cb = q.load_floats(ctx, &cb);
    let square = |ctx: &mut Ctx, x: &[Cell]| {
        q.gate.inner_product(ctx, [Existing(x[0]), Existing(x[1])], [Existing(x[0]), Existing(x[1])])
    };
    let ca_square = square(ctx, &ca);
    let zero = q.gate.is_zero(ctx, ca_square);
    // The `else` branch's CA: 1 when CA is 0 (re^2 + im^2 grows by 2^64).
    let ca = [q.gate.mul_add(ctx, zero, Constant(fe(SCALE)), ca[0]), ca[1]];
    let ca_square = q.gate.mul_add(ctx, zero, Constant(fe(SCALE * SCALE)), ca_square);
    let ca_abs = q.isqrt(ctx, ca_square).cell;
    let cb_square = square(ctx, &cb);
    let cb_abs = q.isqrt(ctx, cb_square).cell;

    let scale = q.add(ctx, ca_abs, cb_abs);
    let scaled_abs_squared = |ctx: &mut Ctx, x: &[Cell]| {
        let ratio: Vec<Cell> = x.iter().map(|v| q.div_by_abs(ctx, *v, scale, scale).cell).collect();
        let ratio_square = square(ctx, &ratio);
        let ratio_abs = q.isqrt(ctx, ratio_square);
        q.mul(ctx, ratio_abs, ratio_abs).cell
    };
    let a2 = scaled_abs_squared(ctx, &ca);
    let b2 = scaled_abs_squared(ctx, &cb);
    let sum = q.add(ctx, a2, b2);
    let root = q.sqrt(ctx, sum);
    let norm = q.mul(ctx, scale, root).cell;
    let alpha: Vec<Cell> = ca.iter().map(|v| q.div_by_abs(ctx, *v, ca_abs, ca_abs).cell).collect();
    let c = q.div_by_abs(ctx, ca_abs, norm, norm).cell;
    // alpha * conj(cb) = (ar br + ai bi) + (ai br - ar bi) i
    let s_re = q.dot(ctx, &alpha, &cb).cell;
    let ai_br = q.gate.mul(ctx, alpha[1], cb[0]);
    let s_im = q.gate.sub_mul(ctx, ai_br, alpha[0], cb[1]);
    let s_im = q.rescale(ctx, s_im).cell;
    let s = [s_re, s_im].map(|v| q.div_by_abs(ctx, v, norm, norm).cell);
    let ca_out = alpha.iter().map(|v| q.mul(ctx, *v, norm).cell).collect::<Vec<_>>();

    // The `if` branch: CA = CB, C = 0, S = 1.
    let ca_out: Vec<Cell> = ca_out.iter().zip(&cb).map(|(v, b)| q.gate.select(ctx, *b, *v, zero)).collect();
    let c = q.gate.select(ctx, Constant(Fr::ZERO), c, zero);
    let s = [q.gate.select(ctx, Constant(fe(SCALE)), s[0], zero), q.gate.select(ctx, Constant(Fr::ZERO), s[1], zero)];
    out.floats(&ca_out);
    out.floats(&cb);
    out.float(c);
    out.floats(&s);
    Ok(())
}
