//! pythran_crotg__CROTG (Pythran crotg), halo2-base baseline.
//! Kernel: BLAS CROTG on complex scalars CA, CB. If `abs(CA) == 0`:
//! `C = 0`, `S = 1`, `CA = CB`. Else `SCALE = |CA| + |CB|`,
//! `NORM = SCALE * sqrt(|CA / SCALE| ** 2 + |CB / SCALE| ** 2)`,
//! `ALPHA = CA / |CA|`, `C = |CA| / NORM`, `S = ALPHA * conj(CB) / NORM`,
//! `CA = ALPHA * NORM`. Returns `(CA, CB, C, S)`. The inputs C and S are
//! overwritten before they are read, so they are not loaded.
//! Complex magnitude and division preserve the frozen Q32 primitive schedule.
//! The branch tests rounded magnitude zero. Each complex product rounds all
//! four signed real products individually; conjugation occurs before rounding.
//! Retain original inactive operands and products. Only primitive zero
//! division is totalized; its nonzero source assertion is branch-guarded.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let ca = inputs.complex("CA")?;
    let cb = inputs.complex("CB")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let ca = q.load_floats(ctx, &ca);
    let cb = q.load_floats(ctx, &cb);
    let magnitude = |ctx: &mut Ctx, x: &[Cell]| {
        let rr = q.mul(ctx, x[0], x[0]).cell;
        let ii = q.mul(ctx, x[1], x[1]).cell;
        let squared = q.add(ctx, rr, ii);
        let squared = q.bound(ctx, squared).cell;
        q.sqrt(ctx, squared).cell
    };
    let ca_abs = magnitude(ctx, &ca);
    let zero = q.gate.is_zero(ctx, ca_abs);
    let active = q.gate.not(ctx, zero);
    let cb_abs = magnitude(ctx, &cb);
    let divide = |ctx: &mut Ctx, numerator: Cell, denominator: Cell| {
        let dzero = q.eq(ctx, denominator, Constant(Fr::ZERO));
        let invalid = q.gate.and(ctx, active, dzero);
        q.gate.assert_is_const(ctx, &invalid, &Fr::ZERO);
        let safe = q.add(ctx, denominator, dzero);
        let masked = q.gate.mul_not(ctx, dzero, numerator);
        q.div_by_abs(ctx, masked, safe, safe).cell
    };
    // Complex/real division: rounded x*d / rounded d*d. No cancellation.
    let complex_div = |ctx: &mut Ctx, x: &[Cell], d: Cell| {
        let denominator = q.mul(ctx, d, d).cell;
        x.iter().map(|v| {
            let numerator = q.mul(ctx, *v, d).cell;
            divide(ctx, numerator, denominator)
        }).collect::<Vec<_>>()
    };
    let scale = q.add(ctx, ca_abs, cb_abs);
    let scaled_abs_squared = |ctx: &mut Ctx, x: &[Cell]| {
        let ratio = complex_div(ctx, x, scale);
        let ratio_abs = magnitude(ctx, &ratio);
        q.mul(ctx, ratio_abs, ratio_abs).cell
    };
    let a2 = scaled_abs_squared(ctx, &ca);
    let b2 = scaled_abs_squared(ctx, &cb);
    let sum = q.add(ctx, a2, b2);
    let root = q.sqrt(ctx, sum);
    let norm = q.mul(ctx, scale, root).cell;
    let alpha = complex_div(ctx, &ca, ca_abs);
    let c = divide(ctx, ca_abs, norm);
    // alpha * conj(cb) = (ar br + ai bi) + (ai br - ar bi) i
    let neg_bi = q.gate.neg(ctx, cb[1]);
    let ar_br = q.mul(ctx, alpha[0], cb[0]).cell;
    let ai_nbi = q.mul(ctx, alpha[1], neg_bi).cell;
    let ar_nbi = q.mul(ctx, alpha[0], neg_bi).cell;
    let ai_br = q.mul(ctx, alpha[1], cb[0]).cell;
    let s_re = q.sub(ctx, ar_br, ai_nbi);
    let s_im = q.add(ctx, ar_nbi, ai_br);
    let s = complex_div(ctx, &[s_re, s_im], norm);
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
