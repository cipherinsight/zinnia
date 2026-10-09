//! npbench_weather_stencils_vadv_vadv__vadv (NPBench weather_stencils/vadv),
//! halo2-base baseline.
//! Kernel: vertical advection, a tridiagonal solve along `k` for every column
//! `(i, j)` of `utens_stage: Float[NI, NJ, NK]` (in place; the output is
//! `utens_stage` after the call, `mutated_outputs`), from `u_stage`, `u_pos`,
//! `utens` (same shape), `wcon: Float[NI + 1, NJ, NK]` and the scalar
//! `dtr_stage`: a forward sweep computing `ccol`, `dcol` with one reciprocal
//! `1.0 / (...)` per level, then back substitution. Every statement is
//! elementwise over `(i, j)`, so each column is solved on its own, with
//! NumPy's rounding: every product and every reciprocal is rounded.
//! The coefficients come from `s = wcon[i + 1, j, k] + wcon[i, j, k]`
//! (`k >= 1`; `wcon[:, :, 0]` is never read, so not loaded):
//! `gcv = 0.25 * s`, `gav = -0.25 * s` (the next level's), and their
//! products with `BET_M = BET_P = 0.5`, which are equal (`cs = ccol`,
//! `as_ = acol`). All four are floors of `s / 4` and `s / 8`, so they come
//! from one decomposition `s = 8 h + r` with the bits of `r` (`split`)
//! instead of four rounded products (1,674,072 advice cells for the circuit
//! against 1,805,697 with four `floor_div_const`). A reciprocal divides by the
//! magnitude of the divisor from a hinted sign and one 64-bit range check
//! (`reciprocal`), with no window check of the divisor (`q.div` after
//! `q.bound`: 1,770,822 advice cells).

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const NI: usize = 15; // = utens_stage_0
const NJ: usize = 15; // = utens_stage_1
const NK: usize = 10; // = utens_stage_2

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let utens_stage = inputs.floats("utens_stage", &[NI, NJ, NK])?;
    let u_stage = inputs.floats("u_stage", &[NI, NJ, NK])?;
    let wcon = inputs.floats("wcon", &[NI + 1, NJ, NK])?;
    let u_pos = inputs.floats("u_pos", &[NI, NJ, NK])?;
    let utens = inputs.floats("utens", &[NI, NJ, NK])?;
    let dtr_stage = inputs.float("dtr_stage")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let mut stage = q.load_floats(ctx, &utens_stage);
    let u = q.load_floats(ctx, &u_stage);
    // wcon[:, :, 0] is never read.
    let wcon: Vec<Option<Cell>> =
        wcon.iter().enumerate().map(|(t, v)| (t % NK != 0).then(|| q.load_float(ctx, *v))).collect();
    let pos = q.load_floats(ctx, &u_pos);
    let tens = q.load_floats(ctx, &utens);
    let dtr = q.load_float(ctx, dtr_stage);

    for i in 0..NI {
        for j in 0..NJ {
            let at = |k: usize| (i * NJ + j) * NK + k;
            let w = |r: usize, k: usize| wcon[(r * NJ + j) * NK + k].expect("k >= 1");
            // coefs[k]: the coefficients of s_k, k >= 1.
            let coefs: Vec<Option<Coefs>> = (0..NK)
                .map(|k| {
                    (k > 0).then(|| {
                        let s = q.add(ctx, w(i + 1, k), w(i, k));
                        split(&q, ctx, s)
                    })
                })
                .collect();
            let coef = |k: usize| coefs[k].expect("k >= 1");
            // dtr_stage * u_pos + utens + utens_stage + correction_term.
            let right = |ctx: &mut Ctx, k: usize, correction: Cell| {
                let scaled = q.mul(ctx, dtr, pos[at(k)]);
                q.sum(ctx, [scaled.cell, tens[at(k)], stage[at(k)], correction])
            };
            let mut ccol: Vec<Cell> = Vec::with_capacity(NK);
            let mut dcol: Vec<Cell> = Vec::with_capacity(NK);

            // k = 0.
            let c = coef(1);
            let bcol = q.sub(ctx, dtr, c.half);
            let minus_cs = q.neg(ctx, c.half);
            let du = q.sub(ctx, u[at(1)], u[at(0)]);
            let correction = q.mul(ctx, minus_cs, du).cell;
            let d = right(ctx, 0, correction);
            let divided = reciprocal(&q, ctx, bcol);
            ccol.push(q.mul(ctx, c.half, divided).cell);
            dcol.push(q.mul(ctx, d, divided).cell);

            // 0 < k < NK - 1, and k = NK - 1 (no ccol, no upper term).
            for k in 1..NK {
                let a = coef(k);
                let last = k == NK - 1;
                let mut bcol = q.sub(ctx, dtr, a.minus_half);
                let du = q.sub(ctx, u[at(k - 1)], u[at(k)]);
                let mut correction = q.mul(ctx, a.half_ceil, du).cell;
                if !last {
                    let c = coef(k + 1);
                    bcol = q.sub(ctx, bcol, c.half);
                    let du = q.sub(ctx, u[at(k + 1)], u[at(k)]);
                    let upper = q.mul(ctx, c.half, du);
                    correction = q.sub(ctx, correction, upper.cell);
                }
                let d = right(ctx, k, correction);
                let lower = q.mul(ctx, ccol[k - 1], a.minus_half);
                let pivot = q.sub(ctx, bcol, lower.cell);
                let divided = reciprocal(&q, ctx, pivot);
                if !last {
                    ccol.push(q.mul(ctx, coef(k + 1).half, divided).cell);
                }
                let carried = q.mul(ctx, dcol[k - 1], a.minus_half);
                let d = q.sub(ctx, d, carried.cell);
                dcol.push(q.mul(ctx, d, divided).cell);
            }

            // Back substitution.
            let mut data = dcol[NK - 1];
            for k in (0..NK).rev() {
                if k < NK - 1 {
                    let product = q.mul(ctx, ccol[k], data);
                    data = q.sub(ctx, dcol[k], product.cell);
                }
                let gap = q.sub(ctx, data, pos[at(k)]);
                stage[at(k)] = q.mul(ctx, dtr, gap).cell;
            }
        }
    }
    out.floats(&stage);
    Ok(())
}

/// The coefficients from one sum `s = wcon[i + 1, j, k] + wcon[i, j, k]`
/// (Q32 codes): with `BET_M = BET_P = 0.5`, `cs = ccol = gcv * 0.5` is
/// `half = floor(floor(s / 4) / 2) = floor(s / 8)` and
/// `as_ = acol = gav * 0.5` is `minus_half = floor(floor(-s / 4) / 2) =
/// -ceil(s / 8)`; `half_ceil = ceil(s / 8) = -as_` is the factor of
/// `-as_ * (...)`. (`gcv`, `gav` themselves are only used through these.)
#[derive(Clone, Copy)]
struct Coefs {
    half: Cell,
    minus_half: Cell,
    half_ceil: Cell,
}

/// `s = 8 h + r_0 + 2 r_1 + 4 r_2` with hinted bits `r_b` and `h` in the
/// window: `s` is an integer below `2^65` in magnitude, so the decomposition
/// is unique. Then `floor(s / 8) = h` and `ceil(s / 8) = h + (r != 0)`.
fn split(q: &Q32, ctx: &mut Ctx, s: Cell) -> Coefs {
    let v = crate::q32::signed(s.value());
    // Hints, pinned as above.
    let h = ctx.load_witness(fe(v.div_euclid(8)));
    let bits: Vec<Cell> = (0..3).map(|b| ctx.load_witness(fe((v.rem_euclid(8) >> b) & 1))).collect();
    for bit in &bits {
        q.gate.assert_bit(ctx, *bit);
    }
    q.bound(ctx, h);
    let weights = [8u64, 1, 2, 4].map(|x| Constant(Fr::from(x)));
    let terms = std::iter::once(h).chain(bits.iter().copied()).map(Existing);
    let recomposed = q.gate.inner_product(ctx, terms, weights);
    ctx.constrain_equal(&recomposed, &s);
    let low = q.gate.or(ctx, bits[0], bits[1]);
    let nonzero = q.gate.or(ctx, low, bits[2]);
    let half_ceil = q.add(ctx, h, nonzero);
    let minus_half = q.neg(ctx, half_ceil);
    Coefs { half: h, minus_half, half_ceil }
}

/// `1.0 / x` (`floor(2^64 / x)`), dividing by `|x|` from a hinted sign `t`:
/// `m = x (1 - 2t)` in `[0, 2^64)` pins `t` for `x != 0` (the wrong sign
/// gives `p - |x|`), and with `|x| < 2^64` the decomposition of
/// [`Q32::div_by_abs`] stays unique (`|q x + r| < 2^128`); `x = 0` has no
/// witness. Cheaper than [`Q32::div`]'s window check of `x` and comparison.
fn reciprocal(q: &Q32, ctx: &mut Ctx, x: Cell) -> Cell {
    let negative = crate::q32::signed(x.value()) < 0;
    // Hint, pinned by the range check below.
    let t = ctx.load_witness(Fr::from(negative as u64));
    q.gate.assert_bit(ctx, t);
    let sign = q.gate.mul_add(ctx, t, Constant(-Fr::from(2)), Constant(Fr::ONE));
    let magnitude = q.gate.mul(ctx, x, sign);
    q.range.range_check(ctx, magnitude, 64);
    q.div_by_abs(ctx, Constant(fe(SCALE)), x, magnitude).cell
}
