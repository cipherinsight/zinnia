// npbench_weather_stencils_vadv_vadv__vadv (NPBench weather_stencils/vadv),
// Noir baseline.
// Kernel: vertical advection, a tridiagonal solve along k for every column
// (i, j) of utens_stage: Float[NI, NJ, NK] (in place; the output is
// utens_stage after the call, `mutated_outputs`), from u_stage, u_pos, utens
// (same shape), wcon: Float[NI + 1, NJ, NK] and the scalar dtr_stage: a
// forward sweep computing ccol, dcol with one reciprocal 1.0 / (...) per
// level, then back substitution. Every statement is elementwise over (i, j),
// so each column is solved on its own, with NumPy's rounding: every product
// and every reciprocal is rounded.
// The coefficients come from s = wcon[i + 1, j, k] + wcon[i, j, k] (k >= 1;
// wcon[:, :, 0] is never read, so it gets no domain check): gcv = 0.25 * s,
// gav = -0.25 * s (the next level's), and their products with
// BET_M = BET_P = 0.5, which are equal (cs = ccol, as_ = acol). All four are
// floors of s / 4 and s / 8, so they come from one decomposition s = 8 h + r
// with the bits of r (`split`) instead of four rounded products. Reciprocal
// uses the frozen nonzero division domain, including its signed endpoint.
// @outputs: float[NI * NJ * NK]

use noir_hw::{canonical_div_nonzero, check_input, check_inputs, mul, window, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global NI: u32 = 15; // = utens_stage_0
global NJ: u32 = 15; // = utens_stage_1
global NK: u32 = 10; // = utens_stage_2

// The coefficients from one sum s (Q32 codes): with BET_M = BET_P = 0.5,
// cs = ccol = gcv * 0.5 is half = floor(floor(s / 4) / 2) = floor(s / 8) and
// as_ = acol = gav * 0.5 is -half_ceil = floor(floor(-s / 4) / 2)
// = -ceil(s / 8); half_ceil = -as_ is the factor of -as_ * (...).
struct Coefs {
    half: Field,
    half_ceil: Field,
}

unconstrained fn split_hint(s: Field) -> (Field, [Field; 3]) {
    // floor(s / 8) and the bits of s mod 8, s in the 65-bit signed range.
    let shifted = (s + 2.pow_32(66)) as u128;
    let h = (shifted >> 3) as Field - 2.pow_32(63);
    let r = shifted & 7;
    (h, [(r & 1) as Field, ((r >> 1) & 1) as Field, ((r >> 2) & 1) as Field])
}

// s = 8 h + r_0 + 2 r_1 + 4 r_2 with bits r_b and h in the window: s is an
// integer below 2^65 in magnitude, so the decomposition is unique. Then
// floor(s / 8) = h and ceil(s / 8) = h + (r != 0).
fn split(s: Field) -> Coefs {
    // Safety: pinned by the bit checks, the window check and the recomposition.
    let (h, r) = unsafe { split_hint(s) };
    for b in 0..3 {
        assert(r[b] * (r[b] - 1) == 0, "split: not a bit");
    }
    let _ = window(h);
    assert(s == 8 * h + r[0] + 2 * r[1] + 4 * r[2], "split: recomposition");
    let low = r[0] + r[1] - r[0] * r[1];
    let nonzero = low + r[2] - low * r[2];
    Coefs { half: h, half_ceil: h + nonzero }
}

fn reciprocal(x: Field) -> Field {
    canonical_div_nonzero(SCALE, x)
}

fn main(
    utens_stage: [[[Field; NK]; NJ]; NI],
    u_stage: [[[Field; NK]; NJ]; NI],
    wcon: [[[Field; NK]; NJ]; NI + 1],
    u_pos: [[[Field; NK]; NJ]; NI],
    utens: [[[Field; NK]; NJ]; NI],
    dtr_stage: Field,
) -> pub [[[Field; NK]; NJ]; NI] {
    // Input domain: every element read is a Q32 code in the window.
    for i in 0..NI {
        for j in 0..NJ {
            check_inputs(utens_stage[i][j]);
            check_inputs(u_stage[i][j]);
            check_inputs(u_pos[i][j]);
            check_inputs(utens[i][j]);
        }
    }
    for i in 0..NI + 1 {
        for j in 0..NJ {
            for k in 1..NK {
                check_input(wcon[i][j][k]);
            }
        }
    }
    check_input(dtr_stage);
    let dtr = dtr_stage;
    let mut result = utens_stage;
    for i in 0..NI {
        for j in 0..NJ {
            let u = u_stage[i][j];
            let pos = u_pos[i][j];
            let tens = utens[i][j];
            let stage = utens_stage[i][j];
            // coefs[k]: the coefficients of s_k, k >= 1 (coefs[0] unused).
            let mut coefs: [Coefs; NK] = [Coefs { half: 0, half_ceil: 0 }; NK];
            for k in 1..NK {
                coefs[k] = split(wcon[i + 1][j][k] + wcon[i][j][k]);
            }
            let mut ccol: [Field; NK] = [0; NK];
            let mut dcol: [Field; NK] = [0; NK];

            // k = 0.
            let c = coefs[1];
            let bcol = dtr - c.half;
            let correction = mul(-c.half, u[1] - u[0]);
            let d = mul(dtr, pos[0]) + tens[0] + stage[0] + correction;
            let divided = reciprocal(bcol);
            ccol[0] = mul(c.half, divided);
            dcol[0] = mul(d, divided);

            // 0 < k < NK - 1, and k = NK - 1 (no ccol, no upper term).
            for k in 1..NK {
                let a = coefs[k];
                let minus_half = -a.half_ceil;
                let mut bcol = dtr - minus_half;
                let mut correction = mul(a.half_ceil, u[k - 1] - u[k]);
                if k < NK - 1 {
                    let c = coefs[k + 1];
                    bcol -= c.half;
                    correction -= mul(c.half, u[k + 1] - u[k]);
                }
                let d = mul(dtr, pos[k]) + tens[k] + stage[k] + correction;
                let divided = reciprocal(bcol - mul(ccol[k - 1], minus_half));
                if k < NK - 1 {
                    ccol[k] = mul(coefs[k + 1].half, divided);
                }
                dcol[k] = mul(d - mul(dcol[k - 1], minus_half), divided);
            }

            // Back substitution.
            let mut data = dcol[NK - 1];
            result[i][j][NK - 1] = mul(dtr, data - pos[NK - 1]);
            for t in 1..NK {
                let k = NK - 1 - t;
                data = dcol[k] - mul(ccol[k], data);
                result[i][j][k] = mul(dtr, data - pos[k]);
            }
        }
    }
    result
}
