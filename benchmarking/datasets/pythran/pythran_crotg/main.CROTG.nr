// pythran_crotg__CROTG (Pythran crotg), Noir baseline.
// Kernel: BLAS CROTG on complex scalars CA, CB. If abs(CA) == 0: C = 0,
// S = 1, CA = CB. Else SCALE = |CA| + |CB|,
// NORM = SCALE * sqrt(|CA / SCALE| ** 2 + |CB / SCALE| ** 2),
// ALPHA = CA / |CA|, C = |CA| / NORM, S = ALPHA * conj(CB) / NORM,
// CA = ALPHA * NORM. Returns (CA, CB, C, S). The inputs C and S are
// overwritten before they are read, so they are not parameters here.
// abs of a complex value is the exact floor square root of re^2 + im^2
// (`isqrt` of the raw sum: one rounding, as abs rounds hypot once). A complex
// value divided or multiplied by a real one is componentwise;
// ALPHA * conj(CB) is a complex product, each part one rescale. Both branches
// are computed and selected; the else branch runs on CA + [CA == 0] (that is,
// 1 when CA is 0), so its divisors are positive for every input. CA == 0 is
// re^2 + im^2 == 0, the same test as abs(CA) == 0. The divisors |CA|, SCALE,
// NORM are positive, so each quotient is `divmod_pos` of v * 2^32 by the
// divisor (no sign test). Where the rounded NORM is 0 (both magnitudes a few
// units of 2^-32) there is no proof.
// @outputs: float[7]

use noir_hw::{check_inputs, divmod_pos, isqrt, mul, rescale, sqrt, SCALE};

fn abs_complex(x: [Field; 2]) -> Field {
    isqrt(x[0] * x[0] + x[1] * x[1])
}

// floor(v * 2^32 / d) for a positive d.
fn div_by_positive(v: Field, d: Field) -> Field {
    let (q, _) = divmod_pos(v * SCALE, d);
    q
}

fn main(CA: [Field; 2], CB: [Field; 2]) -> pub ([Field; 2], [Field; 2], Field, [Field; 2]) {
    // Input domain: every input element is a Q32 code in the window.
    check_inputs(CA);
    check_inputs(CB);
    let zero = CA[0] * CA[0] + CA[1] * CA[1] == 0;
    // The else branch's CA: 1 when CA is 0.
    let ca = [CA[0] + zero as Field * SCALE, CA[1]];
    let ca_abs = abs_complex(ca);
    let cb_abs = abs_complex(CB);
    let scale = ca_abs + cb_abs;
    let a = abs_complex([div_by_positive(ca[0], scale), div_by_positive(ca[1], scale)]);
    let b = abs_complex([div_by_positive(CB[0], scale), div_by_positive(CB[1], scale)]);
    let norm = mul(scale, sqrt(mul(a, a) + mul(b, b)));
    let alpha = [div_by_positive(ca[0], ca_abs), div_by_positive(ca[1], ca_abs)];
    let c = div_by_positive(ca_abs, norm);
    // alpha * conj(CB) = (ar br + ai bi) + (ai br - ar bi) i
    let s_re = rescale(alpha[0] * CB[0] + alpha[1] * CB[1]);
    let s_im = rescale(alpha[1] * CB[0] - alpha[0] * CB[1]);
    let s = [div_by_positive(s_re, norm), div_by_positive(s_im, norm)];
    let ca_out = [mul(alpha[0], norm), mul(alpha[1], norm)];
    if zero {
        (CB, CB, 0, [SCALE, 0])
    } else {
        (ca_out, CB, c, s)
    }
}
