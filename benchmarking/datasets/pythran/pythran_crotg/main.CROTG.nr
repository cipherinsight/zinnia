// pythran_crotg__CROTG (Pythran crotg), Noir baseline.
// Kernel: BLAS CROTG on complex scalars CA, CB. If abs(CA) == 0: C = 0,
// S = 1, CA = CB. Else SCALE = |CA| + |CB|,
// NORM = SCALE * sqrt(|CA / SCALE| ** 2 + |CB / SCALE| ** 2),
// ALPHA = CA / |CA|, C = |CA| / NORM, S = ALPHA * conj(CB) / NORM,
// CA = ALPHA * NORM. Returns (CA, CB, C, S). The inputs C and S are
// overwritten before they are read, so they are not parameters here.
// Q32 magnitude and complex/real division retain every primitive rounding.
// Branch on rounded abs(CA)==0; retain all original inactive products.
// Conjugate CB before rounding the four real products.
// @outputs: float[7]

use noir_hw::{check_inputs, divmod_pos, mul, sqrt, SCALE};

fn abs_complex(x: [Field; 2]) -> Field {
    sqrt(mul(x[0], x[0]) + mul(x[1], x[1]))
}

// floor(v * 2^32 / d) for a positive d.
fn div_by_positive(v: Field, d: Field, active: bool) -> Field {
    assert(!active | (d != 0), "active division by zero");
    let zero = d == 0;
    let (q, _) = divmod_pos((if zero { 0 } else { v }) * SCALE, d + zero as Field);
    q
}

fn complex_div(x: [Field; 2], d: Field, active: bool) -> [Field; 2] {
    let denominator = mul(d, d);
    [div_by_positive(mul(x[0], d), denominator, active), div_by_positive(mul(x[1], d), denominator, active)]
}

fn main(CA: [Field; 2], CB: [Field; 2]) -> pub ([Field; 2], [Field; 2], Field, [Field; 2]) {
    // Input domain: every input element is a Q32 code in the window.
    check_inputs(CA);
    check_inputs(CB);
    let ca = CA;
    let ca_abs = abs_complex(ca);
    let zero = ca_abs == 0;
    let active = !zero;
    let cb_abs = abs_complex(CB);
    let scale = ca_abs + cb_abs;
    let a = abs_complex(complex_div(ca, scale, active));
    let b = abs_complex(complex_div(CB, scale, active));
    let norm = mul(scale, sqrt(mul(a, a) + mul(b, b)));
    let alpha = complex_div(ca, ca_abs, active);
    let c = div_by_positive(ca_abs, norm, active);
    // alpha * conj(CB) = (ar br + ai bi) + (ai br - ar bi) i
    let s_re = mul(alpha[0], CB[0]) - mul(alpha[1], -CB[1]);
    let s_im = mul(alpha[0], -CB[1]) + mul(alpha[1], CB[0]);
    let s = complex_div([s_re, s_im], norm, active);
    let ca_out = [mul(alpha[0], norm), mul(alpha[1], norm)];
    if zero {
        (CB, CB, 0, [SCALE, 0])
    } else {
        (ca_out, CB, c, s)
    }
}
