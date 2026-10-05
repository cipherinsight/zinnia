// DS-1000 case 0326, Noir baseline.
// Kernel: `np.degrees(np.arcsin(value))` for a Float value.
// No proof unless -1 <= value <= 1 (NumPy returns NaN outside).
// arcsin has no exact gadget. The prover supplies y = round(arcsin(value) *
// 2^32); the circuit checks -pi/2 <= y <= pi/2 and |S(y) - value| <= EPS,
// where S(y) = y (1 + u (c3 + u (c5 + u (c7 + u c9)))), u = y^2, is a degree-9
// odd polynomial fitted to sin on [-pi/2, pi/2] (minimax error 4.6e-9; the
// library's degree-14 polynomial is meant for np.sin on [0, pi] and costs 14
// rescales here instead of 6). With its Q32 coefficients S is within 7.3e-9
// of sin on [-pi/2, pi/2], and flooring its 6 products adds at most 40 ulp;
// pointwise the two stay below 1.5e-8, so EPS = 2e-8 accepts the honest y.
// Every accepted y has |sin(y) - value| <= 3.5e-8: it lies within
// 3.5e-8 / sqrt(1 - (|value| + 3.5e-8)^2) rad of arcsin(value), and never
// more than sqrt(2 * 3.5e-8) = 2.7e-4 rad (0.015 degrees, reached at
// |value| = 1). degrees = y * (180 / pi), one rescaled product.
// Range checks use the 64- and 32-bit widths noir_hw already uses (each other
// width costs its own lookup table).
// @outputs: float[1]

use noir_hw::{mul, poly, SCALE};

global HALF_PI_Q: Field = 6746518852; // round(pi / 2 * 2^32)
global EPS_Q: Field = 86; // round(2e-8 * 2^32)
global DEG_PER_RAD_Q: Field = 246083499208; // round(180 / pi * 2^32)
// Q32 coefficients c9, c7, c5, c3, 1 of S (highest first).
global SIN9_Q: [Field; 5] = [11167, -850688, 35790037, -715827472, 4294967296];

fn main(value: Field) -> pub Field {
    // -1 <= value <= 1: value + 1 lies in [0, 2].
    let shifted = value + SCALE;
    shifted.assert_max_bit_size::<64>();
    assert((shifted as u64) <= 2 * 4294967296, "arcsin: |value| > 1");

    // Safety: y is pinned to the window above by y + pi/2 in [0, pi] and
    // S(y) - value + EPS in [0, 2 EPS].
    let y = unsafe { asin_hint(value) };
    let y_shifted = y + HALF_PI_Q;
    y_shifted.assert_max_bit_size::<64>();
    assert((y_shifted as u64) <= 2 * 6746518852, "arcsin: |y| > pi/2");
    let s = mul(y, poly(mul(y, y), SIN9_Q));
    let gap = s - value + EPS_Q;
    gap.assert_max_bit_size::<32>();
    assert((gap as u64) <= 2 * 86, "arcsin: sin(y) differs from value");

    mul(y, DEG_PER_RAD_Q)
}

// round(arcsin(value) * 2^32), by bisection on an accurate sine (Taylor series
// in 2^-60 fixed point) for |value|, then the sign of value.
unconstrained fn asin_hint(value: Field) -> Field {
    let (neg, mag) = split(value);
    let mag = if mag > 4294967296 { 4294967296 } else { mag }; // |value| > 1 has no proof anyway
    let target = mag << 28; // |value| in 2^-60 units
    let half_pi: u128 = 6746518852;
    // Largest m in [0, pi/2] (Q32) with sin(m) <= |value|.
    let mut lo: u128 = 0;
    let mut hi: u128 = half_pi;
    for _ in 0..34 {
        if lo < hi {
            let mid = (lo + hi + 1) / 2;
            if sin_fx60(mid << 28) <= target {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
    }
    // Round to nearest: step up when arcsin(|value|) >= m + 1/2.
    let m = if (lo < half_pi) & (sin_fx60((2 * lo + 1) << 27) <= target) { lo + 1 } else { lo };
    if neg { -(m as Field) } else { m as Field }
}

// sin(x) for x in [0, pi/2], both in 2^-60 fixed point.
unconstrained fn sin_fx60(x: u128) -> u128 {
    let x2 = (x * x) >> 60;
    let mut term = x;
    let mut plus = x;
    let mut minus: u128 = 0;
    for k in 1..13 {
        let k = k as u128;
        term = ((term * x2) >> 60) / (2 * k * (2 * k + 1));
        if k % 2 == 1 {
            minus += term;
        } else {
            plus += term;
        }
    }
    plus - minus
}

// (negative, magnitude) of a Field encoding a signed integer below 2^127.
unconstrained fn split(x: Field) -> (bool, u128) {
    let pos: u128 = x as u128;
    if (pos as Field) == x {
        (false, pos)
    } else {
        (true, (-x) as u128)
    }
}
