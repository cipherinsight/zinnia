// DS-1000 case 0326, Noir baseline.
// Kernel: `np.degrees(np.arcsin(value))` for a Float value.
// No proof unless -1 <= value <= 1 (NumPy returns NaN outside).
// arcsin(v) = 2 atan(t) with t = v / (1 + sqrt(1 - v^2)), |t| <= 1: the square
// root is the exact integer root of 2^64 - v_q^2, t a positive-divisor
// division, atan(t) = t P(t^2) with P a degree-9 Chebyshev fit of atan(t)/t on
// [0, 1] (fit error 2.1e-9 rad), and degrees = atan(t) * 360/pi. Every step is
// a pinned function of the input, so the output is unique; measured error
// against NumPy below 2e-7 degrees on [-1, 1].
// @outputs: float[1]

use noir_hw::{divmod_pos, isqrt, mul, poly, SCALE};

// round(c * 2^32) of P(u) = atan(t)/t, u = t^2, highest degree first.
global ATAN_U_Q: [Field; 10] = [
    -7750000, 47082270, -134277928, 249279572, -361114211, 470446531, -612650751, 858930835,
    -1431654192, 4294967292,
];
global DEGREES_PER_HALF_RADIAN_Q: Field = 492166998415; // round(360 / pi * 2^32)

fn main(value: Field) -> pub Field {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_input(value);
    // -1 <= value <= 1: value + 1 lies in [0, 2] (as a Q32 integer).
    let shifted = value + SCALE;
    shifted.assert_max_bit_size::<64>();
    (2 * SCALE - shifted).assert_max_bit_size::<64>();

    let root = isqrt(SCALE * SCALE - value * value); // sqrt(1 - v^2), exact floor
    let (t, _) = divmod_pos(value * SCALE, root + SCALE);
    let p = poly(mul(t, t), ATAN_U_Q);
    let atan = mul(t, p);
    mul(atan, DEGREES_PER_HALF_RADIAN_Q)
}
