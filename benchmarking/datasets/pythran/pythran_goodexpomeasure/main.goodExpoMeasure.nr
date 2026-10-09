// pythran_goodexpomeasure__goodExpoMeasure (Pythran goodexpomeasure), Noir
// baseline.
// Kernel: the good-exposure measure of inRGB: Int[C, H, W] (only the planes
// 0, 1, 2 are read, and get a domain check) under sigma != 0 (`@requires`; the
// division has no witness for a zero divisor): per pixel
// e_c = np.exp(-(c - 128) ** 2 / sigma) for the three planes,
// x = e_R * e_G * e_B, and the output
// (np.round(x, 2) * (2 ** 8 - 1)).astype(np.uint8).
// c - 128 is an integer, so its square is exact; the quotient by sigma is
// rounded (one sign test of sigma for all pixels), and exp is the shared
// gadget (noir_hw::exp; an argument outside the window has no proof). The two
// products are rounded in NumPy's order.
// Rounds the Q32 product to hundredths (ties to even), truncates 255*n/100,
// and wraps modulo 256, including negative-sigma inputs.
// LIMITATION: Q32 and float64 can disagree at discrete rounding boundaries;
// float64 (n/100)*255 is also just below an integer for e.g. n=12820.
// See BATCH4_VALIDATION.md for the unresolved full-domain limitation.
// @outputs: int[H * W]

use noir_hw::{check_input, check_inputs, divmod_pos, exp, is_neg, mul, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global C: u32 = 16; // = inRGB_0
global H: u32 = 16; // = inRGB_1
global W: u32 = 16; // = inRGB_2

// The planes the kernel reads (R, G, B), and the centre value, from its text.
global PLANES: u32 = 3;
global CENTRE: Field = 128;
global HALF: Field = 2147483648; // 2^31, one half in Q32

unconstrained fn split_hint(v: Field) -> (Field, Field) {
    let w = v as u128;
    ((w >> 32) as Field, (w & 0xffffffff) as Field)
}

unconstrained fn half_hint(m: Field) -> Field {
    ((m as u128) >> 1) as Field
}

unconstrained fn byte_hint(scaled: Field) -> Field {
    ((scaled as u128) / 100) as Field
}

// rint(100 x) for a Q32 x, for a nonnegative Q32 product:
// 100 x + 2^31 = m 2^32 + f with f in [0, 2^32) and m in [0, 2^64) (unique: the
// right side is below 2^96), so m = floor(100 x + 1/2); at a tie (f = 0) an
// odd m steps down to the even neighbour (m = 2 h + odd, h a 64-bit value).
fn rint_hundredths(x: Field) -> Field {
    let shifted = 100 * x + HALF;
    // Safety: pinned by the range checks and the recomposition.
    let (m, f) = unsafe { split_hint(shifted) };
    m.assert_max_bit_size::<64>();
    f.assert_max_bit_size::<32>();
    assert(shifted == m * SCALE + f, "rint: recomposition");
    // Safety: pinned by the bit check and the range check.
    let h = unsafe { half_hint(m) };
    let odd = m - 2 * h;
    assert(odd * (odd - 1) == 0, "rint: parity");
    h.assert_max_bit_size::<64>();
    let n = if f == 0 { m - odd } else { m };

    n
}

unconstrained fn upper_byte_hint(o: Field) -> Field {
    ((o as u128) >> 8) as Field
}

// Truncate 255*n/100, then wrap modulo 256. Each quotient is unique from
// the bounded remainder and recomposition, with all integers below 2^96.
fn to_byte(n: Field) -> Field {
    let scaled = 255 * n;
    // Safety: pinned by the range checks and the recomposition.
    let o = unsafe { byte_hint(scaled) };
    let s = scaled - 100 * o;
    o.assert_max_bit_size::<64>();
    s.assert_max_bit_size::<8>();
    (99 - s).assert_max_bit_size::<8>();
    // Safety: o = 256*upper + byte with bounded upper and byte is unique.
    let upper = unsafe { upper_byte_hint(o) };
    upper.assert_max_bit_size::<64>();
    let byte = o - 256 * upper;
    byte.assert_max_bit_size::<8>();
    byte
}

fn main(inRGB: [[[Field; W]; H]; C], sigma: Field) -> pub [[Field; W]; H] {
    assert(C >= PLANES, "IndexError");
    // Input domain: the planes read hold int64s, sigma a Q32 code.
    for c in 0..PLANES {
        for i in 0..H {
            check_inputs(inRGB[c][i]);
        }
    }
    check_input(sigma);
    // -(d ** 2) / sigma = floor(-d^2 2^64 / sigma): divide by |sigma|, then
    // restore the sign (as noir_hw::div, with the sign test once).
    let negative = is_neg(sigma);
    let magnitude = if negative { -sigma } else { sigma };
    let mut result: [[Field; W]; H] = [[0; W]; H];
    for i in 0..H {
        for j in 0..W {
            let mut e: [Field; PLANES] = [0; PLANES];
            for c in 0..PLANES {
                let d = inRGB[c][i][j] - CENTRE;
                let (quotient, _) = divmod_pos(-(d * d) * SCALE * SCALE, magnitude);
                e[c] = exp(if negative { -quotient } else { quotient });
            }
            let x = mul(mul(e[0], e[1]), e[2]);
            result[i][j] = to_byte(rint_hundredths(x));
        }
    }
    result
}

#[test]
fn rounding_ties_even_and_byte_wrap() {
    assert(rint_hundredths(SCALE / 8) == 12);
    assert(rint_hundredths(3 * SCALE / 8) == 38);
    assert(to_byte(100) == 255);
    assert(to_byte(101) == 1);
    assert(to_byte(200) == 254);
    assert(to_byte(1000) == 246);
}
