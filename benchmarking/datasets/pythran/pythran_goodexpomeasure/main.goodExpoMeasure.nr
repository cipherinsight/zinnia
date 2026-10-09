// Source goodExpo operation order with canonical Q32 exp and rounding.
// Materialize rounded hundredths, multiply255, truncate, then uint8 wrap.
// No rational floor(255*n/100) shortcut.
// @outputs: int[H * W]

use noir_hw::{check_input, check_inputs, divmod_pos, canonical_exp, is_neg, mul, to_int, SCALE};

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

fn main(inRGB: [[[Field; W]; H]; C], sigma: Field) -> pub [[Field; W]; H] {
    assert(C >= PLANES, "IndexError");
    // Input domain: the planes read hold int64s, sigma a Q32 code.
    for c in 0..PLANES {
        for i in 0..H {
            check_inputs(inRGB[c][i]);
        }
    }
    check_input(sigma);
    // Preserve the MulF check after FloatCast/SubF before dividing by sigma.
    let negative = is_neg(sigma);
    let magnitude = if negative { -sigma } else { sigma };
    let mut result: [[Field; W]; H] = [[0; W]; H];
    for i in 0..H {
        for j in 0..W {
            let mut e: [Field; PLANES] = [0; PLANES];
            for c in 0..PLANES {
                let d = inRGB[c][i][j] * SCALE - CENTRE * SCALE;
                let square = mul(d, d);
                let (quotient, _) = divmod_pos(-square * SCALE, magnitude);
                e[c] = canonical_exp(if negative { -quotient } else { quotient });
            }
            let x = mul(mul(e[0], e[1]), e[2]);
            let n = rint_hundredths(x);
            let (rounded, _) = divmod_pos(n * SCALE, 100);
            let integer = to_int(mul(rounded, 255 * SCALE));
            let (_, byte) = divmod_pos(integer, 256);
            result[i][j] = byte;
        }
    }
    result
}
