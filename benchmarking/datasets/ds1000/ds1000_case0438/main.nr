// DS-1000 case 0438, Noir baseline.
// Kernel: `np.corrcoef(post, distance)[0][1]` on two Integer[4].
// Pearson r = Sxy / sqrt(Sxx * Syy) on centred sums (NumPy's population or
// sample factors cancel). Scaled by n^2 = 16 the centred sums are the exact
// integers A = 4 sum(xy) - sum(x) sum(y), B = 4 sum(x^2) - sum(x)^2 and
// C = 4 sum(y^2) - sum(y)^2, so r = A / sqrt(B * C): one Q32 square root of the
// integer B * C and one float division. |r| <= 1 holds exactly (Cauchy-Schwarz
// on the integers, floor square root), so NumPy's clip to [-1, 1] never acts.
// No proof when a variance is 0 (NumPy returns nan) or when B * C >= 2^62,
// where sqrt(B * C) leaves the Q32 window (inputs within +-11585 always fit).
// B and C are range-checked to 64 bits so that B * C cannot wrap the field.
// @outputs: float[1]

use noir_hw::{divmod_pos, from_int, sqrt, SCALE};

fn main(post: [Field; 4], distance: [Field; 4]) -> pub Field {
    let mut sx: Field = 0;
    let mut sy: Field = 0;
    let mut sxy: Field = 0;
    let mut sxx: Field = 0;
    let mut syy: Field = 0;
    for i in 0..4 {
        let (x, y) = (post[i], distance[i]);
        sx += x;
        sy += y;
        sxy += x * y;
        sxx += x * x;
        syy += y * y;
    }
    let a = 4 * sxy - sx * sy;
    let b = 4 * sxx - sx * sx;
    let c = 4 * syy - sy * sy;
    b.assert_max_bit_size::<64>();
    c.assert_max_bit_size::<64>();
    // floor(sqrt(B * C) * 2^32); a zero denominator has no quotient witness.
    let denom = sqrt(from_int(b * c));
    // floor(A * 2^32 * 2^32 / denom), the Q32 quotient of float(A) by it.
    divmod_pos(from_int(a) * SCALE, denom).0
}
