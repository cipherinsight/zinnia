// DS-1000 case 0418, Noir baseline.
// Kernel: `data[:, ::-1][:, :(6 // 2) * 2].reshape(8, -1, 2).mean(axis=-1)`
// with the constant bin_size = 2 on Float[8, 6]: row i, bin k is the mean of
// data[i, 5 - 2k] and data[i, 4 - 2k].
// A Q32 mean of 2 values is floor(sum / 2) (the quotient
// floor(sum * 2^32 / (2 * 2^32))), a division by the constant 2: a 1-bit
// remainder check and a window check on the quotient.
// @outputs: float[24]

use noir_hw::OFFSET;

global ROWS: u32 = 8;
global COLS: u32 = 6;
global BIN_SIZE: u32 = 2;
global BIN_BITS: u32 = 1; // BIN_SIZE = 2^BIN_BITS
global BINS: u32 = 3; // COLS / BIN_SIZE

fn main(data: [Field; 48]) -> pub [Field; 24] {
    let mut means = [0; 24];
    for i in 0..ROWS {
        for k in 0..BINS {
            let at = i * COLS + (COLS - 1) - k * BIN_SIZE; // reversed column 2k
            means[i * BINS + k] = floor_div_pow2::<BIN_BITS>(data[at] + data[at - 1]);
        }
    }
    means
}

// floor(n / 2^K) (Python //) for a signed n: n = q * 2^K + r with r in
// [0, 2^K) and q in the window, so |q * 2^K + r| < p / 2, the identity is an
// integer identity and (q, r) is unique. The constant-divisor gadget noir_hw
// lacks: one K-bit remainder check where divmod_pos spends two 64-bit ones.
fn floor_div_pow2<let K: u32>(n: Field) -> Field {
    // Safety: q and r are pinned by the checks below.
    let (q, r) = unsafe { floor_div_pow2_hint::<K>(n) };
    assert(q * (1 << K) as Field + r == n, "floor_div_pow2: recomposition");
    r.assert_max_bit_size::<K>();
    (q + OFFSET).assert_max_bit_size::<64>();
    q
}

// Shifting a signed n (|n| < 2^127) by 2^127, a multiple of 2^K, makes it a
// u128 without changing its remainder; the quotient shifts by 2^(127 - K).
unconstrained fn floor_div_pow2_hint<let K: u32>(n: Field) -> (Field, Field) {
    let shift: Field = 170141183460469231731687303715884105728; // 2^127
    let d = (1 << K) as u128;
    let m = (n + shift) as u128;
    ((m / d) as Field - shift / (d as Field), (m % d) as Field)
}
