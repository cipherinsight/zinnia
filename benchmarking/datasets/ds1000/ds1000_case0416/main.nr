// DS-1000 case 0416, Noir baseline.
// Kernel: `data[:, :(6 // 2) * 2].reshape(8, -1, 2).mean(axis=-1)` with the
// constant bin_size = 2 on Float[8, 6]: row i, bin k is the mean of
// data[i, 2k] and data[i, 2k + 1].
// A Q32 mean of 2 values is floor(sum / 2) (the quotient
// floor(sum * 2^32 / (2 * 2^32))), a division by the constant 2: a 1-bit
// remainder check and a window check on the quotient.
// @outputs: float[24]

use noir_hw::floor_div_pow2;

global ROWS: u32 = 8;
global COLS: u32 = 6;
global BIN_SIZE: u32 = 2;
global BIN_BITS: u32 = 1; // BIN_SIZE = 2^BIN_BITS
global BINS: u32 = 3; // COLS / BIN_SIZE

fn main(data: [Field; 48]) -> pub [Field; 24] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(data);
    let mut means = [0; 24];
    for i in 0..ROWS {
        for k in 0..BINS {
            let at = i * COLS + k * BIN_SIZE;
            means[i * BINS + k] = floor_div_pow2::<BIN_BITS>(data[at] + data[at + 1]);
        }
    }
    means
}
