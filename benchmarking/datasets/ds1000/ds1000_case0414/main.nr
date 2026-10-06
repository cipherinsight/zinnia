// DS-1000 case 0414, Noir baseline.
// Kernel: `data[:(8 // 4) * 4].reshape(-1, 4).mean(axis=1)` with the constant
// bin_size = 4 on Float[8]: the means of data[0:4] and data[4:8].
// A Q32 mean of 4 values is floor(sum / 4) (the quotient
// floor(sum * 2^32 / (4 * 2^32))), a division by the constant 4: a 2-bit
// remainder check and a window check on the quotient.
// @outputs: float[2]

use noir_hw::floor_div_pow2;

global BIN_SIZE: u32 = 4;
global BIN_BITS: u32 = 2; // BIN_SIZE = 2^BIN_BITS
global BINS: u32 = 2; // 8 / BIN_SIZE

fn main(data: [Field; 8]) -> pub [Field; 2] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(data);
    let mut means = [0; BINS];
    for b in 0..BINS {
        let mut total: Field = 0;
        for k in 0..BIN_SIZE {
            total += data[b * BIN_SIZE + k];
        }
        means[b] = floor_div_pow2::<BIN_BITS>(total);
    }
    means
}
