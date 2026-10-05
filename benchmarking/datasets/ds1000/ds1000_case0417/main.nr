// DS-1000 case 0417, Noir baseline.
// Kernel: `data[::-1][:(8 // 4) * 4].reshape(-1, 4).mean(axis=1)` with the
// constant bin_size = 4 on Float[8]: the means of the reversed data's bins,
// data[7:3:-1] and data[3::-1].
// A Q32 mean of 4 values is floor(sum / 4) (the quotient
// floor(sum * 2^32 / (4 * 2^32))), a division by the constant 4: a 2-bit
// remainder check and a window check on the quotient.
// @outputs: float[2]

use noir_hw::floor_div_pow2;

global BIN_SIZE: u32 = 4;
global BIN_BITS: u32 = 2; // BIN_SIZE = 2^BIN_BITS
global BINS: u32 = 2; // 8 / BIN_SIZE

fn main(data: [Field; 8]) -> pub [Field; 2] {
    let mut means = [0; BINS];
    for b in 0..BINS {
        let mut total: Field = 0;
        for k in 0..BIN_SIZE {
            total += data[7 - (b * BIN_SIZE + k)];
        }
        means[b] = floor_div_pow2::<BIN_BITS>(total);
    }
    means
}
