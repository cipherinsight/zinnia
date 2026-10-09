// DS-1000 case 0414, Noir LITERAL baseline (LITERAL_BASELINES.md).
// Literal translation of kernel.py:
//   bin_size = 4
//   bin_data_mean = data[:(8 // 4) * 4].reshape(-1, 4).mean(axis=1)
// data.size // bin_size, * bin_size and the reshape bound are Python-level
// constants (shapes), so the slice is all of data and the reshape is exact:
// two bins of 4. mean(axis=1) is a reduction (L5): a running sum per bin,
// then one division by the constant bin_size, using
// noir_hw::floor_div_pow2 (bin_size = 4 = 2^2) -- the shared constant-divisor
// gadget, the arithmetic contract, not an optimization.
// @outputs: float[2]

use noir_hw::floor_div_pow2;

global BIN_SIZE: u32 = 4;
global BIN_BITS: u32 = 2; // BIN_SIZE = 2^BIN_BITS
global BINS: u32 = 2; // 8 / BIN_SIZE

fn main(data: [Field; 8]) -> pub [Field; 2] {
    noir_hw::check_inputs(data);
    let mut means = [0; BINS];
    for b in 0..BINS {
        let mut total: Field = 0;
        for k in 0..BIN_SIZE {
            total = total + data[b * BIN_SIZE + k];
        }
        means[b] = floor_div_pow2::<BIN_BITS>(total);
    }
    means
}
