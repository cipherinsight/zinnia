// DS-1000 case 0417: literal translation of kernel.py (LITERAL_BASELINES.md).
// Reverse all eight inputs, reshape into two bins of four, and compute each mean.
// Reversal/slice/reshape are wiring; means sum in index order then floor-divide by four.
// @outputs: float[2]
use noir_hw::floor_div_pow2;

global BIN_SIZE: u32 = 4;
global BIN_BITS: u32 = 2; // BIN_SIZE = 2^BIN_BITS
global BINS: u32 = 2; // 8 / BIN_SIZE

fn main(data: [Field; 8]) -> pub [Field; 2] {
    noir_hw::check_inputs(data);
    // new_data = data[::-1]; reversal and reshape are wiring.
    let mut new_data = [0; 8];
    for i in 0..8 { new_data[i] = data[7 - i]; }
    let mut means = [0; BINS];
    for b in 0..BINS {
        let mut total: Field = 0;
        for k in 0..BIN_SIZE {
            total = total + new_data[b * BIN_SIZE + k];
        }
        means[b] = floor_div_pow2::<BIN_BITS>(total);
    }
    means
}
