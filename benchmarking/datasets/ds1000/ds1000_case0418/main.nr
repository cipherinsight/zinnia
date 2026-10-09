// DS-1000 case 0418: optimized baseline.
// Reverse the six columns, reshape into pairs, and compute means in row order.
// Slice/reversal/reshape are wiring; each mean sums then floor-divides by two.
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
    // new_data = data[:, ::-1]
    let mut new_data = [0; 48];
    for row in 0..8 {
        for col in 0..6 { new_data[row * 6 + col] = data[row * 6 + 5 - col]; }
    }
    let mut means = [0; 24];
    for i in 0..ROWS {
        for k in 0..BINS {
            let at = i * COLS + k * BIN_SIZE;
            means[i * BINS + k] = floor_div_pow2::<BIN_BITS>(new_data[at] + new_data[at + 1]);
        }
    }
    means
}
