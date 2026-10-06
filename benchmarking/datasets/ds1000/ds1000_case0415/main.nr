// DS-1000 case 0415, Noir baseline.
// Kernel: `data[:(8 // 4) * 4].reshape(-1, 4).max(axis=1)` with the constant
// bin_size = 4 on Float[8]: the maxima of data[0:4] and data[4:8].
// Each maximum is noir_hw::max_of: a hint pinned by four range checks and a
// product (cheaper than windowing the four values and comparing three times).
// @outputs: float[2]

use noir_hw::max_of;

fn main(data: [Field; 8]) -> pub [Field; 2] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(data);
    [max_of([data[0], data[1], data[2], data[3]]), max_of([data[4], data[5], data[6], data[7]])]
}
