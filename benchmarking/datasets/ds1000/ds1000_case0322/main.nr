// DS-1000 case 0322, Noir baseline.
// Kernel: `np.argwhere(a == np.min(a))` on Integer[2, 2] (data-dependent length).
// Output: the number of minima, then their (row, col) pairs in row-major
// order, zero-padded to 4 pairs. The minimum is noir_hw::min_of (a hint pinned
// by four range checks and a product). Compaction: noir_hw::compact_rows with
// the constant payload (k // 2, k % 2) for flat index k.
// @outputs: dynamic int[4x2]

use noir_hw::{compact_rows, min_of};

fn main(a: [Field; 4]) -> pub [Field; 9] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);
    let min = min_of(a);
    let mut hit: [Field; 4] = [0; 4];
    let mut w: [[Field; 2]; 4] = [[0; 2]; 4];
    let mut count: Field = 0;
    for k in 0..4 {
        hit[k] = (a[k] == min) as Field;
        w[k] = [hit[k] * ((k / 2) as Field), hit[k] * ((k % 2) as Field)];
        count += hit[k];
    }
    let pairs = compact_rows(hit, w);
    let mut out: [Field; 9] = [0; 9];
    out[0] = count;
    for j in 0..4 {
        out[1 + 2 * j] = pairs[j][0];
        out[2 + 2 * j] = pairs[j][1];
    }
    out
}
