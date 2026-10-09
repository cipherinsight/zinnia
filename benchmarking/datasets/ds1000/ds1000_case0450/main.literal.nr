// DS-1000 case 0450: literal translation of kernel.py (LITERAL_BASELINES.md).
// Broadcast gather is evaluated axis by axis, using native checked indexing.
// @outputs: float[28]

use noir_hw::index_pos;

global D1: u32 = 8;
global D2: u32 = 9;

fn main(a: [Field; 7 * D1 * D2], second: [Field; 2], third: [Field; 2]) -> pub [Field; 28] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);
    noir_hw::check_inputs(second);
    noir_hw::check_inputs(third);

    let mut out: [Field; 28] = [0; 28];
    for i in 0..7 {
        for j in 0..2 {
            // row = a[i, second[j], :]
            let p = index_pos::<8>(second[j], D1);
            let mut row: [Field; 9] = [0; 9];
            for p2 in 0..9 {
                row[p2] = a[i * D1 * D2 + p * D2 + p2];
            }
            for k in 0..2 {
                let qpos = index_pos::<8>(third[k], D2);
                out[i * 4 + j * 2 + k] = row[qpos];
            }
        }
    }
    out
}
