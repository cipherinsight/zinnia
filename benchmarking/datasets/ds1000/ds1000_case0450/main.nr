// DS-1000 case 0450, Noir baseline.
// Kernel: `a[:, np.array(second).reshape(-1, 1), third]` on Float[7, 8, 9] with
// second, third Integer[2]: result[i, j, k] = a[i, second[j], third[k]], shape
// (7, 2, 2), NumPy index rules (negative indices wrap, others out of range
// raise IndexError: no proof).
// Each index is resolved once (noir_hw::index_pos: a hint pinned by 8-bit
// checks of 2 P and L - 1 - P for the position P on an axis of length L) and
// every output is a native array read at i * 72 + 9 P + Q, in range by
// construction. (A custom resolver with 4-bit checks costs 13 gates more: the
// 4-bit range table is extra, the 8-bit one the 64-bit checks build anyway.)
// @outputs: float[28]

use noir_hw::index_pos;

global D1: u32 = 8;
global D2: u32 = 9;

fn main(a: [Field; 7 * D1 * D2], second: [Field; 2], third: [Field; 2]) -> pub [Field; 28] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);
    noir_hw::check_inputs(second);
    noir_hw::check_inputs(third);
    let p = [index_pos::<8>(second[0], D1), index_pos::<8>(second[1], D1)];
    let q = [index_pos::<8>(third[0], D2), index_pos::<8>(third[1], D2)];
    let mut out: [Field; 28] = [0; 28];
    for j in 0..2 {
        for k in 0..2 {
            // offset of (second[j], third[k]) inside a[i], below 72
            let pq = p[j] * D2 + q[k];
            for i in 0..7 {
                out[i * 4 + j * 2 + k] = a[i * D1 * D2 + pq];
            }
        }
    }
    out
}
