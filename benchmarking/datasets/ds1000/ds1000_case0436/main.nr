// DS-1000 case 0436, Noir baseline.
// Kernel: `a.max(axis=1, keepdims=1) == a` on Float[10, 5] (exact equality).
// Each row maximum is noir_hw::max_of (a hint pinned by five range checks and
// a product); the mask compares every value with its row maximum.
// @outputs: bool[50]

use noir_hw::max_of;

fn main(a: [Field; 50]) -> pub [bool; 50] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);
    let mut mask = [false; 50];
    for i in 0..10 {
        let max = max_of([a[5 * i], a[5 * i + 1], a[5 * i + 2], a[5 * i + 3], a[5 * i + 4]]);
        for k in 0..5 {
            mask[5 * i + k] = a[5 * i + k] == max;
        }
    }
    mask
}
