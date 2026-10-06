// DS-1000 case 0314, Noir baseline.
// Kernel: on Float[8, 6], idx = unravel(a.argmax()), a[idx] = a.min(), then
// unravel(a.argmax()) again (C order).
// @outputs: int[2]

use noir_hw::window;

fn main(a: [Field; 48]) -> pub [Field; 2] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);
    let mut w: [u64; 48] = [0; 48];
    for k in 0..48 {
        w[k] = window(a[k]);
    }
    let mut best = w[0];
    let mut low = w[0];
    let mut best_idx: u32 = 0;
    for k in 1..48 {
        if w[k] > best {
            best = w[k];
            best_idx = k;
        }
        if w[k] < low {
            low = w[k];
        }
    }
    // Comparisons only need the order-preserving windows, so the write
    // a[idx] = a.min() is done on them.
    w[best_idx] = low;
    let mut best2 = w[0];
    let mut best2_idx: u32 = 0;
    for k in 1..48 {
        if w[k] > best2 {
            best2 = w[k];
            best2_idx = k;
        }
    }
    [(best2_idx / 6) as Field, (best2_idx % 6) as Field]
}
