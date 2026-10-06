// DS-1000 case 0321, Noir baseline.
// Kernel: `np.unravel_index(a.argmax(), a.shape)` on Float[5, 6]: with k the flat
// index of the first maximum, the coordinates are (k // 6, k % 6).
// @outputs: int[2]

use noir_hw::window;

fn main(a: [Field; 30]) -> pub [Field; 2] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);
    let mut best = window(a[0]);
    let mut best_idx: u32 = 0;
    for k in 1..30 {
        let w = window(a[k]);
        if w > best {
            best = w;
            best_idx = k;
        }
    }
    [(best_idx / 6) as Field, (best_idx % 6) as Field]
}
