// DS-1000 case 0312, Noir baseline.
// Kernel: `np.unravel_index(a.argmax(), a.shape)` on Float[8, 6]: with k the flat
// index of the first maximum, the coordinates are (k // 6, k % 6).
// @outputs: int[2]

use noir_hw::window;

fn main(a: [Field; 48]) -> pub [Field; 2] {
    let mut best = window(a[0]);
    let mut best_idx: u32 = 0;
    for k in 1..48 {
        let w = window(a[k]);
        if w > best {
            best = w;
            best_idx = k;
        }
    }
    [(best_idx / 6) as Field, (best_idx % 6) as Field]
}
