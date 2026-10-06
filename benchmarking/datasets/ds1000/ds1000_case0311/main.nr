// DS-1000 case 0311, Noir baseline.
// Kernel: `np.unravel_index(a.argmax(), a.shape, order='F')` on Float[8, 6]:
// with k the C-order flat argmax, the Fortran-order coordinates are (k % 8, k // 8).
// @outputs: int[2]

use noir_hw::window;

fn main(a: [Field; 48]) -> pub [Field; 2] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);
    let mut best = window(a[0]);
    let mut best_idx: u32 = 0;
    for k in 1..48 {
        let w = window(a[k]);
        if w > best {
            best = w;
            best_idx = k;
        }
    }
    [(best_idx % 8) as Field, (best_idx / 8) as Field]
}
