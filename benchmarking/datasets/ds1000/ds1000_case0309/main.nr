// DS-1000 case 0309, Noir baseline.
// Kernel: `a.argmax()` on Float[8, 6]: the flat index of the first maximum.
// @outputs: int[1]

use noir_hw::window;

fn main(a: [Field; 48]) -> pub Field {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);
    let mut best = window(a[0]);
    let mut best_idx: Field = 0;
    for k in 1..48 {
        // Strictly greater, so ties keep the first index (NumPy's rule).
        let w = window(a[k]);
        if w > best {
            best = w;
            best_idx = k as Field;
        }
    }
    best_idx
}
