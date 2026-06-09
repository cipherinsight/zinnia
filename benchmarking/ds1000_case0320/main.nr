// DS-1000 case 0320 -- Noir baseline.
// np.unravel_index(a.argmin(), a.shape) on Float[5,6].

use noir_fpc::qlt;

global COLS: u32 = 6;

fn main(a: [Field; 30]) -> pub [Field; 2] {
    let mut best_val: Field = a[0];
    let mut best_idx: u32 = 0;
    for k in 1..30 {
        if qlt(a[k], best_val) {
            best_val = a[k];
            best_idx = k;
        }
    }
    [(best_idx / COLS) as Field, (best_idx % COLS) as Field]
}
