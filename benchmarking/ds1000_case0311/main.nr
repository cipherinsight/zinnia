// DS-1000 case 0311 -- Noir baseline.
// np.unravel_index(a.argmax(), a.shape, order='F') on Float[8,6].
// F-order: i = flat % rows, j = flat // rows.

use noir_fpc::qgt;

global ROWS: u32 = 8;

fn main(a: [Field; 48]) -> pub [Field; 2] {
    let mut best_val: Field = a[0];
    let mut best_idx: u32 = 0;
    for k in 1..48 {
        if qgt(a[k], best_val) {
            best_val = a[k];
            best_idx = k;
        }
    }
    let i = best_idx % ROWS;
    let j = best_idx / ROWS;
    [i as Field, j as Field]
}
