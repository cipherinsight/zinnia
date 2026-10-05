// DS-1000 case 0313 -- Noir baseline. a.argmax() on Float[8,6] (flat).

use noir_fpc::qgt;

fn main(a: [Field; 48]) -> pub Field {
    let mut best_val: Field = a[0];
    let mut best_idx: Field = 0;
    for k in 1..48 {
        if qgt(a[k], best_val) {
            best_val = a[k];
            best_idx = k as Field;
        }
    }
    best_idx
}
