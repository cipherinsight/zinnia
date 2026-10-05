// DS-1000 case 0309 -- Noir baseline. argmax of Float[8,6] (returned as flat index).
// Q32 fixed-point: floats arrive as signed Field values = v * 2^32.
// argmax is monotonic-invariant so we can use the signed Q-compare directly.

use noir_fpc::qgt;

fn main(a: [Field; 48]) -> pub Field {
    let mut best_val: Field = a[0];
    let mut best_idx: Field = 0;
    for k in 1..48 {
        let is_new_max = qgt(a[k], best_val);
        if is_new_max {
            best_val = a[k];
            best_idx = k as Field;
        }
    }
    best_idx
}
