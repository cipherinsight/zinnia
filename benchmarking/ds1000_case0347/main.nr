// DS-1000 case 0347 -- Noir baseline. Field convention (CONVENTIONS.md sec 3).
// a[row-1, :] /= divide_number; result = np.multiply.reduce(a[row-1, :])
// on Float[8,6].

use noir_fpc::{qmul, qdiv, Q};

global ROWS: u32 = 8;
global COLS: u32 = 6;

fn main(a: [Field; 48], row: Field, divide_number: Field) -> pub Field {
    let row_idx = row - 1;
    assert(((row_idx as u128) < (ROWS as u128)), "0347: row out of range");
    let dn_fp = divide_number * Q;
    let mut prod: Field = Q;  // multiplicative identity in Q-scale
    for j in 0..COLS {
        // Per-column mux to fetch a[row_idx, j].
        let mut a_at: Field = 0;
        for i in 0..ROWS {
            let is_match: Field = if (i as Field) == row_idx { 1 } else { 0 };
            a_at = a_at + is_match * a[i * COLS + j];
        }
        let v = qdiv(a_at, dn_fp);
        prod = qmul(prod, v);
    }
    prod
}
