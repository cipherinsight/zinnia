// DS-1000 case 0347 -- Noir baseline. Field convention (CONVENTIONS.md sec 3).
// a[row-1, :] /= divide_number; result = np.multiply.reduce(a[row-1, :])
// on Float[8,6].

use noir_fpc::{qmul, qdiv, Q};

global ROWS: u32 = 8;
global COLS: u32 = 6;

fn main(a: [Field; 48], row: Field, divide_number: Field) -> pub Field {
    let row_idx = row - 1;
    // Full numpy semantics: negative row indices wrap. Dual-equality
    // indicators + membership-sum bound (see 0345).
    let mut ind: [Field; 8] = [0; 8];
    let mut hits: Field = 0;
    for i in 0..ROWS {
        let pos_match: Field = if row_idx == (i as Field) { 1 } else { 0 };
        let neg_match: Field =
            if row_idx == ((i as Field) - (ROWS as Field)) { 1 } else { 0 };
        ind[i] = pos_match + neg_match;
        hits = hits + ind[i];
    }
    assert(hits == 1, "0347: row out of range");
    let dn_fp = divide_number * Q;
    let mut prod: Field = Q;  // multiplicative identity in Q-scale
    for j in 0..COLS {
        let mut a_at: Field = 0;
        for i in 0..ROWS {
            a_at = a_at + ind[i] * a[i * COLS + j];
        }
        let v = qdiv(a_at, dn_fp);
        prod = qmul(prod, v);
    }
    prod
}
