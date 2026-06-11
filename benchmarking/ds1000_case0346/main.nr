// DS-1000 case 0346 -- Noir baseline. Field convention (CONVENTIONS.md sec 3).
// a[row-1, :] *= multiply_number; result = cumsum(a[row-1, :]) on Float[8,6].

use noir_fpc::{qmul, qadd, Q};

global ROWS: u32 = 8;
global COLS: u32 = 6;

fn main(a: [Field; 48], row: Field, multiply_number: Field) -> pub [Field; 6] {
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
    assert(hits == 1, "0346: row out of range");
    let mn_fp = multiply_number * Q;
    let mut out: [Field; 6] = [0; 6];
    let mut acc: Field = 0;
    for j in 0..COLS {
        let mut a_at: Field = 0;
        for i in 0..ROWS {
            a_at = a_at + ind[i] * a[i * COLS + j];
        }
        let scaled = qmul(a_at, mn_fp);
        acc = qadd(acc, scaled);
        out[j] = acc;
    }
    out
}
