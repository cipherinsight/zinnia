// DS-1000 case 0346 -- Noir baseline. Field convention (CONVENTIONS.md sec 3).
// a[row-1, :] *= multiply_number; result = cumsum(a[row-1, :]) on Float[8,6].

use noir_fpc::{qmul, qadd, Q};

global ROWS: u32 = 8;
global COLS: u32 = 6;

fn main(a: [Field; 48], row: Field, multiply_number: Field) -> pub [Field; 6] {
    let row_idx = row - 1;
    assert(((row_idx as u128) < (ROWS as u128)), "0346: row out of range");
    let mn_fp = multiply_number * Q;
    let mut out: [Field; 6] = [0; 6];
    let mut acc: Field = 0;
    for j in 0..COLS {
        // Dynamic row index -> per-column mux over ROWS.
        let mut a_at: Field = 0;
        for i in 0..ROWS {
            let is_match: Field = if (i as Field) == row_idx { 1 } else { 0 };
            a_at = a_at + is_match * a[i * COLS + j];
        }
        let scaled = qmul(a_at, mn_fp);
        acc = qadd(acc, scaled);
        out[j] = acc;
    }
    out
}
