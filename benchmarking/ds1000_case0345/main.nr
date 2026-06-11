// DS-1000 case 0345 -- Noir baseline. Field convention (CONVENTIONS.md sec 3).
// a[:, col-1] *= multiply_number; result = cumsum(a[:, col-1]) on Float[8,6].

use noir_fpc::{qmul, qadd, Q};

global ROWS: u32 = 8;
global COLS: u32 = 6;

fn main(a: [Field; 48], col: Field, multiply_number: Field) -> pub [Field; 8] {
    let col_idx = col - 1;
    // Full numpy semantics: negative indices wrap (col = 0 selects the
    // last column). Exact dual-equality identity per slot:
    // match(j) = (idx == j) | (idx == j - COLS) -- no signed-comparison
    // machinery. The membership sum == 1 enforces -COLS <= idx < COLS
    // (numpy IndexError otherwise). Indicators hoisted across rows.
    let mut ind: [Field; 6] = [0; 6];
    let mut hits: Field = 0;
    for j in 0..COLS {
        let pos_match: Field = if col_idx == (j as Field) { 1 } else { 0 };
        let neg_match: Field =
            if col_idx == ((j as Field) - (COLS as Field)) { 1 } else { 0 };
        ind[j] = pos_match + neg_match;
        hits = hits + ind[j];
    }
    assert(hits == 1, "0345: col out of range");
    let mn_fp = multiply_number * Q;
    let mut acc: Field = 0;
    let mut out: [Field; 8] = [0; 8];
    for i in 0..ROWS {
        let mut a_at: Field = 0;
        for j in 0..COLS {
            a_at = a_at + ind[j] * a[i * COLS + j];
        }
        let scaled = qmul(a_at, mn_fp);
        acc = qadd(acc, scaled);
        out[i] = acc;
    }
    out
}
