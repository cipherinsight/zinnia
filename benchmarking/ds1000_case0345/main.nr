// DS-1000 case 0345 -- Noir baseline. Field convention (CONVENTIONS.md sec 3).
// a[:, col-1] *= multiply_number; result = cumsum(a[:, col-1]) on Float[8,6].

use noir_fpc::{qmul, qadd, Q};

global ROWS: u32 = 8;
global COLS: u32 = 6;

fn main(a: [Field; 48], col: Field, multiply_number: Field) -> pub [Field; 8] {
    let col_idx = col - 1;
    assert(((col_idx as u128) < (COLS as u128)), "0345: col out of range");
    let mn_fp = multiply_number * Q;
    let mut acc: Field = 0;
    let mut out: [Field; 8] = [0; 8];
    for i in 0..ROWS {
        // Dynamic column index -> per-row mux over COLS, mirroring the
        // Zinnia emitter's pattern for runtime indexing.
        let mut a_at: Field = 0;
        for j in 0..COLS {
            let is_match: Field = if (j as Field) == col_idx { 1 } else { 0 };
            a_at = a_at + is_match * a[i * COLS + j];
        }
        let scaled = qmul(a_at, mn_fp);
        acc = qadd(acc, scaled);
        out[i] = acc;
    }
    out
}
