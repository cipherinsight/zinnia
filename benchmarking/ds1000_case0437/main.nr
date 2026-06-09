// DS-1000 case 0437 -- Noir baseline. (a.min(axis=1, keepdims=1) == a) on Float[10,5].
// Mirror of 0436 with row-min in place of row-max.

use noir_fpc::{qlt, q_close, TOL_DEFAULT};

global ROWS: u32 = 10;
global COLS: u32 = 5;

fn main(a: [Field; 50]) -> pub [bool; 50] {
    let mut out: [bool; 50] = [false; 50];
    for i in 0..ROWS {
        let base = i * COLS;
        let mut row_min = a[base];
        for j in 1..COLS {
            if qlt(a[base + j], row_min) {
                row_min = a[base + j];
            }
        }
        for j in 0..COLS {
            out[base + j] = q_close(a[base + j], row_min, TOL_DEFAULT);
        }
    }
    out
}
