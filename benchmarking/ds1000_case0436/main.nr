// DS-1000 case 0436 -- Noir baseline. (a.max(axis=1, keepdims=1) == a) on Float[10,5].
// Q32 fixed-point. Per row: scan for max, then per-element equality with tol.

use noir_fpc::{qgt, q_close, TOL_DEFAULT};

global ROWS: u32 = 10;
global COLS: u32 = 5;

fn main(a: [Field; 50]) -> pub [bool; 50] {
    let mut out: [bool; 50] = [false; 50];
    for i in 0..ROWS {
        let base = i * COLS;
        let mut row_max = a[base];
        for j in 1..COLS {
            if qgt(a[base + j], row_max) {
                row_max = a[base + j];
            }
        }
        for j in 0..COLS {
            out[base + j] = q_close(a[base + j], row_max, TOL_DEFAULT);
        }
    }
    out
}
