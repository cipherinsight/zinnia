// DS-1000 case 0454 -- Noir baseline. X / np.abs(X).max(axis=1).reshape(-1,1)
// on Integer[5,4] (Linf row normalization, result Float[5,4]).

use noir_fpc::{qabs, qgt, qdiv, Q};

global ROWS: u32 = 5;
global COLS: u32 = 4;

fn main(x: [Field; 20]) -> pub [Field; 20] {
    let mut x_fp: [Field; 20] = [0; 20];
    let mut abs_fp: [Field; 20] = [0; 20];
    for k in 0..20 {
        x_fp[k] = x[k] * Q;
        abs_fp[k] = qabs(x_fp[k]);
    }
    let mut out: [Field; 20] = [0; 20];
    for i in 0..ROWS {
        let base = i * COLS;
        let mut row_max = abs_fp[base];
        for j in 1..COLS {
            if qgt(abs_fp[base + j], row_max) {
                row_max = abs_fp[base + j];
            }
        }
        for j in 0..COLS {
            out[base + j] = qdiv(x_fp[base + j], row_max);
        }
    }
    out
}
