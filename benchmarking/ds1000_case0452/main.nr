// DS-1000 case 0452 -- Noir baseline. X / np.abs(X).sum(axis=1).reshape(-1, 1)
// on Integer[5,4] (L1 row normalization, result Float[5,4]).
// Q32 fixed-point: input X is integer; promoted to Q-scaled by multiplying by Q.

use noir_fpc::{qadd, qabs, qdiv, Q};

global ROWS: u32 = 5;
global COLS: u32 = 4;

fn main(x: [Field; 20]) -> pub [Field; 20] {
    // Promote int -> Q-scaled float.
    let mut x_fp: [Field; 20] = [0; 20];
    let mut abs_fp: [Field; 20] = [0; 20];
    for k in 0..20 {
        x_fp[k] = x[k] * Q;
        abs_fp[k] = qabs(x_fp[k]);
    }
    let mut out: [Field; 20] = [0; 20];
    for i in 0..ROWS {
        let base = i * COLS;
        let mut l1 = abs_fp[base];
        for j in 1..COLS {
            l1 = qadd(l1, abs_fp[base + j]);
        }
        for j in 0..COLS {
            out[base + j] = qdiv(x_fp[base + j], l1);
        }
    }
    out
}
