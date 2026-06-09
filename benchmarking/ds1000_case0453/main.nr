// DS-1000 case 0453 -- Noir baseline.
// l2 = sqrt((X*X).sum(axis=-1)); result = X / l2 on Integer[5,4].

use noir_fpc::{qadd, qmul, qdiv, qsqrt, Q};

global ROWS: u32 = 5;
global COLS: u32 = 4;

fn main(x: [Field; 20]) -> pub [Field; 20] {
    let mut x_fp: [Field; 20] = [0; 20];
    for k in 0..20 {
        x_fp[k] = x[k] * Q;
    }
    let mut out: [Field; 20] = [0; 20];
    for i in 0..ROWS {
        let base = i * COLS;
        let mut sq_sum: Field = qmul(x_fp[base], x_fp[base]);
        for j in 1..COLS {
            sq_sum = qadd(sq_sum, qmul(x_fp[base + j], x_fp[base + j]));
        }
        let l2 = qsqrt(sq_sum);
        for j in 0..COLS {
            out[base + j] = qdiv(x_fp[base + j], l2);
        }
    }
    out
}
