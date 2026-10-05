// DS-1000 case 0431 -- Noir baseline.
// ~np.logical_and(a > lo, a < hi) on Integer[8] with lo/hi = mean +/- 2*std.

use noir_fpc::{qadd, qsub, qmul, qdiv, qsqrt, qlt, q_from_int, Q};

global N: u32 = 8;

fn main(a: [Field; 8]) -> pub [Field; 8] {
    let mut cells: [Field; 8] = [0; 8];
    for k in 0..N {
        cells[k] = a[k] * Q;
    }
    let mut sum: Field = cells[0];
    for k in 1..N { sum = qadd(sum, cells[k]); }
    let n_q = q_from_int(N as u64);
    let mean = qdiv(sum, n_q);
    let mut sq: Field = {
        let d = qsub(cells[0], mean); qmul(d, d)
    };
    for k in 1..N {
        let d = qsub(cells[k], mean);
        sq = qadd(sq, qmul(d, d));
    }
    let var = qdiv(sq, n_q);
    let std = qsqrt(var);
    let two = q_from_int(2);
    let two_std = qmul(two, std);
    let lo = qsub(mean, two_std);
    let hi = qadd(mean, two_std);
    let mut out: [Field; 8] = [0; 8];
    for k in 0..N {
        let gt_lo = qlt(lo, cells[k]);
        let lt_hi = qlt(cells[k], hi);
        let inside = gt_lo & lt_hi;
        out[k] = if inside { 0 } else { 1 };
    }
    out
}
