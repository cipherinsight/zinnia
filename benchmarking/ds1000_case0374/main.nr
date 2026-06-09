// DS-1000 case 0374 -- Noir baseline. ecdf interpolation.
// Sort grades, evaluate ECDF at 5 query integers.
// Q32 fixed-point.

use noir_fpc::{qlt, qdiv, qmul, q_from_int, Q};

global N: u32 = 50;
global M: u32 = 5;

fn main(grades: [Field; 50], eval: [Field; 5]) -> pub [Field; 5] {
    let mut xs: [Field; 50] = grades;
    for i in 0..N {
        for j in (i + 1)..N {
            if qlt(xs[j], xs[i]) {
                let t = xs[i]; xs[i] = xs[j]; xs[j] = t;
            }
        }
    }
    let n_q = q_from_int(N as u64);
    let mut ys: [Field; 50] = [0; 50];
    for k in 0..N {
        ys[k] = qdiv(q_from_int((k + 1) as u64), n_q);
    }
    let one_q = q_from_int(1);
    let mut out: [Field; 5] = [0; 5];
    for i in 0..M {
        // eval[i] is integer; promote to Q.
        let e = eval[i] * Q;
        let lt_lo = qlt(e, xs[0]);
        let lt_hi = qlt(e, xs[N - 1]);
        let ge_hi = !lt_hi;
        // First k with xs[k] > e.
        let mut found: bool = false;
        let mut best: u32 = 0;
        for k in 0..N {
            let hit = qlt(e, xs[k]);
            if hit & !found {
                best = k;
                found = true;
            }
        }
        // interp = ys[best-1]; if best==0 fallback to ys[0] (only used when not lt_lo/ge_hi).
        let mut interp: Field = ys[0];
        for k in 1..N {
            if best == k {
                interp = ys[k - 1];
            }
        }
        let v_hi: Field = if ge_hi { one_q } else { interp };
        let v: Field = if lt_lo { 0 } else { v_hi };
        out[i] = v;
    }
    out
}
