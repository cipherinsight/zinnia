// DS-1000 case 0373 -- Noir baseline. ecdf: sort grades, output ys = arange(1,n+1)/n.
// Q32 fixed-point. The sort produces xs (50-element selection-sort network)
// but the output ys depends only on length (= 50). We still emit the sort
// to mirror the kernel.py algorithm faithfully (matches the halo2 baseline).

use noir_fpc::{qdiv, qlt, q_from_int};

fn main(grades: [Field; 50]) -> pub [Field; 50] {
    let mut xs: [Field; 50] = grades;
    for i in 0..50 {
        for j in (i + 1)..50 {
            let swap = qlt(xs[j], xs[i]);
            if swap {
                let t = xs[i];
                xs[i] = xs[j];
                xs[j] = t;
            }
        }
    }
    // Touch xs so it isn't optimized out.
    let _last = xs[49];

    let n_q = q_from_int(50);
    let mut ys: [Field; 50] = [0; 50];
    for i in 0..50 {
        ys[i] = qdiv(q_from_int((i + 1) as u64), n_q);
    }
    ys
}
