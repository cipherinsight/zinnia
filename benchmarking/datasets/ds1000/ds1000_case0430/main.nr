// DS-1000 case 0430 -- Noir baseline. Same as 0428 (k=3).

use noir_fpc::{qadd, qsub, qmul, qdiv, qsqrt, q_from_int};

global N: u32 = 30;

fn main(a: [Field; 30]) -> pub [Field; 2] {
    let mut sum: Field = a[0];
    for k in 1..N { sum = qadd(sum, a[k]); }
    let n_q = q_from_int(N as u64);
    let mean = qdiv(sum, n_q);
    let three_q = q_from_int(3);

    let mut sq_a = {
        let d = qsub(a[0], mean); qmul(d, d)
    };
    for k in 1..N {
        let d = qsub(a[k], mean);
        sq_a = qadd(sq_a, qmul(d, d));
    }
    let std_a = qsqrt(qdiv(sq_a, n_q));

    let mut sq_b = {
        let d = qsub(a[0], mean); qmul(d, d)
    };
    for k in 1..N {
        let d = qsub(a[k], mean);
        sq_b = qadd(sq_b, qmul(d, d));
    }
    let std_b = qsqrt(qdiv(sq_b, n_q));

    let lo = qsub(mean, qmul(three_q, std_a));
    let hi = qadd(mean, qmul(three_q, std_b));
    [lo, hi]
}
