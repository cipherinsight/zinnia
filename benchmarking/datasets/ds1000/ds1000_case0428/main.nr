// DS-1000 case 0428 -- Noir baseline. (a.mean() - 3*a.std(), a.mean() + 3*a.std())
// on Float[30]. Q32 fixed-point. kernel.py recomputes a.std() inside the
// tuple expression, so we emit two std computations here too. std uses
// qsqrt; result tolerance loosened to 5e-3 for Newton truncation in qsqrt.

use noir_fpc::{qadd, qsub, qmul, qdiv, qsqrt, q_from_int};

global N: u32 = 30;

fn main(a: [Field; 30]) -> pub [Field; 2] {
    // mean = sum / N
    let mut sum: Field = a[0];
    for k in 1..N {
        sum = qadd(sum, a[k]);
    }
    let n_q = q_from_int(N as u64);
    let mean = qdiv(sum, n_q);

    // std = sqrt(sum((x-mean)^2) / N). Computed twice to mirror kernel.py.
    let three_q = q_from_int(3);
    let mut sq_sum_a = {
        let d = qsub(a[0], mean);
        qmul(d, d)
    };
    for k in 1..N {
        let d = qsub(a[k], mean);
        sq_sum_a = qadd(sq_sum_a, qmul(d, d));
    }
    let var_a = qdiv(sq_sum_a, n_q);
    let std_a = qsqrt(var_a);

    let mut sq_sum_b = {
        let d = qsub(a[0], mean);
        qmul(d, d)
    };
    for k in 1..N {
        let d = qsub(a[k], mean);
        sq_sum_b = qadd(sq_sum_b, qmul(d, d));
    }
    let var_b = qdiv(sq_sum_b, n_q);
    let std_b = qsqrt(var_b);

    let lo = qsub(mean, qmul(three_q, std_a));
    let hi = qadd(mean, qmul(three_q, std_b));
    [lo, hi]
}
