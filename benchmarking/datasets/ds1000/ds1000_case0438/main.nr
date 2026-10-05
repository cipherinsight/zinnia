// DS-1000 case 0438 -- Noir baseline. np.corrcoef(post, distance)[0][1] on Integer[4]x[4].
// Pearson r = cov/(sigma_x * sigma_y) with sigma product = sqrt(varx*vary).

use noir_fpc::{qadd, qsub, qmul, qdiv, qsqrt, q_from_int, Q};

global N: u32 = 4;

fn main(post: [Field; 4], distance: [Field; 4]) -> pub Field {
    let mut xs: [Field; 4] = [0; 4];
    let mut ys: [Field; 4] = [0; 4];
    for k in 0..N {
        xs[k] = post[k] * Q;
        ys[k] = distance[k] * Q;
    }
    let n_q = q_from_int(N as u64);
    let mut sx: Field = xs[0];
    let mut sy: Field = ys[0];
    for k in 1..N {
        sx = qadd(sx, xs[k]);
        sy = qadd(sy, ys[k]);
    }
    let xbar = qdiv(sx, n_q);
    let ybar = qdiv(sy, n_q);
    let mut cov: Field = {
        let dx = qsub(xs[0], xbar);
        let dy = qsub(ys[0], ybar);
        qmul(dx, dy)
    };
    let mut vx: Field = {
        let dx = qsub(xs[0], xbar);
        qmul(dx, dx)
    };
    let mut vy: Field = {
        let dy = qsub(ys[0], ybar);
        qmul(dy, dy)
    };
    for k in 1..N {
        let dx = qsub(xs[k], xbar);
        let dy = qsub(ys[k], ybar);
        cov = qadd(cov, qmul(dx, dy));
        vx = qadd(vx, qmul(dx, dx));
        vy = qadd(vy, qmul(dy, dy));
    }
    let denom = qsqrt(qmul(vx, vy));
    qdiv(cov, denom)
}
