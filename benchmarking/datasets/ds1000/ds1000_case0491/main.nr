// DS-1000 case 0491 -- Noir baseline (faithful np.isclose).
// (~np.isclose(s1, s2)).sum() on Float[6,1] x Float[6,1].
// Faithful to numpy.isclose default tolerance: `|a - b| <= atol + rtol * |b|`
// with atol = 1e-8, rtol = 1e-5. The original baseline used a single coarse
// tolerance (`q_close` with TOL_DEFAULT ~ 1e-3) which is a different
// predicate -- that change rejected fewer pairs for the same fixture.

use noir_fpc::{qabs, qadd, qle, qmul, qsub};

global N: u32 = 6;

// numpy.isclose default in Q32: atol = round(1e-8 * 2^32) = 43,
// rtol = round(1e-5 * 2^32) = 42950.
global ATOL_Q: Field = 43;
global RTOL_Q: Field = 42950;

fn main(s1: [Field; 6], s2: [Field; 6]) -> pub Field {
    let mut count: Field = 0;
    for k in 0..N {
        let d = qsub(s1[k], s2[k]);
        let abs_d = qabs(d);
        let abs_s2 = qabs(s2[k]);
        let thr = qadd(ATOL_Q, qmul(RTOL_Q, abs_s2));
        let close = qle(abs_d, thr);
        count = count + (if close { 0 } else { 1 });
    }
    count
}
