// DS-1000 case 0492 -- Noir baseline (faithful np.isclose, equal_nan=True).
// Same as 0491; with real-valued Q32 inputs there are no NaN witnesses, so
// equal_nan reduces to the same predicate. Uses numpy.isclose's full
// tolerance formula `|a - b| <= atol + rtol * |b|`.

use noir_fpc::{qabs, qadd, qle, qmul, qsub};

global N: u32 = 6;

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
