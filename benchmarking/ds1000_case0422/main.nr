// DS-1000 case 0422 -- Noir baseline.
// np.correlate(a, hstack((b[1:], b)), 'valid') on Float[50] x Float[50].
// c has length 99: c[0..48] = b[1..49], c[49..98] = b[0..49].
// out[k] = sum_n a[n] * c[n + 49 - k] for k in 0..50.

use noir_fpc::{qadd, qmul};

global N: u32 = 50;

fn main(a: [Field; 50], b: [Field; 50]) -> pub [Field; 50] {
    let mut c: [Field; 99] = [0; 99];
    for i in 0..49 { c[i] = b[i + 1]; }
    for i in 49..99 { c[i] = b[i - 49]; }
    let mut out: [Field; 50] = [0; 50];
    for k in 0..N {
        let mut s: Field = qmul(a[0], c[49 - k]);
        for n in 1..N {
            s = qadd(s, qmul(a[n], c[n + 49 - k]));
        }
        out[k] = s;
    }
    out
}
