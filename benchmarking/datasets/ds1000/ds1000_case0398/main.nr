// DS-1000 case 0398 -- Noir baseline.
// B[0] = a*A[0]; B[k] = a*A[k] + b*B[k-1] for k>=1. N=8 sample.
// a_scalar and b_scalar are floats (Q-scaled).

use noir_fpc::{qmul, qadd};

global N: u32 = 8;

fn main(a_arr: [Field; 8], a_scalar: Field, b_scalar: Field, claimed: [Field; 8]) -> pub [Field; 8] {
    let _ = claimed[0];
    let mut b: [Field; 8] = [0; 8];
    b[0] = qmul(a_scalar, a_arr[0]);
    for k in 1..N {
        let term_a = qmul(a_scalar, a_arr[k]);
        let term_b = qmul(b_scalar, b[k - 1]);
        b[k] = qadd(term_a, term_b);
    }
    b
}
