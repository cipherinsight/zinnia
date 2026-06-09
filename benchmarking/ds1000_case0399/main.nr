// DS-1000 case 0399 -- Noir baseline. 3-tap IIR.
// B[0] = a*A[0]; B[1] = a*A[1] + b*B[0]; B[k] = a*A[k] + b*B[k-1] + c*B[k-2].

use noir_fpc::{qmul, qadd};

global N: u32 = 8;

fn main(a_arr: [Field; 8], a_scalar: Field, b_scalar: Field,
        c_arr: [Field; 8], claimed: [Field; 8]) -> pub [Field; 8] {
    let _ = claimed[0];
    let mut b: [Field; 8] = [0; 8];
    b[0] = qmul(a_scalar, a_arr[0]);
    b[1] = qadd(qmul(a_scalar, a_arr[1]), qmul(b_scalar, b[0]));
    for k in 2..N {
        let term_a = qmul(a_scalar, a_arr[k]);
        let term_b = qmul(b_scalar, b[k - 1]);
        let term_c = qmul(c_arr[k], b[k - 2]);
        b[k] = qadd(qadd(term_a, term_b), term_c);
    }
    b
}
