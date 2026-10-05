// DS-1000 case 0459 -- Noir baseline. mean(NA.astype(float)).
// a is unused; NA is integer, cast to Q-scale.

use noir_fpc::{qadd, qdiv, q_from_int, Q};

global N: u32 = 5;

fn main(a: [Field; 5], na: [Field; 5]) -> pub Field {
    let _ = a[0];
    let mut sum: Field = na[0] * Q;
    for k in 1..N {
        sum = qadd(sum, na[k] * Q);
    }
    qdiv(sum, q_from_int(N as u64))
}
