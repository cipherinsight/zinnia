// DS-1000 case 0334 -- Noir baseline. result = mean of three Float[50] vectors.
// Q32 fixed-point per CONVENTIONS.md: every Field value encodes value * 2^32.
// Per cell: out[k] = (a[k] + b[k] + c[k]) / 3.

use noir_fpc::{qadd, qdiv, q_from_int};

fn main(a: [Field; 50], b: [Field; 50], c: [Field; 50]) -> pub [Field; 50] {
    let three = q_from_int(3);
    let mut out: [Field; 50] = [0; 50];
    for k in 0..50 {
        let ab = qadd(a[k], b[k]);
        let sum = qadd(ab, c[k]);
        out[k] = qdiv(sum, three);
    }
    out
}
