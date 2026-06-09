// DS-1000 case 0490 -- Noir baseline.
// per-layer threshold transform on Float[5,4,4]. n1, n2 are Integer[5].

use noir_fpc::{qadd, qlt, Q};

global L: u32 = 5;
global M: u32 = 4;
global N: u32 = 4;

fn main(arr: [Field; 80], n1: [Field; 5], n2: [Field; 5]) -> pub [Field; 80] {
    let mut out: [Field; 80] = [0; 80];
    for i in 0..L {
        let t1 = n1[i] * Q;
        let t2 = n2[i] * Q;
        let thirty: Field = 30 * Q;
        let five: Field = 5 * Q;
        for j in 0..M {
            for k in 0..N {
                let idx = i * M * N + j * N + k;
                let v = arr[idx];
                let lt1 = qlt(v, t1);
                let ge_t2 = !qlt(v, t2);
                let new_v: Field = if lt1 { 0 } else { if ge_t2 { thirty } else { qadd(v, five) } };
                out[idx] = new_v;
            }
        }
    }
    out
}
