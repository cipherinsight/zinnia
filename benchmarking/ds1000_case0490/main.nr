// DS-1000 case 0490 -- Noir baseline.
// per-layer threshold transform on Float[5,4,4]. n1, n2 are Integer[5].
// Mirrors Zinnia's lowering ALGORITHM: all masks are evaluated on the
// row COPY (temp = a.copy()) with shared sign/magnitude projections --
// four comparison families per row (lt t1, ge t2, ge t1, lt t2) -- and
// the three masked assignments are applied as three sequential
// select passes, exactly as the kernel writes them.

use noir_fpc::{qge_signed, qlt_signed, to_signed, Q};

global L: u32 = 5;
global MN: u32 = 16;

fn main(arr: [Field; 80], n1: [Field; 5], n2: [Field; 5]) -> pub [Field; 80] {
    let thirty: Field = 30 * Q;
    let five: Field = 5 * Q;
    let mut out: [Field; 80] = [0; 80];
    for i in 0..L {
        let t1 = n1[i] * Q;
        let t2 = n2[i] * Q;
        let (s1, m1) = to_signed(t1);
        let (s2, m2) = to_signed(t2);
        // temp = a.copy(): project each element once; every mask below
        // reads these frozen projections.
        let mut sgn: [bool; 16] = [false; 16];
        let mut mag: [u128; 16] = [0; 16];
        let mut x: [Field; 16] = [0; 16];
        for k in 0..MN {
            let (s, m) = to_signed(arr[i * MN + k]);
            sgn[k] = s;
            mag[k] = m;
            x[k] = arr[i * MN + k];
        }
        // pass 1: a[temp < t1] = 0
        for k in 0..MN {
            let m: Field = if qlt_signed(sgn[k], mag[k], s1, m1) { 1 } else { 0 };
            x[k] = m * 0 + (1 - m) * x[k];
        }
        // pass 2: a[temp >= t2] = 30
        for k in 0..MN {
            let m: Field = if qge_signed(sgn[k], mag[k], s2, m2) { 1 } else { 0 };
            x[k] = m * thirty + (1 - m) * x[k];
        }
        // pass 3: a[(temp >= t1) & (temp < t2)] += 5
        for k in 0..MN {
            let ma: Field = if qge_signed(sgn[k], mag[k], s1, m1) { 1 } else { 0 };
            let mb: Field = if qlt_signed(sgn[k], mag[k], s2, m2) { 1 } else { 0 };
            let m: Field = ma * mb;
            x[k] = m * (x[k] + five) + (1 - m) * x[k];
        }
        for k in 0..MN {
            out[i * MN + k] = x[k];
        }
    }
    out
}
