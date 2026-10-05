// DS-1000 case 0358 -- Noir baseline.
// index = argsort(a.sum(axis=(1,2))); result = b[index, :, :] on Float[3,3,3]/Int[3,3,3].

use noir_fpc::{qlt, qadd};

global D: u32 = 3;

fn main(a: [Field; 27], b: [Field; 27]) -> pub [Field; 27] {
    let mut sums: [Field; 3] = [0; 3];
    let mut idx: [Field; 3] = [0; 3];
    for i in 0..D {
        let mut s: Field = 0;
        for j in 0..D {
            for k in 0..D {
                s = qadd(s, a[i * D * D + j * D + k]);
            }
        }
        sums[i] = s;
        idx[i] = i as Field;
    }
    for i in 0..D {
        for jj in (i + 1)..D {
            if qlt(sums[jj], sums[i]) {
                let tv = sums[i]; sums[i] = sums[jj]; sums[jj] = tv;
                let ti = idx[i]; idx[i] = idx[jj]; idx[jj] = ti;
            }
        }
    }
    let mut out: [Field; 27] = [0; 27];
    for i in 0..D {
        for m in 0..D {
            let is_eq: Field = if idx[i] == m as Field { 1 } else { 0 };
            for j in 0..D {
                for k in 0..D {
                    out[i * D * D + j * D + k] = out[i * D * D + j * D + k]
                        + b[m * D * D + j * D + k] * is_eq;
                }
            }
        }
    }
    out
}
