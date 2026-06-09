// DS-1000 case 0319 -- Noir baseline.
// c = inverse_permutation(perm); result = a[c, :, :] on Float[8,6,8], perm[8].

global D0: u32 = 8;
global D1: u32 = 6;
global D2: u32 = 8;

fn main(a: [Field; 384], permutation: [Field; 8]) -> pub [Field; 384] {
    let mut c: [Field; 8] = [0; 8];
    for k in 0..D0 {
        let mut acc: Field = 0;
        for j in 0..D0 {
            let is_eq: Field = if permutation[j] == k as Field { 1 } else { 0 };
            acc = acc + (j as Field) * is_eq;
        }
        c[k] = acc;
    }
    let mut out: [Field; 384] = [0; 384];
    for i in 0..D0 {
        for r in 0..D1 {
            for s in 0..D2 {
                let mut v: Field = 0;
                for k in 0..D0 {
                    let is_eq: Field = if c[i] == k as Field { 1 } else { 0 };
                    v = v + a[k * (D1 * D2) + r * D2 + s] * is_eq;
                }
                out[i * (D1 * D2) + r * D2 + s] = v;
            }
        }
    }
    out
}
