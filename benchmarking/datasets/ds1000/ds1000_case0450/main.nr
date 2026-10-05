// DS-1000 case 0450 -- Noir baseline. Field convention (CONVENTIONS.md sec 3).
// a[:, second.reshape(-1,1), third] on Float[7,8,9] -> (7, 2, 2).

global D0: u32 = 7;
global D1: u32 = 8;
global D2: u32 = 9;

fn main(a: [Field; 504], second: [Field; 2], third: [Field; 2]) -> pub [Field; 28] {
    let mut out: [Field; 28] = [0; 28];
    for i in 0..D0 {
        for p in 0..2 {
            for q in 0..2 {
                let target = (i * D1 * D2) as Field + second[p] * (D2 as Field) + third[q];
                let mut v: Field = 0;
                for k in 0..(D0 * D1 * D2) {
                    if (k as Field) == target { v = a[k]; }
                }
                out[i * 4 + p * 2 + q] = v;
            }
        }
    }
    out
}
