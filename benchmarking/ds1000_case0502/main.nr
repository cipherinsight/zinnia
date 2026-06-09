// DS-1000 case 0502 -- Noir baseline. Same as 0501. Field convention.

global D0: u32 = 13;
global D1: u32 = 13;
global D2: u32 = 2;

fn main(a: [Field; 338], b: [Field; 169]) -> pub [Field; 169] {
    let mut out: [Field; 169] = [0; 169];
    for i in 0..D0 {
        for j in 0..D1 {
            let target = (i * D1 * D2) as Field + (j * D2) as Field + b[i * D1 + j];
            let mut v: Field = 0;
            for k in 0..(D0 * D1 * D2) {
                if (k as Field) == target { v = a[k]; }
            }
            out[i * D1 + j] = v;
        }
    }
    out
}
