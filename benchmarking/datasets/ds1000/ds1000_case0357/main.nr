// DS-1000 case 0357 -- Noir baseline. Same as 0355 but with reversed sort indices ([::-1]).

use noir_fpc::qlt;

global D: u32 = 3;

fn main(a: [Field; 27], b: [Field; 27]) -> pub [Field; 27] {
    let mut out: [Field; 27] = [0; 27];
    for j in 0..D {
        for k in 0..D {
            let mut vals: [Field; 3] = [0; 3];
            let mut layer: [Field; 3] = [0; 3];
            for i in 0..D {
                vals[i] = a[i * D * D + j * D + k];
                layer[i] = i as Field;
            }
            for i in 0..D {
                for jj in (i + 1)..D {
                    if qlt(vals[jj], vals[i]) {
                        let tv = vals[i]; vals[i] = vals[jj]; vals[jj] = tv;
                        let tl = layer[i]; layer[i] = layer[jj]; layer[jj] = tl;
                    }
                }
            }
            // Reverse: rev[i] = layer[D-1-i].
            for i in 0..D {
                let li = layer[D - 1 - i];
                let mut v: Field = 0;
                for m in 0..D {
                    let is_eq: Field = if li == m as Field { 1 } else { 0 };
                    v = v + b[m * D * D + j * D + k] * is_eq;
                }
                out[i * D * D + j * D + k] = v;
            }
        }
    }
    out
}
