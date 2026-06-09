// DS-1000 case 0381 -- Noir baseline. np.argsort(a)[::-1][:len(a)] on Float[100].
// Descending argsort.

use noir_fpc::qlt;

global N: u32 = 100;

fn main(a: [Field; 100]) -> pub [Field; 100] {
    let mut vals: [Field; 100] = a;
    let mut idx: [Field; 100] = [0; 100];
    for k in 0..N {
        idx[k] = k as Field;
    }
    for i in 0..N {
        for j in (i + 1)..N {
            if qlt(vals[j], vals[i]) {
                let tv = vals[i]; vals[i] = vals[j]; vals[j] = tv;
                let ti = idx[i]; idx[i] = idx[j]; idx[j] = ti;
            }
        }
    }
    let mut out: [Field; 100] = [0; 100];
    for k in 0..N {
        out[k] = idx[N - 1 - k];
    }
    out
}
