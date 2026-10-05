// DS-1000 case 0382 -- Noir baseline. np.argsort(a) on Float[100].
// Selection sort tracking indices.

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
    idx
}
