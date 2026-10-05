// DS-1000 case 0355 -- Noir baseline.
// sort_indices = np.argsort(a, axis=0) on Float[3,3,3];
// c = b[sort_indices, static_indices[1], static_indices[2]] on Integer[3,3,3].
// For each (j, k), sort the 3 values a[:, j, k] ascending and gather b along axis-0.

use noir_fpc::qlt;

global D: u32 = 3;

fn main(a: [Field; 27], b: [Field; 27]) -> pub [Field; 27] {
    let mut out: [Field; 27] = [0; 27];
    for j in 0..D {
        for k in 0..D {
            // Pull the 3 values along axis-0 with their original layer index.
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
            // Gather b[layer[i], j, k].
            for i in 0..D {
                // layer[i] in {0,1,2}; select via indicator.
                let mut v: Field = 0;
                for m in 0..D {
                    let is_eq: Field = if layer[i] == m as Field { 1 } else { 0 };
                    v = v + b[m * D * D + j * D + k] * is_eq;
                }
                out[i * D * D + j * D + k] = v;
            }
        }
    }
    out
}
