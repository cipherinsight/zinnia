//! DS-1000 case 0458: upper triangle of pairwise Euclidean distances.
// output: [q32]
use crate::q32::{Q32, Q32Trait, ZERO};

#[executable]
fn main(a: Array<Q32>) -> Array<Q32> {
    assert!(a.len() == 8 * 6, "a has the wrong length");
    let mut out = array![];
    for i in 0_u32..8 {
        for j in 0_u32..8 {
            if j < i {
                out.append(ZERO);
                continue;
            }
            let mut squared = ZERO;
            for k in 0_u32..6 {
                let delta = *a.at(j * 6 + k) - *a.at(i * 6 + k);
                squared = squared + delta * delta;
            }
            out.append(squared.sqrt());
        }
    }
    out
}
